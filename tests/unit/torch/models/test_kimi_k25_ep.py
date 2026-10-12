# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Expert-parallel adapter for Kimi-K2.7 (``kimi_k25``), a remote-code VLM on DeepSeek-V3.

The adapter rebuilds the language model as transformers' native deepseek_v3 and installs the
on-the-fly expert decompressor. The checkpoint's remote code is not available here, so the tests
stub the dynamic-module loader with a stand-in remote module. Every process-wide patch is undone
after each test.
"""

import sys
import types
from types import SimpleNamespace

import pytest

transformers = pytest.importorskip("transformers")

from packaging.version import Version

if Version(transformers.__version__) < Version("5.8"):
    pytest.skip(
        "Kimi-K2.5 expert parallelism needs transformers >= 5.8, for the decompression",
        allow_module_level=True,
    )

import transformers.dynamic_module_utils as dmu
from transformers import DeepseekV3Config, DeepseekV3ForCausalLM, PretrainedConfig

from modelopt.torch.models import _onfly_decompress, get_spec
from modelopt.torch.models.kimi_k25 import ep as kimi_k25

QUANT = {"quant_method": "compressed-tensors", "format": "pack-quantized"}
REMOTE = "transformers_modules.Kimi_hyphen_K2_dot_7.0123abcd"  # how the loader names remote code


class _RemoteTextConfig(PretrainedConfig):
    """Stands in for the remote DeepSeek-V3 config Kimi-K2.7 bundles as its ``text_config``."""

    model_type = "kimi_k2_remote_text"


def _vlm_config():
    text = _RemoteTextConfig(
        hidden_size=64,
        num_hidden_layers=2,
        n_routed_experts=8,
        moe_intermediate_size=16,
        auto_map={"AutoConfig": "configuration_deepseek.DeepseekV3Config"},
        quantization_config=QUANT.copy(),
    )
    return SimpleNamespace(text_config=text, quantization_config=QUANT.copy())


@pytest.fixture
def remote(monkeypatch):
    """Stand in for the checkpoint's remote code: importing it registers ``modeling_kimi_k25``,
    whose ``DeepseekV3ForCausalLM`` is the per-expert remote language model."""
    module = types.ModuleType(f"{REMOTE}.modeling_kimi_k25")
    module.DeepseekV3ForCausalLM = type("DeepseekV3ForCausalLM", (), {})
    module.loads = []

    def import_remote_class(class_ref, ckpt_path, **kwargs):
        module.loads.append((class_ref, ckpt_path, kwargs))
        monkeypatch.setitem(sys.modules, module.__name__, module)

    monkeypatch.setattr(dmu, "get_class_from_dynamic_module", import_remote_class)
    monkeypatch.setattr(dmu, "get_cached_module_file", dmu.get_cached_module_file)
    return module


def test_prepare_rebuilds_the_language_model_config_as_native_deepseek_v3(monkeypatch, remote):
    """The text config becomes transformers' own ``DeepseekV3Config`` (no remote ``auto_map``, no
    stale ``quantization_config``) with an EP plan for the fused experts; the decompressor is
    installed while the VLM config still carries the compressed-tensors scheme; the remote-code
    staging shim is left installed; without a checkpoint nothing remote is imported."""
    installs = []
    monkeypatch.setattr(
        _onfly_decompress,
        "install_onfly_decompress",
        lambda cfg: installs.append(getattr(cfg, "quantization_config", None)),
    )
    staging = dmu.get_cached_module_file
    config = _vlm_config()

    assert kimi_k25.prepare(config) is config

    text = config.text_config
    assert type(text) is DeepseekV3Config
    assert getattr(text, "auto_map", None) is None
    assert getattr(text, "quantization_config", None) is None
    assert (text.hidden_size, text.n_routed_experts, text.moe_intermediate_size) == (64, 8, 16)
    assert text.base_model_ep_plan["layers.*.mlp.experts.gate_up_proj"] == "grouped_gemm"
    assert text.base_model_ep_plan["layers.*.mlp.experts.down_proj"] == "grouped_gemm"
    assert installs == [QUANT]
    assert dmu.get_cached_module_file is not staging
    assert remote.loads == []


def test_prepare_with_a_checkpoint_points_the_remote_modeling_at_native_deepseek_v3(
    monkeypatch, remote
):
    """``KimiK25ForConditionalGeneration.__init__`` builds ``DeepseekV3ForCausalLM(text_config)``
    from its own module's namespace; that name must now be transformers' native class, so the
    language model gets fused, EP-capable experts. Other remote modules keep their own class."""
    monkeypatch.setattr(_onfly_decompress, "install_onfly_decompress", lambda cfg: None)
    bystander = types.ModuleType(f"{REMOTE}.modeling_deepseek")
    remote_lm = bystander.DeepseekV3ForCausalLM = type("DeepseekV3ForCausalLM", (), {})
    monkeypatch.setitem(sys.modules, bystander.__name__, bystander)

    kimi_k25.prepare(_vlm_config(), ckpt_path="/ckpt", trust_remote_code=True)

    assert remote.loads == [
        ("modeling_kimi_k25.KimiK25ForConditionalGeneration", "/ckpt", {"trust_remote_code": True})
    ]
    assert remote.DeepseekV3ForCausalLM is DeepseekV3ForCausalLM
    assert bystander.DeepseekV3ForCausalLM is remote_lm


def test_the_swap_fails_loudly_when_the_remote_modeling_did_not_load(monkeypatch):
    """Skipping the swap silently would build the per-expert layout that EP cannot shard."""
    monkeypatch.setattr(dmu, "get_class_from_dynamic_module", lambda *args, **kwargs: None)

    with pytest.raises(RuntimeError, match="modeling_kimi_k25"):
        kimi_k25._patch_kimi_modeling_to_native("/ckpt", True)


def test_prepare_rejects_a_config_without_a_text_config():
    with pytest.raises(ValueError, match="text_config"):
        kimi_k25.prepare(SimpleNamespace())


def test_prepare_refuses_transformers_own_port():
    """From 5.13 a load without ``trust_remote_code`` builds transformers' port, which the patch
    never reaches; ``prepare`` must refuse it rather than run unpatched."""
    native = pytest.importorskip("transformers.models.kimi_k25.configuration_kimi_k25")
    with pytest.raises(ValueError, match="trust_remote_code=True"):
        kimi_k25.prepare(native.Kimi_K25Config(), ckpt_path="/ckpt")


def test_the_kimi_k25_spec_routes_expert_parallelism_through_this_adapter():
    """The checkpoint's code builds no EP-capable experts, so its whole EP setup is ``prepare``."""
    spec = get_spec("kimi_k25")
    assert spec.ep_spec.prepare is kimi_k25.prepare
