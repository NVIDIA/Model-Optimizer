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

"""Expert-parallel adapter for Kimi-K2.5 (``kimi_k25``), a remote-code VLM on DeepSeek-V3.

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
from transformers import DeepseekV3Config, PretrainedConfig

from modelopt.torch.models import _kimi, _onfly_decompress, _remote_code, get_spec
from modelopt.torch.models.kimi_k25 import ep as kimi_k25
from modelopt.torch.utils.moe import _routed_expert_keys, check_distributed_support

QUANT = {"quant_method": "compressed-tensors", "format": "pack-quantized"}
REMOTE = "transformers_modules.Kimi_hyphen_K2_dot_5.0123abcd"  # how the loader names remote code


class _RemoteTextConfig(PretrainedConfig):
    """Stands in for the remote DeepSeek-V3 config Kimi-K2.5 bundles as its ``text_config``."""

    model_type = "kimi_k2_remote_text"


def _vlm_config():
    """Kimi-K2.5's config: the compressed-tensors scheme sits only on the text config."""
    text = _RemoteTextConfig(
        hidden_size=64,
        num_hidden_layers=2,
        n_routed_experts=8,
        moe_intermediate_size=16,
        auto_map={"AutoConfig": "configuration_deepseek.DeepseekV3Config"},
        quantization_config=QUANT.copy(),
    )
    return SimpleNamespace(text_config=text)


@pytest.fixture
def remote(monkeypatch):
    """Stand in for the checkpoint's remote code: importing it registers ``modeling_kimi_k25``,
    whose ``DeepseekV3ForCausalLM`` is the per-expert remote language model. Also restores the
    loader hooks and kept patches prepare installs, and leaves out its compat import shims, which
    the Kimi-VL tests cover for both adapters."""
    module = types.ModuleType(f"{REMOTE}.modeling_kimi_k25")
    module.DeepseekV3ForCausalLM = type("DeepseekV3ForCausalLM", (), {})
    module.loads = []

    def import_remote_class(class_ref, ckpt_path, **kwargs):
        module.loads.append((class_ref, ckpt_path, kwargs))
        monkeypatch.setitem(sys.modules, module.__name__, module)

    monkeypatch.setattr(dmu, "get_class_from_dynamic_module", import_remote_class)
    monkeypatch.setattr(dmu, "get_cached_module_file", dmu.get_cached_module_file)
    monkeypatch.setattr(dmu, "get_class_in_module", dmu.get_class_in_module)
    monkeypatch.setattr(_remote_code, "_KEPT_PATCHES", {})
    monkeypatch.setattr(_kimi, "install_compat_import_shims", lambda: None)
    return module


@pytest.mark.parametrize(
    "quantization_config",
    [
        QUANT,
        # Left to the expert-parallel load's format handling: bf16 experts load as they are, and
        # any other scheme is read from the copy of quantization_config Kimi's config class keeps
        # on the top level.
        None,
        {"quant_method": "fp8", "weight_block_size": [128, 128]},
    ],
    ids=["compressed-tensors", "bf16", "block-fp8"],
)
def test_prepare_rebuilds_the_language_model_config_as_native_deepseek_v3(
    monkeypatch, remote, quantization_config
):
    """The text config becomes transformers' own ``DeepseekV3Config`` (no remote ``auto_map``, no
    stale ``quantization_config``) with an EP plan for the fused experts, naming the experts the
    spec states for a run that skips prepare; the decompressor is installed for compressed-tensors
    experts only, while the text config it replaces still carries their scheme; the remote-code
    staging shim is left installed."""
    installs = []
    monkeypatch.setattr(
        _onfly_decompress,
        "install_onfly_decompress",
        lambda cfg: installs.append(_onfly_decompress._find_quantization_config(cfg)),
    )
    staging = dmu.get_cached_module_file
    config = _vlm_config()
    config.text_config.quantization_config = quantization_config
    if quantization_config is not QUANT:
        config.quantization_config = quantization_config  # the top-level copy the load reads

    assert kimi_k25.prepare(config, ckpt_path="/ckpt", trust_remote_code=True) is config

    text = config.text_config
    assert type(text) is DeepseekV3Config
    assert getattr(text, "auto_map", None) is None
    assert getattr(text, "quantization_config", None) is None
    assert (text.hidden_size, text.n_routed_experts, text.moe_intermediate_size) == (64, 8, 16)
    assert text.base_model_ep_plan["layers.*.mlp.experts.gate_up_proj"] == "grouped_gemm"
    assert text.base_model_ep_plan["layers.*.mlp.experts.down_proj"] == "grouped_gemm"
    assert _routed_expert_keys(text.base_model_ep_plan) == _routed_expert_keys(
        get_spec("kimi_k25").ep_spec.prepared_ep_plan
    )
    assert installs == ([QUANT] if quantization_config is QUANT else [])
    assert dmu.get_cached_module_file is not staging


def test_the_swap_leaves_a_sibling_remote_module_alone(monkeypatch, remote):
    """Kimi-K2.5's ``modeling_kimi_k25`` imports ``modeling_deepseek``, which defines a
    ``DeepseekV3ForCausalLM`` of its own; the swap, scoped to ``modeling_kimi_k25``, must not reach
    it. The swap itself is the shared adapter tests'."""
    monkeypatch.setattr(_onfly_decompress, "install_onfly_decompress", lambda cfg: None)
    bystander = types.ModuleType(f"{REMOTE}.modeling_deepseek")
    remote_lm = bystander.DeepseekV3ForCausalLM = type("DeepseekV3ForCausalLM", (), {})
    monkeypatch.setitem(sys.modules, bystander.__name__, bystander)

    kimi_k25.prepare(_vlm_config(), ckpt_path="/ckpt", trust_remote_code=True)

    assert bystander.DeepseekV3ForCausalLM is remote_lm


def test_a_checkpoint_the_decompressor_refuses_is_refused_before_anything_changes(
    monkeypatch, remote
):
    """The decompressor refuses a checkpoint it cannot decompress as it loads before prepare
    replaces the language model's config or wraps the loader's remote-code staging."""

    def refuse(cfg):
        raise NotImplementedError("the checkpoint has 2 compressed-tensors config groups")

    monkeypatch.setattr(_onfly_decompress, "install_onfly_decompress", refuse)
    staging = dmu.get_cached_module_file
    config = _vlm_config()
    text = config.text_config

    with pytest.raises(NotImplementedError, match="config groups"):
        kimi_k25.prepare(config, ckpt_path="/ckpt", trust_remote_code=True)

    assert config.text_config is text
    assert dmu.get_cached_module_file is staging


def test_a_transformers_too_old_to_decompress_is_refused_before_prepare(monkeypatch):
    """The experts decompress as they load from transformers 5.8: the spec refuses an older
    release before prepare swaps the language model's config."""
    monkeypatch.setattr(transformers, "__version__", "5.7.0")

    with pytest.raises(RuntimeError, match=r"transformers >= 5\.8"):
        check_distributed_support("kimi_k25")


def test_prepare_refuses_transformers_own_port():
    """From 5.13 a load without ``trust_remote_code`` builds transformers' port, which the patch
    never reaches; ``prepare`` must refuse it rather than run unpatched."""
    native = pytest.importorskip("transformers.models.kimi_k25.configuration_kimi_k25")
    with pytest.raises(ValueError, match="runs on the checkpoint's own modeling code"):
        kimi_k25.prepare(native.Kimi_K25Config(), ckpt_path="/ckpt", trust_remote_code=True)
