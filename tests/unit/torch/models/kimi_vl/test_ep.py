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

"""Expert-parallel adapter for Kimi-VL-A3B (``kimi_vl``), a remote-code VLM on a Moonlight MoE.

The adapter rebuilds the language model as transformers' native deepseek_v3; its experts are plain
bf16, so nothing is decompressed. The checkpoint's remote code is not available here, so the tests
stub the dynamic-module loader with a stand-in remote module. Every process-wide patch is undone
after each test.
"""

import contextlib
import copy
import importlib
import json
import shutil
import sys
import types
from types import SimpleNamespace

import pytest
import torch

transformers = pytest.importorskip("transformers")

from packaging.version import Version

if Version(transformers.__version__) < Version("5.7"):
    pytest.skip("Kimi-VL expert parallelism needs transformers >= 5.7", allow_module_level=True)

import transformers.core_model_loading as cml
import transformers.dynamic_module_utils as dmu
import transformers.modeling_utils as mu
import transformers.utils.import_utils as import_utils
from _test_utils.torch import kimi_vl_tiny_modeling
from torch import nn
from transformers import (
    AutoConfig,
    AutoModelForCausalLM,
    DeepseekV3Config,
    DeepseekV3ForCausalLM,
    PretrainedConfig,
)

from modelopt.torch.models import _onfly_decompress, _remote_code, get_spec
from modelopt.torch.models.kimi_k25 import ep as kimi_k25
from modelopt.torch.models.kimi_vl import ep as kimi_vl
from modelopt.torch.utils import distributed as dist
from modelopt.torch.utils.moe import _routed_expert_keys
from modelopt.torch.utils.plugins import model_load_utils

# Symbols the remote modeling imports that newer transformers dropped.
COMPAT_SYMBOLS = (
    ("transformers.utils.import_utils", "is_torch_fx_available"),
    ("transformers.pytorch_utils", "is_torch_greater_or_equal_than_1_13"),
    ("transformers.modeling_attn_mask_utils", "_prepare_4d_causal_attention_mask"),
)


def _restore(monkeypatch, owner, name):
    """Have ``monkeypatch`` put ``owner.<name>`` back after the test, or remove it if it was absent."""
    if hasattr(owner, name):
        monkeypatch.setattr(owner, name, getattr(owner, name))
    else:
        monkeypatch.setattr(owner, name, None, raising=False)
        monkeypatch.delattr(owner, name)


class _RemoteTextConfig(PretrainedConfig):
    """Stands in for the remote DeepSeek-V3 config Kimi-VL bundles as its ``text_config``."""

    model_type = "kimi_vl_remote_text"


def _vlm_config():
    text = _RemoteTextConfig(
        hidden_size=64,
        num_hidden_layers=2,
        n_routed_experts=8,
        moe_intermediate_size=16,
        q_lora_rank=None,
        auto_map={"AutoConfig": "configuration_kimi_vl.DeepseekV3Config"},
    )
    return SimpleNamespace(text_config=text)


def _block_fp8_on_the_text_config_only():
    config = _vlm_config()
    block_fp8 = {"quant_method": "fp8", "weight_block_size": [128, 128]}
    config.text_config.quantization_config = block_fp8
    return config


@pytest.fixture
def remote(monkeypatch, request):
    """Stand in for the checkpoint's remote code: importing it registers its modeling module
    (Kimi-VL's ``modeling_kimi_vl`` unless a test names another), with the inline per-expert
    ``DeepseekV3ForCausalLM`` and a VLM class whose ``tie_weights`` has the old no-kwarg signature,
    as both Kimi checkpoints ship it. Also restores the compat symbols, loader hooks and kept
    patches prepare installs."""
    modeling, vlm_name = getattr(
        request, "param", ("modeling_kimi_vl", "KimiVLForConditionalGeneration")
    )
    for module_name, name in COMPAT_SYMBOLS:
        try:
            owner = importlib.import_module(module_name)
        except ImportError:
            continue
        _restore(monkeypatch, owner, name)
    module = types.ModuleType(f"transformers_modules.Kimi.0123abcd.{modeling}")
    module.remote_lm = type("DeepseekV3ForCausalLM", (), {"__init__": lambda self, config: None})
    module.DeepseekV3ForCausalLM = module.remote_lm
    module.loads, module.tied = [], []

    def build(self, config=None):
        if config is not None:  # as both checkpoints' VLM classes build their language model
            self.language_model = module.DeepseekV3ForCausalLM(config.text_config)

    def tie_weights(self):
        module.tied.append(self)
        return "tied"

    module.vlm = type(vlm_name, (), {"__init__": build, "tie_weights": tie_weights})
    setattr(module, vlm_name, module.vlm)

    def import_remote_class(class_ref, ckpt_path, **kwargs):
        module.loads.append((class_ref, ckpt_path, kwargs))
        monkeypatch.setitem(sys.modules, module.__name__, module)

    monkeypatch.setattr(dmu, "get_class_from_dynamic_module", import_remote_class)
    monkeypatch.setattr(dmu, "get_cached_module_file", dmu.get_cached_module_file)
    monkeypatch.setattr(dmu, "get_class_in_module", dmu.get_class_in_module)
    monkeypatch.setattr(_remote_code, "_KEPT_PATCHES", {})
    return module


def test_prepare_rebuilds_the_language_model_config_without_a_decompressor(monkeypatch, remote):
    """The text config becomes transformers' own ``DeepseekV3Config`` (no remote ``auto_map``) with
    an EP plan for the fused experts, naming the experts the spec states for a run that skips
    prepare; the experts are bf16, so no decompressor is installed; the symbols the remote modeling
    imports exist; the remote-code staging shim is left installed."""
    monkeypatch.delattr(import_utils, "is_torch_fx_available", raising=False)
    monkeypatch.setattr(
        _onfly_decompress,
        "install_onfly_decompress",
        lambda cfg: pytest.fail("Kimi-VL's experts are bf16: there is nothing to decompress"),
    )
    staging = dmu.get_cached_module_file
    config = _vlm_config()

    assert kimi_vl.prepare(config, ckpt_path="/ckpt", trust_remote_code=True) is config

    text = config.text_config
    assert type(text) is DeepseekV3Config
    assert getattr(text, "auto_map", None) is None
    assert (text.hidden_size, text.n_routed_experts, text.moe_intermediate_size) == (64, 8, 16)
    assert text.base_model_ep_plan["layers.*.mlp.experts.gate_up_proj"] == "grouped_gemm"
    assert text.base_model_ep_plan["layers.*.mlp.experts.down_proj"] == "grouped_gemm"
    assert _routed_expert_keys(text.base_model_ep_plan) == _routed_expert_keys(
        get_spec("kimi_vl").ep_spec.prepared_ep_plan
    )
    assert import_utils.is_torch_fx_available() is False
    assert dmu.get_cached_module_file is not staging


# Kimi-K2.5's adapter is Kimi-VL's with on-the-fly decompression added; these hold for both.
_BOTH_ADAPTERS = pytest.mark.parametrize(
    ("adapter", "remote"),
    [
        (kimi_vl, ("modeling_kimi_vl", "KimiVLForConditionalGeneration")),
        (kimi_k25, ("modeling_kimi_k25", "KimiK25ForConditionalGeneration")),
    ],
    ids=["vl", "k25"],
    indirect=["remote"],
)


@_BOTH_ADAPTERS
def test_prepare_with_a_checkpoint_swaps_in_native_deepseek_v3_and_shims_tie_weights(
    monkeypatch, adapter, remote
):
    """The VLM class builds its language model from the module-level ``DeepseekV3ForCausalLM``,
    which must now be transformers' native class. Both checkpoints' remote modeling imports a
    symbol newer transformers dropped, and has an old no-kwarg ``tie_weights`` that must accept the
    ``recompute_mapping`` newer transformers passes. Running prepare again does not wrap
    ``tie_weights`` twice."""
    monkeypatch.delattr(import_utils, "is_torch_fx_available", raising=False)
    monkeypatch.setattr(_onfly_decompress, "install_onfly_decompress", lambda cfg: None)
    adapter.prepare(_vlm_config(), ckpt_path="/ckpt", trust_remote_code=True)
    shim = remote.vlm.tie_weights
    adapter.prepare(_vlm_config(), ckpt_path="/ckpt", trust_remote_code=True)

    vlm_class = f"{remote.__name__.rsplit('.', 1)[-1]}.{remote.vlm.__name__}"
    assert remote.loads == [(vlm_class, "/ckpt", {"trust_remote_code": True})] * 2
    assert import_utils.is_torch_fx_available() is False
    assert remote.vlm.tie_weights is shim
    model = remote.vlm()
    assert model.tie_weights(missing_keys=set(), recompute_mapping=False) == "tied"
    assert remote.tied == [model]


@_BOTH_ADAPTERS
def test_a_load_off_expert_parallelism_builds_the_checkpoint_s_own_language_model(
    monkeypatch, adapter, remote
):
    """The patch outlives the load it was made for. A later load in the process that its plan keeps
    off expert parallelism (replicated, or FSDP2) builds from the config as read, so its VLM gets
    the checkpoint's own language model, as with no patch; only the native config ``prepare`` makes
    builds transformers' native one."""
    monkeypatch.setattr(_onfly_decompress, "install_onfly_decompress", lambda cfg: None)
    as_read = _vlm_config()
    prepared = adapter.prepare(copy.deepcopy(as_read), ckpt_path="/ckpt", trust_remote_code=True)

    with torch.device("meta"):
        assert type(remote.vlm(prepared).language_model) is DeepseekV3ForCausalLM
        assert type(remote.vlm(as_read).language_model) is remote.remote_lm


def _built_by_its_own_code():
    """The VLM the checkpoint's own code builds, on the meta device: each MoE layer of its language
    model holds its routed experts as per-expert modules, at ``language_model.model.layers.<i>.mlp.experts``."""

    def layer():
        mlp = nn.Module()
        mlp.experts = nn.ModuleList(nn.Linear(2, 2, bias=False) for _ in range(8))
        return nn.ModuleDict({"mlp": mlp})

    model = nn.Module()
    model.language_model = nn.Module()
    model.language_model.model = nn.Module()
    model.language_model.model.layers = nn.ModuleList(layer() for _ in range(2))
    model.tie_weights = lambda: None
    return model


@_BOTH_ADAPTERS
@pytest.mark.parametrize(
    ("ranks", "dense", "outcome"),
    [(3, "fsdp", "fsdp2"), (3, "replicate", "refused"), (4, "replicate", "expert-parallel")],
    ids=["fsdp2", "whole", "expert-parallel"],
)
def test_only_an_expert_parallel_plan_patches_the_checkpoint_s_code(
    monkeypatch, tmp_path, adapter, remote, ranks, dense, outcome
):
    """Deciding the plan builds the model the checkpoint's own code builds, importable but unpatched,
    and finds its 8 experts through the plan the spec states. Only an expert-parallel plan (4 ranks)
    then runs ``prepare``, which swaps in the native language model. FSDP2-sharded, or left whole on
    3 ranks, which no EP degree of the 8 experts divides (refused), the run keeps the checkpoint's
    own class, the loader and the config as read."""
    model_type = "kimi_vl" if adapter is kimi_vl else "kimi_k25"
    as_read = _vlm_config()
    as_read.model_type = model_type
    monkeypatch.setattr(dist, "size", lambda group=None: ranks)
    monkeypatch.setattr(
        transformers.AutoConfig, "from_pretrained", lambda *a, **k: copy.deepcopy(as_read)
    )
    monkeypatch.setattr(
        model_load_utils, "build_meta_causal_lm", lambda *a, **k: _built_by_its_own_code()
    )
    hooks = (mu.get_model_conversion_mapping, cml.spawn_materialize)

    if outcome == "refused":
        with pytest.raises(ValueError, match="can be expert-parallel, but the plan resolved"):
            model_load_utils.resolve_distribution_plan(
                str(tmp_path), dense=dense, trust_remote_code=True
            )
    else:
        plan, config = model_load_utils.resolve_distribution_plan(
            str(tmp_path), dense=dense, trust_remote_code=True
        )
        assert plan.is_expert_parallel == (outcome == "expert-parallel")
    assert import_utils.is_torch_fx_available() is False  # importable on every plan
    assert getattr(vars(remote.vlm)["tie_weights"], "_kimi_tie_weights_shim", False)  # buildable
    assert (remote.DeepseekV3ForCausalLM is remote.remote_lm) == (outcome != "expert-parallel")
    if outcome == "expert-parallel":
        assert type(config.text_config) is DeepseekV3Config
        assert config.text_config.base_model_ep_plan
    else:
        assert (mu.get_model_conversion_mapping, cml.spawn_materialize) == hooks
    if outcome == "fsdp2":
        assert type(config.text_config) is _RemoteTextConfig
        assert config.text_config.base_model_ep_plan is None


def test_the_build_step_lets_the_checkpoint_s_own_vlm_build_on_this_transformers(
    monkeypatch, tmp_path
):
    """The plan decision builds the checkpoint's own code on the meta device before ``prepare``
    runs, and Kimi-VL's VLM class has a ``tie_weights`` that takes none of the keyword arguments
    transformers' ``post_init`` passes it. ``make_buildable`` adapts it, and swaps nothing: the
    module still builds the language model from its own ``DeepseekV3ForCausalLM``."""
    for module_name, name in COMPAT_SYMBOLS:
        with contextlib.suppress(ImportError):
            _restore(monkeypatch, importlib.import_module(module_name), name)
    monkeypatch.setattr(dmu, "get_cached_module_file", dmu.get_cached_module_file)
    shutil.copy(kimi_vl_tiny_modeling.__file__, tmp_path / "modeling_kimi_vl.py")
    remote = "modeling_kimi_vl.KimiVLForConditionalGeneration"
    auto_map = {"AutoConfig": "modeling_kimi_vl.KimiVLConfig", "AutoModelForCausalLM": remote}
    text = {"model_type": "deepseek_v3", "hidden_size": 16, "num_hidden_layers": 1}
    stored = {"model_type": "kimi_vl", "auto_map": auto_map, "text_config": text}
    (tmp_path / "config.json").write_text(json.dumps(stored))
    config = AutoConfig.from_pretrained(tmp_path, trust_remote_code=True)

    kimi_vl.make_buildable(config, ckpt_path=str(tmp_path), trust_remote_code=True)
    model = model_load_utils.build_meta_causal_lm(
        str(tmp_path), True, None, copy.deepcopy(config), auto_class=AutoModelForCausalLM
    )

    assert type(model).tie_weights._kimi_tie_weights_shim
    assert sys.modules[type(model).__module__].DeepseekV3ForCausalLM is DeepseekV3ForCausalLM


@_BOTH_ADAPTERS
@pytest.mark.parametrize(
    ("missing", "match"),
    [
        ("module", "could not find a loaded modeling_kimi"),
        ("class", "defines no module-level DeepseekV3ForCausalLM"),
    ],
    ids=["no-module", "no-class"],
)
def test_the_swap_fails_loudly_when_the_remote_modeling_has_nothing_to_swap(
    monkeypatch, adapter, remote, missing, match
):
    """Skipping the swap silently would build the per-expert layout that EP cannot shard: when the
    remote modeling did not load, or names its language model otherwise (a class renamed, or
    imported where the VLM class builds it), which is not the swap already done."""
    if missing == "module":
        monkeypatch.setattr(dmu, "get_class_from_dynamic_module", lambda *args, **kwargs: None)
    else:
        del remote.DeepseekV3ForCausalLM

    with pytest.raises(RuntimeError, match=match):
        adapter.prepare(_vlm_config(), ckpt_path="/ckpt", trust_remote_code=True)


@pytest.mark.parametrize("adapter", [kimi_vl, kimi_k25], ids=["vl", "k25"])
@pytest.mark.parametrize(
    ("config", "kwargs", "match"),
    [
        # transformers' own port (Kimi-K2.5's, from 5.13), which the patch never reaches.
        (PretrainedConfig, {"ckpt_path": "/ckpt", "trust_remote_code": True}, "own port"),
        (SimpleNamespace, {}, "text_config"),
        # Without the checkpoint there is no modeling code to patch, and a half-prepared config
        # would load Kimi's own language model under a plan naming none of its experts.
        (_vlm_config, {"trust_remote_code": True}, "needs the checkpoint's path"),
        # transformers' get_class_from_dynamic_module runs the checkpoint's code without checking
        # trust_remote_code, so prepare must refuse before reaching it.
        (_vlm_config, {"ckpt_path": "/ckpt"}, "trust_remote_code=True"),
        # The native text config drops the scheme, and with no top-level copy for the load to
        # read, the experts would load as stored.
        (
            _block_fp8_on_the_text_config_only,
            {"ckpt_path": "/ckpt", "trust_remote_code": True},
            "on text_config only",
        ),
    ],
    ids=[
        "transformers-port",
        "no-text-config",
        "no-checkpoint",
        "untrusted-checkpoint",
        "quantized-text-config-only",
    ],
)
def test_prepare_refuses_before_patching_anything(monkeypatch, adapter, config, kwargs, match):
    """Before prepare patches anything process-wide, such as the loader's remote-code staging, or
    imports the checkpoint's code."""

    def unstaged_loader(*args, **kwargs):
        """The loader before any staging shim wraps it."""

    monkeypatch.setattr(dmu, "get_cached_module_file", unstaged_loader)
    monkeypatch.setattr(
        dmu,
        "get_class_from_dynamic_module",
        lambda *args, **kwargs: pytest.fail("the checkpoint's code was imported"),
    )

    with pytest.raises(ValueError, match=match):
        adapter.prepare(config(), **kwargs)

    assert dmu.get_cached_module_file is unstaged_loader


@pytest.mark.parametrize(
    ("model_type", "adapter"), [("kimi_vl", kimi_vl), ("kimi_k25", kimi_k25)], ids=["vl", "k25"]
)
def test_the_spec_routes_expert_parallelism_through_its_own_adapter(model_type, adapter):
    """The checkpoint's code builds no EP-capable experts, so its whole EP setup is this adapter's
    ``prepare``: another Kimi adapter's would load it without its own swap or decompression."""
    assert get_spec(model_type).ep_spec.prepare is adapter.prepare
