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

"""What the expert-parallel loader reads off the loaded model and the source checkpoint.

``_routed_expert_modules`` finds the routed-expert containers the EP plan shards,
``_seed_parallel_states`` gives every module the parallel state its amax syncs over, and
``_ckpt_quant_method`` / ``_source_experts_fused_on_disk`` read the checkpoint's native
quantization and the layout its experts are stored in. Everything runs in one CPU process.
"""

import json
from types import SimpleNamespace

import pytest
import torch
from safetensors.torch import save_file
from torch import nn

from modelopt.torch.utils.moe import (
    _ckpt_quant_method,
    _routed_expert_modules,
    _seed_parallel_states,
    _source_experts_fused_on_disk,
)

# ----------------------------------------------------------------------------------- tiny MoE


class _Experts(nn.Module):
    """A layer's routed experts as one grouped-GEMM container (every expert in fused tensors)."""

    def __init__(self):
        super().__init__()
        self.gate_up_proj = nn.Parameter(torch.zeros(4, 4, 2))
        self.down_proj = nn.Parameter(torch.zeros(4, 2, 2))
        self.weight_quantizer = nn.Identity()  # quantizer insertion adds children like this

    def forward(self, hidden_states, top_k_index, top_k_weights, process_group=None):
        return hidden_states


class _SharedExperts(nn.Module):
    """The always-on shared expert MLP: dense, replicated on every rank."""

    def __init__(self):
        super().__init__()
        self.gate_up_proj = nn.Linear(2, 4, bias=False)

    def forward(self, hidden_states):
        return hidden_states


class _MoeLayer(nn.Module):
    def __init__(self):
        super().__init__()
        self.self_attn = nn.Linear(2, 2)
        self.mlp = nn.Module()
        self.mlp.gate = nn.Linear(2, 4, bias=False)  # the router
        self.mlp.experts = _Experts()
        self.mlp.shared_experts = _SharedExperts()


def _model_with_decoder_at(prefix):
    """A two-layer MoE whose decoder (``layers.<i>``) sits under ``prefix`` (``""``: the decoder)."""
    decoder = nn.Module()
    decoder.embed_tokens = nn.Embedding(8, 2)
    decoder.layers = nn.ModuleList(_MoeLayer() for _ in range(2))
    if not prefix:
        return decoder
    root = parent = nn.Module()
    *path, leaf = prefix.rstrip(".").split(".")
    for name in path:
        parent.add_module(name, nn.Module())
        parent = getattr(parent, name)
    parent.add_module(leaf, decoder)
    return root


# The plan load_and_prepare_ep hands over after the a2a swap: base-relative keys (no ``model.``),
# the router replicated, the fused expert params grouped_gemm-sharded, the container a2a-tagged.
_A2A_PLAN = {
    "layers.*.mlp.gate": "torch_a2a_router",
    "layers.*.mlp.experts.gate_up_proj": "grouped_gemm",
    "layers.*.mlp.experts.down_proj": "grouped_gemm",
    "layers.*.mlp.experts": "torch_a2a_experts",
}

# ---------------------------------------------------------------------- _routed_expert_modules


@pytest.mark.parametrize(
    "prefix",
    ["", "model.", "model.language_model.", "language_model.model."],
    ids=["base-model", "causal-lm", "vlm", "legacy-vlm"],
)
def test_routed_experts_are_the_a2a_containers_under_any_prefix(prefix):
    """The base-relative plan keys match the decoder wherever the model nests it, at every layer
    index. The router, the grouped_gemm params and the shared experts are not containers."""
    pytest.importorskip("transformers")
    model = _model_with_decoder_at(prefix)

    found = _routed_expert_modules(model, _A2A_PLAN)

    modules = dict(model.named_modules())
    assert found == {
        name: modules[name] for name in [f"{prefix}layers.{i}.mlp.experts" for i in (0, 1)]
    }


def test_only_the_a2a_experts_style_marks_a_container():
    """A plan whose container still carries the stock ``moe_tp_experts`` style names no experts."""
    pytest.importorskip("transformers")
    plan = {**_A2A_PLAN, "layers.*.mlp.experts": "moe_tp_experts"}

    assert _routed_expert_modules(_model_with_decoder_at("model."), plan) == {}


@pytest.fixture
def quant_wrappers():
    """ModelOpt quant wrappers registered for the experts' and the shared experts' classes.

    Whatever the test leaves registered is unregistered afterwards.
    """
    from modelopt.torch.quantization.nn.modules.quant_module import QuantModule, QuantModuleRegistry

    class _PerExpertQuantWrapper(QuantModule):
        pass

    classes = (_Experts, _SharedExperts)
    QuantModuleRegistry.register({cls: f"test.{cls.__name__}" for cls in classes})(
        _PerExpertQuantWrapper
    )
    yield QuantModuleRegistry
    for cls in classes:
        if cls in QuantModuleRegistry:
            QuantModuleRegistry.unregister(cls)


def test_the_experts_class_loses_its_quant_wrapper_and_other_classes_keep_theirs(quant_wrappers):
    """Quantization would swap a registered wrapper in for the experts, whose forward drops the EP
    process group the a2a transport injects. Only the routed experts' class is unregistered."""
    pytest.importorskip("transformers")

    _routed_expert_modules(_model_with_decoder_at("model."), _A2A_PLAN)

    assert _Experts not in quant_wrappers
    assert _SharedExperts in quant_wrappers


# ----------------------------------------------------------------------- _seed_parallel_states


def test_expert_containers_and_their_children_get_the_expert_state():
    """Everything else, the root and the containers' own parents included, gets the dense state."""
    model = _model_with_decoder_at("model.")
    experts = {f"model.layers.{i}.mlp.experts": model.model.layers[i].mlp.experts for i in (0, 1)}
    dense, expert = object(), object()

    _seed_parallel_states(model, experts, dense_state=dense, expert_state=expert)

    states = {name: module._parallel_state for name, module in model.named_modules()}
    assert {name for name, state in states.items() if state is expert} == {
        "model.layers.0.mlp.experts",
        "model.layers.0.mlp.experts.weight_quantizer",
        "model.layers.1.mlp.experts",
        "model.layers.1.mlp.experts.weight_quantizer",
    }
    assert all(state is dense for state in states.values() if state is not expert)


def test_a_sibling_whose_name_extends_an_expert_container_stays_dense():
    """Qwen2-MoE keeps ``shared_expert_gate`` beside ``shared_expert``: a name that merely starts
    with an expert container's name is not under it."""
    mlp = nn.Module()
    mlp.experts = _Experts()
    mlp.experts_gate = nn.Linear(2, 1)
    dense, expert = object(), object()

    _seed_parallel_states(mlp, ["experts"], dense_state=dense, expert_state=expert)

    assert mlp.experts._parallel_state is expert
    assert mlp.experts_gate._parallel_state is dense


# --------------------------------------------------------------------------- _ckpt_quant_method


@pytest.mark.parametrize(
    ("config", "expected"),
    [
        (SimpleNamespace(quantization_config={"quant_method": "fp8"}), "fp8"),
        (SimpleNamespace(quantization_config=SimpleNamespace(quant_method="mxfp4")), "mxfp4"),
        (
            SimpleNamespace(
                text_config=SimpleNamespace(quantization_config={"quant_method": "fp8"})
            ),
            "fp8",
        ),
        (SimpleNamespace(quantization_config={"weight_block_size": [128, 128]}), None),
        (SimpleNamespace(text_config=None), None),
    ],
    ids=["dict", "object", "vlm-text-config", "no-method", "unquantized"],
)
def test_quant_method_comes_from_the_config_or_a_vlms_text_config(config, expected):
    assert _ckpt_quant_method(config) == expected


def test_the_top_level_quant_method_wins_over_the_text_configs():
    config = SimpleNamespace(
        quantization_config={"quant_method": "mxfp4"},
        text_config=SimpleNamespace(quantization_config={"quant_method": "fp8"}),
    )

    assert _ckpt_quant_method(config) == "mxfp4"


def test_a_transformers_quantization_config_object_reports_the_plain_method_name():
    """A loaded model's config holds transformers' config object, not the JSON dict, and callers
    compare against ``"fp8"`` / ``"mxfp4"``."""
    transformers = pytest.importorskip("transformers")
    config = SimpleNamespace(quantization_config=transformers.FineGrainedFP8Config())

    assert _ckpt_quant_method(config) == "fp8"


# ---------------------------------------------------------------- _source_experts_fused_on_disk

# gpt-oss: per-layer MXFP4 blocks/scales spanning every expert, plus fused biases.
_FUSED_MXFP4 = [
    "model.layers.0.mlp.router.weight",
    "model.layers.0.mlp.experts.gate_up_proj_blocks",
    "model.layers.0.mlp.experts.gate_up_proj_scales",
    "model.layers.0.mlp.experts.gate_up_proj_bias",
    "model.layers.0.mlp.experts.down_proj_blocks",
]
# Llama-4: one bf16 gate_up_proj / down_proj per layer, beside a per-layer shared expert.
_FUSED_BF16 = [
    "language_model.model.layers.0.feed_forward.experts.gate_up_proj",
    "language_model.model.layers.0.feed_forward.experts.down_proj",
    "language_model.model.layers.0.feed_forward.shared_expert.gate_proj.weight",
]
# Qwen3-MoE / DeepSeek / Kimi: one tensor per expert projection.
_PER_EXPERT = [
    "model.layers.0.mlp.gate.weight",
    "model.layers.0.mlp.experts.0.gate_proj.weight",
    "model.layers.0.mlp.experts.0.up_proj.weight",
    "model.layers.0.mlp.experts.0.down_proj.weight",
    "model.layers.0.mlp.experts.1.gate_proj.weight",
]
# Mixtral: per-expert w1 / w2 / w3.
_PER_EXPERT_W123 = [
    "model.layers.0.block_sparse_moe.experts.0.w1.weight",
    "model.layers.0.block_sparse_moe.experts.0.w2.weight",
    "model.layers.0.block_sparse_moe.experts.0.w3.weight",
]
# Fused projections that belong to a shared expert, not to the routed experts.
_FUSED_SHARED_EXPERTS = [
    "model.layers.0.mlp.shared_experts.gate_up_proj.weight",
    "model.layers.0.mlp.shared_experts.down_proj.weight",
    "model.layers.0.mlp.shared_expert.gate_up_proj",
]


def _index(ckpt_dir, keys):
    """Write a sharded checkpoint's ``model.safetensors.index.json`` naming ``keys``."""
    weight_map = dict.fromkeys(keys, "model-00001-of-00002.safetensors")
    (ckpt_dir / "model.safetensors.index.json").write_text(
        json.dumps({"metadata": {}, "weight_map": weight_map})
    )
    return str(ckpt_dir)


@pytest.mark.parametrize(
    ("keys", "fused"),
    [
        (_FUSED_MXFP4, True),
        (_FUSED_BF16, True),
        (_PER_EXPERT, False),
        (_PER_EXPERT_W123, False),
        ([*_PER_EXPERT, "model.layers.1.mlp.experts.gate_up_proj"], False),
        (_FUSED_SHARED_EXPERTS, False),
        (["model.layers.0.mlp.down_proj.weight", "lm_head.weight"], False),
    ],
    ids=[
        "gpt-oss-mxfp4",
        "llama4-bf16",
        "qwen3-per-expert",
        "mixtral-w123",
        "per-expert-wins",
        "fused-shared-experts",
        "dense",
    ],
)
def test_the_expert_layout_is_read_from_the_index(tmp_path, keys, fused):
    assert _source_experts_fused_on_disk(_index(tmp_path, keys)) is fused


@pytest.mark.parametrize(
    ("keys", "fused"), [(_FUSED_MXFP4, True), (_PER_EXPERT, False)], ids=["fused", "per-expert"]
)
def test_without_an_index_the_safetensors_headers_are_read(tmp_path, keys, fused):
    save_file({key: torch.zeros(1) for key in keys}, str(tmp_path / "model.safetensors"))

    assert _source_experts_fused_on_disk(str(tmp_path)) is fused


def test_an_unknown_or_unreadable_checkpoint_counts_as_separated(tmp_path):
    """``False`` keeps the historical per-expert export, and probing must never fail the load."""
    (tmp_path / "config.json").write_text("{}")
    bad_index = tmp_path / "bad_index"
    bad_index.mkdir()
    (bad_index / "model.safetensors.index.json").write_text("{not json")
    truncated = tmp_path / "truncated"
    truncated.mkdir()
    (truncated / "model.safetensors").write_bytes(b"\x08\x00")

    for path in (
        None,
        tmp_path / "missing",
        tmp_path / "config.json",
        tmp_path,
        bad_index,
        truncated,
    ):
        assert _source_experts_fused_on_disk(path and str(path)) is False, path
