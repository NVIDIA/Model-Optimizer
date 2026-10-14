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

"""What the expert-parallel loader reads off the loaded model and its config.

``_routed_expert_modules`` finds the routed-expert containers the EP plan shards,
``_seed_parallel_states`` gives every module the parallel state its amax syncs over, and
``_ckpt_quant_method`` reads the checkpoint's native quantization. Everything runs in one CPU
process.
"""

from types import SimpleNamespace

import pytest
import torch
from torch import nn

from modelopt.torch.utils.distributed import ParallelState
from modelopt.torch.utils.moe import (
    _ckpt_quant_method,
    _experts_sharded_over,
    _routed_expert_modules,
    _seed_parallel_states,
    _transformers_mesh_builder,
    expert_parallel_mesh,
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


# ------------------------------------------------------------------------ expert_parallel_mesh


def test_the_expert_parallel_mesh_is_the_one_the_model_was_loaded_over(monkeypatch):
    """Experts seeded with a live expert-parallel group: the mesh transformers kept on the model."""
    monkeypatch.setattr(torch.distributed, "is_initialized", lambda: True)
    model = _model_with_decoder_at("model.")
    experts = {f"model.layers.{i}.mlp.experts": model.model.layers[i].mlp.experts for i in (0, 1)}
    _seed_parallel_states(
        model,
        experts,
        dense_state=ParallelState(data_parallel_group=object()),
        expert_state=ParallelState(expert_model_parallel_group=object()),
    )
    model._device_mesh = mesh = object()

    assert expert_parallel_mesh(model) is mesh


@pytest.mark.parametrize("ep_group", [None, -1], ids=["not-seeded", "one-rank-ep"])
def test_a_model_without_sharded_experts_has_no_expert_parallel_mesh(monkeypatch, ep_group):
    """An unseeded model, or one whose expert-parallel group is a single rank, is not EP-sharded."""
    monkeypatch.setattr(torch.distributed, "is_initialized", lambda: True)
    model = _model_with_decoder_at("model.")
    if ep_group is not None:
        experts = {"model.layers.0.mlp.experts": model.model.layers[0].mlp.experts}
        _seed_parallel_states(
            model,
            experts,
            dense_state=ParallelState(),
            expert_state=ParallelState(expert_model_parallel_group=ep_group),
        )
    model._device_mesh = object()

    assert expert_parallel_mesh(model) is None


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


def test_from_pretrained_builds_its_mesh_from_the_ep_mesh_it_is_given():
    """Newer transformers ignores ``from_pretrained``'s ``device_mesh`` and builds a world-sized one,
    which shards the experts over every rank when EP is below the world size. During the load it gets
    the EP mesh instead, and its own builder is back afterwards, even when the load raises."""
    pytest.importorskip("transformers")
    mixin = _transformers_mesh_builder()
    if mixin is None:
        pytest.skip("this transformers honors from_pretrained's device_mesh")
    from transformers import PreTrainedModel
    from transformers.distributed import DistributedConfig

    builder = mixin.initialize_distributed_mesh
    ep_mesh = SimpleNamespace(device_type="cpu")
    config = DistributedConfig(enable_expert_parallel=True, tp_plan="auto", tp_size=2)
    with _experts_sharded_over(ep_mesh):
        _, device_map, device_mesh = PreTrainedModel.prepare_distribute_model(config)

    assert device_mesh is ep_mesh
    assert device_map == torch.device("cpu")
    assert mixin.initialize_distributed_mesh is builder
    with pytest.raises(RuntimeError), _experts_sharded_over(ep_mesh):
        raise RuntimeError("the load failed")
    assert mixin.initialize_distributed_mesh is builder
