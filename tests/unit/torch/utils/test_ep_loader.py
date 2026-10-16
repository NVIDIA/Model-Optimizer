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

The loader takes an expert-parallel plan and keeps its experts out of ModelOpt's per-expert quant
wrappers, ``_seed_parallel_states`` gives every module the parallel state its amax syncs over,
``expert_parallel_mesh`` hands calibration the mesh an expert-parallel plan's experts are sharded
over, and ``_ckpt_quant_method`` reads the checkpoint's native quantization. Everything runs in one
CPU process.
"""

from types import SimpleNamespace

import pytest
import torch
from torch import nn

pytest.importorskip("transformers")

from transformers import FineGrainedFP8Config

import modelopt.torch.utils.distributed as dist
from modelopt.torch.utils import moe
from modelopt.torch.utils.distributed import DistributionPlan, set_distribution_plan
from modelopt.torch.utils.moe import expert_parallel_mesh
from modelopt.torch.utils.plugins import model_load_utils
from modelopt.torch.utils.plugins.model_load_utils import (
    _ckpt_quant_method,
    _experts_sharded_over,
    _refuse_unquantizable_experts,
    _seed_parallel_states,
    _transformers_mesh_builder,
    load_and_prepare_ep,
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


# ------------------------------------------------------------------------------- the plan


@pytest.mark.parametrize(
    ("plan", "error", "match"),
    [
        pytest.param(DistributionPlan(2), ValueError, "shards no routed experts", id="no-experts"),
        # FSDP2 within each of several expert-parallel groups: a layout the export cannot write.
        pytest.param(
            DistributionPlan(4, 2, "fsdp"),
            NotImplementedError,
            "expert-data-parallel",
            id="dense-hsdp",
        ),
    ],
)
def test_the_loader_refuses_a_plan_it_cannot_carry_out(plan, error, match):
    """Before anything else: no process group, transformers release or checkpoint is looked at."""
    with pytest.raises(error, match=match):
        load_and_prepare_ep("unused", plan)


def test_the_loader_refuses_a_plan_for_another_world(gloo_world_of_one):
    with pytest.raises(ValueError, match="is for 2 ranks, but this run has 1"):
        load_and_prepare_ep("unused", DistributionPlan(2, ep_size=2))


def test_a_hub_id_is_made_local_before_the_weights_load(monkeypatch):
    """The export reads the tensors the load could not place back from the checkpoint it records,
    which must be a local directory: a Hub ID there leaves the export nothing to read. The load
    resolves it once the plan and the config pass, before anything reads a weight."""

    class _ResolvedError(Exception):
        """Stops the load where it resolves the checkpoint."""

    resolved = []

    def resolve(path, rank):
        resolved.append(path)
        raise _ResolvedError

    monkeypatch.setattr(dist, "is_initialized", lambda: True)
    monkeypatch.setattr(dist, "size", lambda group=None: 2)
    monkeypatch.setattr(torch.distributed, "get_rank", lambda group=None: 0)
    monkeypatch.setattr(moe, "require_transformers", lambda *args: None)
    monkeypatch.setattr(model_load_utils, "_resolve_checkpoint_dir", resolve)

    with pytest.raises(_ResolvedError):
        load_and_prepare_ep(
            "org/model", DistributionPlan(2, ep_size=2), hf_config=SimpleNamespace()
        )

    assert resolved == ["org/model"]


@pytest.mark.parametrize("quant_method", ["gptq", "modelopt", "compressed-tensors"])
def test_a_checkpoint_format_the_load_cannot_dequantize_is_refused(monkeypatch, quant_method):
    """transformers' own quantizer would load it as stored, for PTQ to treat as plain weights; the
    load refuses it before anything is read."""
    resolved = []
    monkeypatch.setattr(dist, "is_initialized", lambda: True)
    monkeypatch.setattr(dist, "size", lambda group=None: 2)
    monkeypatch.setattr(torch.distributed, "get_rank", lambda group=None: 0)
    monkeypatch.setattr(moe, "require_transformers", lambda *args: None)
    monkeypatch.setattr(
        model_load_utils, "_resolve_checkpoint_dir", lambda p, r: resolved.append(p)
    )
    config = SimpleNamespace(quantization_config={"quant_method": quant_method})

    with pytest.raises(NotImplementedError, match=f"cannot dequantize a '{quant_method}'"):
        load_and_prepare_ep("org/model", DistributionPlan(2, ep_size=2), hf_config=config)
    assert resolved == []


# ------------------------------------------------------------------------ quant wrappers


@pytest.fixture
def quant_wrappers():
    """ModelOpt quant wrappers registered for the experts' and the shared experts' classes."""
    from modelopt.torch.quantization.nn.modules.quant_module import QuantModule, QuantModuleRegistry

    class _PerExpertQuantWrapper(QuantModule):
        pass

    classes = (_Experts, _SharedExperts)
    QuantModuleRegistry.register({cls: f"test.{cls.__name__}" for cls in classes})(
        _PerExpertQuantWrapper
    )
    yield QuantModuleRegistry
    for cls in classes:
        QuantModuleRegistry.unregister(cls)


def test_routed_experts_with_a_quant_wrapper_of_their_own_are_refused(quant_wrappers):
    """Quantization would convert them to it, whose forward drops the process group the all-to-all
    transport passes in. The registry serves every model in the process, so it is left as it is; the
    shared experts, which the transport does not manage, may keep a wrapper."""
    model = _model_with_decoder_at("model.")

    with pytest.raises(NotImplementedError, match=r"cannot quantize _Experts: ModelOpt quantizes"):
        _refuse_unquantizable_experts([layer.mlp.experts for layer in model.model.layers])

    assert _Experts in quant_wrappers
    assert _SharedExperts in quant_wrappers


def test_routed_experts_under_the_generic_fused_wrapper_are_accepted():
    """mtq.quantize registers the generic fused-experts wrapper, for the whole process, for every
    fused-experts class it meets, and that wrapper quantizes through the grouped GEMM: a second
    expert-parallel load of the same class is not refused for it."""
    from modelopt.torch.quantization.nn.modules.quant_module import QuantModuleRegistry
    from modelopt.torch.quantization.plugins.huggingface import _QuantFusedExperts

    model = _model_with_decoder_at("model.")
    QuantModuleRegistry.register({_Experts: "test._Experts"})(_QuantFusedExperts)
    try:
        _refuse_unquantizable_experts([layer.mlp.experts for layer in model.model.layers])
    finally:
        QuantModuleRegistry.unregister(_Experts)


def test_transposed_experts_are_refused():
    """The grouped quantization leaves experts stored transposed (GPT-OSS's) unquantized: the model
    would be quantized everywhere but its experts."""
    model = _model_with_decoder_at("model.")
    experts = [layer.mlp.experts for layer in model.model.layers]
    for module in experts:
        module.is_transposed = True

    with pytest.raises(NotImplementedError, match="cannot quantize _Experts: these experts store"):
        _refuse_unquantizable_experts(experts)


# ----------------------------------------------------------------------- _seed_parallel_states


def test_expert_containers_and_their_children_get_the_expert_state():
    """Everything else, the root and the containers' own parents included, gets the dense state; so
    does a sibling whose name merely starts with a container's (Qwen2-MoE keeps
    ``shared_expert_gate`` beside ``shared_expert``)."""
    model = _model_with_decoder_at("model.")
    model.model.layers[0].mlp.experts_gate = nn.Linear(2, 1)
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


# ------------------------------------------------------------------------ expert_parallel_mesh


@pytest.mark.parametrize(
    ("plan", "is_expert_parallel"),
    [(DistributionPlan(2, ep_size=2), True), (None, False), (DistributionPlan(2), False)],
    ids=["expert-parallel", "no-plan", "replicated"],
)
def test_the_expert_parallel_mesh_comes_with_an_expert_parallel_plan_only(plan, is_expert_parallel):
    """An expert-parallel plan: the mesh transformers kept on the model. Any other model has none,
    even with a device mesh on it, as a load through from_pretrained's own TP can leave."""
    model = _model_with_decoder_at("model.")
    if plan is not None:
        set_distribution_plan(model, plan)
    model._device_mesh = mesh = object()

    assert expert_parallel_mesh(model) is (mesh if is_expert_parallel else None)


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
        (
            SimpleNamespace(
                quantization_config={"quant_method": "mxfp4"},
                text_config=SimpleNamespace(quantization_config={"quant_method": "fp8"}),
            ),
            "mxfp4",
        ),
        (SimpleNamespace(quantization_config=FineGrainedFP8Config()), "fp8"),
        (SimpleNamespace(quantization_config={"weight_block_size": [128, 128]}), None),
        (SimpleNamespace(text_config=None), None),
    ],
    ids=[
        "dict",
        "object",
        "vlm-text-config",
        "top-level-wins",
        "transformers-config-object",
        "no-method",
        "unquantized",
    ],
)
def test_quant_method_comes_from_the_config_or_a_vlms_text_config(config, expected):
    """The top-level config first, else a VLM's text config. A loaded model's config holds
    transformers' config object, not the JSON dict, and callers compare against ``"fp8"`` /
    ``"mxfp4"``: it reports the plain method name, not its enum."""
    assert _ckpt_quant_method(config) == expected


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
