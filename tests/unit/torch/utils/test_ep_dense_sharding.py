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

"""How the expert-parallel loader places the dense (non-expert) part of an MoE.

``from_pretrained`` always EP-shards the experts. ``_make_expert_param_matcher`` tells their params
from the dense ones, ``_resolve_dense_strategy`` picks between replicating the dense part and
sharding it, ``_ep_dense_meta_load`` keeps the dense weights off the device during
``from_pretrained`` when they are sharded, and ``_fsdp2_shard_dense`` /
``_load_ep_dense_distributed`` FSDP2-shard them around the experts. The FSDP2 cases run on a
one-rank gloo process group on CPU.
"""

from types import SimpleNamespace

import pytest
import torch
from torch import nn
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.fsdp import FSDPModule
from torch.distributed.tensor import DTensor, Replicate, Shard, distribute_tensor

import modelopt.torch.utils.distributed as mto_dist
from modelopt.torch.utils.moe import (
    _ep_dense_meta_load,
    _fsdp2_shard_dense,
    _load_ep_dense_distributed,
    _make_expert_param_matcher,
    _resolve_dense_strategy,
)

# ------------------------------------------------------------------ _make_expert_param_matcher

# The plan load_and_prepare_ep shards with after the a2a swap: base-relative keys (no ``model.``).
_A2A_PLAN = {
    "layers.*.mlp.gate": "torch_a2a_router",
    "layers.*.mlp.experts.gate_up_proj": "grouped_gemm",
    "layers.*.mlp.experts.down_proj": "grouped_gemm",
    "layers.*.mlp.experts": "torch_a2a_experts",
}


@pytest.mark.parametrize(
    "name",
    [
        "layers.0.mlp.experts.gate_up_proj",
        "model.layers.0.mlp.experts.gate_up_proj",
        "model.layers.17.mlp.experts.down_proj",
        "model.language_model.layers.3.mlp.experts.down_proj",
        "language_model.model.layers.3.mlp.experts.gate_up_proj",
    ],
)
def test_a_planned_expert_param_matches_under_any_prefix_and_layer_index(name):
    pytest.importorskip("transformers")
    plan = {k: v for k, v in _A2A_PLAN.items() if v == "grouped_gemm"}  # the param keys alone

    assert _make_expert_param_matcher(plan)(name)


@pytest.mark.parametrize(
    "name",
    [
        "backbone.layers.1.mixer.experts.up_proj",
        "backbone.layers.1.mixer.experts.3.down_proj.weight",
        "backbone.layers.12.mixer.experts.0.up_proj.bias",
    ],
)
def test_every_param_under_a_planned_experts_container_matches(name):
    """NemotronH tags only the container (``nemotron_experts_ep``, swapped to the a2a style)."""
    pytest.importorskip("transformers")
    is_expert = _make_expert_param_matcher({"layers.*.mixer.experts": "torch_a2a_experts"})

    assert is_expert(name)


@pytest.mark.parametrize(
    "name",
    [
        "model.layers.0.mlp.gate.weight",
        "model.layers.0.mlp.shared_experts.gate_proj.weight",
        "model.layers.0.self_attn.q_proj.weight",
        "model.embed_tokens.weight",
        "lm_head.weight",
    ],
    ids=["router", "shared-experts", "attention", "embedding", "head"],
)
def test_dense_params_do_not_match(name):
    pytest.importorskip("transformers")
    assert not _make_expert_param_matcher(_A2A_PLAN)(name)


@pytest.mark.parametrize("style", ["ep_router", "torch_a2a_router"])
def test_a_router_is_replicated_not_expert_sharded(style):
    pytest.importorskip("transformers")
    assert not _make_expert_param_matcher({"layers.*.mlp.gate": style})(
        "model.layers.0.mlp.gate.weight"
    )


@pytest.mark.parametrize("plan", [None, {}])
def test_without_a_plan_nothing_is_an_expert(plan):
    """No EP credit then: the load-time warmup falls back to stock sizing."""
    pytest.importorskip("transformers")
    assert not _make_expert_param_matcher(plan)("model.layers.0.mlp.experts.gate_up_proj")


# --------------------------------------------------------------------- _resolve_dense_strategy


def _config(quant_method=None, *, on_text_config=False):
    """An HF config whose checkpoint is natively quantized with ``quant_method`` (if any)."""
    quantization_config = {"quant_method": quant_method} if quant_method else None
    if on_text_config:
        return SimpleNamespace(text_config=SimpleNamespace(quantization_config=quantization_config))
    return SimpleNamespace(quantization_config=quantization_config)


@pytest.mark.parametrize(
    ("dense_strategy", "fsdp_dense"), [("replicated", True), ("sharded", False)]
)
def test_an_explicit_dense_strategy_overrides_the_plan(dense_strategy, fsdp_dense):
    assert _resolve_dense_strategy(dense_strategy, fsdp_dense, _config()) == dense_strategy


@pytest.mark.parametrize(("fsdp_dense", "expected"), [(True, "sharded"), (False, "replicated")])
def test_without_an_explicit_strategy_the_plan_decides(fsdp_dense, expected):
    assert _resolve_dense_strategy(None, fsdp_dense, _config()) == expected


@pytest.mark.parametrize("on_text_config", [False, True], ids=["causal-lm", "vlm"])
@pytest.mark.parametrize("rank", [0, 1])
def test_a_block_fp8_checkpoint_keeps_the_dense_part_replicated(
    monkeypatch, capsys, on_text_config, rank
):
    """DCP cannot dequantize block-fp8 dense weights, while ``from_pretrained`` dequantizes them
    as it replicates. Rank 0 says it fell back."""
    config = _config("fp8", on_text_config=on_text_config)
    monkeypatch.setattr(mto_dist, "is_master", lambda group=None: rank == 0)

    assert _resolve_dense_strategy(None, True, config) == "replicated"
    assert ("block-fp8" in capsys.readouterr().out) == (rank == 0)


@pytest.mark.parametrize("quant_method", ["mxfp4", "compressed-tensors"])
def test_other_native_quantization_keeps_the_dense_part_sharded(quant_method):
    assert _resolve_dense_strategy(None, True, _config(quant_method)) == "sharded"


@pytest.mark.parametrize("dense_strategy", ["fsdp", "Sharded"])
def test_an_unknown_dense_strategy_is_rejected(dense_strategy):
    with pytest.raises(ValueError, match="dense_strategy must be one of"):
        _resolve_dense_strategy(dense_strategy, False, _config())


# ------------------------------------------------------------------------- _ep_dense_meta_load


def _is_expert(name):
    return ".experts." in name


def _get_device_lookups():
    """The modules whose ``get_device`` global the state-dict loaders call."""
    import transformers.core_model_loading as core_model_loading
    import transformers.modeling_utils as modeling_utils

    return modeling_utils, core_model_loading


def test_dense_weights_resolve_to_meta_and_the_rest_to_the_stock_device(monkeypatch):
    """Only a dense ``.weight`` / ``.bias`` is sent to meta. An expert param or a buffer goes to
    the stock lookup, with the caller's arguments, in both modules the loaders read it from."""
    pytest.importorskip("transformers")
    stock_calls = []

    def stock_get_device(device_map, key, *args, **kwargs):
        stock_calls.append((key, args, kwargs))
        return device_map[""]

    lookups = _get_device_lookups()
    for module in lookups:
        monkeypatch.setattr(module, "get_device", stock_get_device)
    device_map = {"": "cuda:0"}
    expert = "model.layers.0.mlp.experts.0.gate_proj.weight"
    buffer = "model.layers.0.mlp.gate.e_score_correction_bias"

    with _ep_dense_meta_load(_is_expert):
        for module in lookups:
            assert module.get_device(device_map, "model.layers.0.self_attn.q_proj.weight") == "meta"
            assert module.get_device(device_map, "model.layers.0.self_attn.q_proj.bias") == "meta"
            assert module.get_device(device_map, expert, valid_torch_device=True) == "cuda:0"
            assert module.get_device(device_map, buffer) == "cuda:0"

    assert stock_calls == [(expert, (), {"valid_torch_device": True}), (buffer, (), {})] * 2


def test_get_device_is_restored_on_exit_even_after_an_error():
    pytest.importorskip("transformers")
    lookups = _get_device_lookups()
    stock = [module.get_device for module in lookups]

    with pytest.raises(RuntimeError, match="load failed"), _ep_dense_meta_load(_is_expert):
        assert all(m.get_device is not f for m, f in zip(lookups, stock))
        raise RuntimeError("load failed")

    assert all(m.get_device is f for m, f in zip(lookups, stock))


def test_from_pretrained_leaves_the_dense_weights_on_meta_and_loads_the_rest(tmp_path):
    """Through a real CPU ``from_pretrained``: every dense weight stays on meta for the FSDP2 +
    DCP load to fill, while the MLPs (standing in for experts) and the rotary buffers load."""
    pytest.importorskip("transformers")
    from _test_utils.torch.transformers_models import get_tiny_llama
    from transformers import AutoModelForCausalLM

    src = get_tiny_llama()
    src.save_pretrained(tmp_path)

    with _ep_dense_meta_load(lambda name: ".mlp." in name):
        model = AutoModelForCausalLM.from_pretrained(tmp_path)

    ref_params, ref_buffers = src.state_dict(), dict(src.named_buffers())
    for name, param in model.named_parameters():
        if ".mlp." in name:
            assert torch.equal(param, ref_params[name]), name
        else:
            assert param.is_meta, name
    for name, buffer in model.named_buffers():
        assert torch.equal(buffer, ref_buffers[name]), name


# ------------------------------------------------------------------------- _fsdp2_shard_dense


@pytest.fixture
def fsdp_mesh(skip_on_windows, tmp_path_factory):
    """The dense FSDP mesh of a one-rank gloo process group (FSDP2 needs a process group)."""
    store = tmp_path_factory.mktemp("pg") / "store"
    torch.distributed.init_process_group("gloo", init_method=store.as_uri(), rank=0, world_size=1)
    yield init_device_mesh("cpu", (1,), mesh_dim_names=("dp",))
    torch.distributed.destroy_process_group()


class _Experts(nn.Module):
    def __init__(self):
        super().__init__()
        self.gate_up_proj = nn.Parameter(torch.randn(4, 8, 4))
        self.down_proj = nn.Parameter(torch.randn(4, 4, 4))


class _DecoderLayer(nn.Module):
    def __init__(self):
        super().__init__()
        self.self_attn = nn.Linear(4, 4)
        self.mlp = nn.Module()
        self.mlp.gate = nn.Linear(4, 4, bias=False)  # the router: dense, sharded with its layer
        self.mlp.experts = _Experts()
        self.input_layernorm = nn.LayerNorm(4)


class _MoeLM(nn.Module):
    """A causal-LM-shaped MoE whose ``_no_split_modules`` names its decoder layer, as HF's do."""

    _no_split_modules = ["_DecoderLayer"]

    def __init__(self):
        super().__init__()
        self.model = nn.Module()
        self.model.embed_tokens = nn.Embedding(16, 4)
        self.model.layers = nn.ModuleList(_DecoderLayer() for _ in range(2))
        self.model.norm = nn.LayerNorm(4)
        self.lm_head = nn.Linear(4, 16, bias=False)

    def experts(self):
        return [layer.mlp.experts for layer in self.model.layers]


def test_each_decoder_layer_is_sharded_around_its_experts(fsdp_mesh):
    """A layer's dense params (attention, router, norms) become shards on the FSDP mesh; its
    experts stay as loaded, for EP. The root's embedding, final norm and head stay replicated, and
    a layer's submodules keep their classes (modelopt recognizes them by class)."""
    model = _MoeLM()
    before = {name: param.detach().clone() for name, param in model.named_parameters()}
    experts = {name: param for name, param in model.named_parameters() if _is_expert(name)}

    assert _fsdp2_shard_dense(model, fsdp_mesh, model.experts()) == 2

    assert all(isinstance(layer, FSDPModule) for layer in model.model.layers)
    assert not isinstance(model, FSDPModule)
    assert type(model.model.layers[0].self_attn) is nn.Linear
    for name, param in model.named_parameters():
        if name in experts:
            assert param is experts[name] and not isinstance(param, DTensor), name
        elif name.startswith("model.layers."):
            assert isinstance(param, DTensor), name
            assert param.device_mesh == fsdp_mesh and param.placements == (Shard(0),), name
            assert torch.equal(param.full_tensor(), before[name]), name
        else:
            assert not isinstance(param, DTensor), name


def test_a_layer_holding_only_experts_is_left_unwrapped(fsdp_mesh):
    model = _MoeLM()
    bare = model.model.layers[1]
    bare.self_attn = bare.mlp.gate = bare.input_layernorm = nn.Identity()

    assert _fsdp2_shard_dense(model, fsdp_mesh, model.experts()) == 1
    assert not isinstance(bare, FSDPModule)


def test_a_dense_param_the_loader_already_sharded_stays_as_it_is(fsdp_mesh):
    """A dense param that arrives as a ``Shard`` DTensor of the loader's own plan is not resharded;
    the rest of its layer still is."""
    tp_mesh = init_device_mesh("cpu", (1,), mesh_dim_names=("tp",))
    model = _MoeLM()
    attn = model.model.layers[0].self_attn
    attn.weight = nn.Parameter(distribute_tensor(attn.weight.detach(), tp_mesh, [Shard(1)]))
    presharded = attn.weight

    assert _fsdp2_shard_dense(model, fsdp_mesh, model.experts()) == 2

    assert attn.weight is presharded and attn.weight.placements == (Shard(1),)
    assert model.model.layers[0].mlp.gate.weight.placements == (Shard(0),)


def test_a_replicated_dense_param_is_sharded_over_the_fsdp_mesh_like_a_plain_one(fsdp_mesh):
    ep_mesh = init_device_mesh("cpu", (1, 1), mesh_dim_names=("edp", "ep"))["ep"]
    model = _MoeLM()
    gate = model.model.layers[0].mlp.gate
    weight = gate.weight.detach().clone()
    gate.weight = nn.Parameter(distribute_tensor(weight, ep_mesh, [Replicate()]))

    _fsdp2_shard_dense(model, fsdp_mesh, model.experts())

    assert gate.weight.device_mesh == fsdp_mesh and gate.weight.placements == (Shard(0),)
    assert torch.equal(gate.weight.full_tensor(), weight)


class _VisionBlock(nn.Module):
    def __init__(self):
        super().__init__()
        self.proj = nn.Linear(4, 4)


class _VisionTower(nn.Module):
    def __init__(self):
        super().__init__()
        self.blocks = nn.ModuleList(_VisionBlock() for _ in range(2))


class _MoeVlm(_MoeLM):
    """Kimi's MoonViT style: ``_no_split_modules`` lists the vision tower *and* its blocks."""

    _no_split_modules = ["_VisionTower", "_VisionBlock", "_DecoderLayer"]

    def __init__(self):
        super().__init__()
        self.vision_tower = _VisionTower()


def test_a_listed_container_is_wrapped_once_not_again_through_its_blocks(fsdp_mesh):
    model = _MoeVlm()

    assert _fsdp2_shard_dense(model, fsdp_mesh, model.experts()) == 3

    assert isinstance(model.vision_tower, FSDPModule)
    assert not any(isinstance(block, FSDPModule) for block in model.vision_tower.blocks)


# ------------------------------------------------------------------ _load_ep_dense_distributed


def test_the_dense_load_shards_first_then_asks_for_exactly_the_meta_dense_params(
    fsdp_mesh, monkeypatch, tmp_path
):
    """As ``_ep_dense_meta_load`` leaves the model: dense weights on meta; experts loaded, here
    per-expert ``nn.Linear``s, whose ``.weight`` only the expert matcher tells from a dense one;
    and a Mamba-style ``A_log`` (no ``.weight`` / ``.bias`` suffix) loaded too. The dense layers
    are FSDP2-sharded before the shared meta+DCP loader runs, and it is asked for exactly the meta
    dense params on this rank's GPU. That load materializes on ``cuda:<local_rank>``, so a
    recorder stands in for it here."""
    with torch.device("meta"):
        model = _MoeLM()
    for layer in model.model.layers:
        layer.mlp.experts = nn.ModuleList(nn.Linear(4, 4, bias=False) for _ in range(4))
        layer.A_log = nn.Parameter(torch.zeros(4))
    loads = []

    def record_load(loaded, local_dir, device, *, load_names):
        sharded = [isinstance(layer.self_attn.weight, DTensor) for layer in loaded.model.layers]
        loads.append((local_dir, device, load_names, sharded))

    monkeypatch.setattr(mto_dist, "_meta_dcp_load", record_load)
    monkeypatch.setattr(mto_dist, "local_rank", lambda: 3)

    wrapped = _load_ep_dense_distributed(
        model, fsdp_mesh, str(tmp_path), _is_expert, model.experts()
    )

    assert wrapped == 2
    meta_dense = {name for name, param in model.named_parameters() if param.is_meta}
    assert loads == [(str(tmp_path), torch.device("cuda", 3), meta_dense, [True, True])]
    assert not any(_is_expert(name) or name.endswith("A_log") for name in meta_dense)
