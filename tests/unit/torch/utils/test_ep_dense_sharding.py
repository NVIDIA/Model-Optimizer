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
from the dense ones, the distribution plan says whether the dense part is replicated or sharded,
``_ep_dense_meta_load`` keeps the dense weights off the device during ``from_pretrained`` when they
are sharded, and ``_fsdp2_shard_dense`` /
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

pytest.importorskip("transformers")

import transformers.core_model_loading as core_model_loading
import transformers.modeling_utils as modeling_utils
from _test_utils.torch.transformers_models import get_tiny_llama, get_tiny_qwen3_moe
from safetensors.torch import load_file, save_file
from transformers import AutoModelForCausalLM

import modelopt.torch.utils.moe as moe
import modelopt.torch.utils.plugins.model_load_utils as model_load_utils
from modelopt.torch.utils.distributed import DistributionPlan
from modelopt.torch.utils.plugins.model_load_utils import (
    _ep_aware_caching_allocator_warmup,
    _ep_dense_meta_load,
    _fsdp2_shard_dense,
    _load_ep_dense_distributed,
    _make_expert_param_matcher,
    load_and_prepare_ep,
)

# ------------------------------------------------------------------ _make_expert_param_matcher

# The plan load_and_prepare_ep shards with after the a2a swap: base-relative keys (no ``model.``).
_A2A_PLAN = {
    "layers.*.mlp.gate": "torch_a2a_router",
    "layers.*.mlp.experts.gate_up_proj": "grouped_gemm",
    "layers.*.mlp.experts.down_proj": "grouped_gemm",
    "layers.*.mlp.experts": "torch_a2a_experts",
}


# Only the per-parameter keys: a container key would match the same names on its own.
_PARAM_KEYS = {k: v for k, v in _A2A_PLAN.items() if v == "grouped_gemm"}


@pytest.mark.parametrize(
    ("plan", "name", "expected"),
    [
        pytest.param(_PARAM_KEYS, "layers.0.mlp.experts.gate_up_proj", True, id="key-unprefixed"),
        pytest.param(
            _PARAM_KEYS, "model.language_model.layers.3.mlp.experts.down_proj", True, id="vlm"
        ),
        pytest.param(
            _PARAM_KEYS,
            "language_model.model.layers.3.mlp.experts.gate_up_proj",
            True,
            id="vlm-wrapper",
        ),
        # NemotronH tags only the container; every param under it is an expert's.
        pytest.param(
            {"layers.*.mixer.experts": "torch_a2a_experts"},
            "backbone.layers.1.mixer.experts.3.down_proj.weight",
            True,
            id="under-a-container",
        ),
        pytest.param(
            _A2A_PLAN, "model.layers.0.mlp.shared_experts.gate_proj.weight", False, id="shared"
        ),
        pytest.param(
            {"layers.*.mlp.gate": "ep_router"}, "model.layers.0.mlp.gate.weight", False, id="router"
        ),
        # No EP credit then: the load-time warmup falls back to stock sizing.
        pytest.param(None, "model.layers.0.mlp.experts.gate_up_proj", False, id="no-plan"),
    ],
)
def test_expert_param_matcher(plan, name, expected):
    """A param is an expert's when the plan says so, under any prefix the base-relative keys omit;
    a router is replicated whatever its style. The warmup sizing tests classify a whole MoE's
    experts, router, attention, embedding and head with the same matcher."""
    assert _make_expert_param_matcher(plan)(name) is expected


# ------------------------------------------------------------------ a block-fp8 checkpoint


def test_a_block_fp8_checkpoint_is_refused_as_a_format_whatever_its_dense_layout(
    monkeypatch, gloo_world_of_one
):
    """This load does not dequantize block-fp8 checkpoints at all, so that is the refusal a plan
    sharding their dense part meets, not one advising to replicate it."""
    monkeypatch.setattr("modelopt.torch.utils.distributed.is_initialized", lambda: True)
    monkeypatch.setattr("modelopt.torch.utils.distributed.size", lambda group=None: 2)
    monkeypatch.setattr(moe, "require_transformers", lambda *args: None)
    config = SimpleNamespace(quantization_config={"quant_method": "fp8"})

    with pytest.raises(NotImplementedError, match="cannot dequantize a 'fp8' checkpoint"):
        load_and_prepare_ep("unused", DistributionPlan(2, 2, "fsdp"), hf_config=config)


# ------------------------------------------------------------------------- _ep_dense_meta_load


def _is_expert(name):
    return ".experts." in name


def _get_device_lookups():
    """The modules whose ``get_device`` global the state-dict loaders call."""
    return modeling_utils, core_model_loading


def test_dense_weights_resolve_to_meta_and_the_rest_to_the_stock_device(monkeypatch):
    """Only a dense ``.weight`` / ``.bias`` is sent to meta. An expert param or a buffer goes to
    the stock lookup, with the caller's arguments, in both modules the loaders read it from."""
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


def test_the_dense_load_refuses_a_transformers_without_its_get_device_hook(monkeypatch):
    """Without the hook the dense part would load whole on every rank, the memory sharding it saves."""
    for module in _get_device_lookups():
        monkeypatch.delattr(module, "get_device", raising=False)

    with pytest.raises(RuntimeError, match="get_device"), _ep_dense_meta_load(_is_expert):
        pass


def _get_device_patch():
    return _ep_dense_meta_load(_is_expert), [(m, "get_device") for m in _get_device_lookups()]


def _warmup_sizing_patch():
    if not hasattr(modeling_utils, "get_total_byte_count"):
        pytest.skip("transformers has no get_total_byte_count to size")
    sizing = _ep_aware_caching_allocator_warmup(4, _A2A_PLAN)
    return sizing, [(modeling_utils, "get_total_byte_count")]


@pytest.mark.parametrize(
    "patch", [_get_device_patch, _warmup_sizing_patch], ids=lambda f: f.__name__
)
def test_a_load_time_patch_is_undone_even_when_the_load_raises(patch):
    """``from_pretrained`` reads these transformers globals; a failed load leaves stock ones."""
    context, patched = patch()
    stock = [getattr(module, attr) for module, attr in patched]

    with pytest.raises(RuntimeError, match="load failed"), context:
        assert all(getattr(m, a) is not f for (m, a), f in zip(patched, stock))
        raise RuntimeError("load failed")

    assert all(getattr(m, a) is f for (m, a), f in zip(patched, stock))


def test_from_pretrained_leaves_the_dense_weights_on_meta_and_loads_the_rest(tmp_path):
    """Through a real CPU ``from_pretrained``: every dense weight stays on meta for the FSDP2 +
    DCP load to fill, while the MLPs (standing in for experts) and the rotary buffers load."""
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


def test_a_param_off_the_models_dtype_stays_replicated(fsdp_mesh):
    """FSDP2 shards one dtype per group, and HF keeps a few params in fp32, as a router: such a
    param stays replicated, as the pure-FSDP2 wrap leaves it, and its layer still unshards."""
    model = _MoeLM().to(torch.bfloat16)
    gate = model.model.layers[0].mlp.gate.float()

    assert _fsdp2_shard_dense(model, fsdp_mesh, model.experts()) == 2

    assert not isinstance(gate.weight, DTensor) and gate.weight.dtype == torch.float32
    for layer in model.model.layers:
        layer.unshard()  # FSDP2 checks a group's dtypes here


def test_a_layer_holding_only_experts_is_left_unwrapped(fsdp_mesh):
    model = _MoeLM()
    bare = model.model.layers[1]
    bare.self_attn = bare.mlp.gate = bare.input_layernorm = nn.Identity()

    assert _fsdp2_shard_dense(model, fsdp_mesh, model.experts()) == 1
    assert not isinstance(bare, FSDPModule)


def test_a_dense_part_left_with_nothing_to_shard_is_refused(fsdp_mesh):
    """Experts outweighing the dense part in another dtype put every dense param off the model's
    dominant dtype, out of the shard groups: no layer is left to shard, and the load would hold the
    whole dense part on every rank."""
    model = _MoeLM()
    for experts in model.experts():
        experts.to(torch.bfloat16)

    with pytest.raises(RuntimeError, match="no dense parameter to shard"):
        _fsdp2_shard_dense(model, fsdp_mesh, model.experts())


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


class _ShardMesh:
    """The meshes, as far as the loader reads them before it loads the dense part."""

    device_type = "cpu"

    def __getitem__(self, name):
        return self

    def size(self):
        return 2

    def get_group(self):
        return -1


def test_a_dense_part_from_pretrained_did_not_leave_on_meta_is_refused(
    monkeypatch, gloo_world_of_one
):
    """A transformers that places the dense part without asking ``get_device`` loads it whole on every
    rank, the memory this layout exists to save: the load refuses that before sharding anything."""

    class _DenseLoadedError(Exception):
        """Raised by the dense load: the loader got past the check."""

    def dense_load(*args, **kwargs):
        raise _DenseLoadedError

    loader = SimpleNamespace(__name__="Loader", from_pretrained=lambda path, **kwargs: _MoeLM())
    monkeypatch.setattr("modelopt.torch.utils.distributed.is_initialized", lambda: True)
    monkeypatch.setattr("modelopt.torch.utils.distributed.size", lambda group=None: 2)
    monkeypatch.setattr(moe, "require_transformers", lambda *args: None)
    monkeypatch.setattr(
        "torch.distributed.device_mesh.init_device_mesh", lambda *a, **k: _ShardMesh()
    )
    monkeypatch.setattr(model_load_utils, "ensure_local_checkpoint", lambda p, **k: (None, p))
    monkeypatch.setattr(model_load_utils, "_auto_class", lambda *args, **kwargs: loader)
    monkeypatch.setattr(model_load_utils, "record_unplaced_source_keys", lambda *args: [])
    monkeypatch.setattr(model_load_utils, "_load_ep_dense_distributed", dense_load)
    config = SimpleNamespace(
        base_model_ep_plan={
            "layers.*.mlp.gate": "ep_router",
            "layers.*.mlp.experts": "moe_tp_experts",
            "layers.*.mlp.experts.gate_up_proj": "grouped_gemm",
            "layers.*.mlp.experts.down_proj": "grouped_gemm",
        },
        num_experts=4,
    )

    with pytest.raises(RuntimeError, match="no dense param came back on meta"):
        load_and_prepare_ep("unused", DistributionPlan(2, 2, "fsdp"), hf_config=config)


# ------------------------------------------------------------------ _load_ep_dense_distributed


def test_the_dense_load_shards_first_then_asks_for_exactly_the_meta_dense_params(
    fsdp_mesh, monkeypatch, tmp_path
):
    """As ``_ep_dense_meta_load`` leaves the model: dense weights on meta; experts loaded, here
    per-expert ``nn.Linear``s, whose ``.weight`` only the expert matcher tells from a dense one;
    and a Mamba-style ``A_log`` (no ``.weight`` / ``.bias`` suffix) loaded too. The dense layers
    are FSDP2-sharded before the shared meta+DCP loader runs, and it is asked for exactly the meta
    dense params, on the mesh's device. A recorder stands in for that load."""
    with torch.device("meta"):
        model = _MoeLM()
    for layer in model.model.layers:
        layer.mlp.experts = nn.ModuleList(nn.Linear(4, 4, bias=False) for _ in range(4))
        layer.A_log = nn.Parameter(torch.zeros(4))
    loads = []

    def record_load(loaded, local_dir, device, *, load_names):
        sharded = [isinstance(layer.self_attn.weight, DTensor) for layer in loaded.model.layers]
        loads.append((local_dir, device, load_names, sharded))

    monkeypatch.setattr(model_load_utils, "_meta_dcp_load", record_load)

    wrapped = _load_ep_dense_distributed(
        model, fsdp_mesh, str(tmp_path), _is_expert, model.experts()
    )

    assert wrapped == 2
    meta_dense = {name for name, param in model.named_parameters() if param.is_meta}
    assert loads == [(str(tmp_path), torch.device("cpu"), meta_dense, [True, True])]
    assert not any(_is_expert(name) or name.endswith("A_log") for name in meta_dense)


def test_a_tied_head_the_checkpoint_stores_alone_reaches_both_aliases(fsdp_mesh, tmp_path):
    """A tied checkpoint can store the pair as ``lm_head.weight`` alone, which ``from_pretrained``
    loads into both aliases. With a tied MoE as the expert-parallel load leaves it -- the dense part
    on meta and tied, the experts loaded -- the dense load fills both aliases with it too."""
    src = get_tiny_qwen3_moe(tie_word_embeddings=True)
    src.save_pretrained(tmp_path)
    weights = tmp_path / "model.safetensors"
    stored = load_file(weights)
    stored["lm_head.weight"] = stored.pop("model.embed_tokens.weight")
    save_file(stored, weights, metadata={"format": "pt"})
    with _ep_dense_meta_load(_is_expert):
        model = AutoModelForCausalLM.from_pretrained(tmp_path)
    model.tie_weights()  # as load_and_prepare_ep does once the weights load
    assert model.lm_head.weight is model.model.embed_tokens.weight
    assert model.lm_head.weight.is_meta

    experts = [layer.mlp.experts for layer in model.model.layers]
    _load_ep_dense_distributed(model, fsdp_mesh, str(tmp_path), _is_expert, experts)

    for alias in (model.lm_head.weight, model.model.embed_tokens.weight):
        assert torch.equal(alias, src.lm_head.weight)
