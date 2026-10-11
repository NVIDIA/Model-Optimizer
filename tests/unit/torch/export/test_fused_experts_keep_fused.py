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

"""Keep-fused MoE expert export used by the no-gather FSDP2 writer (CPU, no GPU needed).

The no-gather writer never splits a fused 3-D expert weight while the model is sharded:
``_export_fused_experts_keep_fused`` folds every expert's quantizer into the fused weight in place,
and ``split_fused_experts_state_dict`` / ``_split_local_fused_module`` cut the result into
per-expert keys later. That pipeline must write exactly what the single-process
``_export_fused_experts`` writes, so most tests compare the two byte for byte on a tiny synthetic
fused-experts module quantized with FP8 and NVFP4. Sharded cases run on a gloo process group: one
rank in process, and two spawned ranks where a non-zero expert offset matters.
"""

import copy
import math
import types
import warnings

import pytest
import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
from _test_utils.torch.distributed.utils import spawn_multiprocess_job
from torch.distributed.checkpoint.state_dict import StateDictOptions, get_model_state_dict
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.fsdp import fully_shard
from torch.distributed.tensor import DTensor, Shard, distribute_tensor

pytest.importorskip("transformers")
pytest.importorskip("safetensors")
from safetensors.torch import load_file, save_file

import modelopt.torch.quantization as mtq
from modelopt.torch.export.moe_utils import (
    _dtensor_dim0_offset,
    _export_fused_experts,
    _export_fused_experts_keep_fused,
    _fused_experts_prefixes,
    _split_local_fused_module,
    split_fused_experts_state_dict,
)
from modelopt.torch.export.registry import ExportContext, ExportModuleRegistry
from modelopt.torch.quantization.nn import QuantModuleRegistry
from modelopt.torch.quantization.qtensor import NVFP4QTensor

NUM_EXPERTS = 4
HIDDEN_DIM = 32
INTERMEDIATE_DIM = 16
PREFIX = "model.layers.0.mlp.experts"

FP8_WEIGHT_ONLY_CFG = copy.deepcopy(mtq.FP8_DEFAULT_CFG)
FP8_WEIGHT_ONLY_CFG["quant_cfg"].append({"quantizer_name": "*input_quantizer", "enable": False})

QUANT_CFGS = pytest.mark.parametrize(
    "quant_cfg",
    [mtq.FP8_DEFAULT_CFG, FP8_WEIGHT_ONLY_CFG, mtq.NVFP4_DEFAULT_CFG],
    ids=["fp8", "fp8_weight_only", "nvfp4"],
)


class _KeepFusedGatedExperts(nn.Module):
    """Fused gate+up experts in the transformers 5.x layout (Qwen3-MoE / Mixtral)."""

    def __init__(self, num_experts=NUM_EXPERTS):
        super().__init__()
        self.num_experts = num_experts
        self.gate_up_proj = nn.Parameter(
            torch.randn(num_experts, 2 * INTERMEDIATE_DIM, HIDDEN_DIM) * 0.02
        )
        self.down_proj = nn.Parameter(torch.randn(num_experts, HIDDEN_DIM, INTERMEDIATE_DIM) * 0.02)
        self.act_fn = nn.SiLU()

    def forward(self, hidden_states, top_k_index, top_k_weights):
        out = torch.zeros_like(hidden_states)
        mask = F.one_hot(top_k_index, num_classes=self.num_experts).permute(2, 1, 0)
        for expert_idx in torch.greater(mask.sum(dim=(-1, -2)), 0).nonzero()[:, 0]:
            top_k_pos, token_idx = torch.where(mask[expert_idx])
            gate, up = F.linear(hidden_states[token_idx], self.gate_up_proj[expert_idx]).chunk(
                2, dim=-1
            )
            current = F.linear(self.act_fn(gate) * up, self.down_proj[expert_idx])
            out.index_add_(0, token_idx, current * top_k_weights[token_idx, top_k_pos, None])
        return out


class _KeepFusedUngatedExperts(nn.Module):
    """Ungated fused experts (NemotronH): a single ``up_proj`` and ``down_proj``."""

    def __init__(self, num_experts=NUM_EXPERTS):
        super().__init__()
        self.num_experts = num_experts
        self.up_proj = nn.Parameter(torch.randn(num_experts, INTERMEDIATE_DIM, HIDDEN_DIM) * 0.02)
        self.down_proj = nn.Parameter(torch.randn(num_experts, HIDDEN_DIM, INTERMEDIATE_DIM) * 0.02)
        self.act_fn = nn.SiLU()

    def forward(self, hidden_states, top_k_index, top_k_weights):
        out = torch.zeros_like(hidden_states)
        mask = F.one_hot(top_k_index, num_classes=self.num_experts).permute(2, 1, 0)
        for expert_idx in torch.greater(mask.sum(dim=(-1, -2)), 0).nonzero()[:, 0]:
            top_k_pos, token_idx = torch.where(mask[expert_idx])
            up = self.act_fn(F.linear(hidden_states[token_idx], self.up_proj[expert_idx]))
            current = F.linear(up, self.down_proj[expert_idx])
            out.index_add_(0, token_idx, current * top_k_weights[token_idx, top_k_pos, None])
        return out


LAYOUTS = pytest.mark.parametrize(
    "experts_cls",
    [_KeepFusedGatedExperts, _KeepFusedUngatedExperts],
    ids=["gated", "ungated"],
)


@pytest.fixture(autouse=True)
def _unregister_synthetic_experts():
    """``mtq.quantize`` registers the synthetic classes on the fly; keep the registry clean."""
    yield
    for cls in (_KeepFusedGatedExperts, _KeepFusedUngatedExperts):
        if QuantModuleRegistry.get(cls) is not None:
            QuantModuleRegistry.unregister(cls)


@pytest.fixture
def gloo_world_of_one(skip_on_windows, tmp_path_factory):
    """A one-rank gloo process group and its CPU device mesh."""
    store = tmp_path_factory.mktemp("pg") / "store"
    dist.init_process_group("gloo", init_method=store.as_uri(), rank=0, world_size=1)
    yield init_device_mesh("cpu", (1,))
    dist.destroy_process_group()


def _quantized_experts(experts_cls, quant_cfg, num_experts=NUM_EXPERTS):
    """Build, quantize and calibrate a fused-experts module; every expert sees one token."""
    torch.manual_seed(0)
    model = nn.Module()
    model.experts = experts_cls(num_experts)

    def forward_loop(m):
        hidden = torch.randn(num_experts, HIDDEN_DIM)
        m.experts(hidden, torch.arange(num_experts).unsqueeze(1), torch.ones(num_experts, 1))

    mtq.quantize(model, quant_cfg, forward_loop)
    return model.experts


def _prefixed(state_dict):
    return {f"{PREFIX}.{key}": value for key, value in state_dict.items()}


def _per_expert_export(experts):
    """What the single-process export writes: ``_export_fused_experts`` on a copy."""
    module = copy.deepcopy(experts)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        _export_fused_experts(module, torch.bfloat16)
    return _prefixed(module.state_dict())


def _keep_fused_export(experts):
    """The fused state ``_export_fused_experts_keep_fused`` leaves on a copy."""
    module = copy.deepcopy(experts)
    _export_fused_experts_keep_fused(module, torch.bfloat16)
    return _prefixed(module.state_dict())


def _raw_bytes(tensor):
    return tensor.detach().contiguous().reshape(-1).view(torch.uint8)


def _assert_byte_identical(got, want):
    assert set(got) == set(want), (sorted(set(want) - set(got)), sorted(set(got) - set(want)))
    for key, ref in want.items():
        assert got[key].dtype == ref.dtype, key
        assert got[key].shape == ref.shape, key
        assert torch.equal(_raw_bytes(got[key]), _raw_bytes(ref)), key


def _assert_quantized(state_dict):
    """Guard against a config that quantized nothing, which would make any comparison trivial."""
    weights = [v for k, v in state_dict.items() if k.endswith(".weight")]
    assert weights, "no per-expert weights"
    assert all(v.dtype in (torch.float8_e4m3fn, torch.uint8) for v in weights)


# ------------------------------------------------------------------ keep-fused + split


@LAYOUTS
@QUANT_CFGS
def test_keep_fused_then_split_writes_exactly_the_per_expert_export(experts_cls, quant_cfg):
    experts = _quantized_experts(experts_cls, quant_cfg)
    want = _per_expert_export(experts)
    _assert_quantized(want)

    _assert_byte_identical(split_fused_experts_state_dict(_keep_fused_export(experts)), want)


@LAYOUTS
@QUANT_CFGS
def test_keep_fused_leaves_one_fused_weight_and_stacked_scales_per_projection(
    experts_cls, quant_cfg
):
    experts = _quantized_experts(experts_cls, quant_cfg)
    first = experts._first_proj_attr
    nvfp4 = quant_cfg is mtq.NVFP4_DEFAULT_CFG
    has_input_scale = quant_cfg is not FP8_WEIGHT_ONLY_CFG

    _export_fused_experts_keep_fused(experts, torch.bfloat16)

    suffixes = {"", "_weight_scale"}
    suffixes |= {"_weight_scale_2"} if nvfp4 else set()
    suffixes |= {"_input_scale"} if has_input_scale else set()
    assert set(experts.state_dict()) == {f"{p}{s}" for p in (first, "down_proj") for s in suffixes}
    for proj in (first, "down_proj"):
        weight = getattr(experts, proj)
        assert weight.dim() == 3 and weight.shape[0] == NUM_EXPERTS
        assert weight.dtype == (torch.uint8 if nvfp4 else torch.float8_e4m3fn)
        assert getattr(experts, f"{proj}_weight_scale").shape[0] == NUM_EXPERTS
        if nvfp4:
            assert getattr(experts, f"{proj}_weight_scale_2").shape == (NUM_EXPERTS,)
        if has_input_scale:
            assert getattr(experts, f"{proj}_input_scale").dim() == 0
    # The quantizers' information now lives in the buffers above.
    assert not [n for n, _ in experts.named_children() if "quantizer" in n]


@LAYOUTS
def test_uncalibrated_expert_takes_the_same_weight_derived_amax_as_the_per_expert_export(
    experts_cls,
):
    """An expert calibration never reached falls back to an amax read off its own weight.

    For the gated layout that amax must come from the whole fused ``gate_up`` slice, so gate and up
    keep one shared ``weight_scale_2`` -- the per-expert export guarantees that, and vLLM's W1/W3
    fusion depends on it.
    """
    experts = _quantized_experts(experts_cls, mtq.NVFP4_DEFAULT_CFG)
    first = experts._first_proj_attr
    for proj in (first, "down_proj"):
        getattr(experts, f"{proj}_weight_quantizers")[1]._amax.zero_()
    want = _per_expert_export(experts)

    with pytest.warns(UserWarning, match=r"Expert 1 .* not calibrated"):
        got = split_fused_experts_state_dict(_keep_fused_export(experts))

    _assert_byte_identical(got, want)
    if first == "gate_up_proj":
        gate_ws2 = got[f"{PREFIX}.1.gate_proj.weight_scale_2"]
        assert torch.equal(gate_ws2, got[f"{PREFIX}.1.up_proj.weight_scale_2"])


# ------------------------------------------------------------------ split_fused_experts_state_dict


LOCAL_EXPERTS = 2  # experts in each hand-built fused entry below


def _fused_state_dict(*, gated, rows=4, cols=6):
    """A fused-experts entry with position-coded values, so every slice is identifiable."""
    first = "gate_up_proj" if gated else "up_proj"
    fused_rows = 2 * rows if gated else rows
    n = LOCAL_EXPERTS
    return {
        f"{PREFIX}.{first}": torch.arange(n * fused_rows * cols, dtype=torch.float32)
        .reshape(n, fused_rows, cols)
        .to(torch.float8_e4m3fn),
        # Per-row (per-channel) weight scale: its rows are split along with the weight's.
        f"{PREFIX}.{first}_weight_scale": torch.arange(n * fused_rows).view(n, fused_rows) + 100.0,
        # Per-expert scalars.
        f"{PREFIX}.{first}_weight_scale_2": torch.arange(n) + 7.0,
        f"{PREFIX}.{first}_input_scale": torch.tensor(0.5),
        f"{PREFIX}.down_proj": torch.arange(n * cols * rows, dtype=torch.float32)
        .reshape(n, cols, rows)
        .to(torch.float8_e4m3fn),
        f"{PREFIX}.down_proj_weight_scale": torch.arange(n) + 3.0,
        f"{PREFIX}.down_proj_input_scale": torch.tensor(0.25),
    }


def test_split_cuts_gate_and_up_rows_with_their_scales():
    sd = _fused_state_dict(gated=True)
    fused = dict(sd)
    untouched = torch.ones(3)
    sd["model.layers.0.mlp.gate.weight"] = untouched

    out = split_fused_experts_state_dict(sd)

    gate_up, ws = fused[f"{PREFIX}.gate_up_proj"], fused[f"{PREFIX}.gate_up_proj_weight_scale"]
    for e in range(LOCAL_EXPERTS):
        base = f"{PREFIX}.{e}"
        assert torch.equal(_raw_bytes(out[f"{base}.gate_proj.weight"]), _raw_bytes(gate_up[e, :4]))
        assert torch.equal(_raw_bytes(out[f"{base}.up_proj.weight"]), _raw_bytes(gate_up[e, 4:]))
        assert torch.equal(out[f"{base}.gate_proj.weight_scale"], ws[e, :4])
        assert torch.equal(out[f"{base}.up_proj.weight_scale"], ws[e, 4:])
        # gate and up are halves of one projection: they share its per-tensor second scale.
        for proj in ("gate_proj", "up_proj"):
            assert out[f"{base}.{proj}.weight_scale_2"].item() == 7.0 + e
            assert out[f"{base}.{proj}.input_scale"].item() == 0.5
        down = fused[f"{PREFIX}.down_proj"][e]
        assert torch.equal(_raw_bytes(out[f"{base}.down_proj.weight"]), _raw_bytes(down))
        assert out[f"{base}.down_proj.weight_scale"].item() == 3.0 + e
        assert out[f"{base}.down_proj.input_scale"].item() == 0.25
    assert not any(k in out for k in fused), "fused entries must be consumed"
    assert out["model.layers.0.mlp.gate.weight"] is untouched


def test_split_keeps_an_ungated_up_proj_whole():
    sd = _fused_state_dict(gated=False)
    up = sd[f"{PREFIX}.up_proj"]

    out = split_fused_experts_state_dict(sd)

    assert not any(".gate_proj." in k for k in out)
    for e in range(LOCAL_EXPERTS):
        assert torch.equal(_raw_bytes(out[f"{PREFIX}.{e}.up_proj.weight"]), _raw_bytes(up[e]))
        assert torch.equal(
            out[f"{PREFIX}.{e}.up_proj.weight_scale"], sd[f"{PREFIX}.up_proj_weight_scale"][e]
        )
    assert f"{PREFIX}.up_proj" not in out


def test_split_gives_every_tensor_its_own_storage(tmp_path):
    """Slices alias the fused parent, and gate/up share one scale per expert.

    Aliasing keeps the whole fused tensor alive behind every per-expert key, and a shared scale is
    refused by safetensors (``save_pretrained`` dedups it away instead).
    """
    out = split_fused_experts_state_dict(_fused_state_dict(gated=True))

    storages = [t.untyped_storage().data_ptr() for t in out.values()]
    assert len(set(storages)) == len(storages)
    path = tmp_path / "experts.safetensors"
    save_file(out, str(path))
    reloaded = load_file(str(path))
    assert set(reloaded) == set(out)
    for key, value in out.items():
        assert torch.equal(_raw_bytes(reloaded[key]), _raw_bytes(value)), key


def test_split_leaves_a_state_dict_without_fused_experts_unchanged():
    sd = {
        "model.embed_tokens.weight": torch.ones(4, 2),
        f"{PREFIX}.0.up_proj.weight": torch.ones(4, 2),  # already per-expert
    }

    out = split_fused_experts_state_dict(sd)

    assert out.keys() == sd.keys()
    assert all(out[k] is sd[k] for k in sd)


def test_fused_experts_prefixes_lists_each_fused_module_once_in_order():
    fused = torch.zeros(2, 4, 4)
    sd = {
        "model.layers.1.mlp.experts.gate_up_proj": fused,
        "model.layers.1.mlp.experts.down_proj": fused,
        "model.layers.1.mlp.experts.gate_up_proj_weight_scale": fused,
        "model.layers.0.mlp.experts.up_proj": fused,
        "model.layers.0.mlp.experts.down_proj": fused,
        # Not a fused-experts weight: already split, 2-D, outside an ``experts`` module, no tensor.
        "model.layers.2.mlp.experts.0.down_proj": fused,
        "model.layers.3.mlp.experts.down_proj": torch.zeros(4, 4),
        "model.layers.4.mlp.gate_up_proj": fused,
        "model.layers.5.mlp.experts.down_proj": "not a tensor",
    }
    assert _fused_experts_prefixes(sd) == [
        "model.layers.0.mlp.experts",
        "model.layers.1.mlp.experts",
    ]


@pytest.mark.parametrize(
    ("ep_rank", "total_experts", "first_index"),
    [
        (0, 0, 0),
        # This EP group's slice of a larger expert set: shift by the group base.
        (1, 2 * LOCAL_EXPERTS, LOCAL_EXPERTS),
        (3, 4 * LOCAL_EXPERTS, 3 * LOCAL_EXPERTS),
        # The tensor already spans every expert, so its indices are global.
        (1, LOCAL_EXPERTS, 0),
        # Total unknown: the legacy per-group shift.
        (1, 0, LOCAL_EXPERTS),
    ],
)
def test_split_local_numbers_experts_globally(ep_rank, total_experts, first_index):
    sd = _fused_state_dict(gated=True)
    unshifted = split_fused_experts_state_dict(dict(sd))

    out = _split_local_fused_module(PREFIX, sd, ep_rank=ep_rank, total_experts=total_experts)

    want = {}
    for key, value in unshifted.items():
        expert, tail = key[len(PREFIX) + 1 :].split(".", 1)
        want[f"{PREFIX}.{int(expert) + first_index}.{tail}"] = value
    _assert_byte_identical(out, want)


# ------------------------------------------------------------------ sharded (DTensor) experts


def _shard_fused_weights(experts, mesh):
    for proj in (experts._first_proj_attr, "down_proj"):
        weight = getattr(experts, proj).data
        sharded = distribute_tensor(weight, mesh, [Shard(0)])
        setattr(experts, proj, nn.Parameter(sharded, requires_grad=False))
    return experts


@LAYOUTS
@QUANT_CFGS
def test_keep_fused_on_a_sharded_weight_stays_sharded_and_splits_like_the_per_expert_export(
    gloo_world_of_one, experts_cls, quant_cfg
):
    experts = _quantized_experts(experts_cls, quant_cfg)
    plain = _keep_fused_export(experts)
    sharded = _shard_fused_weights(copy.deepcopy(experts), gloo_world_of_one)

    _export_fused_experts_keep_fused(sharded, torch.bfloat16)

    fused = _prefixed(sharded.state_dict())
    assert set(fused) == set(plain)
    for key, value in fused.items():
        if key.endswith("_input_scale"):  # one scalar shared by every expert
            assert not isinstance(value, DTensor), key
        else:  # weights and per-expert scales stay sharded on the expert axis
            assert isinstance(value, DTensor), key
            assert value.placements == (Shard(0),), key
            assert value.shape == plain[key].shape, key
            value = value.to_local()
        assert torch.equal(_raw_bytes(value), _raw_bytes(plain[key])), key
    assert _dtensor_dim0_offset(fused[f"{PREFIX}.{sharded._first_proj_attr}"]) == 0

    split = _split_local_fused_module(PREFIX, fused)

    assert not any(isinstance(v, DTensor) for v in split.values())
    _assert_byte_identical(split, _per_expert_export(experts))


def test_keep_fused_keeps_the_expert_count_of_an_uneven_shard(gloo_world_of_one):
    """Two experts held of three is what rank 0 holds when three experts split over two ranks."""
    experts = _quantized_experts(_KeepFusedGatedExperts, mtq.FP8_DEFAULT_CFG, num_experts=3)
    for proj in ("gate_up_proj", "down_proj"):
        full = getattr(experts, proj).data
        held = DTensor.from_local(
            full[:2].clone(),
            gloo_world_of_one,
            [Shard(0)],
            run_check=False,
            shape=full.shape,
            stride=full.stride(),
        )
        setattr(experts, proj, nn.Parameter(held, requires_grad=False))

    _export_fused_experts_keep_fused(experts, torch.bfloat16)

    assert experts.gate_up_proj.shape[0] == 3
    assert experts.gate_up_proj_weight_scale.shape[0] == 3


def _two_rank_split_job(rank, world_size):
    mesh = init_device_mesh("cpu", (world_size,))
    for experts_cls, quant_cfg in [
        (_KeepFusedGatedExperts, mtq.FP8_DEFAULT_CFG),
        (_KeepFusedUngatedExperts, mtq.NVFP4_DEFAULT_CFG),
    ]:
        # Every rank calibrates the same seeded module, so all ranks hold identical quantizers.
        experts = _quantized_experts(experts_cls, quant_cfg)
        want = _per_expert_export(experts)
        sharded = _shard_fused_weights(copy.deepcopy(experts), mesh)
        _export_fused_experts_keep_fused(sharded, torch.bfloat16)
        fused = _prefixed(sharded.state_dict())
        first = fused[f"{PREFIX}.{sharded._first_proj_attr}"]
        n_local = first.to_local().shape[0]
        offset = rank * math.ceil(NUM_EXPERTS / world_size)
        assert _dtensor_dim0_offset(first) == offset
        mine = {
            k: v
            for k, v in want.items()
            if offset <= int(k[len(PREFIX) + 1 :].split(".")[0]) < offset + n_local
        }
        assert mine, "this rank owns no experts"

        _assert_byte_identical(_split_local_fused_module(PREFIX, fused), mine)

        # A per-expert scale held as a plain tensor over ALL experts (as keep-fused emits it for an
        # unsharded weight) must be cut to this rank's experts before pairing.
        unsharded = _keep_fused_export(experts)
        plain_scales = {
            k: unsharded[k] if k.endswith(("_weight_scale", "_weight_scale_2")) else v
            for k, v in fused.items()
        }
        _assert_byte_identical(_split_local_fused_module(PREFIX, plain_scales), mine)


def test_two_ranks_fold_and_split_their_own_experts_under_global_indices(skip_on_windows):
    """Rank 1 holds experts 2-3: each must be folded with ITS quantizer and keep its index."""
    spawn_multiprocess_job(2, _two_rank_split_job, backend="gloo")


# ------------------------------------------------------------------ NVFP4 on sharded weights


def test_nvfp4_block_scale_maps_an_all_zero_block_to_one():
    weight = torch.randn(2, 32)
    weight[0, :16] = 0.0  # an all-zero block
    weight[1, :16] = 0.0
    weight[1, 0] = 6.0  # amax 6 at a second-level scale of 1 -> block scale exactly 1

    scale, _ = NVFP4QTensor.get_weights_scaling_factor(weight, 16, torch.tensor(1.0))

    assert scale.float()[0, 0].item() == 1.0
    assert scale.float()[1, 0].item() == 1.0


def test_nvfp4_static_block_scale_maps_a_zero_amax_block_to_one():
    # Two rows of two blocks: a zero-amax block next to ones whose amax maps to a scale of 1.
    quantizer = types.SimpleNamespace(
        block_sizes={-1: 16},
        global_amax=torch.tensor(6.0 * 448.0),
        _amax=torch.tensor([0.0, 6.0, 6.0, 3.0]),
    )

    scale, _ = NVFP4QTensor.get_weights_scaling_factor_from_quantizer(quantizer, torch.randn(2, 32))

    assert scale.float()[0, 0] == scale.float()[0, 1] == scale.float()[1, 0]
    assert scale.float()[1, 1] != scale.float()[1, 0]


def test_nvfp4_block_scale_accepts_a_sharded_weight(gloo_world_of_one):
    weight = torch.randn(4, 32)
    weight[1, :16] = 0.0
    sharded = distribute_tensor(weight, gloo_world_of_one, [Shard(0)])

    want, want_2 = NVFP4QTensor.get_weights_scaling_factor(weight, 16)
    got, got_2 = NVFP4QTensor.get_weights_scaling_factor(sharded, 16)

    assert isinstance(got, DTensor) and got.placements == (Shard(0),)
    assert torch.equal(_raw_bytes(got.to_local()), _raw_bytes(want))
    assert torch.equal(got_2, want_2)


def test_nvfp4_cast_keeps_the_global_shape_of_an_uneven_shard(gloo_world_of_one):
    """``from_local`` would infer local x world rows; an uneven shard must keep the real total.

    Two rows held of a three-row tensor is what one rank holds when three rows split over two.
    """
    local = torch.randn(2, 32) * 4
    sharded = DTensor.from_local(
        local.clone(),
        gloo_world_of_one,
        [Shard(0)],
        run_check=False,
        shape=torch.Size([3, 32]),
        stride=(32, 1),
    )

    packed = NVFP4QTensor._cast_fp4(sharded)

    assert isinstance(packed, DTensor) and packed.placements == (Shard(0),)
    assert packed.shape == (3, 32)
    assert torch.equal(packed.to_local(), NVFP4QTensor._cast_fp4(local.clone()))


# ------------------------------------------------------------------ export handlers


@pytest.mark.parametrize("keep_fused", [True, False], ids=["keep_fused", "split"])
def test_fused_experts_handler_keeps_experts_fused_only_when_asked(keep_fused):
    experts = _quantized_experts(_KeepFusedGatedExperts, mtq.FP8_DEFAULT_CFG)
    container = nn.Module()
    container.experts = experts
    assert ExportContext(model=container, dtype=torch.bfloat16).keep_fused_experts is False
    ctx = ExportContext(model=container, dtype=torch.bfloat16, keep_fused_experts=keep_fused)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        ExportModuleRegistry.match(experts)(PREFIX, experts, ctx)

    names = set(experts.state_dict())
    if keep_fused:
        assert {"gate_up_proj", "gate_up_proj_weight_scale", "down_proj"} <= names
        assert not any(n.startswith("0.") for n in names)
    else:
        assert "0.gate_proj.weight" in names
        assert "gate_up_proj" not in names


def test_quant_linear_handler_packs_an_fsdp2_sharded_nvfp4_weight(gloo_world_of_one):
    """On the no-gather path the model is still sharded when a linear is packed.

    Packing NVFP4 straight off the sharded weight is not possible (the in-place cast on the
    DTensor's local view is refused under autograd), so the handler packs inside a weight-update
    window; the sharded state dict must then carry the packed weight, as an unsharded pack has it.
    """
    torch.manual_seed(0)
    model = nn.Sequential(nn.Linear(32, 16, bias=False), nn.Linear(16, 8, bias=False))
    mtq.quantize(model, mtq.NVFP4_DEFAULT_CFG, lambda m: m(torch.randn(4, 32)))
    ref = copy.deepcopy(model)
    ExportModuleRegistry.match(ref[0])("0", ref[0], ExportContext(model=ref, dtype=torch.bfloat16))
    fully_shard(model[0], mesh=gloo_world_of_one)
    fully_shard(model, mesh=gloo_world_of_one)
    ctx = ExportContext(model=model, dtype=torch.bfloat16, keep_fused_experts=True)

    ExportModuleRegistry.match(model[0])("0", model[0], ctx)

    sd = get_model_state_dict(model, options=StateDictOptions(full_state_dict=False))
    packed = sd["0.weight"]
    assert isinstance(packed, DTensor)
    assert packed.dtype == torch.uint8
    assert torch.equal(packed.full_tensor(), ref[0].weight)
    for name in ("weight_scale", "weight_scale_2", "input_scale"):
        assert torch.equal(_raw_bytes(sd[f"0.{name}"]), _raw_bytes(getattr(ref[0], name))), name


def test_nvfp4_packs_a_sharded_weight_into_the_same_codes(gloo_world_of_one):
    """``DTensor << int`` is a silent no-op, so the nibble pack must not rely on the operator."""
    torch.manual_seed(0)
    weight = torch.randn(4, 32, dtype=torch.bfloat16)
    scale_2 = (weight.abs().amax().float() / (6.0 * 448.0)).reshape(())
    scale = NVFP4QTensor.get_weights_scaling_factor(weight, 16, scale_2)[0]
    want = NVFP4QTensor.quantize(weight, 16, scale, scale_2)[0]._quantized_data

    sharded = distribute_tensor(weight, gloo_world_of_one, [Shard(0)])
    sharded_scale = NVFP4QTensor.get_weights_scaling_factor(sharded, 16, scale_2)[0]
    packed = NVFP4QTensor.quantize(sharded, 16, sharded_scale, scale_2)[0]._quantized_data

    assert torch.equal(packed.full_tensor(), want)
