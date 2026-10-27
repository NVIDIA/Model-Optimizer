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

import contextlib
import copy
import math
import re
import warnings

import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributed.checkpoint.state_dict import StateDictOptions, get_model_state_dict
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.fsdp import FSDPModule, fully_shard
from torch.distributed.tensor import DTensor, Shard, distribute_tensor

pytest.importorskip("transformers")
pytest.importorskip("safetensors")
from safetensors.torch import load_file, save_file

import modelopt.torch.export.hf_export_handlers as handlers
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
from modelopt.torch.quantization.config import QuantizerAttributeConfig
from modelopt.torch.quantization.ggml import GGML_FORMAT_REGISTRY
from modelopt.torch.quantization.nn import QuantModuleRegistry, SequentialQuantizer, TensorQuantizer
from modelopt.torch.quantization.qtensor import NVFP4QTensor
from modelopt.torch.utils.distributed import DistributionPlan, set_distribution_plan

NUM_EXPERTS = 4
HIDDEN_DIM = 32
INTERMEDIATE_DIM = 16
PREFIX = "model.layers.0.mlp.experts"

FP8_WEIGHT_ONLY_CFG = copy.deepcopy(mtq.FP8_DEFAULT_CFG)
FP8_WEIGHT_ONLY_CFG["quant_cfg"].append({"quantizer_name": "*input_quantizer", "enable": False})


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


def _uncalibrated(experts, projections):
    """Expert 1's calibration never ran: zero its weight amax."""
    for proj in projections:
        getattr(experts, f"{proj}_weight_quantizers")[1]._amax.zero_()


def _unquantized(experts, projections):
    """A recipe left these projections' experts unquantized."""
    for proj in projections:
        for quantizer in getattr(experts, f"{proj}_weight_quantizers"):
            quantizer.disable()


def _with_source_scales(experts, projections):
    """A native-FP8 checkpoint's experts keep their source block and activation scales."""
    for proj in projections:
        experts.register_buffer(f"{proj}_scale_inv", torch.ones(NUM_EXPERTS, 1, 1))
        experts.register_buffer(f"{proj}_activation_scale", torch.ones(()))


def _export_cases():
    layouts = [(_KeepFusedGatedExperts, "gated"), (_KeepFusedUngatedExperts, "ungated")]
    cfgs = [
        (mtq.FP8_DEFAULT_CFG, "fp8"),
        (FP8_WEIGHT_ONLY_CFG, "fp8_weight_only"),
        (mtq.NVFP4_DEFAULT_CFG, "nvfp4"),
    ]
    cases = [
        pytest.param(cls, cfg, None, (), id=f"{name}-{cfg_name}")
        for cls, name in layouts
        for cfg, cfg_name in cfgs
    ]
    cases += [
        pytest.param(
            cls,
            mtq.NVFP4_DEFAULT_CFG,
            _uncalibrated,
            ("first", "down_proj"),
            id=f"{name}-uncalibrated",
        )
        for cls, name in layouts
    ]
    cases += [
        pytest.param(
            cls,
            mtq.FP8_DEFAULT_CFG,
            _with_source_scales,
            ("first", "down_proj"),
            id=f"{name}-source-scales",
        )
        for cls, name in layouts
    ]
    cases += [
        pytest.param(
            cls, mtq.FP8_DEFAULT_CFG, _unquantized, projs, id=f"{name}-unquantized-{which}"
        )
        for cls, name in layouts
        for which, projs in (
            ("first", ("first",)),
            ("down", ("down_proj",)),
            ("both", ("first", "down_proj")),
        )
    ]
    return cases


@pytest.mark.parametrize(("experts_cls", "quant_cfg", "change", "projections"), _export_cases())
def test_keep_fused_then_split_writes_exactly_the_per_expert_export(
    experts_cls, quant_cfg, change, projections
):
    """Also when an expert calibration never reached falls back to an amax read off its own weight.

    For the gated layout that amax must come from the whole fused ``gate_up`` slice, so gate and up
    keep one shared ``weight_scale_2``, as the per-expert export guarantees and vLLM's W1/W3 fusion
    needs. And when a recipe leaves a projection's experts unquantized: it has nothing to fold.
    """
    experts = _quantized_experts(experts_cls, quant_cfg)
    if change is not None:
        first = experts._first_proj_attr
        change(experts, [first if proj == "first" else proj for proj in projections])
    want = _per_expert_export(experts)
    if change is None:
        _assert_quantized(want)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")  # an uncalibrated expert warns
        got = split_fused_experts_state_dict(_keep_fused_export(experts))

    _assert_byte_identical(got, want)


@pytest.mark.parametrize("unquantized", [(), ("down_proj",)], ids=["quantized", "down-unquantized"])
def test_keep_fused_then_split_writes_unpacked_native_fp8_experts_as_the_per_expert_export(
    unquantized,
):
    """Native FP8 experts unpack at the default dtype, and the per-expert export reads each expert in
    the export dtype (``get_expert_weight``): the fold reads them alike, for the scales it derives
    and for a projection it leaves unquantized."""
    from transformers import PretrainedConfig
    from transformers.integrations.finegrained_fp8 import FP8Experts

    torch.manual_seed(0)
    config = PretrainedConfig(
        hidden_size=HIDDEN_DIM,
        intermediate_size=INTERMEDIATE_DIM,
        num_experts=NUM_EXPERTS,
        hidden_act="silu",
    )
    experts = FP8Experts(config, block_size=None)
    for name in ("gate_up_proj", "down_proj"):
        weight = getattr(experts, name)
        with torch.no_grad():
            weight.copy_((torch.randn(weight.shape) * 4).to(weight.dtype))
            getattr(experts, f"{name}_scale_inv").fill_(0.01)
    hidden = torch.randn(NUM_EXPERTS, HIDDEN_DIM)
    route = (torch.arange(NUM_EXPERTS).unsqueeze(1), torch.ones(NUM_EXPERTS, 1))
    registered_before = QuantModuleRegistry.get(FP8Experts)
    try:
        mtq.quantize(experts, mtq.NVFP4_DEFAULT_CFG, lambda m: m(hidden, *route))
        experts.unpack_weight()
        _unquantized(experts, unquantized)
        want = _per_expert_export(experts)
        got = split_fused_experts_state_dict(_keep_fused_export(experts))
    finally:
        if registered_before is None and QuantModuleRegistry.get(FP8Experts) is not None:
            QuantModuleRegistry.unregister(FP8Experts)
    _assert_byte_identical(got, want)


@pytest.mark.parametrize(
    "quant_cfg",
    [
        mtq.INT4_BLOCKWISE_WEIGHT_ONLY_CFG,
        mtq.W4A8_AWQ_BETA_CFG,
        mtq.FP8_2D_BLOCKWISE_WEIGHT_ONLY_CFG,
    ],
    ids=["int4-blockwise", "w4a8-awq", "fp8-2d-blockwise"],
)
def test_the_split_cuts_packed_and_block_scales_with_their_rows(monkeypatch, quant_cfg):
    """An INT4 weight packs two rows a byte while its scales stay per row, and an FP8 2-D block
    scale covers 128 rows: gate and up each take the scale rows of their own weight rows. Their
    blocks need dimensions of whole 128-blocks."""
    monkeypatch.setitem(globals(), "HIDDEN_DIM", 128)
    monkeypatch.setitem(globals(), "INTERMEDIATE_DIM", 128)
    experts = _quantized_experts(_KeepFusedGatedExperts, quant_cfg)
    want = _per_expert_export(experts)

    got = split_fused_experts_state_dict(_keep_fused_export(experts))

    _assert_byte_identical(got, want)


@pytest.mark.parametrize("amax_state", ["calibrated", "zero"])
def test_the_fold_derives_a_w4a8_experts_amax_per_quantizer_stage(monkeypatch, amax_state):
    """W4A8's SequentialQuantizer has no amax of its own, and its INT4 block stage rejects a scalar:
    an expert calibration never reached gets a per-block amax there and a scalar on the FP8 stage."""
    experts = _quantized_experts(_KeepFusedGatedExperts, mtq.FP8_DEFAULT_CFG)
    block = 16
    rows = {"gate_up_proj": 2 * INTERMEDIATE_DIM, "down_proj": HIDDEN_DIM}
    blocks = {"gate_up_proj": HIDDEN_DIM // block, "down_proj": INTERMEDIATE_DIM // block}
    for proj in rows:
        for i in range(NUM_EXPERTS):
            int4 = TensorQuantizer(
                QuantizerAttributeConfig(num_bits=4, block_sizes={-1: block, "type": "static"})
            )
            fp8 = TensorQuantizer(QuantizerAttributeConfig(num_bits=(4, 3)))
            fill = 0.0 if amax_state == "zero" else 1.0
            int4.amax = torch.full(
                (rows[proj] * blocks[proj], 1), fill
            )  # as block calibration records
            fp8.amax = torch.tensor(fill)
            getattr(experts, f"{proj}_weight_quantizers")[i] = SequentialQuantizer(int4, fp8)

    seen = []

    def spy(wrapper, dtype, **_kwargs):
        int4, fp8 = wrapper.weight_quantizer
        seen.append((int4._amax.clone(), fp8._amax.clone()))
        wrapper.weight_scale = torch.ones(1)

    monkeypatch.setattr("modelopt.torch.export.unified_export_hf._export_quantized_weight", spy)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        _export_fused_experts_keep_fused(experts, torch.bfloat16)

    assert len(seen) == 2 * NUM_EXPERTS
    assert {amax.shape for amax, _ in seen} == {(64, 1), (32, 1)}
    assert all(torch.all(int4_amax > 0) and torch.all(fp8_amax > 0) for int4_amax, fp8_amax in seen)
    assert any("was not calibrated" in str(w.message) for w in caught) == (amax_state == "zero")


def test_the_fold_puts_its_scales_on_the_weights_device(monkeypatch):
    """Per-tensor FP8 makes each expert's scale from a Python float, on the CPU wherever the weight
    is, and a sharded fold rewraps the scales on the weight's mesh. The weights here sit on meta,
    the one other device a CPU runner has, so a stand-in packs them, its scales on the CPU."""
    experts = _quantized_experts(_KeepFusedGatedExperts, mtq.FP8_DEFAULT_CFG)
    for proj in ("gate_up_proj", "down_proj"):
        setattr(experts, proj, nn.Parameter(getattr(experts, proj).detach().to("meta")))

    def pack(wrapper, dtype, **_kwargs):
        wrapper.weight_scale = torch.tensor(1.0)
        wrapper.weight_scale_2 = torch.tensor(1.0)

    monkeypatch.setattr("modelopt.torch.export.unified_export_hf._export_quantized_weight", pack)
    _export_fused_experts_keep_fused(experts, torch.bfloat16)

    for proj in ("gate_up_proj", "down_proj"):
        for scale in ("weight_scale", "weight_scale_2"):
            assert getattr(experts, f"{proj}_{scale}").is_meta, (proj, scale)


def _mixed(experts):
    experts.down_proj_weight_quantizers[1].disable()


def _native_fp8(experts):
    experts.down_proj = nn.Parameter(
        experts.down_proj.data.to(torch.float8_e4m3fn), requires_grad=False
    )


def _native_fp8_unquantized(experts):
    _native_fp8(experts)
    _unquantized(experts, ["down_proj"])


def _fewer_experts_counted(experts):
    experts.num_experts = NUM_EXPERTS - 1


def _ggml(experts):
    for i in range(len(experts.down_proj_weight_quantizers)):
        experts.down_proj_weight_quantizers[i] = TensorQuantizer(
            QuantizerAttributeConfig(
                num_bits="q8_0",
                block_sizes={-1: GGML_FORMAT_REGISTRY["q8_0"].block_size},
                backend="ggml",
            )
        )


@LAYOUTS
@pytest.mark.parametrize(
    ("change", "match"),
    [
        (_mixed, r"down_proj: weight quantizers \[1\] of 4"),
        (_native_fp8, "native FP8"),
        (_native_fp8_unquantized, "native FP8"),
        (_fewer_experts_counted, "num_experts is 3, but the unsharded weight holds 4 experts"),
        (_ggml, "GGML formats"),
    ],
    ids=["mixed", "native-fp8", "native-fp8-unquantized", "num-experts-short", "ggml"],
)
def test_a_projection_the_fold_cannot_write_is_refused_untouched(experts_cls, change, match):
    """One fused tensor cannot mix quantized and unquantized experts, and native FP8 weights need the
    scale_inv the fold drops, also where a recipe leaves them unquantized. A ``num_experts`` short of
    the weight's rows would fold only the first experts, and a GGML format packs each expert's weight
    with no scale to stack. The refusal comes before the other projection folds."""
    experts = _quantized_experts(experts_cls, mtq.FP8_DEFAULT_CFG)
    change(experts)
    before = {k: v.clone() for k, v in experts.state_dict().items()}

    with pytest.raises(NotImplementedError, match=match):
        _export_fused_experts_keep_fused(experts, torch.bfloat16)

    _assert_byte_identical(experts.state_dict(), before)
    assert getattr(experts, f"{experts._first_proj_attr}_weight_quantizers")[0].is_enabled


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


@pytest.mark.parametrize(
    ("rows", "scale_rows"), [(6, 3), (10, 4)], ids=["block-across-the-cut", "uneven-blocks"]
)
def test_split_refuses_a_scale_block_gate_and_up_would_share(rows, scale_rows):
    """Two-row blocks over 6 rows put gate's last row and up's first under one scale; three-row
    blocks over 10 rows end in a partial block, which no proportion places. Neither half can be given
    scales of its own, so the split refuses rather than round."""
    sd = {
        f"{PREFIX}.gate_up_proj": torch.zeros(1, rows, 4).to(torch.float8_e4m3fn),
        f"{PREFIX}.gate_up_proj_weight_scale": torch.ones(1, scale_rows, 1),
        f"{PREFIX}.down_proj": torch.zeros(1, 4, rows // 2).to(torch.float8_e4m3fn),
    }

    with pytest.raises(NotImplementedError, match=r"cannot be split at the gate\|up boundary"):
        split_fused_experts_state_dict(sd)


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


def _two_rank_split_job(rank, world_size):
    mesh = init_device_mesh("cpu", (world_size,))
    cases = [
        (experts_cls, quant_cfg, NUM_EXPERTS)
        for experts_cls in (_KeepFusedGatedExperts, _KeepFusedUngatedExperts)
        for quant_cfg in (mtq.FP8_DEFAULT_CFG, FP8_WEIGHT_ONLY_CFG, mtq.NVFP4_DEFAULT_CFG)
    ]
    # Three experts over two ranks: rank 0 holds two, rank 1 one. The folded tensors must keep the
    # global expert count, not infer local x world rows.
    cases.append((_KeepFusedGatedExperts, mtq.NVFP4_DEFAULT_CFG, 3))
    for experts_cls, quant_cfg, num_experts in cases:
        # Every rank calibrates the same seeded module, so all ranks hold identical quantizers.
        experts = _quantized_experts(experts_cls, quant_cfg, num_experts)
        want = _per_expert_export(experts)
        plain = _keep_fused_export(experts)
        sharded = _shard_fused_weights(copy.deepcopy(experts), mesh)
        _export_fused_experts_keep_fused(sharded, torch.bfloat16)
        fused = _prefixed(sharded.state_dict())
        assert set(fused) == set(plain)
        first = fused[f"{PREFIX}.{sharded._first_proj_attr}"]
        n_local = first.to_local().shape[0]
        offset = rank * math.ceil(num_experts / world_size)
        assert _dtensor_dim0_offset(first) == offset
        for key, value in fused.items():
            if key.endswith("_input_scale"):  # one scalar shared by every expert
                assert not isinstance(value, DTensor), key
                assert torch.equal(_raw_bytes(value), _raw_bytes(plain[key])), key
            else:  # weights and per-expert scales stay sharded on the expert axis
                assert isinstance(value, DTensor), key
                assert value.placements == (Shard(0),), key
                assert value.shape == plain[key].shape, key
                local = plain[key][offset : offset + n_local]
                assert torch.equal(_raw_bytes(value.to_local()), _raw_bytes(local)), key
        mine = {
            k: v
            for k, v in want.items()
            if offset <= int(k[len(PREFIX) + 1 :].split(".")[0]) < offset + n_local
        }
        assert mine, "this rank owns no experts"

        _assert_byte_identical(_split_local_fused_module(PREFIX, fused), mine)

        # A per-expert scale held as a plain tensor over ALL experts (as keep-fused emits it for an
        # unsharded weight) must be cut to this rank's experts before pairing.
        plain_scales = {
            k: plain[k] if k.endswith(("_weight_scale", "_weight_scale_2")) else v
            for k, v in fused.items()
        }
        _assert_byte_identical(_split_local_fused_module(PREFIX, plain_scales), mine)

    # One expert over two ranks leaves rank 1 none: both ranks refuse, rather than rank 1 alone
    # failing while rank 0 goes on to write.
    lone = _quantized_experts(_KeepFusedGatedExperts, mtq.FP8_DEFAULT_CFG, num_experts=1)
    with pytest.raises(NotImplementedError, match="leaves a rank none"):
        _export_fused_experts_keep_fused(_shard_fused_weights(lone, mesh), torch.bfloat16)

    # Expert parallelism leaves a rank only its own experts' quantizers, in shard order, under a
    # fused weight spanning every expert. Kept last, with the joint decision below: without that
    # pairing rank 1 alone fails, and rank 0 then waits only in the collectives below, which the
    # worker pool ends once a rank has failed.
    experts = _quantized_experts(_KeepFusedGatedExperts, mtq.NVFP4_DEFAULT_CFG)
    plain = _keep_fused_export(experts)
    n_local = NUM_EXPERTS // world_size
    mine = slice(rank * n_local, (rank + 1) * n_local)

    def sharded_keeping_quantizers(kept):
        sharded = _shard_fused_weights(copy.deepcopy(experts), mesh)
        for proj in (sharded._first_proj_attr, "down_proj"):
            name = f"{proj}_weight_quantizers"
            setattr(sharded, name, nn.ModuleList(list(getattr(sharded, name))[kept]))
        return sharded

    with pytest.raises(NotImplementedError, match="3 weight quantizers for 4 experts in 2 shards"):
        _export_fused_experts_keep_fused(sharded_keeping_quantizers(slice(0, 3)), torch.bfloat16)
    sharded = sharded_keeping_quantizers(mine)
    _export_fused_experts_keep_fused(sharded, torch.bfloat16)
    fused = _prefixed(sharded.state_dict())
    assert set(fused) == set(plain)
    for key, value in fused.items():
        if isinstance(value, DTensor):
            assert torch.equal(_raw_bytes(value.to_local()), _raw_bytes(plain[key][mine])), key
        else:
            assert torch.equal(_raw_bytes(value), _raw_bytes(plain[key])), key

    # Each rank sees only its own experts' quantizers there, so the ranks decide on a projection
    # together: rank 1 disabling some or all of its experts' is refused on both ranks, rather than
    # folded on rank 0 alone, and every expert disabled leaves the projection as it is on both.
    for disabled in ([3], [2, 3], [0, 1, 2, 3]):
        sharded = sharded_keeping_quantizers(mine)
        for expert, quantizer in enumerate(sharded.down_proj_weight_quantizers, start=mine.start):
            if expert in disabled:
                quantizer.disable()
        if len(disabled) < NUM_EXPERTS:
            with pytest.raises(NotImplementedError, match=re.escape(f"quantizers {disabled} of 4")):
                _export_fused_experts_keep_fused(sharded, torch.bfloat16)
        else:
            _export_fused_experts_keep_fused(sharded, torch.bfloat16)
            assert hasattr(sharded, "gate_up_proj_weight_scale")
            assert not hasattr(sharded, "down_proj_weight_scale")


def test_two_ranks_fold_and_split_their_own_experts_under_global_indices(cpu_dist_workers):
    """Rank 1 holds experts 2-3: each must be folded with ITS quantizer and keep its index, and the
    fused tensors stay sharded on the expert axis at their global shape, an uneven shard included.
    Its quantizers are every expert's (FSDP2) or, under expert parallelism, its own two, and then
    the ranks decide on each projection's quantization together. A layout that leaves a rank no
    expert, or quantizers that are neither, is refused on every rank."""
    cpu_dist_workers.run(_two_rank_split_job)


@pytest.mark.parametrize(
    ("mesh_shape", "placements"),
    [((1,), [Shard(2)]), ((1, 1), [Shard(0), Shard(0)])],
    ids=["off-the-expert-axis", "over-two-mesh-dims"],
)
def test_a_fused_weight_sharded_other_than_by_expert_is_refused(
    gloo_world_of_one, mesh_shape, placements
):
    """The fold shards the scales on the expert axis alone, and the ranks agree on their experts
    over one mesh dim: the scales of another layout would claim one rank's slice as the whole."""
    experts = _quantized_experts(_KeepFusedGatedExperts, mtq.FP8_DEFAULT_CFG)
    mesh = init_device_mesh("cpu", mesh_shape)
    weight = distribute_tensor(experts.down_proj.data, mesh, placements)
    experts.down_proj = nn.Parameter(weight, requires_grad=False)

    with pytest.raises(NotImplementedError, match="sharded over one mesh dim"):
        _export_fused_experts_keep_fused(experts, torch.bfloat16)


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


def test_keep_fused_refuses_fused_experts_not_named_experts():
    """The split finds a fused weight by its ``experts.<proj>`` key, as the writer does: a container
    named otherwise is refused before anything is folded, not written fused to fail at load."""
    experts = _quantized_experts(_KeepFusedGatedExperts, mtq.FP8_DEFAULT_CFG)
    container = nn.Module()
    container.moe = experts
    before = copy.deepcopy(experts.state_dict())
    ctx = ExportContext(model=container, dtype=torch.bfloat16, keep_fused_experts=True)

    with pytest.raises(NotImplementedError, match="named 'moe'"):
        ExportModuleRegistry.match(experts)("model.layers.0.mlp.moe", experts, ctx)

    torch.testing.assert_close(experts.state_dict(), before, rtol=0, atol=0)


@pytest.mark.parametrize("dense", ["fsdp", "replicate"])
def test_fused_experts_fold_in_a_weight_window_when_the_plan_shards_the_dense_part(
    monkeypatch, dense
):
    """FSDP2 owns the fused weights when it shards the dense part around them: folded outside a
    weight-update window, they would be overwritten with its original shards at the next reshard.
    Outside held windows the handler closes the window itself."""
    events = []

    @contextlib.contextmanager
    def window(root, modules, names=None):
        events.append("open")
        yield
        events.append("close")

    monkeypatch.setattr(handlers, "fsdp2_aware_weight_update", window)
    monkeypatch.setattr(
        handlers, "_export_fused_experts_keep_fused", lambda module, dtype: events.append("fold")
    )
    experts = _quantized_experts(_KeepFusedGatedExperts, mtq.FP8_DEFAULT_CFG)
    container = nn.Module()
    container.experts = experts
    set_distribution_plan(container, DistributionPlan(2, ep_size=2, dense=dense))
    ctx = ExportContext(model=container, dtype=torch.bfloat16, keep_fused_experts=True)

    ExportModuleRegistry.match(experts)(PREFIX, experts, ctx)

    assert events == (["open", "fold", "close"] if dense == "fsdp" else ["fold"])


@pytest.mark.parametrize("held", [True, False], ids=["held", "per-module"])
@pytest.mark.parametrize("root_sharded", [True, False], ids=["fsdp2", "expert-parallel"])
def test_quant_linear_handlers_pack_an_fsdp2_unit_in_one_weight_window(
    monkeypatch, gloo_world_of_one, root_sharded, held
):
    """On the no-gather path the model is still sharded when a linear is packed.

    Packing NVFP4 straight off the sharded weight is not possible (the in-place cast on the
    DTensor's local view is refused under autograd), so the handlers pack inside a weight-update
    window, on the layer's own unit where expert parallelism leaves the root unsharded. The window
    gathers the whole unit, so within held windows it opens once for all of the unit's linears;
    any other dispatch loop gets a window per linear, closed by its handler. Either way the unit is
    sharded again afterwards, and the sharded state dict carries the packed weights, as an
    unsharded pack has them.
    """
    torch.manual_seed(0)
    layer = nn.Sequential(nn.Linear(32, 16, bias=False), nn.Linear(16, 8, bias=False))
    model = nn.Sequential(layer)
    mtq.quantize(model, mtq.NVFP4_DEFAULT_CFG, lambda m: m(torch.randn(4, 32)))
    refs = copy.deepcopy(model)[0]
    for i, ref in enumerate(refs):
        ExportModuleRegistry.match(ref)(
            f"0.{i}", ref, ExportContext(model=ref, dtype=torch.bfloat16)
        )
    fully_shard(layer, mesh=gloo_world_of_one)
    if root_sharded:
        fully_shard(model, mesh=gloo_world_of_one)
    set_distribution_plan(model, DistributionPlan(dense="fsdp"))
    ctx = ExportContext(model=model, dtype=torch.bfloat16, keep_fused_experts=True)
    unshards = []
    unshard = FSDPModule.unshard
    monkeypatch.setattr(
        FSDPModule, "unshard", lambda unit, *a, **k: unshards.append(unit) or unshard(unit, *a, **k)
    )

    with handlers.held_fsdp2_weight_windows(ctx) if held else contextlib.nullcontext():
        for i, linear in enumerate(layer):
            ExportModuleRegistry.match(linear)(f"0.{i}", linear, ctx)

    assert unshards == ([layer] if held else [layer, layer])
    assert all(isinstance(linear.weight, DTensor) for linear in layer)
    sd = get_model_state_dict(model, options=StateDictOptions(full_state_dict=False))
    for i, ref in enumerate(refs):
        packed = sd[f"0.{i}.weight"]
        assert isinstance(packed, DTensor)
        assert packed.dtype == torch.uint8
        assert torch.equal(packed.full_tensor(), ref.weight)
        for name in ("weight_scale", "weight_scale_2", "input_scale"):
            want = _raw_bytes(getattr(ref, name))
            assert torch.equal(_raw_bytes(sd[f"0.{i}.{name}"]), want), name


class GptOssExperts(nn.Module):
    """Named as transformers' fused BMM experts, whose handler packs each of its weights in turn."""


def test_bmm_experts_pack_both_weights_in_their_unit_s_held_window(gloo_world_of_one):
    """``gate_up_proj`` and then ``down_proj`` of one module pack in the unit's held window, which
    the module joins once: listed twice, the window would re-register its weights with FSDP2 a
    second time, off the shards the first pass left."""
    torch.manual_seed(0)
    linears = nn.Sequential(nn.Linear(32, 16, bias=False), nn.Linear(16, 32, bias=False))
    mtq.quantize(linears, mtq.NVFP4_DEFAULT_CFG, lambda m: m(torch.randn(4, 32)))
    experts = GptOssExperts()
    for proj, linear in zip(("gate_up_proj", "down_proj"), linears):
        experts.register_parameter(proj, nn.Parameter(linear.weight.detach().clone()))
        setattr(experts, f"{proj}_weight_quantizer", linear.weight_quantizer)
        setattr(experts, f"{proj}_input_quantizer", linear.input_quantizer)
    ref = copy.deepcopy(experts)
    ExportModuleRegistry.match(ref)(PREFIX, ref, ExportContext(model=ref, dtype=torch.bfloat16))
    layer = nn.Sequential(experts)
    model = nn.Sequential(layer)
    fully_shard(layer, mesh=gloo_world_of_one)
    set_distribution_plan(model, DistributionPlan(dense="fsdp"))
    ctx = ExportContext(model=model, dtype=torch.bfloat16, keep_fused_experts=True)

    with handlers.held_fsdp2_weight_windows(ctx):
        ExportModuleRegistry.match(experts)("0.0", experts, ctx)

    sd = get_model_state_dict(model, options=StateDictOptions(full_state_dict=False))
    for key, want in ref.state_dict().items():
        got = sd[f"0.0.{key}"]
        got = got.full_tensor() if isinstance(got, DTensor) else got
        assert torch.equal(_raw_bytes(got), _raw_bytes(want)), key


def test_nvfp4_packs_a_sharded_weight_into_the_same_codes(gloo_world_of_one):
    """``DTensor << int`` is a silent no-op, so the nibble pack must not rely on the operator.

    The block scales come out sharded like the weight, an all-zero block included: filling it by
    boolean-mask assignment lowers to ``index_put_``, which DTensor cannot dispatch.
    """
    torch.manual_seed(0)
    weight = torch.randn(4, 32, dtype=torch.bfloat16)
    weight[1, :16] = 0.0  # an all-zero block
    scale_2 = (weight.abs().amax().float() / (6.0 * 448.0)).reshape(())
    scale = NVFP4QTensor.get_weights_scaling_factor(weight, 16, scale_2)[0]
    want = NVFP4QTensor.quantize(weight, 16, scale, scale_2)[0]._quantized_data

    sharded = distribute_tensor(weight, gloo_world_of_one, [Shard(0)])
    sharded_scale = NVFP4QTensor.get_weights_scaling_factor(sharded, 16, scale_2)[0]
    packed = NVFP4QTensor.quantize(sharded, 16, sharded_scale, scale_2)[0]._quantized_data

    assert isinstance(sharded_scale, DTensor) and sharded_scale.placements == (Shard(0),)
    assert torch.equal(_raw_bytes(sharded_scale.to_local()), _raw_bytes(scale))
    assert torch.equal(packed.full_tensor(), want)
    # The second-level scale read off the sharded weight itself, as without one handed in.
    assert torch.equal(
        NVFP4QTensor.get_weights_scaling_factor(sharded, 16)[1],
        NVFP4QTensor.get_weights_scaling_factor(weight, 16)[1],
    )


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
