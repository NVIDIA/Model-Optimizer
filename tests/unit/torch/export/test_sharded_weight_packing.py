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

"""Packing weights that are still sharded, as the no-gather export does (CPU, gloo, one and two ranks)."""

import contextlib
import copy

import pytest
import torch
import torch.nn as nn
from _test_utils.torch.distributed.utils import bounded_collectives
from torch.distributed.checkpoint.state_dict import StateDictOptions, get_model_state_dict
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.fsdp import FSDPModule, fully_shard
from torch.distributed.tensor import DTensor, Shard, distribute_tensor

import modelopt.torch.export.hf_export_handlers as handlers
import modelopt.torch.quantization as mtq
from modelopt.torch.export.quant_utils import pack_int4_in_uint8
from modelopt.torch.export.registry import ExportContext, ExportModuleRegistry
from modelopt.torch.quantization.config import QuantizerAttributeConfig
from modelopt.torch.quantization.nn import NVFP4StaticQuantizer
from modelopt.torch.quantization.qtensor import INT4QTensor, MXFP4QTensor, NF4QTensor, NVFP4QTensor
from modelopt.torch.utils.distributed import DistributionPlan, set_distribution_plan


def _raw_bytes(tensor):
    return tensor.detach().contiguous().reshape(-1).view(torch.uint8)


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


def _nvfp4_through_a_static_quantizer(weight):
    whole = weight.full_tensor() if isinstance(weight, DTensor) else weight
    quantizer = NVFP4StaticQuantizer(
        quant_attribute_cfg=QuantizerAttributeConfig(
            num_bits=(2, 1), block_sizes={-1: 16, "type": "static", "scale_bits": (4, 3)}
        )
    )
    quantizer.amax = whole.reshape(-1, 16).abs().amax(-1)
    quantizer.global_amax = whole.abs().amax()
    scale, scale_2 = NVFP4QTensor.get_weights_scaling_factor_from_quantizer(quantizer, weight)
    return NVFP4QTensor.quantize(weight, 16, scale, scale_2)


@pytest.mark.parametrize(
    "pack",
    [
        lambda weight: INT4QTensor.quantize(weight, 16),
        lambda weight: NF4QTensor.quantize(weight, 16, 4),
        lambda weight: MXFP4QTensor.quantize(weight, 32),
        lambda weight: pack_int4_in_uint8(weight, torch.ones(4, 2), 16),
        _nvfp4_through_a_static_quantizer,
    ],
    ids=["int4", "nf4", "mxfp4", "int4-export", "nvfp4-static"],
)
def test_a_packer_that_packs_only_whole_tensors_refuses_a_shard(gloo_world_of_one, pack):
    """Only NVFP4, with the per-block scale it computes off the shard, packs a DTensor shard.

    The other 4-bit packers combine their input with plain tensors, which DTensor refuses deep
    inside, or gather it whole (INT4's fallback, whose ``<<`` also dropped the shift on a DTensor).
    They refuse a shard up front, as NVFP4 does a plain per-block scale, a static quantizer's.
    """
    torch.manual_seed(0)
    weight = torch.randn(4, 32)
    pack(weight)

    with pytest.raises(NotImplementedError):
        pack(distribute_tensor(weight, gloo_world_of_one, [Shard(0)]))


@pytest.mark.parametrize("held", [True, False], ids=["held", "per-module"])
@pytest.mark.parametrize("layout", ["fsdp2", "expert-parallel", "linear-units"])
def test_quant_linear_handlers_pack_an_fsdp2_unit_in_one_weight_window(
    monkeypatch, gloo_world_of_one, layout, held
):
    """On the no-gather path the model is still sharded when a linear is packed.

    Packing NVFP4 straight off the sharded weight is not possible (the in-place cast on the
    DTensor's local view is refused under autograd), so the handlers pack inside a weight-update
    window, on the layer's own unit where expert parallelism leaves the root unsharded. The window
    gathers the whole unit, so within held windows it opens once for all of the unit's linears;
    any other dispatch loop gets a window per linear, closed by its handler. Either way the unit is
    sharded again afterwards, and the sharded state dict carries the packed weights, as an
    unsharded pack has them. A linear that is a unit of its own under an unwrapped root owns its
    weight as the window's root: its packed weight must be handed to FSDP2 all the same.
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
    units = list(layer) if layout == "linear-units" else [layer]
    for unit in units:
        fully_shard(unit, mesh=gloo_world_of_one)
    if layout == "fsdp2":
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

    per_linear = [linear if layout == "linear-units" else layer for linear in layer]
    assert unshards == (list(dict.fromkeys(per_linear)) if held else per_linear)
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


def test_a_module_joins_its_units_held_window_once(gloo_world_of_one):
    """A handler packing several weights of one module asks for the window once per weight.

    The module must join its unit's held window once: listed twice, it would be re-registered with
    FSDP2 a second time when the window closes, from the shard the first registration put back.
    """
    torch.manual_seed(0)
    layer = nn.Sequential(nn.Linear(32, 16, bias=False))
    model = nn.Sequential(layer)
    mtq.quantize(model, mtq.NVFP4_DEFAULT_CFG, lambda m: m(torch.randn(4, 32)))
    ref = copy.deepcopy(layer[0])
    ExportModuleRegistry.match(ref)("0.0", ref, ExportContext(model=ref, dtype=torch.bfloat16))
    fully_shard(layer, mesh=gloo_world_of_one)
    fully_shard(model, mesh=gloo_world_of_one)
    set_distribution_plan(model, DistributionPlan(dense="fsdp"))
    ctx = ExportContext(model=model, dtype=torch.bfloat16, keep_fused_experts=True)

    with handlers.held_fsdp2_weight_windows(ctx):
        ExportModuleRegistry.match(layer[0])("0.0", layer[0], ctx)
        with handlers._fsdp2_weight_window(ctx, layer[0]):  # as for the module's next weight
            pass

    sd = get_model_state_dict(model, options=StateDictOptions(full_state_dict=False))
    assert torch.equal(sd["0.0.weight"].full_tensor(), ref.weight)


class _Unit(nn.Module):
    """An FSDP2 unit of two linears fed the same input; 17 rows shard unevenly."""

    def __init__(self):
        super().__init__()
        self.a = nn.Linear(32, 17, bias=False)
        self.b = nn.Linear(32, 16, bias=False)

    def forward(self, x):
        return self.a(x), self.b(x)


class _TwoUnits(nn.Module):
    def __init__(self):
        super().__init__()
        self.units = nn.ModuleList([_Unit(), _Unit()])

    def forward(self, x):
        return [unit(x) for unit in self.units]


def _two_rank_packing_job(rank, world_size):
    mesh = init_device_mesh("cpu", (world_size,))
    torch.manual_seed(0)
    # Three rows over two ranks, an all-zero block included: rank 0 packs two rows, rank 1 one.
    weight = torch.randn(3, 32, dtype=torch.bfloat16)
    weight[1, :16] = 0.0
    scale_2 = (weight.abs().amax().float() / (6.0 * 448.0)).reshape(())
    scale = NVFP4QTensor.get_weights_scaling_factor(weight, 16, scale_2)[0]
    want = NVFP4QTensor.quantize(weight, 16, scale, scale_2)[0]._quantized_data
    mine = slice(0, 2) if rank == 0 else slice(2, 3)

    sharded = distribute_tensor(weight, mesh, [Shard(0)])
    sharded_scale = NVFP4QTensor.get_weights_scaling_factor(sharded, 16, scale_2)[0]
    packed = NVFP4QTensor.quantize(sharded, 16, sharded_scale, scale_2)[0]._quantized_data

    assert sharded_scale.shape == scale.shape and packed.shape == want.shape
    assert torch.equal(_raw_bytes(sharded_scale.to_local()), _raw_bytes(scale[mine]))
    assert torch.equal(packed.full_tensor(), want)

    # The handlers over two units in held windows, a 17-row weight sharded 9 + 8 among them.
    model = _TwoUnits()
    mtq.quantize(model, mtq.NVFP4_DEFAULT_CFG, lambda m: m(torch.randn(4, 32)))
    linears = [
        (f"units.{u}.{p}", getattr(unit, p)) for u, unit in enumerate(model.units) for p in "ab"
    ]
    refs = copy.deepcopy(model)
    for name, _ in linears:
        ref = refs.get_submodule(name)
        ExportModuleRegistry.match(ref)(name, ref, ExportContext(model=ref, dtype=torch.bfloat16))
    for unit in model.units:
        fully_shard(unit, mesh=mesh)
    fully_shard(model, mesh=mesh)
    set_distribution_plan(model, DistributionPlan(world_size, dense="fsdp"))
    ctx = ExportContext(model=model, dtype=torch.bfloat16, keep_fused_experts=True)

    with bounded_collectives(), handlers.held_fsdp2_weight_windows(ctx):
        for name, linear in linears:
            ExportModuleRegistry.match(linear)(name, linear, ctx)

    sd = get_model_state_dict(model, options=StateDictOptions(full_state_dict=False))
    for name, _ in linears:
        ref = refs.get_submodule(name)
        assert torch.equal(sd[f"{name}.weight"].full_tensor(), ref.weight), name
        for leaf in ("weight_scale", "weight_scale_2", "input_scale"):
            got = _raw_bytes(sd[f"{name}.{leaf}"])
            assert torch.equal(got, _raw_bytes(getattr(ref, leaf))), (name, leaf)


def test_two_ranks_pack_their_shards_into_the_unsharded_bytes(cpu_dist_workers):
    """On two ranks a shard holds other rows than the whole weight: an uneven NVFP4 pack gives the
    plain pack's codes and scales, and the handlers, run over two units, leave the sharded state
    dict holding what an unsharded export packs."""
    cpu_dist_workers.run(_two_rank_packing_job)
