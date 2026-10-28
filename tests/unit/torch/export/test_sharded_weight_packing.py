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

"""Packing weights that are still sharded, as the no-gather export does (CPU, one gloo rank)."""

import contextlib
import copy

import pytest
import torch
import torch.nn as nn
from torch.distributed.checkpoint.state_dict import StateDictOptions, get_model_state_dict
from torch.distributed.fsdp import FSDPModule, fully_shard
from torch.distributed.tensor import DTensor, Shard, distribute_tensor

import modelopt.torch.export.hf_export_handlers as handlers
import modelopt.torch.quantization as mtq
from modelopt.torch.export.registry import ExportContext, ExportModuleRegistry
from modelopt.torch.quantization.qtensor import NVFP4QTensor
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
