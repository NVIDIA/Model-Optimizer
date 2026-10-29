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

"""Packing weights that are still sharded, as the no-gather export does, over NCCL on every GPU."""

import copy

import torch
import torch.nn as nn
from torch.distributed.checkpoint.state_dict import StateDictOptions, get_model_state_dict
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.fsdp import fully_shard
from torch.distributed.tensor import Shard, distribute_tensor

import modelopt.torch.export.hf_export_handlers as handlers
import modelopt.torch.quantization as mtq
from modelopt.torch.export.registry import ExportContext, ExportModuleRegistry
from modelopt.torch.quantization.qtensor import NVFP4QTensor
from modelopt.torch.utils.distributed import DistributionPlan, set_distribution_plan


def _raw_bytes(tensor):
    return tensor.detach().contiguous().reshape(-1).view(torch.uint8).cpu()


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


def _packing_job(rank, world_size):
    mesh = init_device_mesh("cuda", (world_size,))
    torch.manual_seed(0)
    # Two rows a rank, the last rank one, an all-zero block included.
    rows = 2 * world_size - 1
    weight = torch.randn(rows, 32, dtype=torch.bfloat16, device="cuda")
    weight[1, :16] = 0.0
    scale_2 = (weight.abs().amax().float() / (6.0 * 448.0)).reshape(())
    scale = NVFP4QTensor.get_weights_scaling_factor(weight, 16, scale_2)[0]
    want = NVFP4QTensor.quantize(weight, 16, scale, scale_2)[0]._quantized_data
    mine = slice(2 * rank, min(2 * rank + 2, rows))

    sharded = distribute_tensor(weight, mesh, [Shard(0)])
    sharded_scale = NVFP4QTensor.get_weights_scaling_factor(sharded, 16, scale_2)[0]
    packed = NVFP4QTensor.quantize(sharded, 16, sharded_scale, scale_2)[0]._quantized_data

    assert sharded_scale.shape == scale.shape and packed.shape == want.shape
    assert torch.equal(_raw_bytes(sharded_scale.to_local()), _raw_bytes(scale[mine]))
    assert torch.equal(packed.full_tensor().cpu(), want.cpu())

    # The handlers over two units in held windows, a 17-row weight sharded unevenly among them.
    model = _TwoUnits().cuda()
    mtq.quantize(model, mtq.NVFP4_DEFAULT_CFG, lambda m: m(torch.randn(4, 32, device="cuda")))
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

    with handlers.held_fsdp2_weight_windows(ctx):
        for name, linear in linears:
            ExportModuleRegistry.match(linear)(name, linear, ctx)

    sd = get_model_state_dict(model, options=StateDictOptions(full_state_dict=False))
    for name, _ in linears:
        ref = refs.get_submodule(name)
        assert torch.equal(sd[f"{name}.weight"].full_tensor().cpu(), ref.weight.cpu()), name
        for leaf in ("weight_scale", "weight_scale_2", "input_scale"):
            got = _raw_bytes(sd[f"{name}.{leaf}"])
            assert torch.equal(got, _raw_bytes(getattr(ref, leaf))), (name, leaf)


def test_sharded_packing_matches_the_unsharded_bytes_over_nccl(need_2_gpus, dist_workers):
    """An uneven NVFP4 shard packs into the plain pack's codes and scales, and the handlers, run over
    two FSDP2 units in held windows, leave the sharded state dict holding what an unsharded export
    packs."""
    dist_workers.run(_packing_job)
