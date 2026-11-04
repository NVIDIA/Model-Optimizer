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

"""An exported MoE's expert scales come from each expert's own weights, however the model is laid out.

An expert's global weight scale is ``amax / (6 * 448)`` of its own weights, so every expert has its
own. A scale calibrated on another expert's weights moves the dequantized values by a few percent,
inside the tolerance of every check that compares dequantized values, so these compare the exported
scales themselves against the source weights.

The same quantize-and-export runs for each distribution the export tells apart by the model's plan:
whole in one process, FSDP2-sharded, and expert-parallel with the dense part replicated or
FSDP2-sharded. Expert parallelism is the layout in which each rank's per-expert quantizers once
calibrated on other ranks' experts, by way of two routes: indexing the experts' DTensors by local
expert id, and FSDP2's weight-access window gathering them. The FSDP2-sharded dense part is what
reaches the second.
"""

import faulthandler
from contextlib import nullcontext
from unittest import mock

import pytest
import torch
import torch.distributed as dist

pytest.importorskip("transformers")
from _test_utils.torch.export.unified_checkpoint import (
    assert_expert_scales_match_source,
    fused_experts_source_weights,
    load_safetensors_dir,
)
from _test_utils.torch.transformers_models import get_tiny_qwen3_moe
from torch import nn
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.fsdp import fully_shard
from torch.distributed.tensor import Shard, distribute_tensor

import modelopt.torch.quantization as mtq
from modelopt.torch.export import unified_export_hf
from modelopt.torch.export.unified_export_hf import export_hf_checkpoint
from modelopt.torch.utils.distributed import DistributionPlan, ParallelState, set_distribution_plan
from modelopt.torch.utils.plugins.model_load_utils import _fsdp2_shard_dense, _seed_parallel_states

# Each expert's weight scale, so that no two experts share one and an expert taking another's shows.
# The down projections take the gains in reverse: their scales are separate from gate and up's.
_GAINS = (3.0, 1.0, 0.5, 2.0)

_NVFP4_WEIGHTS_ONLY_CFG = {
    **mtq.NVFP4_DEFAULT_CFG,
    "quant_cfg": [
        *mtq.NVFP4_DEFAULT_CFG["quant_cfg"],
        {"quantizer_name": "*input_quantizer", "enable": False},
    ],
}


def _tiny_moe():
    model = get_tiny_qwen3_moe().eval()
    with torch.no_grad():
        for layer in model.model.layers:
            experts = layer.mlp.experts
            for weight, gains in (
                (experts.gate_up_proj, _GAINS),
                (experts.down_proj, _GAINS[::-1]),
            ):
                weight.mul_(torch.tensor(gains, dtype=weight.dtype).view(-1, 1, 1))
    return model


def _shard_experts_like_transformers(model, mesh):
    """Each experts module as transformers' expert parallelism leaves it: its weights DTensors over
    all the experts, sharded on dim 0, with ``num_experts`` the count a rank holds."""
    for layer in model.model.layers:
        experts = layer.mlp.experts
        for name in ("gate_up_proj", "down_proj"):
            weight = getattr(experts, name).detach().clone()
            experts._parameters[name] = nn.Parameter(distribute_tensor(weight, mesh, [Shard(0)]))
        experts.num_experts //= mesh.size()


def _distribute(model, distribution):
    size = dist.get_world_size()
    if distribution == "fsdp2":
        mesh = init_device_mesh("cpu", (size,))
        for layer in model.model.layers:
            fully_shard(layer, mesh=mesh)
        fully_shard(model, mesh=mesh)
        set_distribution_plan(model, DistributionPlan(size, dense="fsdp"))
        return
    ep_mesh = init_device_mesh("cpu", (size,), mesh_dim_names=("ep",))
    _shard_experts_like_transformers(model, ep_mesh)
    experts = [name for name, _ in model.named_modules() if name.endswith(".mlp.experts")]
    dense_fsdp = distribution == "ep+fsdp2"
    if dense_fsdp:
        dp_mesh = init_device_mesh("cpu", (size,), mesh_dim_names=("dp",))
        _fsdp2_shard_dense(model, dp_mesh, [model.get_submodule(name) for name in experts])
    # As the expert-parallel load seeds them: the dense part reduces its amax over the world, the
    # experts over their replicas (none here) and share one input amax over the expert-parallel group.
    _seed_parallel_states(
        model,
        experts,
        dense_state=ParallelState(data_parallel_group=dist.group.WORLD),
        expert_state=ParallelState(
            data_parallel_group=-1, expert_model_parallel_group=ep_mesh.get_group()
        ),
    )
    plan = DistributionPlan(size, ep_size=size, dense="fsdp" if dense_fsdp else "replicate")
    set_distribution_plan(model, plan)


def _quantize_and_export(distribution, export_dir):
    model = _tiny_moe()
    if distribution != "sequential":
        _distribute(model, distribution)
    if distribution.startswith("ep"):
        # The experts' all-to-all forward comes with transformers' expert-parallel load, which a
        # hand-sharded model lacks. So the weights are calibrated alone, and the export's dummy
        # forward, which only groups linears that share an input, is skipped.
        mtq.quantize(model, _NVFP4_WEIGHTS_ONLY_CFG)
        dummy_forward = mock.patch.object(unified_export_hf, "requantize_resmooth_fused_llm_layers")
    else:
        calib = [torch.randint(0, 32, (1, 8)) for _ in range(4)]
        mtq.quantize(model, mtq.NVFP4_DEFAULT_CFG, lambda m: [m(x) for x in calib])
        dummy_forward = nullcontext()
    with dummy_forward:
        export_hf_checkpoint(model, dtype=torch.bfloat16, export_dir=export_dir)


def _quantize_and_export_job(rank, size, distribution, export_dir):
    # A deadlocked rank exits with its stack instead of holding the run for the pool's timeout.
    # Cancelled in ``finally`` because the worker is reused: a leftover timer would kill it later.
    faulthandler.dump_traceback_later(45, exit=True)
    try:
        _quantize_and_export(distribution, export_dir)
        dist.barrier()
    finally:
        faulthandler.cancel_dump_traceback_later()


@pytest.mark.parametrize("distribution", ["sequential", "fsdp2", "ep", "ep+fsdp2"])
def test_exported_expert_scales_come_from_each_experts_own_weights(tmp_path, request, distribution):
    if distribution == "sequential":
        _quantize_and_export(distribution, tmp_path)
    else:
        cpu_dist_workers = request.getfixturevalue("cpu_dist_workers")
        cpu_dist_workers.run(
            _quantize_and_export_job, distribution=distribution, export_dir=tmp_path
        )
    source = fused_experts_source_weights(_tiny_moe())
    # Three projections per expert: every expert of every layer is exported, whichever rank held it.
    checked = assert_expert_scales_match_source(load_safetensors_dir(tmp_path), source)
    assert checked == len(source) // 3
