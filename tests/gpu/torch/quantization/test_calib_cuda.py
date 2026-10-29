# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

"""Calibration tests."""

import torch
import torch.distributed as dist
import torch.nn as nn

import modelopt.torch.quantization as mtq
from modelopt.torch.utils.distributed import DistributionPlan, ParallelState, set_distribution_plan


class _TwoBranchModel(nn.Module):
    """Two parallel linears; only the first is exercised by forward_loop."""

    def __init__(self):
        super().__init__()
        self.calibrated = nn.Linear(16, 16, bias=False)
        self.uncalibrated = nn.Linear(16, 16, bias=False)

    def forward(self, x, branch="calibrated"):
        if branch == "calibrated":
            return self.calibrated(x)
        return self.uncalibrated(x)


def test_awq_lite_uncalibrated_linear_keeps_input_quantizer_enabled():
    """Regression test for NVBug 6143871.

    awq_lite.setup() disables the input_quantizer at the start of search. The
    calibrated branch re-enables it inside postprocess(); the uncalibrated
    branch (no cache-pass tokens, e.g. an MoE expert that never gets routed)
    must do the same — otherwise downstream export (set_expert_quantizer_amax
    + _export_quantized_weight) drops the input_scale buffer and inference
    runtimes that read per-expert input_scale (e.g. TRT-LLM CutlassFusedMoE)
    crash with KeyError on '<idx>.w1.input_scale'.

    Also asserts the export-critical scalar amax invariant (axis=None,
    numel==1) — preprocess_linear_fusion enforces it for fused-expert groups.
    """
    torch.manual_seed(0)
    model = _TwoBranchModel().cuda()

    def _forward_loop(m):
        for _ in range(2):
            m(torch.randn(2, 16, 16, device="cuda"), branch="calibrated")

    mtq.quantize(model, mtq.NVFP4_AWQ_LITE_CFG, _forward_loop)

    assert model.calibrated.input_quantizer.is_enabled
    assert model.uncalibrated.input_quantizer.is_enabled, (
        "Uncalibrated linear's input_quantizer must remain enabled after "
        "awq_lite postprocess so export emits input_scale (NVBug 6143871)."
    )
    uncal_q = model.uncalibrated.input_quantizer
    # When amax exists (cache-hit but search-miss path), it must be the
    # scalar form export expects — preprocess_linear_fusion asserts numel==1.
    # When it's None (truly never routed), set_expert_quantizer_amax will
    # populate it during export.
    if uncal_q.amax is not None:
        assert uncal_q.axis is None
        assert uncal_q.amax.numel() == 1


def _nvfp4_linears(device, count):
    """``count`` copies of one linear: tiny weights, whose FP8 block scales a tighter per-tensor
    amax resolves better, and a large one per row on input channel 0, which a tighter amax clips."""
    model = nn.Sequential(*(nn.Linear(64, 16, bias=False) for _ in range(count))).to(device)
    for linear in model:
        with torch.no_grad():
            linear.weight.copy_(torch.linspace(-1e-3, 1e-3, 16 * 64, device=device).view(16, 64))
            linear.weight[:, 0] = 100.0
    return model


def _nvfp4_rank_input(rank, device):
    x = torch.randn(8, 64, device=device, generator=torch.Generator(device).manual_seed(rank))
    if rank % 2 == 0:
        x[:, 0] = 0  # never reaches the large weights: the tightest clip suits it
    else:
        x[:, 1:] = 0  # reaches only them: no clip suits it
    return x


def _nvfp4_awq_clip_on_every_rank_job(rank, size):
    device = torch.device(f"cuda:{rank}")
    model = _nvfp4_linears(device, 2)
    set_distribution_plan(model, DistributionPlan(size))
    x = _nvfp4_rank_input(rank, device)

    def forward_loop(m):
        m[0](x)
        # The second runs on no tokens on the odd ranks, as an MoE expert no token reaches can.
        m[1](x if rank % 2 == 0 else x[:0])

    mtq.quantize(model, mtq.NVFP4_AWQ_CLIP_CFG, forward_loop)
    amax = torch.stack([linear.weight_quantizer.amax.float().reshape(()) for linear in model])
    clips = [torch.empty_like(amax) for _ in range(size)]
    dist.all_gather(clips, amax)
    assert all(torch.equal(clip, amax) for clip in clips), clips
    # The second's clip is the one the even ranks' tokens pick, calibrated in one process.
    alone = _nvfp4_linears(device, 1)
    alone[0]._parallel_state = ParallelState(data_parallel_group=-1)
    inputs = [_nvfp4_rank_input(even, device) for even in range(0, size, 2)]
    mtq.quantize(alone, mtq.NVFP4_AWQ_CLIP_CFG, lambda m: [m(even) for even in inputs])
    assert amax[1] > 0, amax
    assert torch.equal(amax[1], alone[0].weight_quantizer.amax.float().reshape(())), amax


def test_nvfp4_awq_clip_picks_the_same_clip_on_every_rank(need_2_gpus, dist_workers):
    """NVFP4's awq_clip clips the per-tensor amax. Each rank of a model distributed over several
    ranks calibrates on its own data, and averaging each candidate clip's loss over the ranks gives
    every rank's copy of a weight the same clip, though each rank's own data would pick another. A
    rank that runs a weight on no tokens leaves it the clip the other ranks' tokens pick."""
    dist_workers.run(_nvfp4_awq_clip_on_every_rank_job)
