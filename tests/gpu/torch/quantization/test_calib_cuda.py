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
import torch.nn as nn

import modelopt.torch.quantization as mtq


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
