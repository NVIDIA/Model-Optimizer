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

"""Mixed-precision (NVFP4 + FP8) behavior of mse_calibrate. Requires CUDA because
the NVFP4 forward path uses a fused Triton kernel."""

import pytest
import torch

from modelopt.torch.quantization.nn import NVFP4StaticQuantizer, TensorQuantizer


@pytest.mark.skipif(not torch.cuda.is_available(), reason="NVFP4 path requires CUDA")
def test_mixed_nvfp4_fp8_only_nvfp4_promoted():
    """Mixed NVFP4 + FP8: the NVFP4 layer is promoted to NVFP4StaticQuantizer; the
    FP8 layer is left as a plain TensorQuantizer (max-calibrated amax preserved,
    no MseCalibrator replacement)."""
    import modelopt.torch.quantization as mtq
    from modelopt.torch.quantization.model_calib import mse_calibrate

    # block_size=16 forces linear1.in_features=16.
    class _TwoLayer(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.linear1 = torch.nn.Linear(16, 16, bias=False)
            self.linear2 = torch.nn.Linear(16, 8, bias=False)

        def forward(self, x):
            return self.linear2(self.linear1(x))

    device = torch.device("cuda")
    model = _TwoLayer().to(device)
    inputs = torch.randn(1, 16, device=device)
    config = {
        "quant_cfg": [
            {"quantizer_name": "*", "enable": False},
            {  # Layer 1 — NVFP4 static (eligible for NVFP4StaticQuantizer promotion).
                "quantizer_name": "*linear1.weight_quantizer",
                "cfg": {
                    "num_bits": (2, 1),
                    "block_sizes": {-1: 16, "type": "static", "scale_bits": (4, 3)},
                    "axis": None,
                },
            },
            {  # Layer 2 — FP8 per-tensor (not NVFP4; should be left alone).
                "quantizer_name": "*linear2.weight_quantizer",
                "cfg": {"num_bits": (4, 3), "axis": None},
            },
        ],
        "algorithm": "max",
    }
    mtq.quantize(model, config, forward_loop=lambda m: m(inputs))
    mse_calibrate(model, lambda m: m(inputs), fp8_scale_sweep=True)

    # NVFP4 layer: promoted from TensorQuantizer to NVFP4StaticQuantizer.
    assert isinstance(model.linear1.weight_quantizer, NVFP4StaticQuantizer)
    # FP8 layer: exact type TensorQuantizer — not a subclass, no MseCalibrator
    # replacement, max-calibrated amax kept.
    assert type(model.linear2.weight_quantizer) is TensorQuantizer
