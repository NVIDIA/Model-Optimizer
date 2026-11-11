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

"""GPU tests for local-Hessian calibration's bfloat16 Hessian GEMM."""

import pytest
import torch

import modelopt.torch.quantization as mtq
from modelopt.torch.quantization.model_calib import _LocalHessianAccumulator
from modelopt.torch.quantization.utils import calib_utils


def _fast():
    return calib_utils._has_float32_out_gemm(torch.device("cuda", torch.cuda.current_device()))


def test_bf16_local_hessian_input_gives_the_float32_hessian():
    """``input_dtype=torch.bfloat16`` gives the float32 per-block Hessian, and with float32-output
    GEMMs makes no float32 copy of the inputs."""
    torch.manual_seed(0)
    batches = [torch.randn(4096, 1024, device="cuda", dtype=torch.bfloat16) for _ in range(3)]
    inputs = torch.cat(batches).double().T.reshape(64, 16, -1)
    expected = inputs @ inputs.transpose(-1, -2)
    accumulator = _LocalHessianAccumulator(8, 1024, 16, torch.bfloat16)
    for batch in batches:
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        base = torch.cuda.memory_allocated()
        accumulator.accumulate(batch)
        growth = torch.cuda.max_memory_allocated() - base
    error = (accumulator.hessian_per_block.double() - expected).abs().max()
    # Tensor cores round float32 sums their own way: ~1e-4 of the diagonal, not BF16's 1e-2.
    assert error < 1e-3 * expected.diagonal(dim1=-2, dim2=-1).mean()
    if _fast():
        assert growth < batch.numel() * 4


def test_local_hessian_input_dtype_reaches_the_hessian():
    """A recipe's ``hessian_input_dtype: bfloat16`` reaches local Hessian's accumulator: its Hessian
    matches the float32 one to bfloat16-GEMM rounding, without being the same sums."""
    if not _fast():
        pytest.skip("needs CUDA GEMMs that take out_dtype")
    torch.manual_seed(0)
    inputs = torch.randn(4, 2048, 1024, device="cuda", dtype=torch.bfloat16)
    hessians = {}
    for hessian_input_dtype in ("float32", "bfloat16"):
        model = torch.nn.Linear(1024, 32, bias=False, device="cuda", dtype=torch.bfloat16)
        quant_cfg = {
            "quant_cfg": [
                {"quantizer_name": "*", "enable": False},
                {"quantizer_name": "*weight_quantizer", "cfg": {"num_bits": 8, "axis": 0}},
            ],
            "algorithm": {
                "method": "local_hessian",
                "hessian_input_dtype": hessian_input_dtype,
                "debug": True,
            },
        }
        mtq.quantize(model, quant_cfg, forward_loop=lambda m: m(inputs))
        (accumulator,) = model._local_hessian_accumulators.values()
        hessians[hessian_input_dtype] = accumulator.hessian_per_block
    diagonal = hessians["float32"].diagonal(dim1=-2, dim2=-1).mean()
    assert not torch.equal(hessians["float32"], hessians["bfloat16"])
    assert (hessians["float32"] - hessians["bfloat16"]).abs().max() < 1e-3 * diagonal
