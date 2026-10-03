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

"""Apply each calibration combo's recorded amax to a single fixed test tensor.

This gives an apples-to-apples comparison of calibration-policy quality:
which combo's amax (treated as input_scale = amax / 2688) minimizes MSE on
the same test data?
"""

import math

import torch

from modelopt.torch.quantization.qtensor.nvfp4_tensor import NVFP4QTensor

DEVICE = torch.device("cuda")
NV_BLOCK = 16
DENOM = 6.0 * 448.0  # 2688

# Calibration amaxes measured in each of the three previous runs (at N=2048
# sequences, mean across 3 seeds).
CALIB_AMAXES = {
    "layer 0": {
        "chat only (Nemotron v2)": 4.531,
        "cnn_nemotron_v2_mix": 4.271,
        "nemotron-post-training-v3": 4.156,
    },
    "layer 31": {
        "chat only (Nemotron v2)": 52.000,
        "cnn_nemotron_v2_mix": 50.667,
        "nemotron-post-training-v3": 53.000,
    },
}

# These .pt files are the v3 captures (current state on disk).
PATHS = {
    "layer 0": "scratch/qwen35_9b_mlp_input_layer0.pt",
    "layer 31": "scratch/qwen35_9b_mlp_input_layer31.pt",
}

# We use the last 100 sequences as test (matches N_TEST_SEQS=100 used in the
# most recent v3 sweep run).
N_TEST_SEQS = 100
TEST_SIZE = 1 << 20


def nvfp4_roundtrip(x_bf16, input_scale):
    nv_qt, pb_scale, _ = NVFP4QTensor.quantize(
        x_bf16,
        block_size=NV_BLOCK,
        weights_scaling_factor_2=input_scale.to(x_bf16.device).float(),
    )
    out = nv_qt.dequantize(
        dtype=torch.float32,
        scale=pb_scale,
        double_scale=input_scale.to(x_bf16.device).float(),
        block_sizes={-1: NV_BLOCK},
    )
    return out.float()


def mse(a, b):
    return float(((a.float() - b.float()) ** 2).mean().item())


def main():
    for layer_name, path in PATHS.items():
        seqs = torch.load(path, weights_only=True)
        test_seqs = seqs[-N_TEST_SEQS:]
        flat = torch.cat(test_seqs, dim=0).reshape(-1)
        hidden = test_seqs[0].shape[-1]
        rows = TEST_SIZE // hidden
        test = flat[: rows * hidden].view(rows, hidden).to(DEVICE)
        test_fp32 = test.float()
        sig_pow = (test_fp32**2).mean().item()
        test_amax = test_fp32.abs().max().item()

        print(f"\n=== {layer_name} ===")
        print(f"  test tensor amax={test_amax:.3f}, sig_pow={sig_pow:.3e}")
        print(f"  {'combo':<35}{'amax':>10}{'in_scale':>14}{'MSE':>14}{'SNR(dB)':>10}")
        rows_out = []
        for combo, amax in CALIB_AMAXES[layer_name].items():
            s = torch.tensor(amax / DENOM, device=DEVICE, dtype=torch.float32)
            out = nvfp4_roundtrip(test, s)
            m = mse(test_fp32, out)
            snr = 10 * math.log10(sig_pow / m) if m > 0 else float("inf")
            print(f"  {combo:<35}{amax:>10.3f}{(amax / DENOM):>14.4e}{m:>14.3e}{snr:>10.3f}")
            rows_out.append((combo, amax, m, snr))

        # Find best
        best = min(rows_out, key=lambda r: r[2])
        worst = max(rows_out, key=lambda r: r[2])
        gap = 10 * math.log10(worst[2] / best[2])
        print(f"  --> best: {best[0]} (SNR={best[3]:.3f} dB)")
        print(f"  --> spread (worst-best): {gap:.3f} dB")


if __name__ == "__main__":
    main()
