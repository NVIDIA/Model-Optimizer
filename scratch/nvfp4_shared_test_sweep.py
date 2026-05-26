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

# ruff: noqa: D103

"""NVFP4 input_scale sweep on a SHARED held-out test tensor.

Uses calibration / test captures produced by
``capture_calib_and_test_split.py``:

- Calibration pools per combo:
    qwen35_cnn_nemotron_v2_mix_calib_layer{0,31}.pt
    qwen35_nemotron_post_training_v3_calib_layer{0,31}.pt
- Held-out test pools (positions strictly after each combo's calibration
  range):
    qwen35_cnn_nemotron_v2_mix_test_layer{0,31}.pt
    qwen35_nemotron_post_training_v3_test_layer{0,31}.pt

Test tensor = concatenation of BOTH combos' test pools (256+256=512 samples
of activations not seen during either combo's calibration). First 1M
elements of that concat → fixed (256, 4096) tensor used for every MSE
measurement.

Sweep: for each combo, sample N_seqs ∈ {256, 512, 1024} sequences from its
calib pool (3 seeds each), compute amax → input_scale, quantize the SHARED
test tensor, measure MSE.
"""

import json
import math
from pathlib import Path

import torch

from modelopt.torch.quantization.qtensor.nvfp4_tensor import NVFP4QTensor

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
NV_BLOCK = 16
DEFAULT_DENOM = 6.0 * 448.0  # 2688
TEST_SIZE = 1 << 20  # 1M elements

CALIB_FILES = {
    "cnn_nemotron_v2_mix": (
        Path(__file__).parent / "qwen35_cnn_nemotron_v2_mix_calib_layer0.pt",
        Path(__file__).parent / "qwen35_cnn_nemotron_v2_mix_calib_layer31.pt",
    ),
    "nemotron-post-training-v3": (
        Path(__file__).parent / "qwen35_nemotron_post_training_v3_calib_layer0.pt",
        Path(__file__).parent / "qwen35_nemotron_post_training_v3_calib_layer31.pt",
    ),
}

TEST_FILES = {
    "cnn_nemotron_v2_mix": (
        Path(__file__).parent / "qwen35_cnn_nemotron_v2_mix_test_layer0.pt",
        Path(__file__).parent / "qwen35_cnn_nemotron_v2_mix_test_layer31.pt",
    ),
    "nemotron-post-training-v3": (
        Path(__file__).parent / "qwen35_nemotron_post_training_v3_test_layer0.pt",
        Path(__file__).parent / "qwen35_nemotron_post_training_v3_test_layer31.pt",
    ),
}


# ---------- Quantization roundtrip ------------------------------------------


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


def amax_to_scale(amax):
    return torch.tensor(amax / DEFAULT_DENOM, device=DEVICE, dtype=torch.float32)


def amax_of_seq_subset(calib_seqs, n_seqs, seed):
    """Pick n_seqs random sequences, return amax computed incrementally on CPU."""
    total = len(calib_seqs)
    if n_seqs > total:
        raise RuntimeError(f"n_seqs={n_seqs} > pool size {total}")
    g = torch.Generator(device="cpu").manual_seed(seed)
    idx = torch.randperm(total, generator=g)[:n_seqs].tolist()
    a_max = 0.0
    n_tokens = 0
    for i in idx:
        s = calib_seqs[i]
        a = float(s.abs().max().item())
        a_max = max(a_max, a)
        n_tokens += s.shape[0]
    return a_max, n_tokens


def percentile_cpu(calib_seqs, p):
    flat = torch.cat([s.reshape(-1) for s in calib_seqs])
    abs_vals = flat.float().abs()
    n = abs_vals.numel()
    k = max(1, min(n, round(p * n)))
    return float(abs_vals.kthvalue(k).values.item())


def oracle_input_scale(test_tensor):
    test_fp32 = test_tensor.float()
    test_amax = test_fp32.abs().max().item()
    s_default = test_amax / DEFAULT_DENOM
    best_mse = float("inf")
    best_s = s_default
    for a in torch.logspace(-2, 2, 81):
        s = float(a) * s_default
        m = mse(test_fp32, nvfp4_roundtrip(test_tensor, torch.tensor(s, device=DEVICE)))
        if m < best_mse:
            best_mse = m
            best_s = s
    lo, hi = best_s / 1.5, best_s * 1.5
    for s in torch.linspace(math.log(lo), math.log(hi), 41).exp():
        m = mse(test_fp32, nvfp4_roundtrip(test_tensor, torch.tensor(float(s), device=DEVICE)))
        if m < best_mse:
            best_mse = m
            best_s = float(s)
    return best_s, best_mse


# ---------- Driver ----------------------------------------------------------


def build_shared_test_tensor(layer_idx):
    """Concat all combos' test pools at this layer, take first 1M elements."""
    parts = []
    total_seqs = 0
    for p0, p31 in TEST_FILES.values():
        path = p0 if layer_idx == 0 else p31
        seqs = torch.load(path, weights_only=True)
        total_seqs += len(seqs)
        parts.append(torch.cat(seqs, dim=0))
    test_flat = torch.cat(parts, dim=0).reshape(-1)
    hidden = parts[0].shape[-1]
    assert hidden % NV_BLOCK == 0
    rows = TEST_SIZE // hidden
    if test_flat.numel() < rows * hidden:
        raise RuntimeError(
            f"Need {rows * hidden} elts in shared test pool, have {test_flat.numel()}"
        )
    return test_flat[: rows * hidden].view(rows, hidden).to(DEVICE), total_seqs


def load_calib(combo, layer_idx):
    p0, p31 = CALIB_FILES[combo]
    path = p0 if layer_idx == 0 else p31
    return torch.load(path, weights_only=True)


def main():
    print(f"device: {DEVICE}")
    if DEVICE.type == "cuda":
        print(f"gpu:    {torch.cuda.get_device_name(0)}\n")

    n_seqs_list = [256, 512, 1024]
    n_seeds = 3
    results = []

    for layer_idx in (0, 31):
        layer_name = f"layer {layer_idx}"
        print(f"\n{'=' * 80}\n{layer_name} MLP input — shared held-out test tensor")
        print("=" * 80)

        test, n_test_seqs = build_shared_test_tensor(layer_idx)
        test_fp32 = test.float()
        test_amax = test_fp32.abs().max().item()
        sig_pow = (test_fp32**2).mean().item()
        print(
            f"  shared test tensor: shape={tuple(test.shape)}  "
            f"from {n_test_seqs} held-out sequences\n"
            f"  test amax={test_amax:.3f}  sig_pow={sig_pow:.3e}"
        )

        oracle_s, oracle_mse = oracle_input_scale(test)
        oracle_snr = 10 * math.log10(sig_pow / oracle_mse) if oracle_mse > 0 else float("inf")
        print(
            f"  oracle: input_scale={oracle_s:.3e}  "
            f"oracle_amax_eq={oracle_s * DEFAULT_DENOM:.3f}  "
            f"MSE={oracle_mse:.3e}  ({oracle_snr:.3f} dB)\n"
        )

        layer_rows = []
        for combo in CALIB_FILES:
            calib_seqs = load_calib(combo, layer_idx)
            n_pool = len(calib_seqs)
            print(f"  --- calibrating with {combo} (pool {n_pool} seqs) ---")
            print(
                f"  {'n_seqs':>8}{'amax_mean':>13}{'amax_std':>10}{'MSE_mean':>14}{'SNR(dB)':>10}"
            )
            for n_seqs in n_seqs_list:
                if n_seqs > n_pool:
                    print(f"  skipping n_seqs={n_seqs} (pool only {n_pool})")
                    continue
                amaxes, mses = [], []
                for seed in range(n_seeds):
                    a, _ = amax_of_seq_subset(calib_seqs, n_seqs, seed=1000 + seed)
                    amaxes.append(a)
                    mses.append(mse(test_fp32, nvfp4_roundtrip(test, amax_to_scale(a))))
                a_mean = sum(amaxes) / len(amaxes)
                a_std = (sum((x - a_mean) ** 2 for x in amaxes) / len(amaxes)) ** 0.5
                m_mean = sum(mses) / len(mses)
                snr = 10 * math.log10(sig_pow / m_mean) if m_mean > 0 else float("inf")
                print(f"  {n_seqs:>8}{a_mean:>13.4f}{a_std:>10.4f}{m_mean:>14.3e}{snr:>10.3f}")
                layer_rows.append(
                    {
                        "combo": combo,
                        "n_seqs": n_seqs,
                        "amax_mean": a_mean,
                        "amax_std": a_std,
                        "mse_mean": m_mean,
                        "snr_db": snr,
                    }
                )

            # Percentile baselines on this combo's full pool
            print(f"  {'percentile':>10}{'amax':>14}{'MSE':>14}{'SNR(dB)':>10}")
            for p in [0.99, 0.999, 0.9999, 0.99999]:
                a = percentile_cpu(calib_seqs, p)
                m = mse(test_fp32, nvfp4_roundtrip(test, amax_to_scale(a)))
                snr = 10 * math.log10(sig_pow / m) if m > 0 else float("inf")
                print(f"  {f'p{p:.5f}':>10}{a:>14.4f}{m:>14.3e}{snr:>10.3f}")

        print(
            f"  {'oracle':>10}{oracle_s * DEFAULT_DENOM:>14.3f}{oracle_mse:>14.3e}"
            f"{oracle_snr:>10.3f}"
        )

        results.append(
            {
                "layer_idx": layer_idx,
                "test_amax": test_amax,
                "sig_pow": sig_pow,
                "oracle_input_scale": oracle_s,
                "oracle_mse": oracle_mse,
                "oracle_snr_db": oracle_snr,
                "rows": layer_rows,
            }
        )

    out_path = Path(__file__).parent / "nvfp4_shared_test_sweep_results.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved curves to {out_path}")


if __name__ == "__main__":
    main()
