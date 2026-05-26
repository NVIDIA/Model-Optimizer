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

"""NVFP4 input_scale calibration-size sweep on REAL MLP-input activations.

Reads ``.pt`` files captured by ``capture_qwen35_mlp_activations.py`` from
Qwen3.5-9B's first and last transformer block MLP inputs, then runs the same
"more calibration → MSE" sweep as ``nvfp4_activation_calib_mse.py`` but with
real activations instead of synthetic distributions.

Calibration is by **number of sequences** rather than scalar elements: at
N ∈ {512, 1024, 2048}, the script draws N whole sequences uniformly at random
from the calibration pool (3 seeds), flattens them, computes
``amax = max(|x_calib|)`` → ``input_scale = amax / 2688``, and reports MSE on
the held-out test tensor.
"""

import json
import math
from pathlib import Path

import torch

from modelopt.torch.quantization.qtensor.nvfp4_tensor import NVFP4QTensor

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
NV_BLOCK = 16
FP4_MAX = 6.0
E4M3_MAX = 448.0
DEFAULT_DENOM = FP4_MAX * E4M3_MAX  # 2688

ACTIVATIONS = {
    "layer 0 MLP input": Path(__file__).parent / "qwen35_9b_mlp_input_layer0.pt",
    "layer 31 MLP input": Path(__file__).parent / "qwen35_9b_mlp_input_layer31.pt",
}

TEST_SIZE = 1 << 20  # 1M elements held out for test


# ---------- Quantization roundtrip ------------------------------------------


def nvfp4_roundtrip(x_bf16: torch.Tensor, input_scale: torch.Tensor) -> torch.Tensor:
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


def mse(a: torch.Tensor, b: torch.Tensor) -> float:
    return float(((a.float() - b.float()) ** 2).mean().item())


def amax_to_scale(amax: float) -> torch.Tensor:
    return torch.tensor(amax / DEFAULT_DENOM, device=DEVICE, dtype=torch.float32)


def percentile(t: torch.Tensor, p: float) -> float:
    abs_vals = t.float().abs().reshape(-1)
    n = abs_vals.numel()
    k = max(1, min(n, round(p * n)))
    return float(abs_vals.kthvalue(k).values.item())


# ---------- Load and split real activations ---------------------------------

N_TEST_SEQS = 100  # held-out test sequences (counted from the end of the file)


def load_split(path: Path):
    """Load list-of-(seq_len, hidden) bf16 tensors, split into calib seqs + test seqs."""
    seqs = torch.load(path, weights_only=True)  # list of (S_i, H) bf16
    assert len(seqs) > N_TEST_SEQS, f"Need >{N_TEST_SEQS} captured sequences, have {len(seqs)}"
    calib_seqs = seqs[:-N_TEST_SEQS]
    test_seqs = seqs[-N_TEST_SEQS:]
    return calib_seqs, test_seqs


def get_test_tensor(test_seqs) -> torch.Tensor:
    """Take the first TEST_SIZE elements of the concatenated test sequences."""
    flat = torch.cat(test_seqs, dim=0).reshape(-1)
    if flat.numel() < TEST_SIZE:
        raise RuntimeError(f"Need at least {TEST_SIZE} elements in test pool, have {flat.numel()}")
    hidden = test_seqs[0].shape[-1]
    assert hidden % NV_BLOCK == 0, f"hidden {hidden} not divisible by {NV_BLOCK}"
    rows = TEST_SIZE // hidden
    return flat[: rows * hidden].view(rows, hidden).to(DEVICE)


def amax_of_seq_subset(calib_seqs, n_seqs: int, seed: int) -> tuple[float, int]:
    """Pick n_seqs random sequences, return (amax, n_tokens) computed incrementally.

    Avoids concatenating into a single GPU tensor — only one sequence is moved
    to GPU at a time, then the per-sequence amax is reduced into a running max.
    """
    total = len(calib_seqs)
    if n_seqs > total:
        raise RuntimeError(f"Asked for {n_seqs} seqs but pool only has {total}")
    g = torch.Generator(device="cpu").manual_seed(seed)
    idx = torch.randperm(total, generator=g)[:n_seqs].tolist()
    a_max = 0.0
    n_tokens = 0
    for i in idx:
        s = calib_seqs[i]  # bf16, cpu
        a = float(s.abs().max().item())
        a_max = max(a_max, a)
        n_tokens += s.shape[0]
    return a_max, n_tokens


def percentile_cpu(calib_seqs, p: float) -> float:
    """Percentile of |x| over all calib tokens, computed on CPU to avoid OOM."""
    flat = torch.cat([s.reshape(-1) for s in calib_seqs])  # bf16 on CPU
    abs_vals = flat.float().abs()
    n = abs_vals.numel()
    k = max(1, min(n, round(p * n)))
    return float(abs_vals.kthvalue(k).values.item())


def oracle_input_scale(test_tensor: torch.Tensor) -> tuple[float, float]:
    test_fp32 = test_tensor.float()
    test_amax = test_fp32.abs().max().item()
    s_default = test_amax / DEFAULT_DENOM
    alphas = torch.logspace(-2, 2, 81)
    best_mse = float("inf")
    best_s = s_default
    for a in alphas:
        s = float(a) * s_default
        out = nvfp4_roundtrip(test_tensor, torch.tensor(s, device=DEVICE))
        m = mse(test_fp32, out)
        if m < best_mse:
            best_mse = m
            best_s = s
    lo, hi = best_s / 1.5, best_s * 1.5
    alphas = torch.linspace(math.log(lo), math.log(hi), 41).exp()
    for s in alphas:
        out = nvfp4_roundtrip(test_tensor, torch.tensor(float(s), device=DEVICE))
        m = mse(test_fp32, out)
        if m < best_mse:
            best_mse = m
            best_s = float(s)
    return best_s, best_mse


# ---------- Sweep ----------------------------------------------------------


def run_layer(name: str, path: Path, n_calib_seqs_list, n_seeds: int = 3) -> dict:
    print(f"\n=== {name} ({path.name}) ===")
    calib_seqs, test_seqs = load_split(path)
    test = get_test_tensor(test_seqs)
    test_fp32 = test.float()
    test_amax = test_fp32.abs().max().item()
    sig_pow = (test_fp32**2).mean().item()
    n_calib_pool = len(calib_seqs)
    n_calib_tokens = sum(s.shape[0] for s in calib_seqs)
    n_test_tokens = sum(s.shape[0] for s in test_seqs)
    print(
        f"  calib pool: {n_calib_pool} seqs, {n_calib_tokens} tokens\n"
        f"  test pool:  {len(test_seqs)} seqs, {n_test_tokens} tokens\n"
        f"  test tensor: {test.shape}, amax={test_amax:.3f}, "
        f"sig_pow={sig_pow:.3e}"
    )

    oracle_s, oracle_mse = oracle_input_scale(test)
    oracle_snr = 10 * math.log10(sig_pow / oracle_mse) if oracle_mse > 0 else float("inf")
    print(
        f"  oracle: input_scale={oracle_s:.3e}, "
        f"oracle_amax_eq={oracle_s * DEFAULT_DENOM:.3f}, "
        f"MSE={oracle_mse:.3e} ({oracle_snr:.2f} dB)"
    )

    amax_curve = []
    for n_seqs in n_calib_seqs_list:
        if n_seqs > n_calib_pool:
            print(f"  skipping n_seqs={n_seqs} (pool has only {n_calib_pool})")
            continue
        amaxes = []
        mses = []
        n_tokens_per_draw = []
        for seed in range(n_seeds):
            a, n_tok = amax_of_seq_subset(calib_seqs, n_seqs, seed=1000 + seed)
            amaxes.append(a)
            n_tokens_per_draw.append(n_tok)
            out = nvfp4_roundtrip(test, amax_to_scale(a))
            mses.append(mse(test_fp32, out))
        a_mean = sum(amaxes) / len(amaxes)
        a_std = (sum((x - a_mean) ** 2 for x in amaxes) / len(amaxes)) ** 0.5
        m_mean = sum(mses) / len(mses)
        m_std = (sum((x - m_mean) ** 2 for x in mses) / len(mses)) ** 0.5
        snr_mean = 10 * math.log10(sig_pow / m_mean) if m_mean > 0 else float("inf")
        tok_mean = sum(n_tokens_per_draw) / len(n_tokens_per_draw)
        amax_curve.append(
            {
                "n_calib_seqs": n_seqs,
                "tokens_mean": tok_mean,
                "amax_mean": a_mean,
                "amax_std": a_std,
                "mse_mean": m_mean,
                "mse_std": m_std,
                "snr_mean_db": snr_mean,
            }
        )

    print(
        f"  {'n_seqs':>8}{'tok_avg':>10}{'amax_mean':>14}{'amax_std':>10}"
        f"{'MSE_mean':>14}{'SNR(dB)':>10}"
    )
    for r in amax_curve:
        print(
            f"  {r['n_calib_seqs']:>8}"
            f"{r['tokens_mean']:>10.0f}"
            f"{r['amax_mean']:>14.4f}"
            f"{r['amax_std']:>10.4f}"
            f"{r['mse_mean']:>14.3e}"
            f"{r['snr_mean_db']:>10.2f}"
        )

    print(f"  {'percentile':>10}{'amax':>14}{'MSE':>14}{'SNR(dB)':>10}")
    pct_results = {}
    for p in [0.99, 0.999, 0.9999, 0.99999]:
        a = percentile_cpu(calib_seqs, p)
        out = nvfp4_roundtrip(test, amax_to_scale(a))
        m = mse(test_fp32, out)
        snr = 10 * math.log10(sig_pow / m) if m > 0 else float("inf")
        pct_results[f"p{p:.5f}"] = {"amax": a, "mse": m, "snr_db": snr}
        print(f"  {f'p{p:.5f}':>10}{a:>14.4f}{m:>14.3e}{snr:>10.2f}")
    print(
        f"  {'oracle':>10}{oracle_s * DEFAULT_DENOM:>14.4f}{oracle_mse:>14.3e}{oracle_snr:>10.2f}"
    )

    best_row = min(amax_curve, key=lambda r: r["mse_mean"])
    final_row = amax_curve[-1]
    return {
        "name": name,
        "path": str(path),
        "test_amax": test_amax,
        "test_shape": list(test.shape),
        "sig_power": sig_pow,
        "n_calib_pool_seqs": n_calib_pool,
        "n_calib_pool_tokens": n_calib_tokens,
        "oracle_input_scale": oracle_s,
        "oracle_mse": oracle_mse,
        "oracle_snr_db": oracle_snr,
        "amax_curve": amax_curve,
        "percentile_results": pct_results,
        "best_amax_n_calib_seqs": best_row["n_calib_seqs"],
        "best_amax_snr": best_row["snr_mean_db"],
        "final_amax_snr": final_row["snr_mean_db"],
        "delta_db_best_vs_final": best_row["snr_mean_db"] - final_row["snr_mean_db"],
        "delta_db_oracle_vs_final": oracle_snr - final_row["snr_mean_db"],
    }


def main():
    print(f"device: {DEVICE}")
    if DEVICE.type == "cuda":
        print(f"gpu:    {torch.cuda.get_device_name(0)}")

    # Calibration sample sizes in *sequence* count.
    n_calib_seqs_list = [512, 1024, 2048]

    results = []
    for name, path in ACTIVATIONS.items():
        r = run_layer(name, path, n_calib_seqs_list, n_seeds=3)
        results.append(r)

    # Summary
    print("\n" + "=" * 90)
    print("Summary: Qwen3.5-9B MLP inputs — does 'more calibration is better' hold?")
    print("=" * 90)
    print(f"{'layer':<32}{'best n_seqs':>13}{'best vs max':>14}{'oracle vs max':>16}")
    for r in results:
        print(
            f"{r['name']:<32}"
            f"{r['best_amax_n_calib_seqs']:>13}"
            f"{r['delta_db_best_vs_final']:>13.2f}dB"
            f"{r['delta_db_oracle_vs_final']:>15.2f}dB"
        )

    out_path = Path(__file__).parent / "nvfp4_real_activation_calib_results.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved curves to {out_path}")


if __name__ == "__main__":
    main()
