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

# Research scratch script — relax some style rules that don't add value here.
# ruff: noqa: D103, RUF003

"""Does more calibration always help NVFP4 activation quantization?

NVFP4 activation quantization sets a per-tensor ``input_scale`` from the amax
observed during calibration:

    input_scale = amax_calib / (6 * 448) = amax_calib / 2688.

At inference, only ``input_scale`` is static — per-block FP8 scales are derived
dynamically as ``block_amax / (6 * input_scale)`` and cast to E4M3. With heavy
tailed activations, ``amax_calib`` grows roughly like a Type-II extreme-value
statistic in the number of samples — so calibrating with more sequences / a
longer batch monotonically inflates ``input_scale``. The question:

  Does larger ``input_scale`` (from more calibration) always reduce MSE on
  unseen activations, or does it eventually hurt because every typical block
  gets dynamic per-block scales pushed toward the FP8 subnormal range?

This script tests that on five activation-style synthetic distributions by:

  (1) drawing a fixed "test" tensor from the distribution,
  (2) drawing independent calibration tensors of increasing size,
  (3) taking ``amax_calib`` from each calibration draw → ``input_scale``,
  (4) quantizing the test tensor with that ``input_scale`` (dynamic per-block),
  (5) measuring MSE / SNR on the test tensor.

We also compare against:
  - **percentile** calibration (99.0 / 99.9 / 99.99 percentile of the calib
    sample, instead of the absolute max),
  - **oracle** calibration (sweep ``input_scale`` directly on the test tensor
    and pick the MSE-minimizer; an unrealistic upper bound).
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


# ---------- Quantization roundtrip ------------------------------------------


def nvfp4_roundtrip(x_bf16: torch.Tensor, input_scale: torch.Tensor) -> torch.Tensor:
    """Quantize x with the supplied input_scale (static), dynamic per-block scales."""
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


def snr_db(ref: torch.Tensor, approx: torch.Tensor) -> float:
    sig = (ref.float() ** 2).mean().item()
    err = ((ref.float() - approx.float()) ** 2).mean().item()
    if err <= 0:
        return float("inf")
    if sig <= 0:
        return float("-inf")
    return 10.0 * math.log10(sig / err)


def amax_to_scale(amax: float) -> torch.Tensor:
    return torch.tensor(amax / DEFAULT_DENOM, device=DEVICE, dtype=torch.float32)


# ---------- Activation distributions -----------------------------------------
# Each generator(n_elements) returns a 1D bf16 sample of size n_elements drawn
# i.i.d. from the underlying activation distribution. Shapes are handled by the
# caller (test tensor is reshaped to 2D for blocking).


def _dist_post_layernorm(n: int) -> torch.Tensor:
    """Post-LayerNorm activation: ~N(0,1) with a light heavy tail from rare spikes.

    Most LayerNorm outputs are O(1). 0.1% of values get a 10x kick to mimic the
    occasional "outlier feature" channel that gets amplified by residual streams.
    """
    x = torch.randn(n, device=DEVICE)
    mask = torch.rand(n, device=DEVICE) < 1e-3
    x[mask] *= 10.0
    return x.bfloat16()


def _dist_per_channel_outlier(n: int) -> torch.Tensor:
    """SmoothQuant-style: a handful of channels carry 50-100x larger activations.

    We model channels as a length-1024 hidden dim. ~1% of channels are 'hot'
    (100x scaling). Tokens (rows) are gaussian in their channel scale.
    """
    hidden = 1024
    rows = max(n // hidden, 1)
    x = torch.randn(rows, hidden, device=DEVICE)
    # Pick a small fixed set of hot channels (fixed across rows — that's the point)
    n_hot = max(hidden // 100, 1)
    hot = torch.randperm(hidden, device=DEVICE)[:n_hot]
    x[:, hot] *= 100.0
    return x.reshape(-1)[:n].bfloat16()


def _dist_per_token_outlier(n: int) -> torch.Tensor:
    """A small fraction of TOKENS (rows) are 'hot' — every channel pumped up.

    Activation entering attention after a residual that hit a peaky token.
    """
    hidden = 1024
    rows = max(n // hidden, 1)
    x = torch.randn(rows, hidden, device=DEVICE) * 0.5
    n_hot = max(rows // 50, 1)
    hot_rows = torch.randperm(rows, device=DEVICE)[:n_hot]
    x[hot_rows] *= 30.0
    return x.reshape(-1)[:n].bfloat16()


def _dist_post_gelu(n: int) -> torch.Tensor:
    """Post-GeLU: half-Gaussian-ish positive tail, rare negative bulge.

    GeLU(x) ≈ x for large x, ≈ 0 for very negative x. Distribution of GeLU(N(0,2))
    is one-sided with a heavy positive tail.
    """
    pre = torch.randn(n, device=DEVICE) * 2.0
    # GeLU approx: x * 0.5 * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
    cube = pre * pre * pre
    inner = math.sqrt(2.0 / math.pi) * (pre + 0.044715 * cube)
    x = 0.5 * pre * (1.0 + torch.tanh(inner))
    return x.bfloat16()


def _dist_pareto_mixture(n: int) -> torch.Tensor:
    """Heavy-tailed mixture: 99% N(0, 1) + 1% Pareto-like outliers.

    Designed so that ``amax`` grows monotonically with sample size n: more
    samples → likelier to hit a Pareto extreme.
    """
    bulk = torch.randn(n, device=DEVICE)
    out_mask = torch.rand(n, device=DEVICE) < 0.01
    # Pareto(alpha=1.5): heavy tail, finite mean but no variance.
    u = torch.rand(int(out_mask.sum().item()), device=DEVICE).clamp(min=1e-7)
    pareto = (u ** -(1.0 / 1.5)) * torch.sign(torch.randn_like(u))
    bulk[out_mask] = pareto * 3.0
    return bulk.bfloat16()


def _dist_rare_giant_spike(n: int) -> torch.Tensor:
    """Pathological 'training instability spike' pattern.

    Bulk is N(0, 1). At rate 1e-7, a single value spikes to ±1e6. Most calib
    draws will miss the spike (so amax ≈ 5 — true tail of the bulk), but a
    LARGE calib will hit it (so amax explodes to 1e6, inflating input_scale).
    The test tensor here is *deliberately* drawn so it does not contain the
    spike — modelling 'calibration overshoots inference' for a transient
    training artifact.
    """
    x = torch.randn(n, device=DEVICE)
    mask = torch.rand(n, device=DEVICE) < 1e-7
    n_spikes = int(mask.sum().item())
    if n_spikes > 0:
        x[mask] = 1e6 * torch.sign(torch.randn(n_spikes, device=DEVICE) + 1e-9)
    return x.bfloat16()


def _dist_lognormal_heavy(n: int) -> torch.Tensor:
    """Log-normal magnitude × random sign. sigma=3 gives a wickedly heavy tail.

    log-normal sigma=3: 99.99% of values are < ~20; 1-in-1e8 hits ~1e6.
    """
    g = torch.randn(n, device=DEVICE) * 3.0
    mag = torch.exp(g)  # log-normal
    sign = torch.sign(torch.randn(n, device=DEVICE) + 1e-9)
    return (sign * mag).bfloat16()


def _dist_rare_giant_spike_test(n: int) -> torch.Tensor:
    """Test-tensor variant of the giant-spike distribution: spike-free.

    Models the 'calibration overshoots inference' regime: at training time, a
    rare giant spike got captured and inflated input_scale; at inference, no
    such spike appears, so the test tensor is just N(0, 1).
    """
    return torch.randn(n, device=DEVICE).bfloat16()


# Tuple form: (name, calib_gen, test_gen). test_gen defaults to calib_gen.
DISTRIBUTIONS = [
    ("post-LayerNorm + rare 10x spikes", _dist_post_layernorm, None),
    ("per-channel outlier (1% chans, 100x)", _dist_per_channel_outlier, None),
    ("per-token outlier (2% tokens, 30x)", _dist_per_token_outlier, None),
    ("post-GeLU one-sided heavy tail", _dist_post_gelu, None),
    ("Pareto(1.5) heavy-tail mixture", _dist_pareto_mixture, None),
    (
        "rare giant spike (1e-7, mag=1e6, test=clean)",
        _dist_rare_giant_spike,
        _dist_rare_giant_spike_test,
    ),
    ("log-normal magnitude (sigma=3)", _dist_lognormal_heavy, None),
]


# ---------- Calibration baselines --------------------------------------------


def amax_calib(calib_sample: torch.Tensor) -> float:
    """Standard amax calibration: the absolute max of the sample."""
    return float(calib_sample.float().abs().max().item())


def percentile_calib(calib_sample: torch.Tensor, p: float) -> float:
    """Percentile-based calibration: more robust to rare outliers.

    ``torch.quantile`` has an internal size cap, so we use ``kthvalue`` on the
    sorted absolute values for large inputs.
    """
    abs_vals = calib_sample.float().abs().reshape(-1)
    n = abs_vals.numel()
    k = max(1, min(n, round(p * n)))
    return float(abs_vals.kthvalue(k).values.item())


def oracle_input_scale(test_tensor: torch.Tensor) -> tuple[float, float]:
    """Sweep input_scale directly on the test tensor and return the MSE minimum.

    Returns (best_input_scale, best_mse). This is the unrealistic upper bound:
    it 'cheats' by seeing the test tensor before picking the scale.
    """
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
    # Refine
    lo, hi = best_s / 1.5, best_s * 1.5
    alphas = torch.linspace(math.log(lo), math.log(hi), 41).exp()
    for s in alphas:
        out = nvfp4_roundtrip(test_tensor, torch.tensor(float(s), device=DEVICE))
        m = mse(test_fp32, out)
        if m < best_mse:
            best_mse = m
            best_s = float(s)
    return best_s, best_mse


# ---------- Driver -----------------------------------------------------------


def run_distribution(
    name: str, calib_gen, test_gen, calib_sizes: list[int], n_seeds: int = 3
) -> dict:
    """Run the calibration-size sweep for one distribution.

    Strategy: one large test tensor, multiple seeds of fresh calibration draws
    at each size. Report mean and std of test-MSE across seeds.
    """
    print(f"\n=== {name} ===")

    # Test tensor: large enough that block statistics are well-resolved.
    torch.manual_seed(12345)  # fixed test tensor across all settings
    gen_for_test = test_gen if test_gen is not None else calib_gen
    test = gen_for_test(1 << 20).reshape(1024, 1024)  # 1M elements
    test_fp32 = test.float()
    test_amax = test_fp32.abs().max().item()
    sig_pow = (test_fp32**2).mean().item()

    oracle_s, oracle_mse = oracle_input_scale(test)
    oracle_snr = 10 * math.log10(sig_pow / oracle_mse) if oracle_mse > 0 else float("inf")
    print(
        f"  Test tensor amax={test_amax:.3f}, signal_pow={sig_pow:.3e}, "
        f"oracle MSE={oracle_mse:.3e} ({oracle_snr:.2f} dB), oracle_S={oracle_s:.3e}"
    )

    # amax-based sweep across calib sizes (averaged over seeds)
    amax_curve = []  # list of dicts per calib size
    for n_calib in calib_sizes:
        amax_samples = []
        mses = []
        for seed in range(n_seeds):
            torch.manual_seed(100 + seed)
            calib = calib_gen(n_calib)
            a = amax_calib(calib)
            amax_samples.append(a)
            out = nvfp4_roundtrip(test, amax_to_scale(a))
            mses.append(mse(test_fp32, out))
        a_mean = sum(amax_samples) / len(amax_samples)
        a_std = (sum((x - a_mean) ** 2 for x in amax_samples) / len(amax_samples)) ** 0.5
        m_mean = sum(mses) / len(mses)
        m_std = (sum((x - m_mean) ** 2 for x in mses) / len(mses)) ** 0.5
        snr_mean = 10 * math.log10(sig_pow / m_mean) if m_mean > 0 else float("inf")
        amax_curve.append(
            {
                "n_calib": n_calib,
                "amax_mean": a_mean,
                "amax_std": a_std,
                "mse_mean": m_mean,
                "mse_std": m_std,
                "snr_mean_db": snr_mean,
            }
        )

    # Percentile baselines: use the LARGEST calib size and a few percentiles
    n_pct_calib = max(calib_sizes)
    pct_results = {}
    for p in [0.99, 0.999, 0.9999]:
        pct_amaxes = []
        pct_mses = []
        for seed in range(n_seeds):
            torch.manual_seed(100 + seed)
            calib = calib_gen(n_pct_calib)
            a = percentile_calib(calib, p)
            pct_amaxes.append(a)
            out = nvfp4_roundtrip(test, amax_to_scale(a))
            pct_mses.append(mse(test_fp32, out))
        m_mean = sum(pct_mses) / len(pct_mses)
        a_mean = sum(pct_amaxes) / len(pct_amaxes)
        snr_mean = 10 * math.log10(sig_pow / m_mean) if m_mean > 0 else float("inf")
        pct_results[f"p{p:.4f}"] = {
            "amax_mean": a_mean,
            "mse_mean": m_mean,
            "snr_mean_db": snr_mean,
        }

    print(f"  {'n_calib':>10}{'amax_mean':>14}{'MSE_mean':>14}{'SNR(dB)':>10}")
    for row in amax_curve:
        print(
            f"  {row['n_calib']:>10}"
            f"{row['amax_mean']:>14.4f}"
            f"{row['mse_mean']:>14.3e}"
            f"{row['snr_mean_db']:>10.2f}"
        )

    print(f"  {'percentile':>10}{'amax_mean':>14}{'MSE_mean':>14}{'SNR(dB)':>10}")
    for k, v in pct_results.items():
        print(f"  {k:>10}{v['amax_mean']:>14.4f}{v['mse_mean']:>14.3e}{v['snr_mean_db']:>10.2f}")
    print(
        f"  {'oracle':>10}{oracle_s * DEFAULT_DENOM:>14.4f}{oracle_mse:>14.3e}{oracle_snr:>10.2f}"
    )

    # Find amax-calibration MSE minimum (best n_calib)
    best_amax_row = min(amax_curve, key=lambda r: r["mse_mean"])
    final_amax_row = amax_curve[-1]
    delta_db_best_vs_final = best_amax_row["snr_mean_db"] - final_amax_row["snr_mean_db"]
    delta_db_oracle_vs_final = oracle_snr - final_amax_row["snr_mean_db"]

    return {
        "name": name,
        "test_amax": test_amax,
        "sig_power": sig_pow,
        "oracle_input_scale": oracle_s,
        "oracle_mse": oracle_mse,
        "oracle_snr_db": oracle_snr,
        "amax_curve": amax_curve,
        "percentile_results": pct_results,
        "best_amax_n_calib": best_amax_row["n_calib"],
        "best_amax_mse": best_amax_row["mse_mean"],
        "final_amax_mse": final_amax_row["mse_mean"],
        "delta_db_best_vs_final": delta_db_best_vs_final,
        "delta_db_oracle_vs_final": delta_db_oracle_vs_final,
    }


def main():
    print(f"device: {DEVICE}")
    if DEVICE.type == "cuda":
        print(f"gpu:    {torch.cuda.get_device_name(0)}")

    # Calibration sample sizes — log-spaced from a single sequence (~1k tokens
    # × 1k hidden) up to a large multi-batch (~32M elements, enough for the
    # giant-spike case at p=1e-7 to be likely to capture a spike).
    calib_sizes = [
        1024,  # ~1 short sequence
        4096,  # 1 batch
        16384,  # ~16 sequences
        65536,
        262144,
        1048576,  # ~1M elements
        4194304,  # ~4M
        16777216,  # ~16M
        33554432,  # ~32M (probability of seeing a 1e-7 spike: ~96%)
    ]

    results = []
    for name, calib_gen, test_gen in DISTRIBUTIONS:
        r = run_distribution(name, calib_gen, test_gen, calib_sizes, n_seeds=3)
        results.append(r)

    # Final cross-distribution summary
    print("\n" + "=" * 90)
    print("Summary: does 'more calibration is always better' hold for amax-based input_scale?")
    print("=" * 90)
    print(f"{'distribution':<40}{'best n_calib':>14}{'best vs max':>14}{'oracle vs max':>16}")
    for r in results:
        print(
            f"{r['name']:<40}"
            f"{r['best_amax_n_calib']:>14}"
            f"{r['delta_db_best_vs_final']:>13.2f}dB"
            f"{r['delta_db_oracle_vs_final']:>15.2f}dB"
        )
    print()
    print("'best vs max' = SNR gain of best amax-calib size over the largest calib size.")
    print("'oracle vs max' = SNR gain of the unrealistic oracle scale over the largest amax calib.")
    print()

    # Save sweep curves to JSON for plotting
    out_path = Path(__file__).parent / "nvfp4_activation_calib_results.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Saved sweep curves to {out_path}")


if __name__ == "__main__":
    main()
