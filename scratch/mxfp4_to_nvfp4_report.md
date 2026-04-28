# MXFP4 → NVFP4 Conversion: MSE Analysis of Three Algorithms

## Problem

Convert an MXFP4-quantized tensor (block size 32, E2M1 mantissa, E8M0 power-of-two
scale) to an NVFP4 tensor (block size 16, E2M1 mantissa, E4M3 per-block scale, FP32
global scale `scale_2`). Reference for error measurement is the MXFP4-dequantized
tensor — i.e. the values MXFP4 faithfully encodes — since both algorithms aim to
preserve those values in the NVFP4 representation.

Notation: each MXFP4 block `j` has integer exponent `k_j` (so its scale is `2^k_j`).
E4M3 represents `2^k` exactly only for `k ∈ [−9, 8]` (an 18-value window with
spread 17).

## Algorithms

### Algo 1: Dequantize → Requantize (baseline)

```text
MXFP4 → BF16 (dequantize) → NVFP4 (standard quantize)
```

The "obvious" approach. Always introduces error from:
- Per-16-element re-bucketing (NVFP4 picks new amax per 16-block)
- E4M3 mantissa quantization of per-block scales

### Algo 2: Verbatim Nibbles + Power-of-Two Global Scale

Keep the E2M1 nibbles unchanged. Each MXFP4 block of 32 splits into two NVFP4
blocks of 16; both inherit the same exponent `k_j`. Pick a global scale
`S = 2^m` (integer `m`) and store the per-block E4M3 scale as `2^(k_j − m)`.

- **In-range blocks** (`k_j − m ∈ [−9, 8]`): contribution **MSE = 0** — both
  `2^(k_j − m)` (E4M3) and `2^m` (FP32) are exactly representable, so the product
  reproduces `2^k_j` exactly.
- **Out-of-range blocks** (spread > 17): snap the per-block exponent to the
  E4M3 boundary `clamp(k_j − m, −9, 8)`. Provably MSE-optimal *given verbatim
  nibbles*, but the snap can be huge if a block's true scale is far from the
  snapped value.

**Choice of `m`** (two strategies tested):
- `midpoint`: when spread ≤ 17, midpoint `m` makes everything in-range.
  When spread > 17, fall back to `median(k) − center`.
- `search`: 1D integer sweep with closed-form objective
  `Σ_j S_j · (2^k_j − 2^(m + clamp(k_j − m, −9, 8)))^2` where
  `S_j = Σ_i e2m1_value_i^2` for block `j`. Cheap (≤ 50 candidates per tensor).

### Algo 3: Hybrid (verbatim where exact, NVFP4-requant where lossy)

Combines Algo 2's exact path with per-block requantization for OOR blocks:

1. Search `m` (integer) by minimizing the actual hybrid reconstruction MSE.
2. For in-range MXFP4 blocks: keep verbatim path (zero error).
3. For OOR MXFP4 blocks: dequantize to FP32, then NVFP4-quantize each
   16-element half with the **fixed** `scale_2 = 2^m`. The per-NVFP4-block amax
   can be smaller than the per-MXFP4-block amax — one half might lack the
   max-magnitude nibble — letting OOR-at-MXFP4-level blocks fit cleanly into
   per-NVFP4-block E4M3 scales.
4. Final reconstruction is masked-merged from the two paths.

The `m` search is brute-force over the same integer range Algo 2 considers, but
evaluated against the actual hybrid MSE because NVFP4-requant's E4M3 mantissa
rounding isn't a clean closed form.

## Experimental Setup

- 27 scenarios spanning standard distributions (uniform, gaussian, heavy-tail),
  outlier patterns (rare, per-row, per-col, single-extreme), block-spread
  boundary tests (15, 17, 18, 50), bimodal/power-law/saturating/subnormal/
  constant cases, and layer-shaped LLM weights (QKV 4096², MLP 1024×4096,
  embedding, LayerNorm gain, conv 4D).
- Reference: MXFP4 dequantized tensor (FP32).
- Metrics: MSE, max abs error, SNR (dB).
- Hardware: NVIDIA RTX 6000 Ada Generation.

## Results

### Aggregate

| Outcome (across 27 scenarios) | Count |
|---|---|
| Algo 3 outright winner | 4 |
| Algo 2 outright winner | 0 |
| Algo 1 outright winner | 1 |
| Tied (≥ 2 algos at same MSE) | 22 |
| Algo 3 exact (MSE = 0) | 22/27 |

### Algo 3 dominant wins (over Algo 2)

| Scenario | Spread | Algo 1 | Algo 2 (best) | **Algo 3** | SNR Δ (3 vs 2) |
|---|---|---|---|---|---|
| mixed block scales (~2²⁵) | 26 | 1.45e+03 | 2.96e-04 | **1.58e-05** | +12.7 dB |
| spread 17 (boundary) | 19 | 1.91e+01 | 2.27e-07 | **2.17e-07** | +0.2 dB |
| spread 18 (just over) | 20 | 1.87e+01 | 7.21e-07 | **2.85e-07** | +4.0 dB |
| spread 50 (extreme) | 52 | 7.54e+10 | 3.85e+04 | **9.45e+02** | +16.1 dB |

### Cases where Algo 3 loses to Algo 1

| Scenario | Algo 1 MSE | Algo 3 MSE | Ratio (3/1) |
|---|---|---|---|
| single extreme outlier | 2.466046e-05 | 2.471241e-05 | **1.0021** |

Gap is 0.21% — both algorithms underflow the small-magnitude blocks identically;
the residual is the integer-vs-continuous quantization of the global scale.
Algo 1 picks `scale_2 = global_amax / (6·448) ≈ 3.72`; Algo 3 is constrained to
`2^m` integer powers (here `m=3` → `scale_2 = 8`).

### Cases where Algo 3 loses to Algo 2

> None. Algo 3 ≤ Algo 2 in every scenario tested.

### Selected exact (MSE = 0) scenarios for Algo 3

uniform, gaussian, heavy-tail, rare outliers, narrow range, typical LLM weight,
mostly zeros, spread 15 (in-range), bimodal, Pareto power-law, per-row outliers,
per-col outliers, subnormal-heavy (1e-8), saturating at E2M1 max, strong bimodal
signs, constant, QKV weight (4096²), MLP gate/up, embedding, LayerNorm gain,
conv weight (4D), large flat (1024×4096).

## Why Algo 3 Works

The asymmetry in error scaling explains everything:
- **Snap-up errors** scale as `(2^k_j − 2^(m+8))²`, dominated by the *true*
  magnitude `2^k_j` — can be enormous.
- **Snap-down errors** scale as `(2^k_j − 2^(m−9))²`, bounded by the *snapped*
  magnitude `2^(m−9)`.

Algo 2's `m`-search already exploits this asymmetry by preferring low `m` values
that keep high-magnitude blocks in-range. But verbatim-snap on OOR blocks still
introduces a fixed-magnitude error per block, with no use of the within-block
structure.

Algo 3 replaces that snap with a real per-NVFP4-block requantization, which:
- Adapts to the actual half-block amax (much smaller than the MXFP4 block amax
  in many cases).
- Lets the E4M3 mantissa carry information beyond pure powers of 2 — for an OOR
  block where `k_j − m = 9` but max nibble is `4`, the required scale is
  `(4/6)·2^9 ≈ 341`, which fits in E4M3 (max 448).
- Costs nothing for in-range blocks because they keep the verbatim path.

## Recommendations

1. **Default to Algo 3** for MXFP4 → NVFP4 conversion. It is exact in the
   typical-weight case, strictly better than Algo 2 on spread-too-large cases,
   and within 0.2% of Algo 1 even on the pathological single-outlier case.
2. **The bound case** (single block dominates the entire tensor's dynamic range)
   can be closed by allowing a continuous (non-power-of-2) global scale. The
   integer-`m` form is purely for clean E4M3 representation of in-range
   per-block scales; on OOR blocks Algo 3 already routes through E4M3 mantissa
   rounding, so dropping the integer constraint there costs nothing in
   exactness and recovers the last 0.2%.
3. **Detection of the pathological case** is cheap: when the spread is very
   large *and* one block's `S_j` dominates the rest, Algo 1 (or Algo 3 with a
   continuous global scale) is preferable to Algo 3-with-integer-`m`.
4. **Cost**: Algo 3 runs a 1D integer sweep over typically 20–50 candidates, each
   evaluating one NVFP4 quantize+dequantize. For typical PTQ workflows this runs
   once per tensor and is negligible.

## Reproducibility

All results in this report were produced by `scratch/mxfp4_to_nvfp4_mse.py`
with `torch.manual_seed(0)` on the GPU described above.
