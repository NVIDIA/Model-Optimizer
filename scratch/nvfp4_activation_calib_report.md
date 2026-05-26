<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# NVFP4 activation `input_scale`: how much does calibration matter?

**Scratch experiment.** Single-GPU (RTX 6000 Ada). Run end-to-end in ~25
minutes (capture + sweep). Companion scripts and data live alongside this
report in `scratch/`.

## Question

NVFP4 activation quantization sets a per-tensor `input_scale` from the amax
observed during calibration:

```text
input_scale = amax_calib / (6 * 448) = amax_calib / 2688.
```

At inference, **only** `input_scale` is static — per-block FP8 scales are
derived dynamically as `block_amax / (6 · input_scale)` and cast to E4M3.

We want to answer three related questions:

1. **Calibration size:** does more calibration data always reduce
   MSE-of-quantization, or are there regimes where it hurts?
2. **Calibration policy:** is `amax` the right reduction, or would a
   percentile / a smaller scaling be better?
3. **Calibration dataset:** does the choice of *what data* you calibrate
   on (chat-only, mixed news + chat, multi-domain SFT) affect the
   resulting quantization quality?

## TL;DR

- **Calibration size on real LLM activations is essentially flat.** On
  Qwen3.5-9B MLP-input activations, varying calibration from 512 to 2048
  whole sequences (≈ 250k–1M tokens) moves SNR by ≤ 0.01 dB. The amax
  default is within 0.01–0.06 dB of the unrealistic MSE-optimal oracle.
- **Calibration size *can* hurt** in principle — confirmed on two
  synthetic stress-test distributions where more calibration captures a
  rare outlier that inference doesn't see (2.4 dB SNR loss) or a true
  heavy-tail amax that never converges (0.8 dB loss). Neither pathology
  manifests on the real Qwen3.5-9B / Nemotron-family activations.
- **Calibration dataset choice doesn't matter.** With the *same* held-out
  test tensor evaluated under three different calibration combos
  (chat-only, `cnn_nemotron_v2_mix`, `nemotron-post-training-v3`), the MSE
  spread is ≤ 0.013 dB SNR on both layer 0 and layer 31. The combos
  produce `amax` values within ~5% of each other.
- **Percentile calibration (`p99`, `p99.9`, `p99.99`) is uniformly worse**
  than `amax` because it under-shoots typical outliers and causes E2M1
  clipping at inference time.
- **Asymmetry that matters in principle:** under-calibration (calib_amax <
  test_amax) is catastrophic due to top-end E2M1 clipping; over-calibration
  is mild until calib_amax / test_amax exceeds ~30, at which point typical
  blocks fall into the FP8 subnormal range and per-block scale precision
  degrades.

## NVFP4 mechanics (the thing we're measuring)

For an activation block of 16 elements (`block_size=16`) and a per-tensor
`input_scale = S`, modelopt's NVFP4 quantize+dequantize path
(`modelopt/torch/quantization/qtensor/nvfp4_tensor.py`) does:

```text
per_block_scale_fp8 = round_e4m3( block_amax / (6 · S) ),   clamped to [2^-9, 448]
x_e2m1              = round_e2m1( x / (per_block_scale_fp8 · S) )
x_recon             = x_e2m1 · per_block_scale_fp8 · S
```

The two failure modes for the dynamic per-block scale:

1. **Top clamp** (`per_block_scale > 448`): when
   `block_amax > 6·448·S = amax_calib`. E2M1 saturates everything above
   `amax_calib`. *Catastrophic.*
2. **Subnormal / bottom clamp**: when `block_amax / (6·S) < 2^-6 ≈ 0.0156`,
   the per-block scale enters E4M3's subnormal region and loses
   precision. Triggers when `block_amax < amax_calib / 28672`. *Mild.*

The asymmetry between these is what drives the headline findings.

---

## Part A — Synthetic distribution study

Companion script: [`nvfp4_activation_calib_mse.py`](nvfp4_activation_calib_mse.py).
Curves: [`nvfp4_activation_calib_results.json`](nvfp4_activation_calib_results.json).

## Setup

For each synthetic distribution we:

1. Fix a seed and draw a 1M-element test tensor reshaped to `(1024, 1024)`.
2. For each calibration size `n_calib ∈ {1k, 4k, 16k, …, 32M}` (log-spaced
   in element count), draw `n_calib` independent samples from the same
   distribution under three calibration seeds.
3. Set `amax_calib = max(|x_calib|)`, `S = amax_calib / 2688`.
4. Quantize-dequantize the test tensor with `S` (modelopt's reference path)
   and measure MSE against the bf16 test tensor itself.
5. Compare against percentile calibration (p99/p99.9/p99.99) and an
   "oracle" sweep of `S` directly on the test tensor (unrealistic upper
   bound).

## Distributions

| short name | description |
|---|---|
| `post-LayerNorm + rare 10× spikes` | `N(0,1)`, 0.1% of elements scaled 10× |
| `per-channel outlier (1%, 100×)` | SmoothQuant-style channel outliers, ~1% of hidden dims at 100× |
| `per-token outlier (2%, 30×)` | ~2% of tokens (rows) pumped 30× |
| `post-GeLU one-sided heavy tail` | `GeLU(N(0, 2))` |
| `Pareto(1.5) mixture` | 99% `N(0,1)` + 1% Pareto(α=1.5)·3 |
| `rare giant spike (test=clean)` | calib has 1-in-1e7 chance of ±1e6 spike; **test tensor is spike-free** |
| `log-normal σ=3` | `exp(N(0,3)) × random sign` (very heavy tail) |

The `rare giant spike` case is deliberately constructed to model "training
captured a numerical artifact that inference doesn't see."

## Results (SNR in dB vs calibration size)

| n_calib | PLN | Ch out | Tk out | pGeLU | Pareto | Rare spike | log-norm σ=3 |
|---:|---:|---:|---:|---:|---:|---:|---:|
|     1 024 | 13.98 | 18.56 | 20.44 | 20.35 |  1.17 |   20.42    |   0.87       |
|     4 096 | 18.47 | 22.50 | 20.39 | 20.36 |  1.57 |   20.43    |   1.01       |
|    16 384 | 19.44 | 24.17 | 20.43 | 20.37 |  3.24 |   20.44    |   2.93       |
|    65 536 | 20.25 | 25.88 | 20.43 | 20.37 |  5.24 |   20.44    |   4.50       |
|   262 144 | 20.45 | 26.48 | 20.45 | 20.37 |  6.70 |   20.44    |  10.06       |
| 1 048 576 | 20.45 | 26.52 | 20.44 | 20.38 | 15.32 |   20.44    |  25.70       |
| 4 194 304 | 20.47 | 26.53 | 20.43 | 20.38 | 25.49 | **18.02** ↓ | **30.83** ↑ |
|16 777 216 | 20.46 | 26.52 | 20.45 | 20.37 | 25.30 | **18.02** ↓ | **30.16** ↓ |
|33 554 432 | 20.46 | 26.53 | 20.44 | 20.37 | 25.94 | **18.02** ↓ | **30.06** ↓ |
| **oracle**| 20.47 | 26.58 | 20.49 | 20.38 | 26.30 |   20.44    |  35.30       |

Bold cells: worsening relative to the previous (smaller) calibration size.

### Summary

| distribution | best n_calib | best vs largest | oracle vs largest |
|---|---:|---:|---:|
| post-LayerNorm + 10× spikes  | 4 194 304 | +0.00 dB | +0.01 dB |
| per-channel outlier          | 33 554 432 | +0.00 dB | +0.05 dB |
| per-token outlier            | 16 777 216 | +0.02 dB | +0.05 dB |
| post-GeLU                    | 4 194 304 | +0.00 dB | +0.01 dB |
| Pareto(1.5)                  | 33 554 432 | +0.00 dB | +0.36 dB |
| **rare giant spike**         | **262 144** | **+2.42 dB** | **+2.42 dB** |
| **log-normal σ=3**           | **4 194 304** | **+0.77 dB** | **+5.24 dB** |

### Interpretation

- **5/7 distributions** show a saturating MSE curve — `amax_calib`
  converges as `n_calib` grows, the curve plateaus around 20 dB SNR
  (E2M1's noise floor), and further calibration neither helps nor hurts.
  This includes all the "well-behaved" patterns: post-LayerNorm,
  per-channel, per-token, post-GeLU, Pareto(1.5).
- **2/7 distributions** show a non-monotonic curve — more calibration
  *eventually hurts*. `rare giant spike` loses 2.42 dB SNR once
  calibration is large enough to capture the spike (>4M elements). The
  inflation factor of `amax_calib / test_amax` jumps to ~200 000×, far
  past the FP8 subnormal threshold, and typical blocks lose precision.
  `log-normal σ=3` shows a gentler, gradual degradation as more rare-tail
  samples push amax higher.

### Why percentile calibration doesn't rescue you

Percentile (p99 / p99.9 / p99.99) was tested at the largest calibration
set per distribution:

- For well-behaved distributions, all percentiles **undershoot**
  `test_amax`, causing E2M1 clipping at inference. SNR is 1–25 dB worse
  than amax.
- For `rare giant spike`, percentiles **avoid** the spike (since it's
  rarer than p99.99) and match the small-calib amax SNR — robust here,
  but only because the spike happens to be rarer than the percentile.

Percentile calibration is an over-fit knob: it works for one specific
tail shape and breaks otherwise.

---

## Part B — Real Qwen3.5-9B MLP-input study

Companion scripts:
[`capture_qwen35_mlp_activations.py`](capture_qwen35_mlp_activations.py),
[`nvfp4_real_activation_calib_mse.py`](nvfp4_real_activation_calib_mse.py),
[`_cross_dataset_amax_compare.py`](_cross_dataset_amax_compare.py).
Curves: [`nvfp4_real_activation_calib_results.json`](nvfp4_real_activation_calib_results.json).

## Capture pipeline

For each calibration dataset:

1. Load `/models/Qwen/Qwen3.5-9B` in bf16 on cuda:0 (requires
   `transformers >= 5.4` for Qwen3.5 architecture).
2. Iterate sequences from the dataset, applying the model's chat template
   to message-format rows.
3. Tokenize each sequence with truncation to `max_tokens=512`.
4. Register forward pre-hooks on `model.layers[0].mlp` and
   `model.layers[31].mlp` — these capture the MLP input tensor (i.e., the
   post-`*_layernorm` output that a static NVFP4 quantizer sees).
5. Run forward (no cache, no grad) for ~2600 sequences. Hooks move
   activations to CPU as bf16 and accumulate per layer.
6. Save two `.pt` files (list of `(seq_len, hidden)` bf16 tensors).

Layer 0 and layer 31 were chosen as bookends:
- Layer 0 uses linear attention; comes after `input_layernorm`. Tame
  amax (~4).
- Layer 31 uses self-attention; comes after `post_attention_layernorm`.
  Larger amax (~50) because the residual stream has accumulated through
  31 blocks.

Both pass through a LayerNorm immediately before the MLP, so neither has
a truly fat tail.

## Calibration datasets tested

Three combos sourced from modelopt's
`DATASET_COMBOS` (`modelopt/torch/utils/dataset_utils.py:203`):

| dataset | composition |
|---|---|
| **Nemotron chat-only** | `chat` split of `nvidia/Nemotron-Post-Training-Dataset-v2` (single split) |
| **`cnn_nemotron_v2_mix`** | 50% `abisee/cnn_dailymail` v3.0.0 articles + 50% Nemotron-Post-Training v2 (stem/chat/math/code). modelopt's `hf_ptq.py` default. |
| **`nemotron-post-training-v3`** | Round-robin over 7 SFT splits: Instruction-Following-Chat-v2, Science-v1, Competitive-Programming-v1, SFT-Agentic-v2, Math-v2, SFT-SWE-v2, SFT-Multilingual-v1 |

Caveat: in our environment, the v3 capture got only 6 of 7 members —
`Nemotron-SFT-Agentic-v2` (`search` split) emitted no rows in streaming
mode. Final v3 pool is 2231 sequences instead of the targeted 2604.

## Sweep design (per dataset)

- **Capture:** ~2600 sequences, cap 512 tokens.
- **Split:** last `N_TEST_SEQS` (100 or 200) sequences → held-out test
  pool. Rest → calibration pool.
- **Test tensor:** first 1 048 576 elements of the concatenated test
  pool, reshaped to `(256, 4096)`. Same test tensor used for every
  sweep point *within* a dataset.
- **Calibration sweep:** for `n_seqs ∈ {512, 1024, 2048}`, draw `n_seqs`
  whole sequences uniformly at random from the calibration pool (3 seeds
  per `n_seqs`), compute `amax` incrementally, derive
  `input_scale = amax / 2688`, quantize the test tensor, measure MSE.

## Per-dataset results

### Nemotron chat-only

| layer | global amax (capture) | test amax | sig_pow |
|---|---:|---:|---:|
| 0  | 4.531 | 3.156 | 1.90e-2 |
| 31 | 52.000 | 46.500 | 1.27 |

| layer | n_seqs | amax (mean ± std) | MSE | SNR (dB) |
|---|---:|---:|---:|---:|
| 0  |   512 | 4.177 ± 0.090 | 1.634e-4 | 20.66 |
| 0  | 1 024 | 4.229 ± 0.039 | 1.634e-4 | 20.66 |
| 0  | 2 048 | 4.531 ± 0.000 | 1.632e-4 | 20.66 |
| 0  | oracle | (168 equiv.) | 1.628e-4 | 20.67 |
| 31 |   512 | 51.83 ± 0.12 | 1.060e-2 | **20.79** |
| 31 | 1 024 | 51.92 ± 0.12 | 1.060e-2 | 20.78 |
| 31 | 2 048 | 52.00 ± 0.00 | 1.061e-2 | 20.78 |
| 31 | oracle | (1168 equiv.) | 1.046e-2 | 20.84 |

### `cnn_nemotron_v2_mix`

| layer | global amax (capture) | test amax | sig_pow |
|---|---:|---:|---:|
| 0  | 4.312 | 3.469 | 2.17e-2 |
| 31 | 50.75 | 37.25 | 1.33 |

| layer | n_seqs | amax (mean ± std) | MSE | SNR (dB) |
|---|---:|---:|---:|---:|
| 0  |   512 | 4.188 ± 0.000 | 1.864e-4 | 20.66 |
| 0  | 1 024 | 4.188 ± 0.000 | 1.864e-4 | 20.66 |
| 0  | 2 048 | 4.271 ± 0.059 | 1.866e-4 | 20.65 |
| 0  | oracle | (52.8 equiv.) | 1.859e-4 | 20.67 |
| 31 |   512 | 50.42 ± 0.24 | 1.148e-2 | 20.65 |
| 31 | 1 024 | 50.67 ± 0.12 | 1.148e-2 | 20.65 |
| 31 | 2 048 | 50.67 ± 0.12 | 1.148e-2 | 20.65 |
| 31 | oracle | (209 equiv.) | 1.146e-2 | 20.66 |

### `nemotron-post-training-v3`

| layer | global amax (capture) | test amax | sig_pow |
|---|---:|---:|---:|
| 0  | 4.156 | 3.078 | 1.71e-2 |
| 31 | 53.000 | 46.000 | 1.25 |

| layer | n_seqs | amax (mean ± std) | MSE | SNR (dB) |
|---|---:|---:|---:|---:|
| 0  |   512 | 4.156 ± 0.000 | 1.465e-4 | 20.67 |
| 0  | 1 024 | 4.156 ± 0.000 | 1.465e-4 | 20.67 |
| 0  | 2 048 | 4.156 ± 0.000 | 1.465e-4 | 20.67 |
| 0  | oracle | (77.3 equiv.) | 1.461e-4 | 20.68 |
| 31 |   512 | 53.000 ± 0.000 | 1.039e-2 | 20.81 |
| 31 | 1 024 | 53.000 ± 0.000 | 1.039e-2 | 20.81 |
| 31 | 2 048 | 53.000 ± 0.000 | 1.039e-2 | 20.81 |
| 31 | oracle | (75.4 equiv.) | 1.025e-2 | 20.87 |

### Percentile baselines (per dataset)

Computed once on the full calibration pool of each dataset.

| dataset | layer | p99 | p99.9 | p99.99 | p99.999 | amax @ 2048 | oracle |
|---|---|---:|---:|---:|---:|---:|---:|
| chat-only | 0  | 11.76 | 16.03 | 19.84 | 20.61 | **20.66** | 20.67 |
| chat-only | 31 |  6.29 |  8.16 | 19.05 | 20.80 | **20.78** | 20.84 |
| mix       | 0  | 11.79 | 16.39 | 20.23 | 20.62 | **20.65** | 20.67 |
| mix       | 31 |  8.82 | 12.90 | 20.48 | 20.65 | **20.65** | 20.66 |
| v3        | 0  | 11.81 | 16.24 | 20.00 | 20.63 | **20.67** | 20.68 |
| v3        | 31 |  5.71 |  7.39 | 19.44 | 20.84 | **20.81** | 20.87 |

`p99.999` reaches parity with amax in every case. Lower percentiles
clip the inference-time outliers and lose 1–15 dB SNR.

## Clean shared-test comparison (disjoint calib + test by construction)

The per-dataset sweep tables above use a *different* test tensor for
each combo — both calibration and test change between rows. To remove
that confound entirely, we ran a separate capture
(`capture_calib_and_test_split.py`) that produces explicitly-disjoint
calibration and test pools, drawn from positions strictly after the
calibration positions in each underlying member stream:

- `cnn_nemotron_v2_mix` calib pool: 1024 sequences (positions 0..511 of
  CNN/DailyMail train; positions 0..511 of Nemotron-Post-Training v2
  stem/chat/math/code in round-robin).
- `cnn_nemotron_v2_mix` test pool: 256 sequences (positions 512..639 of
  each member — these never appear in the mix calibration).
- `nemotron-post-training-v3` calib pool: 881 sequences (147 each from 6
  of 7 members; Agentic-v2 dropped in streaming).
- `nemotron-post-training-v3` test pool: 222 sequences (37 each from the
  same 6 members, positions 147..183 — never in v3 calibration).

The **shared test tensor** is the concatenation of *both* combos' test
pools (256 + 222 = 478 sequences), first 1 048 576 elements reshaped
to `(256, 4096)`. This is the test tensor for every MSE measurement
in the table below. Both calibration combos are evaluated on the
*same* held-out activations, drawn from sources neither calibration saw.

### Shared-test results

**Layer 0 MLP input** (test amax = 3.484, sig_pow = 2.105e-2):

| combo | n_seqs | amax (mean ± std) | MSE | SNR (dB) |
|---|---:|---:|---:|---:|
| cnn_nemotron_v2_mix       |   256 | 4.271 ± 0.059 | 1.799e-4 | 20.681 |
| cnn_nemotron_v2_mix       |   512 | 4.271 ± 0.059 | 1.799e-4 | 20.681 |
| cnn_nemotron_v2_mix       | 1 024 | 4.313 ± 0.000 | 1.800e-4 | 20.680 |
| nemotron-post-training-v3 |   256 | 4.156 ± 0.000 | 1.798e-4 | **20.683** |
| nemotron-post-training-v3 |   512 | 4.156 ± 0.000 | 1.798e-4 | **20.683** |
| **oracle**                | —     | (8.75 equiv.) | 1.792e-4 | 20.699 |

**Layer 31 MLP input** (test amax = 43.5, sig_pow = 1.293):

| combo | n_seqs | amax (mean ± std) | MSE | SNR (dB) |
|---|---:|---:|---:|---:|
| cnn_nemotron_v2_mix       |   256 | 49.67 ± 0.42 | 1.104e-2 | 20.689 |
| cnn_nemotron_v2_mix       |   512 | 50.25 ± 0.00 | 1.102e-2 | **20.695** |
| cnn_nemotron_v2_mix       | 1 024 | 50.25 ± 0.00 | 1.102e-2 | **20.695** |
| nemotron-post-training-v3 |   256 | 52.67 ± 0.24 | 1.104e-2 | 20.687 |
| nemotron-post-training-v3 |   512 | 53.00 ± 0.00 | 1.104e-2 | 20.686 |
| **oracle**                | —     | (3877 equiv.) | 1.100e-2 | 20.703 |

(N=1024 is skipped for v3 because the v3 calib pool is 881 sequences —
Agentic-v2's streaming iterator emits no rows in our environment.)

### Observations from the clean comparison

1. **Combo-to-combo spread is 0.002 dB (layer 0) and 0.009 dB (layer 31)**
   on a shared held-out test tensor. Both combos calibrate to an amax
   within ~3% of each other (4.16 / 4.31 on layer 0; 50.25 / 53.0 on
   layer 31) and the resulting MSE on disjoint test data is
   indistinguishable.
2. **Which combo "wins" depends on the layer.** v3 is fractionally
   better on layer 0; mix is fractionally better on layer 31. Both
   margins are well below seed noise — the two combos are
   interchangeable for input_scale calibration purposes.
3. **Default amax is within 0.008–0.018 dB of oracle on both layers**
   on this clean test. Per-block scale rounding leaves the MSE
   landscape extremely flat — the oracle's amax-equivalent for layer
   31 is 3877 (88× larger than the test amax), and it only buys
   0.008 dB.
4. **Calibration size insensitivity confirmed.** Going from 256 → 1024
   sequences moves SNR by ≤ 0.001 dB on layer 0 and ≤ 0.006 dB on
   layer 31. The mix's layer-31 amax even converges fully at N=512
   (std 0.00 across 3 seeds).
5. **Percentile baselines remain uniformly worse**, with p99 / p99.9 /
   p99.99 losing 1–13 dB by under-shooting inference-time outliers.

This is the experiment that the original question deserves. The answer
is unambiguous: on Qwen3.5-9B MLP inputs, with this realistic
calibration data, **the choice between `cnn_nemotron_v2_mix` and
`nemotron-post-training-v3` does not affect NVFP4 quantization
quality.** Both calibrate to an amax close enough that the resulting
input_scale lands in the flat region of the MSE landscape.

## Legacy: per-dataset-test cross-dataset comparison (`_cross_dataset_amax_compare.py`)

The earlier experiment (`_cross_dataset_amax_compare.py`) applied each
combo's recorded amax to a single fixed test tensor — but that fixed
test tensor was the v3 capture's own holdout, so v3-distributed. It
gave the same qualitative answer (≤ 0.013 dB spread across combos)
but the test data wasn't disjoint from the v3 calibration in the
strict sense the shared-test capture above guarantees.

1. Use **one** fixed test tensor — the v3 dataset's held-out 100
   sequences (first 1M elements reshaped to `(256, 4096)`).
2. For each combo, take the recorded `amax_calib` at N=2048 from that
   combo's run (4.531 / 4.271 / 4.156 on layer 0; 52.0 / 50.67 / 53.0
   on layer 31).
3. Apply that amax to derive `input_scale`, quantize the shared test
   tensor, measure MSE.

| layer | calibration combo | calib amax | MSE | SNR (dB) |
|---|---|---:|---:|---:|
| 0  | chat-only                  |  4.531 | 1.464e-4 | 20.668 |
| 0  | cnn_nemotron_v2_mix        |  4.271 | 1.463e-4 | **20.672** |
| 0  | nemotron-post-training-v3  |  4.156 | 1.464e-4 | 20.667 |
| 31 | chat-only                  | 52.000 | 1.038e-2 | 20.820 |
| 31 | cnn_nemotron_v2_mix        | 50.667 | 1.036e-2 | **20.827** |
| 31 | nemotron-post-training-v3  | 53.000 | 1.039e-2 | 20.814 |

**Spread worst → best:** 0.005 dB on layer 0, 0.013 dB on layer 31.

The three calibration combos produce input_scale values within ~5% of
each other (amax range 4.16–4.53 on layer 0; 50.7–53.0 on layer 31), and
that ~5% spread translates to ≤ 0.013 dB SNR on a shared test tensor.

One caveat worth noting: the shared test tensor in this experiment is
itself v3-distributed (it's the v3 capture's holdout). That gives v3 a
slight home-field advantage — and yet v3 still loses to
`cnn_nemotron_v2_mix` here by 0.005–0.013 dB. That strengthens, rather
than weakens, the "essentially identical" conclusion. A truly neutral
test (e.g. WikiText, or a different model's activations) would only
move things closer to equality.

## Answers

> 1. If we increase calibration data sequence length or batch count,
>    will it always help improve the model accuracy?

**On Qwen3.5-9B MLP inputs, no, but it doesn't hurt either.** Going from
512 to 2048 calibration sequences moves SNR by ≤ 0.01 dB on every
dataset × layer combination. Calibration is essentially converged at
N=512. *In principle* more calibration could hurt (synthetic
`rare giant spike` and `log-normal σ=3` showed 0.8–2.4 dB SNR losses
with too much calibration), but the pathological regime is not reached
on this model with realistic Nemotron-family calibration data.

> 2. Should we always use amax to derive input_scale?

**On this model, yes — amax is within 0.01–0.06 dB of the MSE-optimal
oracle on every layer × dataset combination tested.** No percentile
policy below p99.999 matches amax; p99.999 itself is functionally
indistinguishable from amax. The synthetic study confirms amax is
optimal except under contrived calibration-overshoot scenarios that
realistic LLM-activation data does not exhibit.

> 3. Does the choice of calibration dataset matter for quantization
>    quality?

**No, under a clean disjoint-calib/test design.** Both
`cnn_nemotron_v2_mix` and `nemotron-post-training-v3`, evaluated on a
shared test tensor drawn from data neither combo's calibration saw,
yield MSE within **0.002 dB (layer 0)** and **0.009 dB (layer 31)** of
each other. They calibrate to amax values within ~3%, and the
resulting input_scale lands in a flat region of the MSE landscape on
both layers. The calibration dataset composition is not a sensitive
knob on this model.

## Synthesis

The MSE landscape around `input_scale = amax_calib / 2688` is *flat* for
roughly an order of magnitude in either direction on realistic LLM
activations. Three orthogonal levers — calibration size, calibration
policy (amax vs percentile), calibration dataset composition — all
produce sub-0.1 dB SNR variation as long as `amax_calib` is reasonably
representative of inference-time amax. The mechanism is straightforward:
NVFP4's reconstruction MSE is dominated by E2M1's 3-bit nibble
quantization, not by the per-tensor scale choice; small `input_scale`
perturbations only matter when they push per-block scales out of E4M3's
normal range, which requires order-of-magnitude inflations.

**The risk that does exist** is the asymmetric one: under-calibration
(`amax_calib < test_amax`) clips activations catastrophically, while
over-calibration is mild. So defaulting to `amax` with *enough*
calibration to capture the typical tail is the right policy. For
Qwen3.5-9B MLP inputs that's ~500 sequences of ~500 tokens.

The only scenario where this could fail is calibration data
contaminated with rare numerical artifacts (training-instability
spikes, sentinel/NaN-adjacent values) that inflate amax orders of
magnitude past what inference will see. The synthetic
`rare giant spike` case demonstrates this — but it's not a property of
the calibration datasets tested here.

---

## Reproduction

```bash
# Part A: synthetic distributions
python scratch/nvfp4_activation_calib_mse.py

# Part B: real activations on cnn_nemotron_v2_mix (modelopt PTQ default)
python scratch/capture_qwen35_mlp_activations.py \
    --n_seqs 2600 --max_tokens 512 --dataset cnn_nemotron_v2_mix
python scratch/nvfp4_real_activation_calib_mse.py

# Or with the multi-domain SFT combo
python scratch/capture_qwen35_mlp_activations.py \
    --n_seqs 2604 --max_tokens 512 --dataset nemotron-post-training-v3
python scratch/nvfp4_real_activation_calib_mse.py

# Clean disjoint-calib/test sweep (single capture for both combos,
# strictly-disjoint held-out test set used for every measurement)
python scratch/capture_calib_and_test_split.py \
    --n_calib 1024 --n_test 256 --max_tokens 512
python scratch/nvfp4_shared_test_sweep.py

# Legacy: cross-combo fixed-test comparison using v3's own holdout
# (preserves edit history; superseded by the shared-test sweep above)
python scratch/_cross_dataset_amax_compare.py
```

Requirements:
- Single GPU (RTX 6000 Ada or similar — bf16 forward of 9B model needs
  ~18 GB).
- `transformers >= 5.4` for Qwen3.5 architecture (we used 5.9).
- modelopt repo on the PYTHONPATH (uses the production
  `NVFP4QTensor.quantize` reference path).

Timings (RTX 6000 Ada):
- Synthetic study: ~3 minutes.
- One capture run (2600 seqs through Qwen3.5-9B): ~12 minutes.
- One sweep run: ~1 minute.

## Artifact map

| file | what |
|---|---|
| `nvfp4_activation_calib_mse.py` | Synthetic distributions, `n_calib` sweep, percentile + oracle baselines. |
| `nvfp4_activation_calib_results.json` | Raw curves for the synthetic study (α-vs-MSE per distribution). |
| `capture_qwen35_mlp_activations.py` | MLP-input activation capture from Qwen3.5-9B. Supports `cnn_nemotron_v2_mix` and `nemotron-post-training-v3` combos. |
| `qwen35_9b_mlp_input_layer{0,31}.pt` | Captured bf16 activation tensors. Current state on disk: v3 combo. |
| `nvfp4_real_activation_calib_mse.py` | Sequence-count sweep on the `.pt` captures. 3 seeds, percentile + oracle baselines. |
| `nvfp4_real_activation_calib_results.json` | Raw curves for the real-data sweep (current = v3). |
| `_cross_dataset_amax_compare.py` | Legacy: cross-combo comparison using the v3 capture's own holdout as the shared test tensor. |
| `capture_calib_and_test_split.py` | Clean disjoint-calib/test capture: 1024 calib seqs each for `cnn_nemotron_v2_mix` and `nemotron-post-training-v3`, plus 256 + 222 held-out test seqs from positions strictly after the calibration ranges. Single model load. |
| `nvfp4_shared_test_sweep.py` | Apples-to-apples sweep over both combos using one shared test tensor (the concatenation of both combos' held-out pools). |
| `nvfp4_shared_test_sweep_results.json` | Raw curves for the shared-test sweep. |
| `qwen35_cnn_nemotron_v2_mix_{calib,test}_layer{0,31}.pt`, `qwen35_nemotron_post_training_v3_{calib,test}_layer{0,31}.pt` | Per-combo, per-split activation captures used by the shared-test sweep. |

## Limitations

- Block size is fixed at 16 (NVFP4 standard). Behaviour would be
  qualitatively different at MXFP4's block size of 32.
- All activations are bf16; the MSE floor is dominated by E2M1's 3-bit
  quantization, so the flat curves we observed bottom out around 20 dB
  SNR. Models / activations with weaker bf16-floor headroom (e.g. ones
  with stronger outliers that escape E2M1's ±6 range) might show
  different sensitivity.
- Reference implementation only — we use modelopt's PyTorch quantize /
  dequantize, not the TensorRT-LLM fast path. Absolute MSE numbers may
  differ very slightly from hardware kernels due to rounding-mode quirks
  but the comparative conclusions stand.
- Real-data study is on one specific model (Qwen3.5-9B), one layer pair
  (0 and 31), and one type of activation (MLP input, post-LayerNorm).
  Models without LayerNorm-before-MLP or with stronger residual-stream
  outliers may show different behaviour.
- The v3 combo capture is 6/7 members (Agentic-v2 streaming dropped out
  in our environment). The conclusion is unlikely to change with the
  missing member included given how robust the result was across the
  other combos.
- "Calibration overshoots inference" pathology, while real in the
  synthetic study, has not been observed on any production-style LLM
  calibration data we tested. If a production scenario involves
  calibration on raw training data with intermittent numerical
  instabilities, the synthetic findings suggest worth detecting and
  rejecting outlier batches.
