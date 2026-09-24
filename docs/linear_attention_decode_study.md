# Decode-aware QAT qualification study

## Fixed protocol

The study uses `arcee-ai/AFM-4.5B-Base-KDA-Only` at revision
`01ad2e06ee4f1214193c17b69e09105a9b257e80` and Wikitext-2 raw at revision
`b08601e04326c79dfdd32d625aee71d232d685c3`. The actual checkpoint has 36 KDA
layers, hidden size 2560, 20 heads, and key/value dimension 128. Parameters,
source, dataset files, and token blocks are bound by stored hashes.

`decode_study_plan.json` fixes selection before held-out test access. Every
block contains 129 input tokens: 64 explicit prefix tokens followed by a decode
suffix. The loss scores labels at positions 64 through 128 (65 predictions per
block); it does not average prefix loss into decode quality. Controls and
candidates use the exact prefill solve, FP32 working arithmetic, BF16 outer
autocast, and seed 2026.

Validation uses 32 blocks (2,080 scored tokens) without training. State FP8 and
replay must each pass an NLL upper-bound margin of 0.02 against exact. Decay
selection chooses the coarsest passing grid among 1/256, 1/1024, and 1/4096,
requiring the same margin against both state FP8 and exact. Eligible modes then
receive matched 32-step attention-only QAT pilots and 64 held-out test blocks.
These descriptive paired-block bootstrap bounds are not broad quality guarantees.

## Validation selection

| Policy | Suffix perplexity | Mean NLL increase vs exact | 95% interval |
| --- | ---: | ---: | --- |
| Exact | 9.019945 | 0 | — |
| Token state FP8 | 9.023610 | 0.000406 | [-0.002017, 0.002718] |
| State FP8 + log grid 1/256 | 9.029220 | 0.001028 | [-0.001118, 0.002986] |
| State FP8 + log grid 1/1024 | 9.035839 | 0.001761 | [-0.000115, 0.003723] |
| State FP8 + log grid 1/4096 | 9.024412 | 0.000495 | [-0.001268, 0.002299] |
| Replay FP8, window 8 | 9.019499 | -0.000049 | [-0.001867, 0.001630] |

All families passed the validation gate. The predeclared coarsest-grid rule
selected **1/256**; its paired comparison against state FP8 also passed. The
held-out set was frozen to exact, token state FP8, state FP8 with grid 1/256,
and replay FP8 with window 8, before accessing test data.

## Held-out training pilot

All four selected modes completed 32 attention-only updates on 32 training
blocks (2,080 scored training tokens) using FP32 AdamW, learning rate 1e-5,
zero weight decay, gradient clipping at 1.0, and non-reentrant activation
checkpointing. All 994,043,088 trainable attention parameters had finite gradients
on every update, and the query projection changed in each run. Evaluation used
64 held-out blocks (4,160 scored tokens), with identical token hashes and
training order across modes.

| Policy | PPL before | PPL after | Post-training NLL delta vs exact | 95% interval |
| --- | ---: | ---: | ---: | --- |
| Exact | 12.899382 | 12.641259 | 0 | — |
| Token state FP8 | 12.904970 | 12.656691 | 0.001220 | [-0.000290, 0.002703] |
| State FP8 + grid 1/256 | 12.915614 | 12.653026 | 0.000930 | [-0.000802, 0.002673] |
| Replay FP8, window 8 | 12.909424 | 12.646554 | 0.000419 | [-0.000566, 0.001410] |

All three approximations met the 0.02 held-out pilot margin after training; the
largest upper bound was 0.002703. Their intervals include zero, so this study does
not establish a quality advantage for any approximation. All modes, including
the exact control, improved after fine-tuning. These small short-context pilots
verify model-level training behavior and bounded measured differences; they do
not demonstrate recovery of a substantial accuracy gap or broad downstream quality.

## Complete training cost

The synthetic H100 workload is `[B,T,H,D] = [1,257,4,128]`, with 64 prefix tokens,
FP32 working inputs, TF32 disabled, three warmups, and 20 interleaved samples.
The table includes the complete prefix/suffix forward and backward, including
returned final-state gradients. Speedups compare each fused implementation with
its matching direct Torch numerical reference, with descriptive paired-median
bootstrap intervals. Memory is peak allocated memory above the live input baseline.

| Policy | Torch wall ms | Fused wall ms | Speedup [95% interval] | Torch / fused extra MiB |
| --- | ---: | ---: | --- | ---: |
| GDN exact | 188.830 | 4.758 | 39.64 [39.52, 40.19] | 103.75 / 14.68 |
| GDN token | 224.532 | 7.239 | 31.02 [30.73, 31.06] | 103.75 / 14.68 |
| GDN decay | 231.847 | 7.321 | 31.70 [31.57, 31.82] | 103.75 / 14.69 |
| GDN replay | 325.854 | 6.206 | 52.47 [51.12, 52.73] | 272.08 / 14.94 |
| KDA exact | 193.183 | 11.110 | 17.40 [17.33, 17.53] | 152.35 / 63.88 |
| KDA token | 228.732 | 11.265 | 20.32 [20.25, 20.43] | 152.35 / 63.88 |
| KDA decay | 235.972 | 11.310 | 20.87 [20.77, 20.98] | 152.35 / 64.26 |
| KDA replay | 327.361 | 11.569 | 28.34 [28.16, 28.39] | 321.54 / 64.51 |

The exact FLA reference, including conversion of Q/K/V to BF16, took 1.806 ms
for GDN and 2.347 ms for KDA. Fused numerical emulation remains slower than that
reference. These measurements establish a reduction in QAT emulation overhead,
not native low-precision execution, inference latency, or compressed-cache cost.
Every policy first passed checks of outputs, final state, and all input gradients
against the matching Torch reference; source hashes accompany the raw samples.

## Validation boundary

The H100 regression suite passed 148 cases, with nine expected skips for FP32
Hopper/TileLang modes that are rejected before launch. It includes GDN TP=1/2
sharded save/restore and optimizer updates, KDA layer training/restore, explicit
phase handling, and custom backward tests. All eight long-trajectory cases also
passed the tightened `1e-4` relative-norm gate on H100 and SM86.

Local checks passed 139 unit cases plus two re-encoding continuation cases,
44 kernel/layer cases, and four additional full-width/value-tail cases covering
32- and 128-column codec blocks (also passed on H100), plus two composed-policy
cases (also passed on H100). The combined H100 total is **154 passed, nine
expected skips**; the local GPU total is 50 passed. The full Sphinx build passed with FLA imports
disabled. Measurements pin retained source snapshots. Final cleanup moved import
boundaries and added explanatory comments; non-import AST equality was verified
and focused regression/import checks were rerun. The retained benchmark harness
differs only in the order of two imported typing names, with AST equality
verified after normalizing that order.

Model results apply to this checkpoint, short context, small corpus slice, and
seed. Serving cache integration, real compression, long-context quality, and
multi-model recovery remain unqualified.
