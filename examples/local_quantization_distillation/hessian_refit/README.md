# Closed-form refit of native scale fields (experimental)

[Return to the API guide](../README.md).

Bypass repairs a PTQ student with optimizer updates on each unit's native scale fields
against the teacher's unit output. With the codes frozen the same objective can be solved
directly: the unit output is linear in the last linear's scale steps and smooth in the
earlier ones. This directory holds the solvers and drivers used to measure how far a
closed-form and second-order refit gets next to Bypass.

It is research code, not an API: one GPU per process, GGML IQ formats only, gated-MLP
units (`gate_proj`, `up_proj`, `down_proj`) and routed-expert MoE units with fused
`gate_up` and `down` tensors.

## Method

- **Last linear, exact.** For frozen integer levels, `down_proj`'s contribution to the unit
  error is quadratic in its per-group steps. A per-row least squares with the student Gram
  and the student/teacher cross Gram solves them, and an alternating exact block-scale
  solve and per-group enumeration projects onto the native hierarchy of an FP16 block scale
  times a 4-bit local multiplier (`block_ls.py`).
- **Earlier linears, Gauss-Newton.** For `gate_proj` and `up_proj` a per-row joint
  Gauss-Newton step in scale space is damped, line searched exactly under the linearized
  model (rows couple through the shared down projection, so the raw step overshoots) and
  projected in its own metric (`block_gn.py`). It alternates with the exact down solve.
  There is no learning rate, schedule or clipping.
- **Routed experts, offline.** Each expert is refit on the tokens routed to it, weighted by
  the squared routing weight, from stored unit inputs (`refit_moe.py`). Gate/up steps can
  move only the block scales (`block`), every step with a Kronecker-factored curvature
  (`kfac`) or every step with the exact per-row-pair curvature from a token subsample
  (`group`). Optional gain steps rescale block scales with codes frozen.

## Files

| File | Purpose |
|---|---|
| `block_ls.py` | Group quadratics from Grams, continuous solve, projection onto the native hierarchy |
| `block_gn.py` | Gauss-Newton statistics, exact line search and projection for gated-MLP units |
| `bypass_baseline.py` | GPTQ, held-out KLD, Bypass, held-out KLD for one model and format |
| `refit_dense.py` | GPTQ, KLD, closed-form refit of dense units, KLD, optional Bypass before or after |
| `refit_moe.py` | Offline per-expert refit from captured unit inputs and exported payloads |
| `summarize.py` | One table from the `metrics.json` files the dense drivers write |

## Usage

Run from the repository root with a corpus from `prepare_data`.

```bash
python -m examples.local_quantization_distillation.hessian_refit.bypass_baseline \
  --model /path/to/teacher --data /path/to/packed-corpus --format iq2_xxs \
  --out runs/bypass/iq2_xxs --updates 512

python -m examples.local_quantization_distillation.hessian_refit.refit_dense \
  --model /path/to/teacher --data /path/to/packed-corpus --format iq2_xxs \
  --out runs/refit/iq2_xxs --interior gn --gn-damp 0.1 --gn-iters 6 --gn-packs 16 \
  --gn-rotate 1 --gn-ls-packs 4 --gram-packs 32 --gn-down-packs 16 --gn-monitor 0 \
  --validation-packs 32 --updates 8

python -m examples.local_quantization_distillation.hessian_refit.summarize runs/bypass runs/refit
```

`--interior none` refits only `down_proj`; `--gn-granularity block` moves only block scales
of `gate_proj` and `up_proj`; `--updates N` runs Bypass from the refit state.

The expert solver works on files, so the 8-GPU model never has to be loaded with it:

```bash
python -m examples.local_quantization_distillation.hessian_refit.refit_moe \
  --capture /path/to/capture --payloads /path/to/payloads --source /path/to/teacher-checkpoint \
  --out runs/experts --layers 3,4,5 --format iq2_s --prox 0.1 \
  --interior block,kfac,kfac --iters 3 --fit-packs 384
```

| Input | Layout |
|---|---|
| `--payloads` | `layer-LL-rank-RR.pt`: `format`, `total`, `start`, `count`, and for `gate_up` and `down` a uint8 `[count, bytes]` payload plus its `[rows, cols]` shape |
| `--capture` | `{fit,validation}/layer-LL/pack-NNNNN.pt`: `x` `[tokens, hidden]` unit inputs of valid tokens, `idx` `[tokens, top_k]` routed experts, `w` `[tokens, top_k]` routing weights, optionally `y` teacher unit outputs for a self-check |
| `--source` | Teacher safetensors with `...layers.L.mlp.experts.E.{gate,up,down}_proj.weight` and FP8 `_scale_inv` companions |
| Output | `layer-LL.pt` with the refit `gate_up` and `down` payloads of every expert (codes unchanged) and `layer-LL.json` with per-expert and held-out unit reports |

The hooks that export payloads, capture unit inputs and load refit scale fields back into a
distributed student ran from a cluster-specific driver and are not part of this directory.

## Results

All numbers are held-out KL divergence of the teacher to the candidate over the full
vocabulary (`evaluate_pair`), as a change against the GPTQ student, on NVIDIA GB300 with
pinned Transformers 5.23. GPU-hours count the repair only.

### Qwen3.5-2B, 24 MLP units, 128 held-out packs

| Format | Bypass, 512 updates (1.18 GPU-h) | Refit, 6 iterations (0.5 GPU-h) | Refit, 3 iterations + 8 Bypass updates |
|---|---|---|---|
| IQ1_S | -48.0% | -47.7% | -47.5% |
| IQ2_XXS | -25.9% | -23.8%, -25.2% with 8 Bypass updates | -25.0% |
| IQ2_S | -23.7% | -18.5%, -20.7% with 8 Bypass updates | -21.2% |

Bypass alone on IQ2_S: 8 updates -7%, 32 -17%, 64 -21%, 128 -22%, 512 -24%.

### Qwen3.5-9B

| Format | Bypass, 512 updates (3.0 GPU-h) | Refit, 3 iterations | Refit + 8 Bypass updates |
|---|---|---|---|
| IQ1_S | -55.1% | -52.4% | |
| IQ2_XXS | -40.2% | -35.1% | -37.5% |
| IQ2_S | -37.1% | -31.7% | -33.7% |

The refit takes 2.9 to 4.2 GPU-hours here: the per-row statistics grow with the square of
the groups per row, so at this size it is no cheaper than Bypass.

### GLM-5.3-Flash, IQ2_S, 42 MoE units of 288 routed experts

In-model on 8 GPUs, 128 held-out packs, FP8 teacher (perplexity 4.652). Unit error is the
mean change in validation unit MSE. The refit's GPU-hours include capturing the inputs.

| State | GPU-h | 32K-token packs | Unit error | KLD | Perplexity |
|---|---|---|---|---|---|
| GPTQ | | | | 0.4263 | 4.681 |
| Bypass, 64 updates | 7.2 | 512 | -11.2% | 0.4106 (-3.7%) | 4.653 |
| Refit | 8.3 | 128 | -11.4% | 0.4092 (-4.0%) | 4.694 |
| Refit | 11.9 | 384 | -12.8% | 0.4068 (-4.6%) | 4.689 |
| Refit + 64 Bypass updates at `scale_lr=0.003` | 18.8 | 896 | -13.2% | 0.4044 (-5.1%) | 4.676 |
| Bypass, 512 updates | 43.3 | 4096 | -15.7% | 0.3961 (-7.1%) | 4.683 |

- At 7 to 8 GPU-hours the refit and Bypass repair the same units by the same amounts, and
  the refit needs a quarter of the data. Bypass pulls ahead as the budget grows.
- The exact down solve gives about three quarters of the refit's gain in typical units.
  Block-scale, Kronecker-factored and exact per-group gate/up steps end within a point of
  each other, unlike on Qwen, where the per-group step carries most of the gain.
- Lightly routed experts are data-starved: the fit set is 8 points better than held-out at
  128 packs, and three times the data adds 1 to 3 points.

### Starting from a gain-corrected PTQ

A least-squares encode shrinks each linear's output. GPTQ with a derived super-block scale
and a per-row gain step (the GPTQ follow-up described in PR #3351, run here as a prototype on
an earlier commit of this branch) lifts each expert's output gain from about 0.90 to 0.99.

| State (KLD) | Stock GPTQ start | Gain-corrected start |
|---|---|---|
| PTQ | 0.4263 | 0.4203 |
| Bypass, 64 updates | 0.4106 | 0.4089 |
| Refit, 384 packs | 0.4068 | 0.4083 |
| Refit + 64 Bypass updates | 0.4044 | 0.4076 |
| Bypass, 512 updates | 0.3961 | 0.3968 |

Both repairs refit scales against the teacher output, so they absorb the head start. In
the few units where the refit gains 25 to 37%, the stock PTQ is wrongly scaled, and the gain
correction recovers half to four fifths of that at PTQ time.

## What did not work

- The down solve alone stops near -9% KLD at 2 bits on Qwen (-26% at IQ1_S).
- A per-linear least squares on `gate_proj` and `up_proj` does nothing: their effect is
  only visible in the unit output.
- A Gauss-Newton step without the line search overshoots by 1.5 to 4 times.
- Bypass at the default `scale_lr` from a refit state first undoes it: the first Adagrad
  update moves every scale by about 1%. A third of the rate composes cleanly.
- Forcing unit gain 1 per expert after the refit raises KLD from 0.4068 to 0.4178; matching
  energy leaves it unchanged.

## Limits

- Stored unit inputs cost about 9 GB per 32K-token pack for the 42 GLM units. A streaming
  variant would pay the same teacher forward that is 55% of Bypass's time.
- Only KLD, perplexity and unit MSE were measured; no downstream benchmark.
- Perplexity and KLD disagree across gain states: perplexity falls as output gain rises,
  KLD does not.
