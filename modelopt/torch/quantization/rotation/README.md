# Rotation Folding + Learning (SpinQuant/QuaRot R1 + R2) — Design

Status: offline R1/R2 only (fold + Cayley-SGD learner); online R3/R4 are out of scope.
Module: `modelopt.torch.quantization.rotation` (`fold.py` + `learn.py` + `sgdg.py`).

## Goal

Apply SpinQuant/QuaRot-style rotations as a **pre-quantization checkpoint transform**: a
global orthogonal rotation R1 of the residual stream plus a per-layer head-space rotation R2
on the v_proj → o_proj path are folded into the HF model weights in place. The rotated model
is functionally identical to the original (up to one float64 → original-dtype round-trip per
weight) but its activation/weight distributions are flatter, i.e. easier to quantize. The
transform is **orthogonal to qformat by construction**: it edits the checkpoint before
`mtq.quantize` runs, so every existing quant config, calibrator, exporter, and runtime works
unchanged on the rotated model.

Mechanics are ported from a validated clean-room reference implementation
(fp32-equivalence validated on Qwen3-0.6B/1.7B — WikiText-2 PPL delta −0.0022%
(17.8337 → 17.8333), GSM8K checked; clean-room w.r.t. Meta's SpinQuant code).

## Non-goals (explicit)

- **No online transforms** — R3 (post-RoPE QK rotation) and R4 (down_proj activation
  Hadamard) need runtime kernels; folding only their weight halves destroys the model.
- **No QuantAlgo / mode-registry / config-class integration** — one plain function.
- **No exporter or runtime changes** — the output is still a vanilla HF checkpoint.
- **No CLI** — callers script it.

## API

```python
from modelopt.torch.quantization.rotation import (
    fold_rotations, fold_seam_diags, learn_rotations,
    QuantObjective, RotationSet, SGDG,
    W4A4_G128_OBJECTIVE, INT8_DEFAULT_OBJECTIVE, W16A4_ASYM_R4G_OBJECTIVE,
    SEAM_DIAG_LR,
)

rotations = fold_rotations(model, mode="hadamard", seed=0, use_r2=True)
# model mutated in place (config.tie_word_embeddings forced False);
# rotations: {"R1": fp64 cpu [hidden, hidden],
#             "model.layers.{i}.self_attn.R2": fp64 cpu [head_dim, head_dim]}

rs = learn_rotations(
    model, calib_loader, steps=150, lr=1.5, mode="hadamard",
    objective_cfg=W4A4_G128_OBJECTIVE,        # or any QuantObjective / None
    seed=0, init_rotations=None, log_every=10,
    teacher=None, kd_alpha=0.5, kd_temp=2.0,  # optional KD objective
)  # -> RotationSet: .rotations (R.bin convention), .history, .meta,
   #    .seam_diags (transform-QAT only, else None)

fold_rotations(model, R1=rs.R1, R2=rs.R2)     # bake a checkpoint
fold_seam_diags(model, rs.seam_diags, smax=256)  # transform-QAT: bake seam scales
```

Pipeline order (load-bearing): untie tied embeddings with a real clone → seed the global
torch RNG → fuse RMSNorm gains into downstream linears (fused norms become exactly ones;
pure RMSNorm commutes with an orthogonal rotation of its input) → apply R1 → apply R2.
All math in float64, cast back to the original dtype.

## Reader/writer orientation

| Weight | Role vs residual stream | Transform |
|---|---|---|
| `embed_tokens` | writer (rows are stream vectors) | `E ← E @ R1` |
| `q/k/v_proj`, `gate/up_proj` | reader (input side) | `W ← W @ R1` |
| `o_proj`, `down_proj` | writer (output side) | `W ← R1ᵀ @ W`, `b ← R1ᵀ b` |
| `lm_head` (after final-norm fusion) | reader | `W ← W @ R1` |
| `v_proj` (R2) | output side, per **KV** head | `W_h ← R2ᵀ @ W_h` (row blocks) |
| `o_proj` (R2) | input side, per **Q** head | `W[:, h·d:(h+1)·d] ← … @ R2` (col blocks) |
| `q_norm`/`k_norm` (Qwen3) | per-head, post-q/k_proj head space | **never fused, never rotated** |

One R2 is shared by all heads of a layer, which is what keeps GQA/`repeat_kv` exact.

## Arch-mapping registry

`_ARCH_REGISTRY` is one small dict keyed on the model **class name** (`LlamaForCausalLM`,
`Qwen3ForCausalLM`), with fields: `has_qk_norm`, `head_dim` (callable on the config — must
prefer `config.head_dim`; `hidden_size // num_attention_heads` is wrong for Qwen3-0.6B: 64
vs the true 128), and `norm_edges` (RMSNorm → downstream-linear fusion edges). Any other
class raises `NotImplementedError`. Adding a standard-layout HF decoder
(`model.model.{embed_tokens,layers,norm}` + `model.lm_head`) is one dict entry.

## Equivalence gates

1. **Generation gate** — every R asserts `max |R Rᵀ − I| < 1e-10` in fp64 at build time.
2. **In-function post-conditions** — fused norms exactly ones; Qwen3 q/k_norm bitwise
   untouched; every parameter shape unchanged.
3. **Unit gate** — fp32 logits before vs after fold agree to `atol=1e-4` (measured ~3e-7);
   `tests/unit/torch/quantization/test_rotation_fold.py`.
4. **Model gate (external, before any PTQ)** — WikiText-2 PPL of the rotated fp checkpoint
   must match the original to noise.

## External-matrix fold path (learned rotations enter here)

`fold_rotations(model, R1=..., R2=...)` folds externally supplied matrices through the
exact same pipeline (untie → fuse → rotate). The seed path is unchanged and bitwise
reproducible; the external path skips the RNG entirely (global RNG state untouched) and
gates each matrix at `max |R Rᵀ − I| < 1e-4` — the *trained-rotation* deployability
tolerance (fp32 Cayley iterates drift ~1e-7/step; 150-step reference runs measure ~5e-5)
instead of the 1e-10 fresh-draw gate. `R2` accepts a layer-ordered sequence, an
int-keyed dict, or the returned-dict key convention (`model.layers.{i}.self_attn.R2`).
Unit gate: matrices returned by the seed path, fed back through `R1=`/`R2=` on an
identical model, reproduce every parameter bitwise.

## Learned rotations — `learn.py`

`learn_rotations(model, calib_loader, steps=150, lr=1.5, mode="hadamard",
objective_cfg=..., teacher=None, ...) -> RotationSet` learns the same offline pair the
fold applies (SpinQuant, arXiv:2405.16406):

- **What is learned**: the global residual-stream rotation **R1 `[hidden, hidden]`** plus
  one per-layer head-space rotation **R2 `[head_dim, head_dim]`** on the v_proj → o_proj
  path, as fp32 parameters on the **Stiefel manifold**, updated by **Cayley SGD** (the
  `SGDG` optimizer of Li et al., MIT-licensed, ported self-contained into `sgdg.py` — no new
  deps; the original momentum dead-store quirk is reproduced and documented, so momentum is inert
  exactly as in the official trainer).
- **Objective**: next-token **CE of the fake-quantized rotated model** on calibration
  text. Each step assembles every rotated effective weight out-of-place per fold.py's
  orientation table (readers `W @ R1`, writers `R1ᵀ @ W`, embed/lm_head included, v/o R2
  block mechanics), applies the objective's weight fake-quant with a straight-through
  estimator, reparametrizes the *frozen* model with those tensors
  (`torch.nn.utils.stateless._reparametrize_module`, spanning forward and backward), and
  fake-quantizes activations via pre-hooks on the 7×n_layers target linears. Gradients
  reach only R1/R2 (asserted at step 0; plus the seam-diag leaves under transform-QAT).
  Cosine lr decay over `steps`.
- **KD objective (optional)**: `teacher=` a frozen reference model (typically a
  bf16 copy of the SAME checkpoint) switches the loss to
  `(1−kd_alpha)·CE + kd_alpha·kd_temp²·KL(student ‖ teacher)`, teacher logits under
  `no_grad` on the same batch. The teacher is never reparametrized/fused/modified
  (unit-asserted). `teacher=None` (default) is bitwise the plain-CE trainer.
  `meta["kd"]` records `{alpha, temp}` when active.
- **Transform-QAT (optional, `QuantObjective.learn_seam_diag=True`)**: jointly learns
  OSTQuant-style per-input-channel diagonal scales at the two rotation-SURVIVING seams
  (down_proj input `[intermediate]`, o_proj input `[n_kv·head_dim]`, GQA-exact) as
  `log s` leaves (init 0 = identity; no extra RNG consumed — the R trajectory stream is
  unchanged). They live in a separate plain-Adam group at `SEAM_DIAG_LR` (1e-2, same
  cosine schedule — never the SGDG stiefel group), are applied in the effective-weight
  assembly with the SmoothQuant-style prefold structure (prefold-inside / rotation-outside), and
  export as `RotationSet.seam_diags` (fp64, positive by construction). Bake with
  `fold_seam_diags(model, seam_diags, smax=256)` (fp64 exact identities; the smax
  ceiling is the fp16-subnormal guard). `save()`/`load()` round-trips them under a
  reserved key; legacy flat R.bins load with `seam_diags=None`.
- **Init**: random(-sign) **Hadamard** (or Haar `"random"`) drawn with the SAME seeded
  global-RNG draw order as `fold_rotations` — `steps=0` returns bitwise the fold's seed
  draws (unit-gated), so trained and random rotations share one provenance contract.
  Warm starts enter via `init_rotations` (R.bin key convention, ortho-gated).
- **Model side effects** (identical to fold's pre-rotation steps, so learn-then-fold on
  one object or fold-on-a-fresh-copy are both valid): untie with a real clone, RMSNorm
  gains fused (idempotent), params frozen. Weights themselves are never rewritten —
  rotated weights exist only inside the per-step reparametrization. Qwen3 q/k_norm
  bitwise untouched (asserted; head-space, post-projection — reused arch spec).
- **Final retraction (load-bearing)**: raw fp32 Cayley iterates drift off the manifold,
  and the max-entry residual is **basis-dependent**: field measurements on 150-step R1s
  give `max|RᵀR−I| ≈ 4.6e-5 / 8.0e-5` (the form the step audit reports) but
  `max|RRᵀ−I| ≈ 7.3e-4 / 1.6e-3` — 10–20× larger — and the fold orientation consumes
  the `RRᵀ` form (reader/writer seams compose to `x R1 R1ᵀ Wᵀ`). Before returning,
  every trained matrix is polar-projected to the nearest orthogonal matrix
  (`R = UΣVᵀ → UVᵀ`, float64; per-entry move ≈ drift/2, far below one bf16 ulp; the
  same retraction semantics SGDG applies stochastically during training, applied once
  deterministically at the end). Post-retraction residual ~1e-14 both forms; raw
  drift + projection distance recorded in `meta["final_retraction"]`. The predecessor
  consumer of R.bin files handled the same asymmetry by widening
  its gate to 1e-3; the module closes the drift instead of widening the gate.
- **Output**: `RotationSet` — float64 CPU dict in the fold/R.bin key convention plus
  per-step `history` and `meta` (final static-A8 amax and the retraction log included
  when applicable). `save()`/`load()` round-trips the flat dict bitwise; `load()`
  refuses off-manifold matrices (`orthogonalize=True` retracts raw legacy R.bins).
  Feed `fold_rotations(model, R1=rs.R1, R2=rs.R2)` to bake a checkpoint.

Architecture knowledge is REUSED from `fold.py` (`_ARCH_REGISTRY`, norm edges, head_dim
resolution, q/k_norm exclusion, untie handling) — defined once, imported by the learner.

### Pluggable fake-quant objectives (`QuantObjective`)

Default numerics: symmetric integer QDQ, ModelOpt max-calibration semantics —
`s = amax/(2^{b-1}−1)` (clamped 1e-12), round-half-even, clamp `[−2^{b-1}, 2^{b-1}−1]`;
STE backward. Two paper-protocol extensions:

- **`a_asym=True`** — per-token dynamic ASYMMETRIC min-max affine activation QDQ,
  matching the official SpinQuant `ActQuantizer` (`sym=False`, `clip_ratio=1`)
  bit-for-bit: zero-inclusive token range, all-zero-token fallback to `[−1, 1]`,
  `scale=(max−min)/(2^b−1)`, `zp=round(−min/scale)` (unit-tested against a line-for-line
  reference transcription). Per-token-dynamic mode only.
- **`r4_in_graph=True`** — the online R4 down_proj Hadamard placed in the TRAINING
  graph only (input hook `x @ H` before act-QDQ + effective down_proj columns `@ H`; a
  functional-identity pair that only the quantizers see). The deployed fold never sees
  H (unit-asserted). Power-of-2 seam dims only (`_walsh_hadamard`; Llama-3.2 8192 ✓,
  Qwen3's 6144 needs the unimplemented had-K composition).

| preset | weights | activations | role |
|---|---|---|---|
| `W4A4_G128_OBJECTIVE` | int4 sym per-group g128 (in-dim) | int4 sym **per-token dynamic** | SpinQuant-paper-style W4A4; reference-trainer comparison axis |
| `INT8_DEFAULT_OBJECTIVE` | int8 sym **per-out-channel** | int8 sym **per-tensor STATIC** | the axes of ModelOpt `INT8_DEFAULT_CFG` — the deployment cell where random rotations barely help |
| `W16A4_ASYM_R4G_OBJECTIVE` | none (W16 in the loss) | int4 **asym per-token dynamic** + R4-in-graph | the official GPTQ-deploy training objective ("Cayley on 16-4-KV", paper Table 3) — kept for ablation; measured HARMFUL for R1R2-only deployment (objective-lever anchor below) |
| custom `QuantObjective(...)` | any bits, per-group or per-channel | per-token dynamic (sym/asym) / per-tensor static / off; `learn_seam_diag`, `r4_in_graph` | e.g. exact reference-trainer replica = `w_bits=4, w_group=None, a_bits=4, per_token_dynamic`; recommended W4A4 objective = `w4a4_g128 + a_asym=True`, built via `QuantObjective(name="w4a4_g128_asym", w_bits=4, w_group=128, a_bits=4, a_asym=True)` — NOT the shipped default: `W4A4_G128_OBJECTIVE` stays symmetric for paper-protocol comparability |

Coverage: the 7 per-layer projections only; embeddings/lm_head never quantized (matches
`INT8_DEFAULT_CFG`'s `*lm_head*` exclusion and the deployed ModelOpt cells). Static-A8
scale scope is `a_static_scope="batch"` by DEFAULT — a fresh amax per calib batch, the
stationary surrogate that tracks the moving rotation (post-hoc max calibration of the
folded ckpt is what deployment sees anyway). The literal running-max semantics
(`"run"`) is kept for ablation only: measured NON-stationary under a moving rotation
(loss drifts up as stale outliers pin the scale).

### Lineage diff — official SpinQuant vs. our internal reference trainer vs. this module

The middle column is an internal reference reimplementation (not shipped); it is kept in
the table because it is where the numerics of this module were first validated.

| axis | official `optimize_rotation.py` | internal reference trainer (not shipped) | `learn.py` (this module) |
|---|---|---|---|
| trained params | R1 + per-layer R2, fp32 `RotateModule`s | R1 + per-layer R2, fp32 `nn.Parameter`s | same as reference |
| R2 size derivation | `hidden_size // num_attention_heads` (**wrong for Qwen3-0.6B**: 64 vs true 128) | `config.head_dim` (asserted) | fold.py registry (`config.head_dim` preferred/asserted) |
| model plumbing | forked `modeling_llama_quant.py` with online rotation modules + QuantizeLinear wrappers (Llama only) | stock HF model, frozen; per-step effective-weight assembly + reparametrize | same as reference, arch registry Llama+Qwen3 |
| quantizer in the loss | their QuantLinear: asym per-token A4 (`--a_asym`), clip-searched weights (`--w_clip`); trained vs **activation-quant-only** network when pairing with GPTQ | v1: per-out-channel sym W4 (max scale, ±7 clamp) + per-token dynamic sym A4 STE hooks; v2: clip-searched W + asym A4 | pluggable `QuantObjective` (table above); sym, ModelOpt max-scale numerics, `[−2^{b−1}, 2^{b−1}−1]` |
| calib data | WikiText-2 train, seq 2048, 800 samples | GSM8K-train blocks, 512 tokens (an earlier variant used WikiText-2) | caller-supplied loader (any re-iterable of input_ids) |
| budget | 100 steps, effective bs 8 (1×8 GPUs), lr 1.5 | 150 steps, bs 4, seq 512, lr 1.5 | defaults steps=150, lr=1.5; caller-set |
| lr schedule | cosine (HF Trainer `--lr_scheduler_type cosine`) | cosine (explicit) | cosine (explicit, same formula) |
| optimizer | SGDG stiefel=True | SGDG port (stiefel branch verified bitwise vs official; momentum dead-store documented) | same port, self-contained |
| norm fusion / untie | `prepare_model` fuses norms; lm_head cloned for 3.2 | norm-fusion helper shared with the reference fold path; untie clone | reused `fold.py` helpers; untie clone |
| dtype discipline | bf16 model, fp32 R | bf16 model, fp32 R, TF32 force-off, bf16-grid store points mirroring the offline fold (bitwise store-point anchor) | model dtype preserved (fp32 CPU tests / bf16 GPU), fp32 R; no intermediate-grid mirroring (functional, not bitwise, contract) |
| output | `R.bin` (fp32 state-dict values) | `R.bin` fp64 + loss.jsonl + config.json | `RotationSet` (fp64, R.bin-compatible `save()`, history+meta in-object) |
| paper/no-online-Hadamard anchor | ~~v1 Table 3: R1-only 9.6~~ **superseded by v4 (ICLR'25) Table 8**: Llama-3.2-1B W4A4KV16 "SpinQuant no had" (= learned R1R2-only, GPTQ+w_clip+a_asym) wiki **48.4** vs fp **13.4** = **3.61×**; "SpinQuant had" (+online R3/R4) 15.3 = 1.14× | reference-trainer Llama-3.2-1B W4A4: 23.32 (windowed protocol, bf16 8.59) | measured (module training runs + official-harness alignment matrix) |

### Anchor correction — paper v4 (ICLR'25) no-had W4A4 band

The v1-based anchor above was misleading: arXiv 2405.16406 **v4** shows that a learned
R1R2-only ("no had") rotation at **W4A4 is NOT close to the fp baseline on any model** —
Llama-3.2-1B: 48.4 vs fp 13.4 (3.61×); LLaMA-2-7B: 9.2 vs 5.5 (1.67×); L3-8B: 18.6 vs 6.1
(3.05×). "No-had ≈ fp" is the paper's **W4A8** claim (1B: 15.3 vs 13.4). Closing the W4A4 gap
requires the online R3/R4 Hadamard ("had" scheme). Cross-checked on the official harness
(external alignment study, Llama-3.2-1B):
this module's learned R.bin dropped into official `ptq.py` gives no-had 61.2 (RTN+clip+asym)
vs random-no-had 109.2 vs none 256.1, and 18.8 with online R4 — same structure as the paper.
Budget note: paper Table 11 shows rotation quality saturates at 100 iters / 128 samples, so
the module's default 150 steps is not the binding factor at W4A4; the eval recipe
(GPTQ/clip/asym + online R4) is.

### Objective-lever anchor — Llama-3.2-1B W4A4 no-had

Measured on Llama-3.2-1B W4A4KV16, official released harness, no-had (R1R2-only)
deployment, GPTQ+clip+asym eval (external measurement campaign):

| training objective | no-had GPTQ PPL | verdict |
|---|---|---|
| `w4a4_g128` (sym, the shipped default — kept for paper-protocol comparability) | 57.27 | regression band |
| **`w4a4_g128 + a_asym=True`** | **48.89 — within 1% of the paper's 48.4** | **recommended W4A4 objective** (not the shipped default; see the objective table) |
| `+ r4_in_graph` (W4 in loss) | 59.55 | HARMFUL — trains a grid deployment never uses |
| `W16A4_ASYM_R4G` (Table-3 objective) | 93.72 | HARMFUL — replicates the official-trainer arm4 negative (cross-trainer consistency) |

The residual vs the paper's normalized multiple (5.01× vs 3.61× own-fp) equals the
internal-vs-released fp-anchor discrepancy (13.4/9.7611 = 1.37×) documented in the anchor
correction above — on the released harness, parity within ~1% is the achievable ceiling.

Online transforms require exporter + runtime kernel support (TRT-LLM) and remain a
separate track; `learn.py` trains only the foldable pair R1/R2.
