# PTQ Recipes & Schemes

This doc walks through the **PTQ quantization schemes** in two parts: the
model-agnostic recipes under [`general/ptq/`](general/ptq/) (the recommended
starting point for any model), and then the
[model-specific recipes](#model-specific-recipes-huggingface-and-models) under
`huggingface/` and `models/` — comparing each to its general baseline and
explaining why it deviates.

---

## General recipes

The general recipes are model-agnostic. Each file name combines a
**formats + scope** (what gets quantized, and to what format) with a
**KV-cache mode**, optionally with an **algorithm** (calibration variant):

```text
<formats-scope>-<kv-mode>[-<algorithm>].yaml
            nvfp4_experts_only - kv_fp8_cast
```

Pick one model-body scheme + one KV-cache scheme; the shipped files are the
supported combinations.

---

### The shipped recipes

<details>
<summary>All 18 <code>general/ptq/</code> recipes (click to expand)</summary>

| Recipe | Model body | KV cache | Calibration |
|--------|-----------|----------|-------------|
| `fp8_default-kv_fp8` | FP8 W8A8, all linears | FP8 (calibrated) | max |
| `fp8_default-kv_fp8_cast` | FP8 W8A8, all linears | FP8 (constant amax) | max |
| `nvfp4_default-kv_fp8` | NVFP4 W4A4, all linears | FP8 (calibrated) | max |
| `nvfp4_default-kv_fp8_cast` | NVFP4 W4A4, all linears | FP8 (constant amax) | max |
| `nvfp4_default-kv_nvfp4_cast` | NVFP4 W4A4, all linears | NVFP4 (constant amax) | max |
| `nvfp4_default-kv_none-gptq` | NVFP4 W4A4 (static W), all linears | none | GPTQ (layerwise) |
| `nvfp4_mlp_only-kv_fp8` | NVFP4 W4A4, MLP + MoE experts | FP8 (calibrated) | max |
| `nvfp4_mlp_only-kv_fp8-novit` | NVFP4 W4A4, MLP + MoE experts (VL vision tower excluded) | FP8 (calibrated) | max |
| `nvfp4_mlp_only-kv_fp8_cast` | NVFP4 W4A4, MLP + MoE experts | FP8 (constant amax) | max |
| `nvfp4_mlp_only_mse-kv_fp8_cast` | NVFP4 W4A4, MLP + MoE experts | FP8 (constant amax) | MSE + FP8 sweep |
| `nvfp4_experts_only-kv_fp8` | NVFP4 W4A4, MoE experts only | FP8 (calibrated) | max |
| `nvfp4_experts_only-kv_fp8_cast` | NVFP4 W4A4, MoE experts only | FP8 (constant amax) | max |
| `nvfp4_experts_only-kv_fp8_layerwise` | NVFP4 W4A4, MoE experts only | FP8 (calibrated) | max, layerwise |
| `nvfp4_experts_only_mse-kv_fp8_cast` | NVFP4 W4A4, MoE experts only | FP8 (constant amax) | MSE + FP8 sweep |
| `nvfp4_omlp_only-kv_fp8` | NVFP4 W4A4, o_proj + MLP/MoE | FP8 (calibrated) | max |
| `nvfp4_omlp_only-kv_fp8_cast` | NVFP4 W4A4, o_proj + MLP/MoE | FP8 (constant amax) | max |
| `nvfp4_weight_only-kv_fp16` | NVFP4 W4A16, weights only | none (BF16/FP16) | max |
| `nvfp4_weight_only-kv_fp8_cast` | NVFP4 W4A16, weights only | FP8 (constant amax) | max |
| `int4_blockwise_weight_only` | INT4 W4A16, block 128, weights only | none | max |

</details>

---

### Model-body schemes

The body scheme is the main lever: it trades accuracy against memory/throughput
by choosing **which parts of the model drop to low precision** and **whether
activations are quantized too** (W4A4/W8A8 vs weight-only W4A16).

#### Full-model schemes (quantize everything)

- **`fp8_default`** — **per-tensor** FP8 E4M3 **W8A8** on every linear (attention
  q/k/v/o + MLP/MoE) — one scale per weight/activation tensor. The safest
  aggressive option: FP8 has a wide dynamic range, so accuracy loss is usually
  negligible. Needs Hopper+ for FP8 kernels. Good default when the target
  hardware is FP8-class and you want the broadest speedup.
- **`nvfp4_default`** — NVFP4 (E2M1, block-16, FP8 block scales) **W4A4** on every
  linear. The most aggressive scheme — 4-bit weights *and* activations everywhere
  — for maximum memory/throughput on Blackwell+. Highest risk of accuracy loss;
  if it regresses, fall back to one of the scoped schemes below rather than
  abandoning NVFP4.

#### Scoped schemes (quantize part of the model)

- **`nvfp4_experts_only`** — NVFP4 W4A4 on **MoE routed experts only**
  (`*mlp.experts*`, `*block_sparse_moe*`). Dense layers, shared experts, and
  attention stay BF16. **The most recommended NVFP4 recipe for MoE models**: it's
  the narrowest, most accuracy-preserving NVFP4 scope, so it recovers the most
  accuracy — while still compressing well, because the routed experts are usually
  the largest share of the model's total weights.
- **`nvfp4_mlp_only`** — NVFP4 W4A4 on **all MLP/FFN compute**: dense MLP layers,
  MoE routed experts, and `block_sparse_moe` blocks. Attention stays BF16.
  **Recommended for dense models**: most FLOPs/params live in the MLP, so this
  captures most of the win while leaving the sensitive attention path untouched
  for accuracy.
- **`nvfp4_omlp_only`** — NVFP4 W4A4 on **MLP/MoE plus the attention output
  projection** (`o_proj`), but *not* q/k/v. A middle ground between `mlp_only`
  and `default`: adds the o_proj GEMM (often safe) without quantizing the more
  sensitive q/k/v projections.

> **Scope vs. compression.** These schemes keep the accuracy-sensitive attention
> path (or the whole dense path) at the model's original precision — BF16 for most
> checkpoints — and quantize only the FFN/expert weights, which dominate params and
> compute. That's good for accuracy, but the left-out layers stay uncompressed. How
> much that matters is **model-dependent**: on MoE models the routed experts are the
> vast majority of the weights, so leaving attention in BF16 costs almost nothing on
> disk; on some dense models the attention projections are large enough that they
> noticeably bound the checkpoint size. *If* the attention weights are large and you
> want to compress them, we recommend adding an **FP8** rule for the attention
> projections (keep NVFP4 on the MLP/experts) rather than leaving them BF16 — FP8
> keeps that sensitive path at a safer precision than NVFP4 while still halving those
> weights vs. BF16.

#### Weight-only schemes (W4A16 — activations stay BF16)

Quantize weights only; activations run in BF16. This shrinks the model
(memory-bound decode win) with much lower accuracy risk than W4A4, and **needs no
calibration forward pass**.

These are usually recommended for **low-concurrency deployments** — edge and
on-device/client use cases — where the workload is memory-bandwidth-bound and
shrinking the weights is the main win. For **high-concurrency data-center
serving**, prefer a scheme that also quantizes activations (the W4A4/W8A8 body
schemes above): at large batch sizes the GEMMs become compute-bound, so low-bit
activations and tensor-core math are what deliver the throughput.

- **`nvfp4_weight_only`** — NVFP4 weights, BF16 activations. Memory savings of
  4-bit weights without the activation-quantization risk.
- **`int4_blockwise_weight_only`** — INT4 weights, block size 128, BF16
  activations. Classic W4A16 weight compression; works without NVFP4-class
  hardware.

---

### KV-cache schemes

The `kv_*` suffix controls how the attention KV cache is quantized — independent
of the body scheme. Quantizing the KV cache reduces memory at long context.

- **`kv_fp8_cast`** — FP8 KV with a **constant amax** (cast mode): skips KV
  calibration entirely. Cheaper to produce and the safe default for KV. For most
  models it is **as accurate as the calibrated `kv_fp8`** below, so prefer it
  unless you have a specific reason to calibrate KV scales. Hopper+.
- **`kv_nvfp4_cast`** — NVFP4 KV cache with constant amax. More aggressive KV
  compression (4-bit); combines with any body scheme. Blackwell+.
- **`kv_fp8`** — FP8 E4M3 KV cache with **calibrated** per-tensor amax. The KV
  scales are measured during the calibration pass. Hopper+.

> **`kv_fp8_cast` vs `kv_fp8`:** both produce an FP8 KV cache. `_cast` uses a
> fixed scale and skips the KV calibration step (faster, no extra data
> dependence); plain `kv_fp8` calibrates the scale from data. The cast version
> usually matches calibrated accuracy, so start with `kv_fp8_cast`.

---

### Calibration variants

How the quantization scales are searched. The default (no suffix) is `max`.

- **`max`** (default) — amax/max calibration. Fast, one calibration pass; the
  baseline choice.
- **`mse`** (e.g. `nvfp4_mlp_only_mse`, `nvfp4_experts_only_mse`) — MSE search
  for **static** NVFP4 weight scales, with an FP8-scale sweep over the e4m3 scale
  values. The MSE search applies to the weights; activations are still max
  (amax) calibrated as in the default recipes. Costs more calibration time but
  recovers accuracy NVFP4 W4A4 can lose under plain max. Reach for it when a
  `max` recipe regresses.
- **`gptq`** (`nvfp4_default-kv_none-gptq`) — GPTQ layerwise calibration of the
  weight scales; writes layerwise checkpoints. GPTQ is best established for
  **INT4 weight-only** quantization; its effectiveness on **NVFP4** weight
  quantization varies model by model — it tends to help most when the other
  recipes show a larger accuracy loss. Applying GPTQ to **MoE** models is still
  an open research topic and needs extra recipe tuning.
- **`layerwise`** (`nvfp4_experts_only-kv_fp8_layerwise`) — max calibration done
  one decoder layer at a time to **lower peak memory**; same numerics as the
  non-layerwise variant.

These can also be **stacked** when a single method isn't enough — e.g. `mse` +
`gptq` combines an MSE-searched weight scale with GPTQ's layerwise update.

---

### Choosing a general recipe

1. **Match the format to your hardware/target.** FP8 (`fp8_default`) on Hopper+;
   NVFP4 (`nvfp4_*`) on Blackwell+; weight-only (`*_weight_only`) when you want
   compression with minimal risk or lack NVFP4-class kernels.
2. **Start from the most accurate scope, then quantize more toward your
   memory/performance target.** For **low-concurrency** deployments (edge,
   on-device/client), start from a **weight-only** recipe (`nvfp4_weight_only` /
   `int4_blockwise_weight_only`) — shrinking weights is the main win there. For
   higher-concurrency serving, begin with the narrowest activation-quantized
   scope — `nvfp4_experts_only` for MoE, `nvfp4_mlp_only` for dense — then widen
   (`mlp_only` → `omlp_only` → `default`) only as far as your memory/throughput
   target requires, checking accuracy as you go.
3. **Recover accuracy via calibration before backing off the scope.** If a
   wider-scope recipe regresses, switch its `max` to the `mse` variant before
   retreating to a narrower scope.
4. **Pick KV by deployment.** `kv_fp8_cast` is the safe default (usually as
   accurate as calibrated `kv_fp8`); use `kv_nvfp4_cast` for maximum KV
   compression.

---

## Model-specific recipes (`huggingface/` and `models/`)

The general recipes above are **model-agnostic**: they select layers by wildcard
(`*mlp*`, `*self_attn*`, `*[kv]_bmm_quantizer`) and lean on the shared
`default_disabled_quantizers` exclusions, so the same file works on any
architecture whose module names follow the usual conventions. A recipe only
earns a place under `huggingface/<model_type>/` or `models/<checkpoint>/` when a
model has to **deviate** from that baseline. The deviations come in four kinds:

| Kind | What changes vs. the general recipe | Examples |
|------|-------------------------------------|----------|
| **Architecture-aware `quant_cfg`** | Per-sub-module format choices a single wildcard scheme can't express | `qwen3_5`, `qwen3_5_moe` |
| **Algorithm override** | Same numerics & scope, but the *calibration algorithm* is tweaked because the default breaks or regresses | `gemma`, `mpt` |
| **Extra exclusions** | Adds disabled-quantizer patterns so non-language branches stay full precision | `nemotron_vl`, `phi4mm` |
| **Checkpoint mirror** | A mixed-precision map reproducing one published checkpoint exactly | `models/Nemotron-3-Super-120B-A12B` |

The numerics and standard exclusions are still inherited from `configs/`
wherever possible — the model folder captures *only* the delta. Each `<task>/`
folder carries a `README.md` spelling out that delta.

### Architecture-aware `quant_cfg` — `qwen3_5`, `qwen3_5_moe`

`huggingface/qwen3_5/ptq/w4a16_nvfp4-fp8_attn-kv_fp8_cast` (and its MoE twin,
which shares the same `quant_cfg` snippet) is a **mixed scheme no single general
body covers**: NVFP4 **W4A16** on MLP / expert projection weights and `lm_head`,
**FP8** on self-attention *and* the large linear-attention projections
(`in_proj_qkv`, `in_proj_z`, `out_proj`), plus FP8 KV cast. It also disables
architecture-specific submodules that aren't in the reference recipe
(`linear_attn.in_proj_a/b`, `conv1d`, and any `visual`/`mtp` siblings).

*Why special:* these are hybrid **linear-attention + softmax-attention** models.
A general scheme would apply one format per wildcard class; this architecture
needs FP8 for the big linear-attention projections but NVFP4 for MLP weights, and
needs the linear-attention conv/gate submodules left alone. The dense and MoE
families share the identical wildcard rules, so one snippet drives both.

On Qwen3.5 / Qwen3.6 this W4A16 recipe **usually does not regress accuracy**
versus the official checkpoint, and it is designed for **best performance in
low-concurrency use cases** (weight-only on the MLP keeps the memory-bound decode
path fast without quantizing activations).

A lighter case: **`step3p5/Step3.5-Flash/ptq/nvfp4-mlp-only`** is close to
`general/ptq/nvfp4_mlp_only` (NVFP4 on MoE/MLP weights+inputs, FP8 KV) but pinned
to one released checkpoint and carrying instance-specific disables
(`share_expert`, `moe.gate`, the conv1d branches).

### Algorithm overrides — `gemma`, `mpt`

These quantize the **same layers** as the general recipes; only the
`quantize.algorithm` block differs, to work around model-specific numerics:

- **`gemma/ptq/w4a8_awq-kv_fp8_cast`** (INT4 block weights + FP8 inputs + FP8 KV
  cast) and **`mpt/ptq/w4a8_awq-kv_fp8_cast`** use `awq_lite` with `alpha_step: 1`
  instead of the default AWQ search. The default search overflows the TRT-LLM
  kernels on these models; the coarser sweep avoids it without measurably hurting
  accuracy.
- **`gemma/ptq/int8_sq-kv_fp8_cast`** (INT8 per-channel weights + INT8 inputs +
  FP8 KV cast) sets SmoothQuant `alpha: 0.5` instead of the default `1.0` —
  Gemma 7B regresses at `1.0`, and `0.5` recovers it.

*Why special:* identical scope/numerics to a general scheme, but a general
recipe's default algorithm would overflow or regress here.

### Extra exclusions (multimodal) — `nemotron_vl`, `phi4mm`

Both are **numerically identical** to `general/ptq/nvfp4_default-kv_fp8_cast`
(NVFP4 W4A4 + FP8 KV cast). What makes them special is a model-local
`disabled_quantizers.yaml` unit that *extends* the standard exclusions so only
the **language decoder** is quantized:

- **`nemotron_vl`** (vision-language, incl. Nemotron-Parse) adds
  `*vision*`, `*image*`, `*radio*`, `*visual*`, `*encoder*`, `*model_encoder*`.
- **`phi4mm`** (Phi-4-Multimodal) adds `*speech*`, `*audio*`, `*image*`,
  `*vision*`.

*Why special:* a general recipe would happily quantize the vision/audio
encoders, regressing those modalities. The extra patterns keep them in full
precision; everything else matches the general recipe.

### Checkpoint mirror — `models/Nemotron-3-Super-120B-A12B`

The `models/` tier reproduces a **single published checkpoint's** quant config
verbatim. `super-nvfp4.yaml` mirrors
`nvidia/NVIDIA-Nemotron-3-Super-120B-A12B-NVFP4` exactly — a hybrid **Mamba-MoE**
with a hand-mapped, **per-component** precision scheme:

- MoE routed experts → NVFP4 W4A4, `group_size 16`, **static** weight scales
- shared experts and Mamba `in/out_proj` → FP8 per-tensor
- KV cache → FP8
- attention q/k/v, MTP head, `lm_head`, latent-MoE, Mamba conv1d → **BF16**

*Why special:* unlike any general recipe, it **mixes FP8 and NVFP4 across
different component types** and hardcodes the precise published layout (matched on
both HF and Megatron-Core module names) rather than a portable wildcard scheme.
`super-nvfp4.yaml` uses MSE calibration with an FP8-scale sweep (matches the
release); `super-nvfp4-max-calib.yaml` is the identical layer map under plain
`max` calibration, kept for comparison.

---

For the full catalog and how to pick a starting recipe for a given model, see
[`README.md`](README.md).
