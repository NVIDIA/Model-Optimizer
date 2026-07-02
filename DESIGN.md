# Attention quantization + skip-softmax sparsity (vLLM)

## Goal

Serve a checkpoint that is **both attention-quantized and skip-softmax-sparse**,
composed in a single attention pass (prefill *and* decode), driven entirely by
the exported `*_bmm_quantizer` / sparse-attention config — no `MODELOPT_ATTN_*`
env knobs. Skip-softmax selects which KV tiles to compute; quantization controls
how the attention operands are represented. The two are orthogonal in code and
compose in one Triton kernel launch.

This is a **fake-quantization accuracy study**: operands are rounded to the
NVFP4/FP8 grid and dequantized back to a float type for the matmul. Real KV-cache
*memory* savings (packed FP4) are a deferred deployment concern (see "Build status").
The matmul precision is **phase-specific** — bf16 tensor-core in prefill, fp32 in
decode — see "Matmul precision by phase" below.

## The two matmuls set the quantization axis

NVFP4 blocks (16 elements sharing an E4M3 scale) must lie along the GEMM
**contraction** axis. Attention has two matmuls with *different* contraction
axes:

| BMM | operands | contraction axis | block-16 axis |
|-----|----------|------------------|---------------|
| BMM1 `Q · Kᵀ` | Q, K | head_dim | head_dim |
| BMM2 `P · V`  | P, V | **keys**     | **keys**      |

A vLLM decode step writes **one token = one key** at a time. So a per-token
cache write can form the **head_dim** axis (fully present per token) but *not*
the **keys** axis (a 16-key block spans 16 decode steps). That single fact
decides where each operand is quantized.

## Per-operand mechanism

| operand | BMM | block axis | where quantized | producible at write? |
|---------|-----|-----------|-----------------|----------------------|
| Q | 1 | head_dim | pre-step (`_QuantVLLMAttention.forward`, `q_bmm_quantizer`) | n/a (current query) |
| K | 1 | head_dim | pre-step → written to cache (`k_bmm_quantizer`) | yes (quantize-once) |
| V | 2 | keys | **on-write bake (both phases)** (`v_bmm_quantizer` → `_v_qdq_nvfp4`) | no |
| P | 2 | keys | **in-kernel** (`p_bmm_quantizer` → `_p_qdq_nvfp4`) | no (P is transient) |
| skip | – | – | in-kernel tile selection on the quantized scores | – |

Principle: quantize each operand once, on its correct axis, as close to the
write as that axis allows; the operands whose axis a per-token write cannot form
(P always; V at decode) are handled in/around the kernel.

- **Q, K** stay on the `_QuantVLLMAttention` pre-step: head_dim is present per
  token, so the pre-step is quantize-once-at-write for free and reuses standard
  ModelOpt machinery with stock vLLM cache writes.
- **P** is fake-quantized **in-kernel** by `_p_qdq_nvfp4` (plain max, P ≥ 0;
  16-blocks along keys). It is transient, so in-kernel is its only home. It quantizes
  the *tile-local unnormalized* numerator `exp2(scores − m_new)` (vs the running max,
  before the `/l` normalization), so the autotuned prefill `BLOCK_N` and the
  auto-selected decode split count perturb the quantized P — E4M3 scale rounding is
  not scale-equivariant across the online-softmax max-shift. This is inherent to
  flash-attention P fake-quant (the mni reference and real FP4-attention hardware
  quantize tile-local P too), ~cos 0.9998 on the output (inside the NVFP4 quant
  floor), **not a bug**. For bitwise reproducibility across runs/hardware, pin the
  tiling (`PYTEST_VERSION` → fixed `BLOCK_M/BLOCK_N`; `num_kv_splits` → fixed decode
  splits) — a determinism knob, not an accuracy fix.
- **V** is fake-quantized along the keys axis by `_v_qdq_nvfp4` (`abs` for signed
  V; 16-blocks along axis 0 of the loaded tile `[BLOCK_N keys, BLOCK_D head_dim]`;
  masked-to-0 loads keep a partial tail from poisoning a block amax). *Where* it
  runs differs by phase — see below.

## Matmul precision by phase (prefill bf16 tensor-core, decode fp32)

The NVFP4/FP8 operand *grid* is identical in both phases, but the matmuls run at
different precision, by design:

- **Prefill** (a GEMM over `BLOCK_M` query rows) uses `tl.dot` on **bf16** operands
  for both `Q·Kᵀ` and `P·V` (`p.to(v.dtype)` before BMM2). Tensor-core throughput,
  at the cost of rounding the dequantized operands to bf16.
- **Decode** (a GEMV — one query row) uses **fp32** elementwise reductions for both
  matmuls (`tl.sum(q[:,None]*kᵀ)`, `tl.sum(p[:,None]*v)`); P stays fp32 (V is still
  dequantized to bf16 to match the on-write cache, then upcast). `tl.dot` is
  wasteful at M=1, and fp32 is more accurate: vs an fp64-exact reference the
  fp32-elementwise decode is **5.70e-8** (the fp32 floor) while a bf16/tf32x3
  `tl.dot` is **1.39e-4** (~2400× worse); the fp32 decode also reproduces the
  reference branch's default decode bit-for-bit.

Consequence: the **P dtype entering `P·V` differs** — bf16 in prefill, fp32 in
decode. This is an intentional phase-specific asymmetry (prefill trades precision
for GEMM throughput; decode, cheap at M=1, keeps full fp32), not a bug. Note
`tl.dot` does not *require* bf16 (it also accepts fp32 via
`input_precision=tf32|tf32x3|ieee`); the bf16 prefill path is a throughput choice,
not a hard constraint. A `P·V`-tile micro-benchmark (A6000, vs fp64, V held at
bf16) shows fp32 `tf32x3` is ~9000× more accurate than bf16 on the tile (rel-L2
~1.6e-7 vs ~1.5e-3) at ~1.3× the *tile* matmul latency. **That gain does not reach
the attention output.** Re-measured on **B200** (full-quant prefill, 4096-tok GQA),
fp32 `tf32x3` prefill is **2.4× end-to-end** — not the 1.3× tile ratio; the tf32x3
3-pass dominates the whole kernel — for a **negligible** accuracy change: cos-vs-dense
is unchanged (**0.9918** either way), because the NVFP4 *quantization* error (~8e-3)
dominates the output and the bf16-accumulation artifact (~1.5e-3) washes out after
softmax normalization. **So prefill `P·V` stays bf16**: fp32 accumulation is more
faithful to native FP4 in principle, but here it costs ~2.4× for no measurable eval
benefit.

## V: baked on write for both phases (required for decode)

The keys axis means V cannot be quantized by a per-token write, so it is
fake-quantized around the kernel. But the *cost* differs sharply by phase:

- **Prefill** is a single pass that touches each tile once → `O(S)` either way. It
  bakes its complete tiles on write *before* the kernel and reads them as-is
  (`V_CACHE_QUANTIZED`, same as decode), re-FQ'ing only the trailing partial, so V is
  **quantized exactly once** — a *chunked* prefill never re-FQ's an earlier chunk's
  baked tiles. (That double-quant is usually a no-op anyway: NVFP4 QDQ is a near
  fixed-point — measured `maxabs 0` on B200 random data — but the gate makes
  quantize-once exact.)
- **Decode** is autoregressive: every step re-reads the whole growing cache, so
  on-read re-FQ's the entire cache each step → `Σ O(s) = O(S²)`, and it is almost
  all redundant (a written token is immutable, so re-quantizing token 5 at steps
  6…1000 yields the identical result 995×).

That `O(S²)` is not academic: it **made long-context evals (HLE) infeasible** —
the original on-read decode design timed out, which is why PR #1635 switched to
**on-write** (≈8–18× decode-kernel speedup). So for the long-context campaign,
decode V **must** be on-write:

- **Bake** complete 128-token V tiles once, in place in the paged cache
  (`fake_quant_v_onwrite`, tile size `_ONWRITE_BLOCK_N = 128`, driven by
  `v_bmm_quantizer`); each 128-token tile is internally 8 × 16-key NVFP4 blocks.
  Prefill bakes the prompt so decode inherits a pre-quantized cache; decode bakes
  each newly-complete tile.
- The decode kernel reads complete tiles **as-is** (`V_CACHE_QUANTIZED`) and
  re-FQ's only the trailing `s mod 128` partial tile via `_v_qdq_nvfp4` → `O(S)` total.
- Needs the graph-safe `(batch, n_kv, 1)` decode-grid repair so the bake kernel
  composes with the captured decode step.

Because written tokens are immutable, on-write reproduces the on-read fake-quant
**bit-for-bit** at the V‑value level. Validated on B200 (NVFP4, fp16/bf16 cache):
(A) FQ-all on the baked cache equals FQ-all on the raw cache — `maxabs = 0`; (B)
incremental per-tile baking (the decode pattern) yields a bit-identical cache to a
single-shot bake — **no cross-step accumulation**. The dequantized V is stored at
the buffer (cache) dtype, which the NVFP4 dequant already hits exactly. The
attention *output* of read-as-is vs FQ-on-read can differ by ≤~1e-5 — fp32-reduction
scheduling between the two compiled kernel variants on **identical** V values, not a
quantization difference. So this is a **pure speedup, no accuracy change**. K escapes
the problem entirely: the pre-step quantizes each K once. (This is "Option 3" of the
trailing-block methods: quantize the tail from pristine bf16 each step — no
accumulation, uniform-precision kernel.)

## Fidelity to true NVFP4

True NVFP4 = E2M1 element × dynamic E4M3 per-16 block scale (`amax(block)/6`) ×
FP32 per-tensor global. The **per-16 block scale is the real quantizer** and is
computed dynamically per block in-kernel for both P and V; the partial trailing
tile gets its scale from its own valid keys (zeros from masked loads never raise
the amax). For V the per-tensor global barely matters — the block amax carries
the range and V does not saturate E4M3 — so `v_qdq_amax=None` uses the constant
`1.0` global. A frozen first-chunk global is the only scheme that diverges (it
saturates E4M3 on long context) and is intentionally not used.

The per-block scale is clamped to `[2**-9, 448]` in `fp8_quantize_scale`, matching
the canonical NVFP4 path (`qtensor/nvfp4_tensor.py`): the upper bound caps at the
E4M3 max, and the `2**-9` lower bound floors an underflowed block at the smallest
subnormal instead of letting the scale round to 0 (which would zero the block). A
block below the floor still rounds to 0 on the E2M1 grid; one in the `[~1.8e-7,
2.2e-6]` band reconstructs nonzero (e.g. amax `2e-6` → `2.179827e-6`). This keeps the
in-kernel fake-quant bit-consistent with the exported NVFP4 checkpoint in the
underflow regime, and lets the P/V and Q/K (`fp4_kernel_hopper.py`) paths share one
scale contract without the ad-hoc `<1e-5 → 1.0` guard. On the output it is a no-op:
underflowed P/V blocks carry weights `< ~2.2e-6` (keys many log2-units below the row
max), a B200 A/B (current-vs-clamped, up to 98% of P underflowing) is bit-identical,
and Q/K never underflow on real data — it is a fidelity/consistency guarantee, not an
accuracy change.

## Supported configuration (fail-loud)

The served checkpoint runs the ModelOpt attention kernel only when **every** requested layer
satisfies all of these; anything else fails at startup with a single layer-qualified error (no
requested quant or sparsity transform is ever silently dropped):

- Decoder self-attention (`AttentionType.DECODER`) on the FlashAttention or FlashInfer backend.
- Q/K/P/V use the supported fake-quant recipe: dynamic block-16 NVFP4 (P/V BMM2 may also be
  per-tensor FP8), Q per-step dynamic, K/V calibrated-or-default global scale.
- FP16/BF16 KV cache (every `fp8*` cache dtype is rejected); cache page size a multiple of 16.
- `decode_context_parallel_size == 1`; no sliding window, ALiBi, logits soft-cap, or sinks.
- No prefix caching, cross-layer KV sharing, KV connector, speculative decoding, or cascade.

Launch with `--enforce-eager` and prefix caching disabled until the retained code has CUDA-graph
evidence. Cascade is disabled automatically when quant is active — but disabling cascade does **not**
make prefix-cache storage reuse safe (V is baked in place; a shared prefix would read another
request's quantized V), which is why prefix caching is rejected outright.

## Code layout

```text
kernels/quantization/attention/p_qdq.py   _p_qdq_nvfp4 (P BMM2 helper)
kernels/quantization/attention/v_qdq.py   _v_qdq_nvfp4 (V BMM2 helper)
kernels/quantization/common/nvfp4_quant.py  nvfp4_scalar_qdq + fp8_quantize_scale ([2^-9,448] clamp)
kernels/common/attention/triton_fa.py     prefill kernel: P_QDQ + V_QDQ constexprs
kernels/common/attention/decode_attention.py paged decode kernel + fake_quant_v_onwrite (128-token V bake)
quantization/plugins/vllm.py
    _import_attention_module   select the vLLM Attention class by exported symbol
    _QuantVLLMAttention        Q/K head_dim pre-step; holds q/k/v/p_bmm_quantizer
sparsity/attention_sparsity/plugins/vllm.py
    _p_qdq_from_layer / _v_qdq_from_layer  read p/v_bmm_quantizer -> (mode, amax); fail-loud on unmapped
    ModelOptSparseAttentionImpl / ModelOptSparseFlashInferImpl  P/V + skip in both kernels
    select_sparse_impl_cls / _clone_sparse_impl / patch_flashinfer_metadata_builder  backend routing
examples/vllm_serve/quant_sparse_attn_worker.py
    _preflight_quant_sparse_attn            pure support-matrix validation (fail-loud, no mutation)
    _prepare_ / _commit_quant_sparse_attn   two-pass atomic install (build all, then assign + flip V gate)
```

## Build status

Implemented and validated: paged decode kernel + skip-softmax; in-kernel `P_QDQ`/`V_QDQ` in
prefill and decode; **decode V on-write** (`fake_quant_v_onwrite` 128-token tiles +
`V_CACHE_QUANTIZED` gating + graph-safe `(batch, n_kv, 1)` grid), reproducing on-read FQ
bit-for-bit (incremental == single-shot; output diff ≤~1e-5 is fp32-reduction scheduling, not a
value change); plugin wiring that drives P/V from the exported `p/v_bmm_quantizer`; a unified serve
worker that attaches attention quant (impl-swap default, composes with a realquant checkpoint; a
legacy `mtq` mode) then installs the sparse impl; FlashAttention **and** FlashInfer backends.

Gate A safety: a pure capability **preflight** validates the support matrix above and fails at
startup with one layer-qualified error; installation is a two-pass **prepare/commit** transaction,
so a failure leaves every original impl and V-ownership gate unchanged.

Validated on **B200** (on-write V numerics) and **GB300 / sm_100** (aws-cmh, vLLM 0.22): the NVFP4
attention kernel suite (68 tests, incl. the V-bake lifecycle) and the install/logic suite (32 tests)
pass, with no sm_100 kernel fault.

End-to-end serve evidence (GB300 / sm_100, Qwen3-8B bf16, vLLM 0.22 / Triton 3.6 / CUDA 13,
`ATTN_QUANT_MODE=impl_swap --enforce-eager --no-enable-prefix-caching --enable-chunked-prefill`):
the impl-swap installs NVFP4 on all 36 attention layers and prefill / decode / chunked-prefill
produce coherent output on **both** the FlashAttention and FlashInfer backends ("The capital of
France is" → " Paris"; correct counting continuation). FlashInfer (the deployed path) runs
prefill/decode/chunked ≈ 0.11 / 1.8 / 0.5 s; the FlashAttention path is slower, dominated by
first-request Triton autotune under eager. Output equivalence to bf16 is covered separately by the
eval numbers (AA-LCR, HLE), not a per-token tolerance here.

Remaining (Gate B/C follow-ups): CUDA-graph capture validation for the retained path (the E2E above
ran under `--enforce-eager`); a framework-neutral numerical spec + immutable per-layer plan consumed
at runtime (so serving reads plans, not mutable quant modules); calibrate skip-softmax on the
**quantized** model; real packed-FP4 KV-cache memory savings.
