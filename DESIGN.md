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

## V: block-16 finalization with a pristine tail

V's NVFP4 blocks run along the key axis, so a per-token cache write cannot form a
complete scale group. Quantizing the entire cache on every decode step would cost
`Σ O(s) = O(S²)`, while reading the current partial group as BF16 during attention
would make the `P @ V` operands mixed precision. The serving path therefore uses the
paged-cache tail itself as the high-precision buffer:

1. Every complete 16-key group remains QDQ in the paged cache; the incomplete
   `s mod 16` group remains pristine FP16/BF16 between attention calls.
2. Before attention, newly completed groups are QDQ once from their pristine values.
3. Prefill and decode read completed groups as-is and QDQ only the pristine partial
   group in registers. Thus every valid V value entering `P @ V` is QDQ.

This keeps the attention arithmetic uniform without an auxiliary buffer, temporary
cache mutation, or restore kernel. It also avoids cumulative `QDQ(QDQ(V))` error, and
the complete history is quantized once, so decode remains `O(S)` in V quantization work.
NVFP4 QDQ is not generally idempotent: for example,
`0.017578125 -> 0.015625 -> 0.01171875`, which is why preserving the pristine tail is
a numerical requirement rather than only a performance optimization.

## Fidelity to true NVFP4

True NVFP4 = E2M1 element × dynamic E4M3 per-16 block scale (`amax(block)/6`) ×
FP32 per-tensor global. The **per-16 block scale is the real quantizer** and is
computed dynamically in the attention kernel for P and the partial V group, and in
the cache preparation kernel for complete V groups. The partial V group gets its scale from its valid keys; masked
positions are zero and never raise the amax. For V the per-tensor global barely matters — the block amax carries
the range and V does not saturate E4M3 — so `v_qdq_amax=None` uses the constant
`1.0` global. A frozen first-chunk global is the only scheme that diverges (it
saturates E4M3 on long context) and is intentionally not used.

The per-block scale is clamped to `[2**-9, 448]` in `fp8_quantize_scale`, matching
the canonical NVFP4 path (`qtensor/nvfp4_tensor.py`): the upper bound caps at the
E4M3 max, and the `2**-9` lower bound floors an underflowed block at the smallest
subnormal instead of letting the scale round to 0 (which would zero the block). A
block below the floor still rounds to 0 on the E2M1 grid; one in the `[~1.8e-7,
2.2e-6]` band reconstructs nonzero (e.g. amax `2e-6` → `2.179827e-6`). This keeps the
fake-quant path bit-consistent with the exported NVFP4 checkpoint in the
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
- `decode_context_parallel_size == 1`; no dual batch overlap, sliding window, ALiBi,
  logits soft-cap, or sinks.
- No prefix caching, cross-layer KV sharing, KV connector, speculative decoding, or cascade.

Launch with `--enforce-eager` and prefix caching disabled until CUDA-graph capture and shared-cache
reuse have been validated for this path. Cascade is disabled automatically when quant is active.

## Code layout

```text
kernels/quantization/attention/p_qdq.py   _p_qdq_nvfp4 (P BMM2 helper)
kernels/quantization/attention/v_qdq.py   _v_qdq_nvfp4 (V BMM2 helper)
kernels/quantization/common/nvfp4_quant.py  nvfp4_scalar_qdq + fp8_quantize_scale ([2^-9,448] clamp)
kernels/common/attention/triton_fa.py     prefill kernel: P QDQ + on-read partial-V QDQ
kernels/common/attention/decode_attention.py paged decode + block-16 V finalization/partial-V QDQ
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

Implemented: paged decode + skip-softmax; in-kernel P QDQ; block-16 V finalization and on-read
partial-group QDQ for prefill and decode; plugin wiring that drives P/V from the exported
`p/v_bmm_quantizer`; and a unified serve worker for FlashAttention and FlashInfer.

Gate A safety: a pure capability **preflight** validates the support matrix above and fails at
startup with one layer-qualified error; installation is a two-pass **prepare/commit** transaction,
so a failure leaves every original impl and V-ownership gate unchanged.

The eager orchestration, install, and non-NVFP4 kernel tests pass locally. The block-16 finalization,
on-read tail selection, and canonical cache-value conformance matrix require SM89+/B200 and must be
rerun there before the new lifecycle is considered hardware-validated; earlier B200/GB300 evidence
used the superseded 128-token implementation.

Earlier GB300 end-to-end serving confirmed the impl-swap and both backend integrations, but used the
superseded 128-token lifecycle. Prefill, decode, chunked-prefill, and decode latency must be rerun for
this block-16 implementation.

Remaining (Gate B/C follow-ups): CUDA-graph capture validation for the retained path (the E2E above
ran under `--enforce-eager`); a framework-neutral numerical spec + immutable per-layer plan consumed
at runtime (so serving reads plans, not mutable quant modules); calibrate skip-softmax on the
**quantized** model; real packed-FP4 KV-cache memory savings.
