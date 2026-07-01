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
*memory* savings are a downstream deployment concern (see "On-write V, deferred").
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
| V | 2 | keys | **on-write bake (decode) / on-read (prefill)** (`v_bmm_quantizer` → `_v_qdq_nvfp4`) | no |
| P | 2 | keys | **in-kernel** (`p_bmm_quantizer` → `_p_qdq_nvfp4`) | no (P is transient) |
| skip | – | – | in-kernel tile selection on the quantized scores | – |

Principle: quantize each operand once, on its correct axis, as close to the
write as that axis allows; the operands whose axis a per-token write cannot form
(P always; V at decode) are handled in/around the kernel.

- **Q, K** stay on the `_QuantVLLMAttention` pre-step: head_dim is present per
  token, so the pre-step is quantize-once-at-write for free and reuses standard
  ModelOpt machinery with stock vLLM cache writes.
- **P** is fake-quantized **in-kernel** by `_p_qdq_nvfp4` (plain max, P ≥ 0;
  16-blocks along keys). It is transient, so in-kernel is its only home.
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

- **Bake** complete 16-key V blocks once, in place in the paged cache
  (`fake_quant_kv_onwrite` V-path, driven by `v_bmm_quantizer`). Prefill bakes the
  prompt so decode inherits a pre-quantized cache; decode bakes each newly-complete
  block.
- The decode kernel reads complete blocks **as-is** (`V_CACHE_QUANTIZED`) and
  re-FQ's only the trailing `s mod 16` tile via `_v_qdq_nvfp4` → `O(S)` total.
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

## Code layout

```text
quantization/attention/p_qdq.py     _p_qdq_nvfp4 (P), _v_qdq_nvfp4 (V) — BMM2 helpers
quantization/common/nvfp4_quant.py  nvfp4_scalar_qdq (elementwise primitive)
common/attention/triton_fa.py       prefill kernel: P_QDQ + V_QDQ constexprs
common/attention/decode_attention.py paged decode kernel: P_QDQ + V_QDQ + skip
sparsity/attention_sparsity/plugins/vllm.py
    _p_qdq_from_layer / _v_qdq_from_layer  read p/v_bmm_quantizer -> (mode, amax)
    ModelOptSparseAttentionImpl.forward     threads p_qdq + v_qdq + skip into both kernels
```

## Build status

Implemented: paged decode kernel + skip-softmax; in-kernel `P_QDQ` and `V_QDQ`
helpers in both prefill and decode; plugin wiring that drives P/V quant from the
exported `p/v_bmm_quantizer` and engages the kernel for quant-only launches.
**Decode V on-write** (`fake_quant_v_onwrite` + `V_CACHE_QUANTIZED` gating + graph-safe
`(batch, n_kv, 1)` grid): prefill bakes the prompt's complete tiles after its on-read
attention; decode bakes each newly-complete tile and reads complete tiles as-is,
re-FQ'ing only the trailing tile → `O(S)`. **Validated on B200** as exactly Option 3
(bake == on-read FQ bit-for-bit; incremental == single-shot cache; output diff ≤~1e-5
is fp32-reduction scheduling, not a value change).

Remaining:

- Un-skip `p_bmm_quantizer` / `v_bmm_quantizer` in the vLLM reload + add their
  slots to `_QuantVLLMAttention`, so the exported config reaches the served layer.
- Unified serve worker: quant restore (gives `_QuantVLLMAttention` with Q/K
  pre-step quant) then swap in `ModelOptSparseAttentionImpl`.
- Calibrate skip-softmax on the **quantized** model (the skip decision sees
  quantized `Q·K` scores).
- GPU validation of the full quant + skip path on B200; README / CHANGELOG.
