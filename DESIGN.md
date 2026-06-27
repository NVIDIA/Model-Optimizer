# Attention quantization + skip-softmax sparsity (vLLM)

## Goal

Serve a checkpoint that is **both attention-quantized and skip-softmax-sparse**,
composed in a single attention pass (prefill *and* decode), driven entirely by
the exported `*_bmm_quantizer` / sparse-attention config — no `MODELOPT_ATTN_*`
env knobs. Skip-softmax selects which KV tiles to compute; quantization controls
how the attention operands are represented. The two are orthogonal in code and
compose in one Triton kernel launch.

This is a **fake-quantization accuracy study**: operands are rounded to the
NVFP4/FP8 grid and dequantized to bf16 for the matmul. Real KV-cache *memory*
savings are a downstream deployment concern (see "On-write V, deferred").

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

## V: on-read for prefill, on-write for decode (required)

The keys axis means V cannot be quantized by a per-token write, so it is
fake-quantized around the kernel. But the *cost* differs sharply by phase:

- **Prefill** is a single pass that touches each tile once → on-read FQ is
  `O(S)`. Fine; the kernel FQs V tiles as it reads them.
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

Because written tokens are immutable, on-write is **bit-identical** to on-read for
every complete block (same 16 elements → same amax → same result) and the trailing
tile is handled identically — so this is a **pure speedup, no accuracy change**.
K escapes the problem entirely: the pre-step quantizes each K once. (This is
"Option 3" of the trailing-block methods: quantize the tail from pristine bf16
each step — no accumulation, uniform-precision kernel.)

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

Remaining:

- **Decode V → on-write** (the `O(S²)` fix above). Port the `fake_quant_kv_onwrite`
  V-path + `V_CACHE_QUANTIZED` boundary gating + graph-safe decode grid from
  `kaix/sparse_attn_integration` onto this config-driven chassis (V-only; K/Q stay
  pre-step, P stays straight). The current in-kernel `V_QDQ` stays as the
  **prefill** path and as the **trailing-tile** FQ; the decode kernel flips from
  "FQ every tile" to "read baked blocks + FQ the trailing tile." Numerically
  identical, so no accuracy rework.
- Un-skip `p_bmm_quantizer` / `v_bmm_quantizer` in the vLLM reload + add their
  slots to `_QuantVLLMAttention`, so the exported config reaches the served layer.
- Unified serve worker: quant restore (gives `_QuantVLLMAttention` with Q/K
  pre-step quant) then swap in `ModelOptSparseAttentionImpl`.
- Calibrate skip-softmax on the **quantized** model (the skip decision sees
  quantized `Q·K` scores).
- GPU validation of the full quant + skip path on B200; README / CHANGELOG.
