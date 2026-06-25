# Design: Attention Quantization + Sparsity (vLLM serving)

Status: finalized 2026-06-24. Target: PR consolidating the sparse-attention serving
stack (#1622/#1624/#1634/#1635) onto current `main`, adding NVFP4/FP8 attention-operand
quantization that composes with attention sparsity.

## Goal

Serve a checkpoint that is **both attention-quantized and attention-sparse** under
vLLM: fake-quantize the attention BMM operands (Q, K, V, and softmax P) and run
sparse attention (N:M sparse softmax / skip-softmax) in the **same** attention op —
driven entirely by the exported ModelOpt `*_bmm_quantizer` config (no env knobs).

## Key decision

Reuse ModelOpt's existing machinery rather than a bespoke in-kernel quantizer:

| Operand | Where it is quantized | Mechanism |
|---|---|---|
| **Q** | `_QuantVLLMAttention.forward` (pre-step, on the tensor) | `q_bmm_quantizer` |
| **K, V** | `_QuantVLLMAttention.forward` (pre-step, on the tensor) | `k/v_bmm_quantizer` → vLLM writes the quantized K/V into the paged cache |
| **P** | inside the sparse Triton kernel | `P_QDQ` (PR #1757), driven by `layer.p_bmm_quantizer` |
| **Sparsity** | inside the sparse Triton kernel | existing `sparse_kw` (N:M, skip-softmax) |

Q/K/V are quantized as a pre-step on the tensors (the standard `_QuantVLLMAttention`
path, already used by the fakequant serve). P is the only operand that exists only
inside the attention kernel, so it must be quantized there — which is exactly what
its in-kernel `P_QDQ` (added in PR 1757) does. Sparsity also lives in the kernel,
so they compose.

## Why it composes (verified against vLLM v1)

`FlashAttentionBackend.forward_includes_kv_cache_update = False` (split mode), and the
ModelOpt sparse impl/backend inherit that plus the inherited `do_kv_cache_update`.
The runtime chain for one layer:

```text
_QuantVLLMAttention.forward(q, k, v):
    q, k, v = q/k/v_bmm_quantizer(q, k, v)     # modelopt TensorQuantizer (pre-step)
    super().forward(q, k, v)  ==  vLLM Attention.forward:
        unified_kv_cache_update(k, v)  -> do_kv_cache_update -> reshape_and_cache_flash
                                       # writes the QUANTIZED k, v into the paged cache
        unified_attention(q, k, v)     -> ModelOptSparseAttentionImpl.forward
                                       # reads the quantized cache; sparse kernel + P_QDQ
```

Consequences:
- **KV-cache quant is "on-write" for free**: each token's quantized K/V is written
  to the cache exactly once (per-token scale), and the kernel reads it with no
  per-step re-quant. This makes a dedicated `fake_quant_kv_onwrite` kernel and a
  `per_page_scale` policy unnecessary (the per-write scale also avoids the
  frozen-scale saturation that per-page existed to fix).
- **No double-quant**: the module quantizes Q/K/V, the kernel quantizes only P.

## Serve-path wiring (one worker)

1. Run the ModelOpt quantizer restore (reuse the FakeQuant prolog): attention layers
   become `_QuantVLLMAttention` carrying `q/k/v/p_bmm_quantizer`; linears are
   quantized if the checkpoint has them.
2. Swap `.impl` -> `ModelOptSparseAttentionImpl` on those layers.
3. Order: quant restore **then** impl swap (the `_QuantVLLMAttention` layer stays
   `isinstance(Attention)` in-place and keeps its `.impl`, so the swap still matches).

The sparse impl reads `layer.p_bmm_quantizer` at forward time and maps it to a
`p_qdq` mode/amax exactly like `_QuantAttention._p_qdq_mode()`; Q/K/V need no kernel
work (already quantized upstream / in the cache).

## Build / change / delete

Build:
- Port #1757's `P_QDQ` into the decode kernel (`decode_attention.py`); today it is
  prefill-only in `triton_fa._attn_fwd`.
- `ModelOptSparseAttentionImpl`: read `layer.p_bmm_quantizer` -> `p_qdq`/`p_qdq_amax`,
  thread into the prefill (`triton_attention`) and decode calls.
- Unified serve worker (quant restore + sparse impl swap).
- Un-skip `p_bmm_quantizer` in `vllm_reload_utils._convert_key_for_vllm` and add a
  `p_bmm_quantizer` slot to `_QuantVLLMAttention`.

Delete:
- `kernels/quantization/attention/nvfp4_fakequant.py`, `softmax_fakequant.py`.
- `fake_quant_kv_onwrite` + `per_page_scale` + device-scale plumbing in
  `decode_attention.py`.
- In `triton_fa.py`: our `NVFP4_Q/K/V/P`, `MIXED_FP16`, `PER_PAGE_SCALE`,
  `*_global_scale`, `_resolve_softmax_modes` — revert to main's `P_QDQ` only
  (keep the orthogonal fixes: int64 pointer cast, autotune-key bucketing,
  `dense_recent_tokens=128`).
- `plugins/vllm.py`: `parse_attn_quant_env` + the env / global-scale / on-write path.
- The two root scripts and the untracked `test_onwrite_*` tests.

Keep:
- The sparse-attention chassis (already on main) + the decode kernel (sparse decode)
  - paged calibration.

## Numerics & scope

- Q/K/V via `TensorQuantizer`; P via #1757's shared `fp4_round_magnitude` (RNE).
  **No mni/attnOpt bit-exactness** (the research-only mixed-precision softmax datapath
  is also dropped).
- Linear-layer quant rides along via the quant restore; "attention-only" quant is a
  recipe that enables only the `*_bmm_quantizer`s.

## Open implementation detail (not a blocker)

The unified worker must make the quant restore and the sparse impl swap agree on the
same `Attention` class across vLLM versions (the worker has a legacy-vs-current import
shim). Runtime composition is verified; this is a setup-time wiring check.
