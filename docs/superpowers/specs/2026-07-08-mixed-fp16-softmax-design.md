# Mixed-FP16 Softmax Design

## Goal

Add the reference mixed-FP16 online-softmax path to the compact NVFP4 attention worker while preserving one public `QuantSparseAttnWorker` and the current fixed block-16 NVFP4 Q/K/P/V recipe.

The worker must select either the existing FP32 softmax or mixed-FP16 softmax before model installation and CUDA-graph capture. This is inference-only support for regular decoder self-attention through the existing FlashAttention and FlashInfer adapters.

## Configuration Boundary

Q/K/P/V formats remain `TensorQuantizer` configuration because they describe tensor QDQ operations. Mixed-FP16 softmax is a kernel compute policy: it changes the two online-softmax exponentials but is not a tensor QDQ format.

Use one compact-only environment selector:

```text
MODELOPT_ATTN_SOFTMAX_MODE=fp32
MODELOPT_ATTN_SOFTMAX_MODE=mixed_fp16
```

The default is `fp32`. Any other value fails before an attention layer is mutated. The launcher must forward the selector to Ray workers so every tensor-parallel worker installs the same static policy.

The worker resolves the value once and stores the normalized string in the existing immutable launch-policy snapshot:

```python
impl.quant_kw["softmax_mode"] = "fp32" | "mixed_fp16"
```

No additional worker class, softmax `TensorQuantizer`, generic per-point precision matrix, or integration-branch compatibility alias is added.

## Numerical Contract

`fp32` preserves current behavior.

`mixed_fp16` implements the reference path:

- Convert the input of `exp2(scores - new_max)` from FP32 to FP16 with round-to-nearest-even, execute native `ex2.approx.f16`, and return its FP16-valued result as FP32.
- Apply the same native FP16 operation to the online correction `exp2(old_max - new_max)`.
- Sum probabilities into the denominator in FP32 without another rounding step.
- Keep the denominator state, weighted-value accumulator, and matrix accumulators in FP32.
- Apply existing NVFP4 P QDQ after the mixed-FP16 probability computation and after the unquantized FP32 denominator sum.
- Keep split-K reconciliation in FP32. Mixed FP16 applies only inside each split's online-softmax loop.
- Leave Q/K/P/V QDQ, V-cache finalization, sparse masking, and skip-softmax decisions unchanged.

The mode is a Triton compile-time constant. It introduces no device scalar, host synchronization, or graph-time mutable state.

## Data Flow

```text
MODELOPT_ATTN_SOFTMAX_MODE
  -> worker validation and normalization
  -> impl.quant_kw["softmax_mode"]
  -> _ResolvedForward
  -> shared FlashAttention/FlashInfer _forward_modelopt path
  -> triton_fa.attention / decode_attention.attention_decode
  -> MIXED_FP16 constexpr
```

The mode counts as an active ModelOpt transform. A non-FP32 mode must therefore never delegate to the backend's native dense path, including FlashInfer mixed prefill/decode batches.

## API And Error Handling

The public prefill and decode kernel wrappers accept `softmax_mode="fp32"` and validate against the two supported values. Internal Triton kernels receive only a boolean `MIXED_FP16` constexpr.

Autograd with `mixed_fp16` raises `NotImplementedError`; backward recomputation is outside this inference-focused change. Existing autograd behavior for `fp32` remains unchanged.

Worker configuration is read before plan installation. Invalid configuration fails atomically, without converting modules, replacing implementations, or partially installing a policy.

## Testing

Keep checked-in coverage focused on behavioral contracts:

- Worker selection defaults to `fp32`, accepts `mixed_fp16`, rejects invalid modes atomically, and forwards the selector to Ray workers.
- FlashAttention and FlashInfer routes propagate the mode for prefill, decode, and FlashInfer mixed batches without native fallback.
- Prefill matches a mixed-FP16 reference while composing with NVFP4 Q/K/P/V and 2:4 sparsity.
- Fixed 32-split decode matches a split-local mixed-FP16 reference with FP32 split reconciliation.
- The native FP16 `ex2` helper covers representative finite inputs and masked negative infinity.
- Mixed-FP16 autograd is rejected before launch.
- Default `fp32` routing and numerics remain unchanged.

CUDA-graph support is claimed only after a capture/replay smoke confirms that the static mode is preserved.

## Non-Goals

- Porting the integration branch's `MODELOPT_ATTN_FP16_SOFTMAX` alias or generic `MODELOPT_ATTN_SOFTMAX_QUANT` matrix.
- Supporting independently configurable DIFF, EXP2, DELTA, ALPHA, ACC, reciprocal, or output formats.
- Changing the fixed NVFP4 recipe, split count, tile sizes, sparse behavior, or supported vLLM attention envelope.
- Implementing mixed-FP16 backward kernels.
- Adding the design document to the final pull request.
