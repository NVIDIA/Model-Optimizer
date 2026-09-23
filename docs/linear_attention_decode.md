# Experimental GDN/KDA decode-aware QAT

This extension models token-state writes, rounded KDA log retention, and encoded-update replay during training. Select the exact-prefix backend, an explicit decode policy, and a prefix length per sequence. The default configuration remains unchanged.

```python
import torch
import modelopt.torch.quantization as mtq
from modelopt.torch.quantization.linear_attention import linear_attention_training_phase

mtq.quantize(model, {
    "quant_cfg": [
        {"quantizer_name": "*", "enable": False},
        {"quantizer_name": "*kda_state_quantizer", "cfg": {
            "num_bits": 8, "unsigned": False, "narrow_range": True,
            "type": "dynamic", "axis": (0, 1),
        }},
    ],
    "algorithm": None,
    "linear_attention": [{"module_name": "*", "cfg": {
        "backend": "matmul",
        "state": {"block_v": 64},
        "decode": {
            "mode": "replay", "implementation": "triton", "readout": "stored",
            "replay": {"window": 8, "factor_qdq": True, "encoding": "once"},
        },
    }}],
})
# One sequence with 64 prefix tokens; all remaining tokens use the decode policy.
# Keep phase metadata active during activation-checkpoint recomputation.
with linear_attention_training_phase(model, [64]):
    with torch.autocast("cuda", dtype=torch.bfloat16):
        loss = model(input_ids=ids, labels=labels, use_cache=False).loss
    loss.backward()
```

For GDN, use `*gdn_state_quantizer`. This example selects INT8 state and replay anchors.
Use `num_bits=(4, 3)` to retain FP8 E4M3 state; replay factors have their own FP8 toggle. Policies persist through ModelOpt save/restore; per-batch phase lengths are runtime metadata and must be supplied for each workload. The context restores previous lengths on exit, supports nesting, and never patches process-global functions. Concurrent forwards on the same model instance with different phase contexts are unsupported.

## Numerical contract

For activated input tensors, a token computes `D = exp(g)`, `R = D * S`, `u = beta * (v - k^T R)`, and `W = R + k u^T`. GDN uses one log retention per head; KDA uses one per key channel. The output reads either the working state `W` or the stored state, according to `readout`. All operations use FP32 working values (FP64 is supported by the Torch reference).

- **Token mode:** fake-quantize `W` at every state write when the module state quantizer is enabled. The next token consumes that stored value.
- **Decay approximation:** an optional positive `decay_log_step` rounds log retention to the nearest grid point before exponentiation. Its derivative is an identity straight-through estimator (STE); no hidden clamping is applied. Prefix gates use the exact, unrounded computation.
- **Replay mode:** maintain an anchor and ordered encoded rank-one updates. Encode each key per head and each update per value block, with optional FP8 E4M3 QDQ. Compute a new residual from the reconstructed current trajectory and the encoded key. At `window` updates, encode the current state as a new anchor and clear the buffer. Anchor QDQ follows the state quantizer toggle; factor QDQ has its own toggle.
- **Initial state:** `quantize_initial` applies once at the first nonempty decode handoff. Continuation and empty calls do not create extra writes. Prefix state QDQ is independently selected by `prefill_state_qdq`.

State scale grouping is one dynamic scale for each `[Dk, block_v]` head tile. Update scales span one value block; key scales span the complete key vector. Zero tiles use scale one. Codec scales are detached, and rounded values have identity STE gradients. State codecs support E4M3 nearest-even rounding (including subnormals) and signed
symmetric INT8. INT8 uses FP32 `amax / 127` scales, zero point zero, codes in
`[-127, 127]`, nearest-even rounding, and saturation. Replay key/update factors
retain their independent E4M3 or identity format. Key reductions use a fixed pairwise FP32 tree; this avoids changing recurrence rounding merely by changing the reduction schedule.

The public numerical APIs `recurrent_decode_reference` and `recurrent_decode` accept `state_format="fp8_e4m3"` (default) or `"int8"` and one aligned `[T,H,D]` sequence and return outputs plus a `LinearAttentionCarry`. It retains anchor values/scales, buffered key/update values/scales, log retentions, position, cursor, and a policy signature. Returned tensors preserve gradients through prefix handoff, refreshes, and continuation. Restart with a new carry to reset; incompatible policy, state format, or state shape is rejected.

The Torch reference can explicitly re-encode stored entries using `encoding="reencode"`. It re-encodes the already stored values and does not regenerate factors from a teacher trajectory. The fused implementation supports encode-once replay. Carrying its reconstructed state incrementally applies the same ordered transitions as replaying the fixed entries; it does not measure serving-time reconstruction cost.

## Training implementation and limits

The CUDA Triton implementation supports FP32 working inputs, key dimensions up to 128, scalar/channel gates, and value blocks 16/32/64/128 with masked tails. It saves a state every eight tokens and recomputes intermediate states in backward. This internal checkpoint interval does not change write cadence. Its custom backward supports first-order gradients; use the Torch reference for higher-order differentiation.

Direct prefill APIs accept `prefill_lengths`, grouped query/key heads, both state layouts, packed sequences, empty entries, and full/zero prefixes. Positive prefixes use exact chunk algebra; only their optional state-write QDQ is configurable. WY/operand QDQ is rejected for decode policies in this delivery. FLA KDA layer integration requires 0.5.1 and `use_cache=False`; serving cache objects are rejected. The numerical carry API provides explicit continuation, but this change does not implement a serving cache or export format. The original GDN chunk path has prior Megatron tensor-parallel qualification; the extracted decode path still needs distributed requalification; context parallelism remains rejected.

All encoded values remain floating tensors. This is training-time fake quantization and approximation, with no compressed storage or inference speed claim. Use `examples/llm_qat/linear_attention/benchmark_decode.py` for matched-policy training overhead and `decode_study_plan.json` for the fixed model-quality protocol. The exact inverse is the only supported prefix solve. INT8 model-quality and distributed training qualification remain pending.

The previous stacked implementation
[reported FP8 results](https://github.com/NVIDIA/Model-Optimizer/pull/2509).
Those results predate this exact-prefix extraction and do not qualify INT8 quality.
The vLLM cache adapter and multi-GPU serving validation remain separate D1 work.
