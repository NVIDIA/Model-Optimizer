# Experimental GDN/KDA decode-aware QAT

This extension models token-state writes, rounded KDA log retention, and encoded-update replay during training. Select the existing materialized prefill backend, an explicit decode policy, and a prefix length per sequence. The default configuration remains unchanged.

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
- **Decay approximation:** an optional positive `decay_log_step` rounds log retention to the nearest grid point before exponentiation. Its derivative is an identity straight-through estimator (STE); no hidden clamping is applied. Prefix gates retain the prefill policy.
- **Replay mode:** maintain an anchor and ordered encoded rank-one updates. Encode each key per head and each update per value block, with optional FP8 E4M3 QDQ. Compute a new residual from the reconstructed current trajectory and the encoded key. At `window` updates, encode the current state as a new anchor and clear the buffer. Anchor QDQ follows the state quantizer toggle; factor QDQ has its own toggle.
- **Initial state:** `quantize_initial` applies once at the first nonempty decode handoff. Continuation and empty calls do not create extra writes. Prefix state QDQ is independently selected by `prefill_state_qdq`.

State scale grouping is one dynamic scale for each `[Dk, block_v]` head tile. Update scales span one value block; key scales span the complete key vector. Zero tiles use scale one. Codec scales are detached, and rounded values have identity STE gradients. State codecs support E4M3 nearest-even rounding (including subnormals) and signed
symmetric INT8. INT8 uses FP32 `amax / 127` scales, zero point zero, codes in
`[-127, 127]`, nearest-even rounding, and saturation. Replay key/update factors
retain their independent E4M3 or identity format. Key reductions use a fixed pairwise FP32 tree; this avoids changing recurrence rounding merely by changing the reduction schedule.

The public numerical APIs `recurrent_decode_reference` and `recurrent_decode` accept `state_format="fp8_e4m3"` (default) or `"int8"` and one aligned `[T,H,D]` sequence and return outputs plus a `LinearAttentionCarry`. It retains anchor values/scales, buffered key/update values/scales, log retentions, position, cursor, and a policy signature. Returned tensors preserve gradients through prefix handoff, refreshes, and continuation. Restart with a new carry to reset; incompatible policy, state format, or state shape is rejected.

The Torch reference can explicitly re-encode stored entries using `encoding="reencode"`. It re-encodes the already stored values and does not regenerate factors from a teacher trajectory. The fused implementation supports encode-once replay. Carrying its reconstructed state incrementally applies the same ordered transitions as replaying the fixed entries; it does not measure serving-time reconstruction cost.

## INT8 state with Hadamard rotation

Set `decode.state_codec="int8_hadamard32"` with an INT8 state quantizer to enable
32-point orthonormal Sylvester Hadamard transforms along the **value** dimension.
This codec supports both token writes and replay anchors. The default `"tile"`
codec retains the existing FP8/INT8 behavior above.

```python
# Execution policy; pair with the INT8 quantizer attributes in the example above.
policy = {
    "backend": "matmul",
    "state": {"block_v": 64},
    "decode": {
        "mode": "replay",
        "implementation": "triton",
        "state_codec": "int8_hadamard32",
        "readout": "working",
        "replay": {"window": 8, "factor_qdq": False, "encoding": "once"},
    },
}
```

For key-first state `S`, the internal state is `S @ H`, with a block-diagonal
Hadamard matrix `H`. Inputs `v` and replay update vectors use the same basis;
queries, keys, and scalar/channel decay gates keep their original coordinates.
Outputs are transformed back. The first nonempty decode call transforms the
incoming state once; continuation retains the rotated anchor and updates without
another rotation or initial-state quantization. `carry.value_basis` records the
basis. `carry.reconstruct()` returns the **original** basis; passing
`original_basis=False` exposes the internal basis.

The state codec uses one scale per **key channel and 32 contiguous values**, even
when the execution tile `state.block_v` is 64 or 128. For each group, compute
`scale=max(amax/127, 6e-8)` in FP32, divide by that scale, round half ties away from
zero, and saturate codes to `[-127,127]`. Store the scale in FP16 and reconstruct
with that FP16 scale. Scale metadata has shape `[H,Dk,Dv/32]`. Rounding has
identity STE and scales are detached. Basis transforms use their actual linear
derivatives. Replay factor QDQ remains independently configurable; the example
disables it to isolate state/anchor quantization.

The Hadamard basis and INT8 scale/rounding conventions follow
[quantized-replayssm at 29c35508](https://github.com/mxinO/quantized-replayssm/blob/29c355086070d2c7973550d3670171af53433533/vllm/model_executor/layers/fla/ops/fused_recurrent_replayssm.py).
Model-quality evaluation of this ModelOpt implementation remains pending.

`Dv` must be divisible by 32, and `state.block_v` must be 32, 64, or 128. Prefix
computation remains in the original basis with the exact solve; this codec starts
at decode handoff and rejects `prefill_state_qdq=True`. An all-prefix or empty
sequence does not create a decode quantization event. The policy is saved and
restored with the model. A complete example is
`examples/llm_qat/linear_attention/configs/kda_decode_replay_int8_hadamard.json`.

## Training implementation and limits

The CUDA Triton implementation supports FP32 working inputs, key dimensions up to 128, scalar/channel gates, and value blocks 16/32/64/128 with masked tails. It saves a state every eight tokens and recomputes intermediate states in backward. This internal checkpoint interval does not change write cadence. Its custom backward supports first-order gradients; use the Torch reference for higher-order differentiation.

Direct prefill APIs accept `prefill_lengths`, grouped query/key heads, both state layouts, packed sequences, empty entries, and full/zero prefixes. Positive prefixes are evaluated together so prefill quantizer scale domains remain shared. FLA KDA layer integration requires 0.5.1 and `use_cache=False`; serving cache objects are rejected. The numerical carry API provides explicit continuation, but this change does not implement a serving cache or export format. Megatron GDN tensor parallelism is covered by dedicated tests; context parallelism remains rejected.

All encoded values remain floating tensors. This is training-time fake quantization and approximation, with no compressed storage or inference speed claim. Use `examples/llm_qat/linear_attention/benchmark_decode.py` for matched-policy training overhead and `decode_study_plan.json` for the fixed model-quality protocol. The exact inverse remains selected in these studies; the earlier rejected Neumann candidate is not enabled.

See [the qualification study](linear_attention_decode_study.md) for the fixed
model/data protocol, quality comparisons, and measured training overhead.
The [vLLM fakequant adapter](linear_attention_vllm.md) quantizes incoming recurrent
state before native prefill/decode calls. Its invocation-boundary cadence differs
from these training decode/replay policies.
