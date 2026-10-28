# State quantization alignment between training and serving

GDN/KDA state QAT must reproduce the state consumed by serving. This requires
matching the quantization boundaries, scale groups, native arithmetic, and output
read timing. The implementation combines a native chunked prefix with a recurrent
suffix. Each stage runs one native forward; prefix backward uses upstream FLA
helpers, and decode uses an explicit recurrence backward.

## Why chunk-only quantization does not match decode

For one head, the recurrent state has shape `[K, V]`: key channels by value
channels. Each token updates this state and reads an output. Quantize/dequantize
(QDQ) rounds the state; the training API retains floating-point values.

Let `F_t` be token `t`'s update and `Q` be QDQ. Quantizing once at a chunk boundary
and quantizing every decode state generally give different results:

$$
Q(F_2(F_1(S))) \ne Q(F_2(Q(F_1(S)))).
$$

The second decode update consumes the first update's rounded state. Chunk-only
QDQ omits that intermediate perturbation. For a toy update `S = S + 0.6`, starting
at zero and rounding to the nearest integer:

| Event | QDQ after two tokens | QDQ between tokens |
| --- | --- | --- |
| First working state | 0.6 | 0.6 |
| State consumed by token 2 | 0.6 | 1.0 |
| Second working state | 1.2 | 1.6 |
| Rounded state for the next token | 1 | 2 |

This is an illustration of rounding placement, not the actual INT8 scaling rule.
A straight-through estimator (STE) changes the backward approximation; it cannot
repair this forward mismatch.

Matching the QDQ schedule alone is insufficient. BF16 casts, reduction order,
normalization, and gate evaluation can change values near quantization thresholds.
An algebraically equivalent FP32 recurrence can follow a different rounded
trajectory. Training therefore imports the selected serving forward kernels.

## Match each training phase to serving

All tokens come from the dataset through teacher forcing. A per-sequence boundary
selects native prefill followed by native recurrence:

```text
prompt tokens                         completion tokens
[native chunked prefill] -> encoded handoff -> [native token updates]
       training prefix                           training suffix
```

| Phase | Training behavior | Serving behavior being reproduced |
| --- | --- | --- |
| Fresh prefix | Native chunked prefill without internal state QDQ | One native prompt-prefill call |
| Continuation prefix | Encode its incoming nonzero state | Quantize the state supplied to a continuation-prefill call |
| Handoff | Encode final prefix state when the suffix is nonempty | Quantize incoming state before the first decode token |
| Suffix | Native updates with state QDQ after every token | The selected decode/cache implementation |

The prefix stays chunked because it represents serving prefill. Adding QDQ after
every 64-token training chunk would introduce boundaries absent from a single
native prefill call. A prompt split across multiple serving prefill calls requires
matching each actual incoming-state boundary; one prefix length alone does not
encode the scheduler's call sequence.

`linear_attention_training_phase(model, prefill_lengths)` supplies the boundary.
Keep it active through forward and backward so activation recomputation sees the
same split. The example masks QAT/QAD loss to the suffix; gradients remain
connected through the handoff into the prefix. An all-prefix sequence has no
suffix handoff QDQ and supplies no suffix training signal.

## Why state QDQ before and after an update can align

For ordinary token-state QDQ, training stores rounded state after each update;
a state-only serving adapter can round its floating cache before the next call.
These placements align when they apply the same deterministic QDQ once to the
same working state. The current output is read before checkpoint rounding.

Let `S_P` be the prefix's final state, `C_t` the training carry, `R_t` the serving
cache, and `U_t` the working state. Training computes:

$$
C_0 = Q(S_P),\qquad
U_t = F_t(C_{t-1}),\qquad
o_t = \operatorname{read}_t(U_t),\qquad
C_t = Q(U_t).
$$

Serving computes:

$$
R_0 = S_P,\qquad
U'_t = F_t(Q(R_{t-1})),\qquad
o'_t = \operatorname{read}_t(U'_t),\qquad
R_t = U'_t.
$$

Initially `C_0 = Q(R_0)`. If the token updates consume equal states, identical
native arithmetic gives `U_t = U'_t`, equal outputs, and `C_t = Q(R_t)`. The
invariant therefore holds for the next token. Raw stored caches need not match;
the states consumed by the recurrence must match.

This argument requires matching inputs, initial states, QDQ format/grouping,
runtime settings, and update/readout arithmetic. It does not justify applying
QDQ twice or comparing different kernel specializations.

## INT8 state QDQ and backward

State QDQ uses TensorQuantizer. The block32 INT8 recipe uses one scale per key
row and 32 value channels and encodes at handoff and every suffix token.

| Stage | Forward | Backward |
| --- | --- | --- |
| Prefix | One native vLLM chunked prefill | Reconstruct intra-chunk factors, then upstream FLA backward |
| Suffix token | One native vLLM decode | Explicit recurrence adjoint |
| State QDQ | ModelOpt TensorQuantizer | Identity STE |

Backward reconstruction avoids retaining a second full FLA forward graph. Decode
also has no second Torch recurrence in forward. These first-order approximations
propagate gradients through the prefix, handoff and recurrent writes; they are
not exact derivatives of floating-point rounding or a discontinuous quantizer.

Matching the selected serving trajectory does not establish the best QAD training
strategy. A controlled comparison with chunk-boundary QDQ training is deferred.

## Validation results

The single-forward refactor passed 27 GPU tests on an H100 with vLLM 0.30.0 and
FLA 0.5.1, including native output/state equality, independent backward
comparisons, handoff gradients and padding. A fresh-cache run completed those
tests in 9m32s; this is test runtime, not model throughput.

The current public profile targets vLLM 0.30.0 and FLA 0.5.1. Twelve focused
GDN/KDA kernel tests passed on an RTX A6000 for the parent implementation. They
cover native output/state agreement, QDQ placement, handoff gradients, raw-gate
requirements, and BF16 projection inputs. This is kernel-level evidence, not a
completed Bridge training or serving-engine benchmark.

Both GDN Bridge example tests passed on an RTX A6000 with the dependency pair
listed in the [example requirements](README.md#requirements). They use a tiny
random Qwen3.5 model and mock data, with one optimizer step followed by one
resumed step.

| Check | QAT | QAD |
| --- | --- | --- |
| Enabled INT8 state QDQ and student weight update | Passed | Passed |
| Save, exact weight/policy restore, and another optimizer step | Passed | Passed |
| HF export, Transformers weight loading, and Megatron reimport with metadata | Passed | Passed |
| Effective parameters, logits, and attention gradients after HF round trip | Exact | Exact |
| Frozen, unquantized teacher | Not applicable | Passed |

HF weights alone do not enable state quantization in Transformers or vLLM. The
exact numerical round-trip check restores ModelOpt metadata after reimporting
those weights into Megatron.

Older vLLM results do not qualify this updated workflow. KDA Bridge training
also requires a provider for the chosen checkpoint.

Kernel agreement does not establish full-model serving equivalence. Model
conversion, projections, convolution, normalization, engine-managed multi-call
prefill, cache-slot reuse, and scheduler batching still need matched integration
tests. Pretrained-model quality recovery, multi-GPU scaling, and training
throughput also require separate validation. Long recurrent suffixes can be slow
with the current Python orchestration and Torch backward implementation.
