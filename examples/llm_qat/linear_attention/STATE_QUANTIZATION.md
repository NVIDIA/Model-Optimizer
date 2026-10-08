# State quantization alignment between training and serving

GDN/KDA state QAT must reproduce the state consumed by serving. This requires
matching the quantization boundaries, scale groups, native arithmetic, and output
read timing. The implementation combines a native chunked prefix with a recurrent
suffix and supplies a differentiable Torch adjoint for training.

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
ReplaySSM also reconstructs state from a checkpoint and a weighted BF16 update
ring. An algebraically equivalent FP32 recurrence can follow a different rounded
trajectory. Training therefore imports the selected serving forward kernels.

## Match each training phase to serving

All tokens come from the dataset through teacher forcing. A per-sequence boundary
selects native prefill followed by native recurrence:

```text
prompt tokens                         completion tokens
[native chunked prefill] -> encoded handoff -> [native token/replay updates]
       training prefix                           training suffix
```

| Phase | Training behavior | Serving behavior being reproduced |
| --- | --- | --- |
| Fresh prefix | Native chunked prefill without internal state QDQ | One native prompt-prefill call |
| Continuation prefix | Encode its incoming nonzero state | Quantize the state supplied to a continuation-prefill call |
| Handoff | Encode final prefix state when the suffix is nonempty | Quantize incoming state before the first decode token |
| Suffix | Native updates with token writes or replay checkpoint refreshes | The selected decode/cache implementation |

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

## INT8, Hadamard, and ReplaySSM

Ordinary state QDQ uses TensorQuantizer. The block32 INT8 recipe uses one scale
per key row and 32 value channels and encodes at handoff and every suffix token.

The native ReplaySSM profile instead rotates groups of 32 value channels into a
Hadamard basis, then scales, rounds to INT8, and reconstructs the checkpoint.
Hadamard rotation and quantization are separate operations. Checkpoints use a
scale per key row and 32-value group, with FP16 scale metadata.

With `replay_window=1`, every token refreshes the checkpoint. Larger windows keep
that checkpoint and append each native key/update vector once in BF16, then
refresh at the window boundary. GDN uses scalar decay; KDA's channel-dependent
decay acts on the key axis while Hadamard rotates the value axis. The update ring,
rounding, and checkpoint schedule all need to match serving.

The adapter imports the native encoder and recurrent kernels from a compatible
quantized-ReplaySSM fork. These interfaces, including KDA vector-gate support,
are separate from the public vLLM profile and ordinary state-only fake quantization.
Temporary native cache buffers protect tensors needed by backward. The persistent
training carry remains floating fake-QDQ data.

The Torch adjoint uses saved forward values and identity STE through casts and
QDQ. This is a training approximation to rounding, with gradients through the
prefix, handoff, token writes, and checkpoint refreshes. It is not an exact
derivative of the discontinuous quantizer.

## Validation results

The following checks used RTX A6000 GPUs. The replay and training matrix used
Torch 2.9.1, FLA 0.5.1, public vLLM 0.15.1 prefill, and compatible native ReplaySSM
sources with local KDA extensions. Results describe the tested implementation
snapshots and environments.

| Check | Result |
| --- | --- |
| 22 GDN/KDA native replay cases, including windows 1/4/16, nonzero states, partial prefixes, longer sequences, and QDQ-off controls | Exact outputs, checkpoint values/scales, and BF16 ring entries; finite gradients into nonempty prefixes |
| Four split/resume cases with Q/K normalization | Exact outputs, final states, and gradients; empty calls perform no write |
| Four GDN Bridge runs: QAT/QAD with Hadamard token mode and replay window 4 | One-step student updates; QAD teachers remain frozen and unquantized |
| Four KDA Megatron layer runs with training or distillation losses | Student updates and gradients into the prefix; separate from Bridge trainer validation |
| Public-vLLM GDN/KDA checks on 0.15.1, 0.20.0, and 0.30.0 | Exact native output/final-state comparisons, QDQ placement/effect, and gradients in the focused kernel tests |

The Bridge checks used the example's training entry point, tiny random weights,
and mock data, including activation recomputation and the distributed optimizer.
The tested Bridge provider supported GDN; KDA requires its own compatible
provider. Matching versions and arithmetic settings remains necessary even where
more than one runtime version passes kernel tests.

These checks resolve the reproduced kernel/cache mismatch for the tested cases.
They do not establish full-model serving equivalence: model conversion,
projections, convolution, normalization, engine-managed multi-call prefill,
cache-slot reuse, and scheduler batching still need matched integration tests.
Pretrained-model quality recovery, multi-GPU scaling, and training throughput
also require separate validation. Long recurrent suffixes can be slow with the
current Python orchestration and Torch backward implementation.
