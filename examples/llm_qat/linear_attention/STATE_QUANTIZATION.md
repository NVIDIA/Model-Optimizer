# State quantization alignment between training and serving

State quantization changes the recurrent state consumed by later tokens. QAT must
reproduce serving's quantization boundaries, scale groups, and output-read timing
to train against the same numerical behavior. Applying QDQ every 64 tokens during
training does not reproduce applying it every decode token during serving.

The implementation uses a chunked prefix matching serving prefill, followed by a
recurrent suffix matching serving decode. Matching the QDQ schedule alone was
insufficient: small differences in prefill and token-update arithmetic crossed
INT8 rounding thresholds and accumulated into different states.

The opt-in `decode.precision="vllm_0_15"` profile in #2519 now uses pinned native
forward kernels imported from the pinned vLLM dependency, with a differentiable
Torch adjoint. No server is needed. Independent training and
serving executions match bitwise for GDN and KDA in the previously failing
257-token-prefix/256-token-suffix cases and a new-seed 129/512 check. Nonzero-state
512-token decode also matches exactly. These results resolve the reproduced
kernel/state-quantization mismatch for the pinned runtime and tested settings;
they do not establish full-model checkpoint equivalence or QAT/QAD quality recovery.
See [the native-arithmetic validation](#native-arithmetic-fix-in-2519-on-2026-10-05)
and [the direct-import rerun](#direct-vllm-imports) below.

## Why the mismatch happens

For one head, the recurrent state has shape `[K, V]`: key channels by value
channels. Each token updates this state and reads an output from it. QDQ means
quantize then dequantize: the rounded values remain in floating-point tensors.
This discussion does not imply compressed state storage or integer arithmetic
throughout the recurrence.

Without QDQ, chunked prefill and token recurrence implement the same state-update
algebra, apart from floating-point differences. Intermediate QDQ changes the
inputs of subsequent updates. Rounding only the final chunk state omits those
intermediate perturbations.

Consider the toy update `H = H + 0.6`, starting at zero. Round to the nearest
integer to illustrate the effect; this is not the actual INT8 scaling rule.

| Event | Two-token chunk, QDQ at the end | Decode, QDQ before each token |
| --- | --- | --- |
| First updated state | 0.6 | 0.6 |
| State consumed by token 2 | 0.6 | 1.0 |
| Second updated state | 1.2 | 1.6 |
| Rounded state for subsequent use | 1 | 2 |

Writing `F_t` for a token's update and `Q` for QDQ, the corresponding rounded
two-token states are generally different:

$$
Q(F_2(F_1(H_0))) \ne Q(F_2(Q(F_1(H_0)))).
$$

The mismatch exists in forward. A straight-through estimator (STE) supplies a
useful backward approximation through QDQ; it does not repair a different forward
trajectory.

## Proposed training behavior

Supply all tokens through normal teacher forcing, but split the attention
computation at the prompt/completion boundary. For example:

```text
160 training tokens, all supplied by the dataset

tokens 1–128                       tokens 129–160
[64-token chunk][64-token chunk] → [token][token] ... [token]
       prefill prefix       state       decode-aware suffix
                           handoff
```

The prefix follows serving prefill. The suffix performs a recurrent update and
state QDQ for every token, with the same format, grouping, and readout as serving.
No text-generation loop is needed. Labels remain teacher-forced; this example's
QAT/QAD loss mask selects the suffix. Gradients flow through the suffix and the
state handoff into the prefix. State QDQ uses identity STE.

`linear_attention_training_phase(model, prefill_lengths)` supplies the split to
converted layers. Keep the context active through backward when activation
checkpointing recomputes the forward. The split alone does not enable a quantizer
or select matching quantization settings.

### Why the prefix can stay chunked

Training prefix must match serving **prefill**; training suffix must match serving
**decode**. Serving does not normally decode the prompt one token at a time.

For a fresh prompt handled by one native prefill call, the current state-only
adapter quantizes the zero incoming state and then runs native prefill without
additional internal state QDQ. The training prefix should therefore use
`prefill_state_qdq=False`. Enabling QDQ after every 64-token training chunk would
introduce boundaries absent from this serving path.

A prompt split across multiple native prefill calls, or resumed from nonzero
state, requires QDQ at each actual incoming-state boundary. A prefix length by
itself does not describe that schedule. These cases need additional integration
and parity validation.

## How the decode states align

The following contract covers ordinary token-mode state QDQ. It assumes identical
quantizer behavior, inputs, initial state, and update/readout arithmetic. It does
not establish equivalence for the Hadamard or ReplaySSM policies.

| Event | Training | State-only serving |
| --- | --- | --- |
| Decode handoff | Quantize the prefix's final state | Quantize incoming cached state before the first decode call |
| Update | Consume rounded state | Consume rounded state |
| Output | Read working state, before write QDQ | Read native working state |
| Next-token state | Quantize updated state and retain it | Retain updated state; quantize before the next update |

Let `H_P` be the prefix's final state. Training retains a quantized carry `C`,
whereas serving retains an unrounded floating cache `R`. `W` denotes working state.
With `readout="working"`, training computes:

$$
C_0 = Q(H_P),\qquad
W_t = F_t(C_{t-1}),\qquad
o_t = \operatorname{read}_t(W_t),\qquad
C_t = Q(W_t).
$$

Serving computes:

$$
R_0 = H_P,\qquad
W'_t = F_t(Q(R_{t-1})),\qquad
o'_t = \operatorname{read}_t(W'_t),\qquad
R_t = W'_t.
$$

Initially `C_0 = Q(R_0)`, so the first updates receive identical states. Equal
working states give equal outputs and `C_1 = Q(R_1)`. The same argument applies to
each subsequent token. Quantizing after one update is equivalent to quantizing
before the next when the same deterministic QDQ is applied exactly once and no
intervening operation changes that state or the quantizer configuration.

Compare **states consumed after QDQ**, rather than raw caches: `C_t` and `R_t`
represent different points in the computation. Different native arithmetic can
still introduce rounding differences; runtime parity must use appropriate
tolerances and sufficiently long trajectories.

## Shared recipe and remaining gaps

The alignment target for a fresh, single-call prefill followed by plain token
decode is `quantize_initial=True`, `prefill_state_qdq=False`,
`readout="working"`, and matching state quantizer settings. Additional replay,
Hadamard, and gate-rounding policies must remain disabled for this comparison.
The opt-in recipes below implement this contract using standard TensorQuantizer
`block_sizes={-1: 32}`: one dynamic INT8 scale per 32 value channels of each key
row. The same grouping applies to training's `[H, K, V]` and serving's
`[N, H, K, V]` state tensors. Both use signed narrow-range INT8 and identity STE.

Training uses `general/ptq/linear_attention_state_int8_block32_dynamic`:

```python
import modelopt.torch.quantization as mtq
from modelopt.recipe import load_recipe
from modelopt.torch.quantization.linear_attention import linear_attention_training_phase

recipe = load_recipe("general/ptq/linear_attention_state_int8_block32_dynamic")
mtq.quantize(model, recipe.quantize.model_dump())
with linear_attention_training_phase(model, prefill_lengths):
    loss = compute_suffix_loss(model, batch)
    loss.backward()
```

On the vLLM branch, select
`examples/vllm_serve/linear_attention_state_int8_block32.yaml` with the existing
fake-quant worker. That adapter still applies only TensorQuantizer before native
prefill/decode; native kernels are unchanged. These additions are local changes
on the separate training and serving branches; downstream rebases remain deferred.

The training recipe selects `decode.precision: vllm_0_15`. This CUDA BF16 profile
covers both phases. It uses the pinned prefill and token-update arithmetic,
including reductions and exponentials, instead of approximating it with Torch
casts. The native FLA kernels are adapted into ModelOpt; training does not import
vLLM. FLA 0.5.1 supplies the compatible triangular solve and indexing helpers.
No state QDQ occurs inside the fresh prefill. A continuation prefill with an
incoming state quantizes that state once at entry, matching the serving adapter.

The Megatron adapter retains raw Q/K so prefill can normalize into BF16 while
decode normalizes inside the FP32 update. It also matches GDN's BF16 beta and
native gate activation; KDA retains FP32 beta and its native channelwise gate.
This profile requires Megatron's unfused input-preparation hook. Context
parallelism and fused Megatron input preparation remain unsupported. The kernel
envelope is `K <= 256`, equal K/V dimensions for KDA, and power-of-two K when
normalization is enabled.

Backward uses Torch operations evaluated along the actual rounded state
trajectory, with identity STE at operand-rounding and TensorQuantizer boundaries.
The prefix preserves its native cumulative gates, inverse, WY factors, incoming
chunk states, and updated values. The suffix differentiates its update at the
actual incoming rounded state and uses the native working state for readout.
This is an explicit training surrogate, not a native vLLM backward kernel.

`precision: full` retains the FP32/FP64 Torch reference. Other existing recipes
and default policies retain their behavior:

- **Readout:** training defaults to `readout="stored"`, which reads
  after state QDQ. The state-only serving schedule above needs working-state
  readout, selected explicitly by the new recipe.
- **Scale grouping:** the legacy training path applies the quantizer separately
  to `[K, block_v]` tiles. Serving calls it directly on `[N, H, K, V]`. With
  `axis=(0, 1)`, serving uses one scale per entire head state, whereas training
  uses multiple scales if `V > block_v`. The settings must describe the same
  logical groups; using the same quantizer class alone is insufficient. The new
  recipe supplies explicit value blocks and bypasses this legacy tile grouping.
- **Prefill and arithmetic:** the opt-in profile passes the native comparisons
  recorded below; the original full-precision prefix retains its measured
  discrepancy. Multi-call prefill still needs matching incoming-state boundaries.
  Matching an arithmetic profile does not imply bitwise agreement across all
  kernels, shapes, hardware, or versions.
- **Other codecs:** the default INT8 training recipe includes Hadamard rotation;
  the current state-only vLLM adapter does not implement that policy or ReplaySSM.
  The ordinary INT8 results below do not validate that default recipe.

The relevant training code is
[decode.py](../../../modelopt/torch/quantization/linear_attention/decode.py),
[training.py in #2519](https://github.com/NVIDIA/Model-Optimizer/pull/2519),
and [utils.py](../../../modelopt/torch/quantization/linear_attention/utils.py).
The serving adapter is on the separate
[vLLM PR](https://github.com/NVIDIA/Model-Optimizer/pull/2541).

## Recorded numerical results

### CPU state schedule check on 2026-10-05

The [reproduction script](check_state_quantization.py) runs the actual Torch
`recurrent_decode` implementation against a small CPU recurrence that applies
`TensorQuantizer` before each update. The reference uses the same key-reduction
order to isolate QDQ placement and grouping. It does not load vLLM, run native
serving kernels, evaluate a model, or test gradients.

The fixed fixture uses FP32 on CPU, seed 514, 16 tokens, two heads, `K=16`, and
`V=128`. Quantization is dynamic signed narrow-range INT8 with `axis=(0, 1)` and
identity STE. Initial values in the first and second 64-channel regions are
scaled by 0.05 and 4, respectively, to expose grouping differences. Gates are
scalar for GDN and per-key-channel for KDA.

The initial nonzero state is generated directly, so this check isolates the
decode suffix; it does not validate prefill or the handoff from a real prompt.
Errors are maximum absolute differences across all 16 steps. State errors compare
the training carry with the serving reference's state after its next QDQ.

| Variant | GDN output error | GDN state error | KDA output error | KDA state error |
| --- | ---: | ---: | ---: | ---: |
| Matching groups, working readout | 0 | 0 | 0 | 0 |
| Matching groups, stored readout | 0.02254343 | 0 | 0.01966226 | 0 |
| Training uses two tiles, working readout | 0.05220260 | 0.25671503 | 0.05183376 | 0.21353874 |

For matching groups, the training `block_v=128` covers the complete head state,
matching the serving reference. The mismatched variant uses `block_v=64` only on
the training side. This choice establishes grouping equivalence for this fixture;
it is not a general fix for arbitrary value dimensions. Stored readout changes
outputs without changing the future state trajectory, explaining the zero state
errors in that row. Grouping differences change both outputs and future states.

[Saved results](state_quantization_results.json) include full-precision numbers,
the PyTorch version, and hashes of the imported ModelOpt source files. The run used
the local #2519 checkout based on `1da0f35a1584e9eccfcd14c50c706518ad1737b3`
with uncommitted TensorQuantizer/grouping changes. That commit alone does not
identify the tested implementation; use the recorded source hashes. The vLLM
source review used `864973e890b092216aa6094ae7ce9c5aa0083262` and was read-only.
The example branch has not been rebased onto those local training changes.

To repeat the check with the intended training checkout, run from the repository
root containing this example, using an environment with ModelOpt dependencies:

```bash
PYTHONPATH=/path/to/pr2519-checkout python \
  examples/llm_qat/linear_attention/check_state_quantization.py \
  --output /tmp/state_quantization_results.json
```

Confirm that the source hashes match before comparing with the recorded row.
Once the stack contains the same implementation, use `PYTHONPATH="$PWD"`.
The script reports differences; native runtime equivalence must be checked
separately.

### Initial native GPU checks on 2026-10-05, before the prefill profile

The pinned vLLM source (`930288170c31e8568290fff407dca8caf17d16ad`,
`0.15.2.dev16`) supplied the actual chunked and recurrent GDN/KDA kernels. The
production state-only adapter called those kernels. This was a kernel/adapter
check, without an LLM engine, scheduler, checkpointed language model, or quality
benchmark. It combined the local #2519 training implementation with the #2541
adapter without rebasing either branch.

The fixture used an RTX A6000, PyTorch 2.9.1+cu128, seed 514, batch 1, two heads,
`K=64`, and `V=128` for GDN or `V=64` for KDA. Q/K/V and beta were BF16; gates and
state were FP32. Q/K normalization happened once before both paths. A fresh
73-token prefix was followed by eight decode tokens. A separate eight-token
case started from an identical random nonzero state. Both paths used the shared
block32 recipe. The predeclared elementwise gate was `rtol=0.03, atol=0.003`.

| Initial condition for the suffix | GDN final consumed-state max error | KDA final consumed-state max error | Output/state gate |
| --- | ---: | ---: | --- |
| Independent Torch/native 73-token prefill | 0.00730869 | 0.00928746 | **Fail**: all eight consumed states and final state; outputs pass |
| Identical nonzero random initial state | 0.00215688 | 0.000000179 | Pass for eight tokens |
| Identical native prefill state supplied to both suffixes | 0.000000179 | 0.000000179 | Pass for eight tokens |

In every case, a separate one-token comparison reset Torch to that token's
incoming native state. All these local BF16 output comparisons were bitwise
identical. Training and serving quantizers also produced exactly equal rounded
states for the same input despite their different input ranks.

The shared-handoff diagnostic produced bitwise-equal suffix outputs across all
eight tokens for both architectures. It supports the explanation that prefill
arithmetic is responsible for the larger trajectory difference in this fixture.
It is not a training solution: copying the serving state bypasses the original
training prefix and does not establish its backward correctness.

The independent prefill outputs and unquantized handoff states passed the initial
gate, but their small differences crossed INT8 rounding thresholds. The resulting
state RMS differences remained below 1%; that does **not** override the failed
elementwise gate. No tolerance was relaxed, and no full trajectory parity claim
is made. Even the same-state recurrence can accumulate small floating-point
rounding differences, as the nonzero GDN case shows.

A small serving regression test additionally checks direct prefill states,
gathered decode slots, untouched inactive slots, and changes in active batch size.
This exposed a fixed-shape block-layout cache in TensorQuantizer. Dynamic
quantization now recomputes its layout, including padding; static quantization
retains its fixed-shape requirement. The fix lives in TensorQuantizer rather than
in a separate linear-attention quantizer.

### Initial Megatron QAT and cost check on 2026-10-05

A converted Megatron GDN layer and KDA layer each completed forward, backward,
and an SGD update with the shared recipe. Input and parameter gradients were
finite, projection weights changed, and distributed checkpoint save/restore
reproduced outputs exactly, including the quantizer grouping and decode policy.
After the checks completed, a checkpoint helper reported an NFS temporary-directory
cleanup error during shutdown; the parent exited successfully. The log is retained.
This is a layer-level QAT smoke check, not a run of the Bridge example, QATTrainer,
QADTrainer, distillation, or a model-quality comparison.

For a preliminary cost measurement, each case used 128 tokens, batch 1, hidden
size 64, BF16 autocast, a fixed loss over the last eight outputs, and SGD with
zero learning rate to preserve weights during timing. Times include forward,
backward, and optimizer work, after two warmup steps, with five synchronized
measurements. The baseline used the native unquantized chunked path; quantized
cases used the Torch prefix and suffix. Thus their difference includes the
prefix backend as well as recurrence/QDQ overhead.

| Path | GDN median step (ms) | KDA median step (ms) |
| --- | ---: | ---: |
| Native unquantized chunked | 13.72 | 18.66 |
| Torch prefix + 8-token QDQ suffix | 49.15 | 161.34 |
| Torch prefix + 16-token QDQ suffix | 53.47 | 160.97 |
| Torch prefix + 32-token QDQ suffix | 79.85 | 218.54 |

These are short single-layer measurements, not full-model throughput. Samples
and peak allocated memory are retained in the result artifact; variation and
prefix work prevent interpreting their differences as per-token kernel cost.
The Torch suffix is a correctness reference and needs fused forward/backward
work before claiming efficient training.

The [GPU result artifact](state_quantization_gpu_results.json) records summaries,
failed checks, timing samples, source hashes, and paths to the local reproduction
scripts and full logs. Revisions alone do not identify these uncommitted changes.
The earlier CPU artifact remains a separate historical experiment.

### Historical Torch prefill emulation on 2026-10-05

Tracing the intermediate tensors found two concrete sources of the discrepancy:

- GDN rounds `beta * key` to BF16 when both inputs are BF16, before multiplying
  by the FP32 gate. Rounding only the final gated product is a different operation.
  The inverse, WY factors, state operands, and updated values also have native
  BF16 rounding sites that the original full-precision prefix omitted.
- The pinned KDA kernel uses different computations within and between 16-token
  blocks. Its cross-block matrix products consume TF32-truncated operands. Merely
  adding BF16 casts, or switching to FLA 0.5.1, did not pass the original state gate.
  Intermediate traces isolated this difference; emulating TF32 truncation reduced it.

The earlier experimental `prefill_precision` profile handled these sites in the Torch prefix functions.
It was replaced by the native-arithmetic profile after the long-trajectory failure below. That
historical implementation imported no vLLM training dependency and changed no serving kernel. It preserved
beta's incoming dtype: native Qwen3Next GDN serving uses BF16 beta, while native
Kimi KDA serving computes beta in FP32. These checks supply identical prepared
inputs to both paths; full-model projection, normalization, gate-activation,
and checkpoint-conversion equivalence are separate requirements.

All rows below compare **independently computed** training and native prefill
states. No native state is substituted into training. The original elementwise
`rtol=0.03, atol=0.003` gate was retained for outputs and consumed states.

| Case | Beta dtype | Prefix + suffix | Final consumed-state max error | Final state relative RMS | All output/state gates |
| --- | --- | ---: | ---: | ---: | --- |
| GDN, seed 514 | bfloat16 | 73 + 8 | 2.09e-07 | 2.134e-05% | Pass |
| KDA, seed 514 | bfloat16 | 73 + 8 | 0.0037 | 0.08979% | Pass |
| GDN, seed 991 | bfloat16 | 129 + 16 | 0.0038 | 0.02386% | Pass |
| KDA, seed 991 | float32 | 129 + 16 | 4.53e-06 | 9.548e-05% | Pass |

The gate is `abs(training - native) <= 0.003 + 0.03 * abs(native)` per element;
an absolute difference above `0.003` can pass because of the relative term.

The seed-514 fixture also passed nonzero-initial-state decode and both BF16/FP32
beta input variants for each architecture. Its FP32-beta values were converted
from the original BF16 fixture to isolate dtype promotion. The seed-991 KDA case
uses newly generated FP32 beta values, with no BF16 conversion. Local single-token
output checks, reset to the same incoming native state, were bitwise equal in
the seed-514 cases. The longer GDN case had a maximum local difference of
`6.10e-5`, still within the gate. The complete trajectories pass tolerance;
they are not generally bitwise identical.

Backward validation keeps the rounding semantics explicit. Rounding uses identity
STE, and PyTorch differentiates the surrounding products and triangular solve.
Six directional finite-difference checks with frozen rounding offsets passed
`rtol=0.02, atol=0.003`; this validates the STE surrogate, not a derivative of
discontinuous rounding. An elementwise comparison to the unrounded FP32
recurrence's gradients failed for a few near-zero components and is retained as
a diagnostic, not reported as gradient equality. Two minimal tests check forward
agreement, finite gradients, and nonzero prefix-key gradients from a suffix loss.
The focused CPU suite passed 43 tests.

Both real Megatron attention layers also passed the shared-recipe optimizer and
distributed-checkpoint smoke checks with the new profile. The restored policy
includes `prefill_precision`, and restored outputs match exactly. This still
does not run the full Bridge example or establish QAT/QAD model-quality recovery.
The profile is a Torch numerical reference; extra casts and KDA's materialized
cross-block arithmetic add cost. The new artifact includes preliminary step times
using the same small layer workload as above; no throughput improvement is claimed.

[Prefill alignment results](prefill_alignment_results.json) retain the original
failed candidates, all eight native cases, STE checks, training measurements,
source hashes, and local reproduction paths. These are local changes on #2519
and the #2657 example branch; no downstream rebase was performed.

### Historical failure on extended trajectories on 2026-10-05

The short checks above were followed by a new seed (1701), a 257-token prefix,
and 256 decode tokens. Shapes, state grouping, and the original elementwise
`rtol=0.03, atol=0.003` criterion were retained. GDN uses BF16 beta; KDA uses
directly generated FP32 beta. **Both independent-prefill runs failed the state
criterion.** All 256 suffix output comparisons still pass for each architecture.

| Case | First failing consumed-state comparison | Final state relative RMS | Output criterion | State criterion |
| --- | ---: | ---: | --- | --- |
| GDN, independent prefill | After decode update 2 | 1.814% | Pass | Fail |
| KDA, independent prefill | After decode update 9 | 1.853% | Pass | Fail |
| GDN, native handoff substituted for diagnosis | After decode update 20 | 1.186% | Pass | Fail |
| KDA, native handoff substituted for diagnosis | After decode update 44 | 1.853% | Pass | Fail |
| GDN, state QDQ disabled | None | 0.00007598% | Pass | Pass |
| KDA, state QDQ disabled | None | 0.00007297% | Pass | Pass |

The native-handoff cases are diagnostic interventions, not proposed training
solutions. They show that prefill differences are not the only source: separate
Torch/native decode arithmetic can diverge even from the same initial handoff.
Local one-token checks reset to the same incoming native state still pass.
With state QDQ disabled, both complete trajectories pass and final-state maximum
errors are below `1.5e-6`. Together these controls support accumulation of small
arithmetic differences through INT8 rounding as a cause of the failed state gate.
They do not demonstrate a model-quality regression or quantify its severity.

Replacing the Torch state update with `addcmul`, moving query scaling before the
readout reduction, and subsequently changing `exp` to `exp2(g * log2(e))` were
tested as scratch candidates. Neither candidate passed both long shared-handoff
state checks. A `torch.compile` token-update candidate also failed the shared and
independent-handoff state checks. No candidate was installed into the production
training code, and the tolerance was not changed.

An actual one-layer Megatron GPT forward with the shared recipe exposed another
prepared-input difference. Its GDN beta is FP32; the native Qwen3Next gate returns
BF16. All 81 beta elements differed, with a maximum difference of `0.001946`;
casting the Megatron beta to BF16 made this fixture exactly equal. Log-decay
differences were below `1.9e-9`. Prefill Q/K normalization matched exactly, but
native decode normalizes inside its FP32 update, whereas Megatron passes
BF16-normalized Q/K. Holding all other inputs fixed, changing this normalization
site changed the one-token state by up to `0.000173`. Each local comparison
passes tolerance, but those inputs are not identical.

This audit runs a complete tiny Megatron model and native vLLM gate/normalization
helpers; it does not run a matched full vLLM model or checkpoint conversion.
The previous kernel fixtures deliberately supplied identical prepared inputs and
therefore did not cover these differences. Aligning gate dtype and normalization
must accompany any change to the recurrent arithmetic.

The [extended result artifact](extended_state_alignment_results.json) records the
failed cases, controls, candidate results, prepared-input audit, source hashes,
and local reproduction paths. The earlier passing short-case artifact remains
unchanged. Full-model Megatron/vLLM parity and QAT/QAD quality recovery remain
unvalidated. Passing the short cases was insufficient to claim the mismatch solved;
the replacement below also passes the formerly failing long cases.

### Native arithmetic fix in #2519 on 2026-10-05

The correctness fix stays in #2519. #2562 remains the follow-up for a more
efficient fused training backward. The failed Torch precision emulation was
removed. The serving plugin and its native kernels were not modified by this fix.

Both sides compute their own prefill and advance their own states. Validation
never substitutes a native handoff into training. Serving uses the #2541 adapter
with unmodified vLLM kernels at
`930288170c31e8568290fff407dca8caf17d16ad` (`0.15.2.dev16`). Training imports only
ModelOpt's adapted arithmetic and FLA. State quantization is the standard dynamic
signed narrow-range INT8 TensorQuantizer with `block_sizes={-1: 32}`.

| Case | Seed | Prefix + decode tokens | Outputs | Every consumed state |
| --- | ---: | ---: | --- | --- |
| GDN and KDA, original failing case | 1701 | 257 + 256 | Bitwise equal | Bitwise equal |
| GDN and KDA, raw Q/K normalization | 2903 | 129 + 512 | Bitwise equal | Bitwise equal |
| GDN and KDA, nonzero initial decode state | 2903 | 0 + 512 | Bitwise equal | Bitwise equal |
| GDN and KDA, K=128 and K=256, nonzero continuation prefill | 3871 | 129 + 256 | Bitwise equal | Bitwise equal |

All cases retain the original `rtol=0.03, atol=0.003` acceptance gate and pass the
stronger zero-error check. The validation host is an RTX A6000 with PyTorch
2.9.1+cu128, Triton 3.5.1, and FLA 0.5.1; other GPU/compiler combinations require
their own qualification. The fixture uses BF16 Q/K/V, FP32 gates/state, two heads,
K=64/128/256, V=128 for GDN and V=K for KDA. GDN beta is BF16; KDA beta is FP32.
Training's
stored carry is compared with serving's cache after its next TensorQuantizer
call, because the latter stores the raw update and quantizes before consumption.
Native defaults are retained:
`FLA_USE_FAST_OPS=0`, `USE_DEFAULT_FLA_NORM=0`, and `FLA_GDN_FIX_BT=0`.

Twelve directional finite-difference checks pass for the prefix and token-update
STE surrogates across both architectures (`rtol=1e-5, atol=1e-7`). The checks freeze
forward-rounding offsets, so they measure the declared identity STE rather than
an undefined derivative at quantization discontinuities. Each converted Megatron
layer also passes forward/backward, an SGD weight update, and distributed
checkpoint save/restore with bitwise-equal restored output. The KDA layer uses
selective activation recomputation, with the phase context kept active through
backward. These are layer-level training checks, not full QATTrainer/QADTrainer
quality experiments.

An additional comparison starts from the actual converted Megatron layers,
including their preparation hooks. With seed 2903 and 73 prefix + 64 decode
tokens, both GDN and KDA match native prefill output, handoff, all token outputs,
and every consumed state bitwise (133 GDN and 131 KDA comparisons). GDN's native
beta and log-gate outputs also match exactly. Each training layer makes 65 state
quantizer calls: the handoff plus 64 updates. Megatron Core is pinned at
`ac100f773f9d8fac918ce3e64698be4ed84fa428`. This compares the attention boundary,
not two independently loaded full models.

The permanent regression coverage remains two small GPU cases, one per model,
checking QDQ call count, changed suffix outputs, grad/no-grad forward equality,
finite gradients, and gradient flow from suffix loss into prefix keys. Compilation
runs in a module fixture outside the test-call timer. The focused CPU suite passes
42 tests, plus the dynamic TensorQuantizer shape regression.
[Native alignment results](native_state_alignment_results.json) contain
source hashes, per-case results, reproduction paths, and retained failed attempts.
Historical result files above are unchanged.

### Historical FLA reuse

The compatibility profile imports FLA 0.5.1 wherever its arithmetic can preserve
the serving results. ModelOpt keeps the TensorQuantizer calls, phase handling,
and training gradient adapter.

| Operation | Implementation |
| --- | --- |
| GDN and KDA token update | Imported FLA KDA recurrent kernel, with the serving tile size and launch settings. GDN repeats its scalar decay across key channels. |
| GDN cumulative sum | Imported FLA scalar scan. |
| GDN WY construction | Imported FLA helper; convert cumulative natural-log gates to log2 at its boundary. |
| KDA output | Imported FLA GLA output helper; scale and round Q before the call, and pass an output scale of one. |
| Triangular solve and sequence indexing | Already imported from FLA. |

This removes four copied Triton kernels and three files, reducing the `serving/`
directory from 2,448 to 1,797 lines. It does not change QDQ placement or the
identity-STE backward.

Some prefill adaptations remain because their FLA replacements failed the
numerical comparison. These include KDA's vector cumulative sum and
decay-weighted keys, prefill normalization, gate activation, and GDN's state and
output calculations. In particular, changing where a BF16 cast or gate scaling
occurs can change the next INT8 state even when the formulas are equivalent.
The shared state helper also retains FP32 intermediates for the gradient adapter.

The dependency remains pinned to FLA 0.5.1. Source review of releases 0.4.0 through
0.5.1 did not identify a complete compatible replacement: older releases lack
APIs used by ModelOpt's existing integration, and the reviewed releases scale
GDN's key-product matrix after the dot product, whereas the pinned serving path
rounds beta-weighted keys before it. Importing the full FLA chunk operator would
therefore require another alignment change, not just a version downgrade.

After the refactor, all ten trajectory cases above and both converted Megatron
layer comparisons still match bitwise: 11,042 output/state comparisons in total.
The twelve directional gradient checks, optimizer updates, and exact checkpoint
round trips pass for both architectures. The existing 43 CPU tests and two GPU
tests pass; no permanent tests were added. GPU test calls take 0.06 and 0.09
seconds after compilation in the existing fixture. These results have the same
RTX A6000 and pinned-runtime scope as the original checks.
[FLA reuse results](fla_reuse_results.json) record the new source hashes,
replacement probes (including rejected candidates), version audit, and reruns.
The original native-alignment receipt remains unchanged.

## Direct vLLM imports

The current implementation imports vLLM's forward kernels directly instead of
maintaining adapted copies. The refactor reduced `serving/` from nine files and
1,797 lines to three files and 227 lines before the version-check cleanup.
All copied vLLM Triton kernel bodies are removed.
ModelOpt keeps a small wrapper around the imported chunk-state kernel to save
FP32 chunk-start states and residual values for backward; the native output
kernels still receive BF16 casts. ModelOpt's Torch adjoint and TensorQuantizer
QDQ remain responsible for gradients and the quantization schedule.

Only `decode.precision="vllm_0_15"` requires the optional vLLM dependency. The
[example installation instructions](README.md#run-the-example) pin the public
`vllm==0.15.1` release. The version pin is maintained in the example requirements;
the runtime checks dependency availability. The shared
[`with_vllm_defaults.sh`](with_vllm_defaults.sh) launcher sets and logs the five
default FLA settings before starting training, native-kernel tests, or vLLM
evaluation. It replaces inherited overrides; the library no longer rejects
environment settings at import. Use it for the server/worker processes on every
node, rather than only an evaluation client. Save its stderr log with results;
matching settings does not replace numerical qualification for each GPU/runtime.
Training imports kernels without launching a vLLM server.
Training and serving share kernel code but compute their states independently.

The refactor passes the same validation on RTX A6000:

- All ten trajectory cases and two converted Megatron-layer comparisons remain
  bitwise equal: **11,042 output/state comparisons**.
- All twelve directional checks of the STE surrogate gradient pass; the largest
  absolute directional-derivative error is `3.76e-10`.
- GDN and KDA each complete a layer backward/optimizer update and an exact
  distributed-checkpoint/ModelOpt-state round trip.
- The 43 CPU tests and two GPU tests pass. The two existing GPU cases now live
  under `tests/gpu_vllm`; environments missing the required kernel API skip them.
  Compilation stays in the shared module fixture, with test calls taking 0.10
  and 0.09 seconds in this run.
- Optional-dependency and revision/flag rejection checks passed at that revision;
  the latter guards have since been removed. Source hooks passed.

[Direct-import results](vllm_import_results.json) contain the source hashes,
comparison summaries, gradient/training results, and local reproduction scripts.
The earlier receipts remain historical evidence for the implementations they
validated. This rerun establishes kernel/layer correctness for the tested runtime
and block32 INT8 configuration; it does not measure full-model quality or speed.

### Public release pin verification on 2026-10-06

The earlier receipts used local fork revision `930288170`; that revision is not
available from the public vLLM GitHub repository. The install requirement now
pins **`vllm==0.15.1`**, whose public source is
[`1892993bc`](https://github.com/vllm-project/vllm/tree/1892993bc18e243e2c05841314c5e9c06a80c70d).
All seventeen FLA/GDN source files used by these checks match the earlier fork
byte for byte. The version guard used for this validation was subsequently removed;
the example requirements own the pin, and tests check kernel API availability.

All ten trajectory cases and both converted Megatron-layer comparisons were
rerun with the public checkout: **11,042 comparisons remain bitwise equal**.
Both layers again complete backward, optimizer update, and exact checkpoint
restore; the two focused GPU cases and dependency guards also pass. These runs
reuse the existing local compiled extensions, whose relevant source files also
match the release. The unchanged Torch adjoint retains its prior twelve
passing directional checks; the 43 CPU tests were not rerun for this pin-only
correction. At validation time, the `serving/` adapter contained 227 lines.

[Public-release results](vllm_public_release_results.json) record the source
comparison, rerun summaries, dependency checks, source hashes, and local scripts.
The earlier receipts remain unchanged and describe their earlier implementations.

## Remaining validation and optimization

1. Qualify matched full-model Megatron/vLLM weights, projections, convolution,
   output normalization, and checkpoint conversion. Kernel/layer-boundary parity
   does not establish full-model or scheduler equivalence.
2. Exercise engine-managed multi-call prefill, cache-slot reuse, and production
   batching. Explicit continuation states exercise the numerical boundary but
   do not test vLLM's scheduler.
3. Run the Bridge QAT/QAD example on a fixed model and dataset, then compare
   generation quality with the unquantized baseline.
4. Optimize forward/backward in #2562 against the same numerical gates. Preserve
   QDQ at every decode token and gradients through the prefill handoff. No training
   speedup follows from this correctness result; old timing tables measure the
   replaced implementation.

## Related analysis

The [Mamba state quantization analysis](https://docs.google.com/document/d/1Wlc_QVIcQevlu_kSDvb7c3rEi5B1RDoBp9h3DKSwWr0/edit)
describes the analogous mismatch for FP16 Mamba state storage. It identifies
chunk-boundary quantization as approximate and recurrent training as a faithful
but potentially expensive approach. Its section on unrolling quantization error
is a crossed-out heading without an algorithm or result. The prefix/suffix
proposal here applies recurrent training only to the continuation. The Mamba
analysis does not validate GDN/KDA INT8 or FP8 kernels.
