# ReplaySSM state QAT

This optional extension adds checkpoint-and-update replay to the
[GDN/KDA QAT and QAD example](README.md). The existing INT8/Hadamard profile is
implemented as a comparison baseline. The two-precision GDN profile uses a ModelOpt Triton forward kernel shared by
training and serving, with a window-wise training adjoint.
The experimental serving worker below binds the two-precision cache to vLLM's
request cache manager.

## Existing INT8/Hadamard profile

The native adapter imports the compatible quantized-ReplaySSM fork used by this
implementation. Public vLLM 0.30 and upstream ReplaySSM do not expose that fork's
checkpoint-writing, INT8-scale, Hadamard, and KDA APIs. Install the compatible
fork in a separate environment; changing an import path does not provide those
capabilities. This dependency is not part of the example's public requirements.

Use the same QAT/QAD CLI as the main example, with
`--recipe general/ptq/linear_attention_state_int8_dynamic`. That recipe enables
dynamic INT8 state quantization and a normalized Hadamard transform over groups
of 32 value channels. Checkpoints use one FP16 scale per key row and 32-value group. The native codec owns checkpoint encoding; TensorQuantizer
owns the enable/disable setting and validates the requested format.

The built-in recipe uses `replay_window=32`: one checkpoint refresh followed by
up to 31 replay tokens. To run a controlled window-size comparison, modify the
recipe's execution policy explicitly:

```python
from modelopt.recipe import load_recipe

cfg = load_recipe("general/ptq/linear_attention_state_int8_dynamic").quantize.model_dump()
cfg["linear_attention"][0]["cfg"]["replay_window"] = 8
```

Pass `cfg` to `mtq.quantize`, or save the equivalent YAML for the CLI's `--recipe`.
The window accepts 1–64 tokens and is independent of the 64-token prefill chunk.
The value dimension must be divisible by 32. The suffix requires power-of-two
key dimensions of at least 32 and raw GDN gate projections; KDA requires the
fork's channel-wise gate extension.

At handoff, the profile encodes one shared read/update checkpoint. Between
refreshes it saves BF16 keys and beta-scaled residual updates with FP32 log
decays. Reconstruction applies the intervening decay to the checkpoint and
saved updates. The output reads the working state before checkpoint rounding;
a full window refreshes the checkpoint and clears the update ring. Thus the
quantized checkpoint influences subsequent residuals as well as outputs.

`linear_attention/replay.py` owns the differentiable recurrence and
`kernels/quantization/linear_attention/serving/replay.py` calls the fork's native
forward. The native values are attached to a Torch surrogate gradient. This is
training emulation with floating tensors, not a claim of compressed training
storage or serving speed. The fork's checkpoint codec also handles incoming
continuation-prefill state; a fresh prefix has no internal checkpoint QDQ.

## Two-precision GDN profile

The goal is to reduce read precision without feeding the low-precision read
copy back into the recurrence checkpoint. This changes the state machine, not
just its quantizer setting.

ModelOpt implements the GDN replay equations internally in `replayssm.py` and
uses its existing E4M3/E2M1 quantization helpers. The external
[ReplaySSM reference](https://github.com/Johnny-Liou/ReplaySSM/tree/73a9592e9d8a2268620e8883a82f5cec7442ed36/emulation)
was used for numerical comparison; its `emulation` package is not a runtime or
regular-test dependency. Quantized values remain in FP32 tensors, so this is
numerical emulation rather than compressed cache storage.

Select `--recipe general/ptq/gdn_replayssm_fp8_read` or
`--recipe general/ptq/gdn_replayssm_nvfp4_read` in the main QAT/QAD example.
The recipe sets `precision="replayssm_two_precision"`, `replay_window=16`, and:

| Quantizer | Setting | Checkpoint layout |
| --- | --- | --- |
| `gdn_state_quantizer` | Disabled: recurrence stays FP32 | `[H,V,K]` internally |
| `gdn_read_quantizer` (FP8 recipe) | Dynamic E4M3, `block_sizes={-1: 128}`, identity STE | One scale per 128 key channels of each value row |
| `gdn_read_quantizer` (NVFP4 recipe) | Dynamic E2M1, block16 E4M3 scales, identity STE | Groups of 16 key channels; one FP32 global scale per request and head at each refresh |

NVFP4 uses the existing TensorQuantizer configuration and ModelOpt rounding helpers.
A second kernel reduces each head's state amax independently and refreshes the read
copy at initialization and each 16-token flush. Training and the vLLM cache
adapter use the same mode. The FP32 recurrence anchor never uses the quantized read copy.

Disabling the read quantizer retains the same replay schedule with FP32 reads
inside an explicit training phase. The initial profile requires GDN, normalized
Q/K, K=128, and `scale=K**-0.5`. It accepts `output_final_state=False`
(the default); exporting a dense state from an incomplete window is not supported.
KDA and lower-precision recurrence anchors remain follow-ups.

### Stochastic NVFP4 reads

The `gdn_replayssm_nvfp4_read` recipe supports optional stochastic rounding:
set `rounding: stochastic` in its `*gdn_read_quantizer` configuration. The default
is nearest. Only E2M1 read values change; E4M3 block scales retain nearest rounding
and the recurrence anchor remains FP32. Reads refresh at initialization and each
16-token replay flush. Backward uses saved read values with the identity STE and
draws no new random numbers.

For example, create a resolved recipe usable by training and the ReplaySSM worker:

```python
import yaml
from pathlib import Path
from modelopt.recipe import load_recipe

recipe = {
    "metadata": {"recipe_type": "ptq"},
    "quantize": load_recipe("general/ptq/gdn_replayssm_nvfp4_read").quantize.model_dump(),
}
for rule in recipe["quantize"]["quant_cfg"]:
    if rule.get("quantizer_name") == "*gdn_read_quantizer":
        rule["cfg"]["rounding"] = "stochastic"
Path("nvfp4_stochastic.yaml").write_text(yaml.safe_dump(recipe))
# Training: mtq.quantize(model, load_recipe("nvfp4_stochastic.yaml").quantize)
# Serving: RECIPE_PATH=/absolute/path/nvfp4_stochastic.yaml with ReplaySSMWorker.
```

The native adapter derives one request seed from PyTorch's device RNG and uses
logical state coordinates plus the refresh count. Cache-slot reordering and CUDA
graph replay therefore preserve each request's random sequence. Repeatability
requires the same seed, request initialization order and shapes; it does not imply
bit-exact agreement with the generic Torch sampler. ModelOpt checkpoints save the
rounding policy; training checkpoints must additionally save the RNG state.
Standard inference export rejects stochastic rounding; the ReplaySSM worker uses
the explicit recipe above to install its supported read-rounding policy.

### Backend installation

Install the main example's dependencies for QAT/QAD and public vLLM 0.30 for the
chunked prefix and serving worker. Decode-only calls require PyTorch and Triton;
they need no external ReplaySSM checkout or additional `PYTHONPATH` entry.
The existing INT8/Hadamard fork uses a separate environment.

Maintain two checkpoints formed from the same refreshed, unrounded state `S`:

```text
anchor = Q_recurrence(S)
read_copy = Q_read(S)
```

Never form `read_copy` from the rounded anchor, or overwrite the anchor with the
read copy. This implementation uses an FP32 anchor and an FP8 or NVFP4 read copy.
Lower-precision anchors require their own upstream grouping and rounding schedule.

The forward schedule is:

```text
S = native_chunked_prefill(prefix)
for each suffix token:
    save raw key/value and gate/beta for this window
    if this is the first decode token:
        output, update = read_and_update(S, token)
        anchor, read_copy = encode_independently(S)
    elif this token closes the replay window:
        S_new, output = solve_window(anchor, raw_window_inputs)
        anchor, read_copy = encode_independently(S_new)
        clear_window()
        continue
    else:
        output, update = read_with_cached_updates(read_copy, update_ring, token)
    save_update(update)
```

The window solve must recompute residual updates from raw inputs and the anchor;
it must not refresh the anchor using updates derived from the read copy. Retain
both raw inputs and cached updates because they serve different purposes. Match
the reference's first-token behavior and flush-output arithmetic explicitly.
Initially use its fixed 16-token window. KDA channel-wise decay is a separate
extension requiring its own reference and validation; upstream GDN evidence does
not qualify KDA.

## Forward, backward and checkpoint ownership

The chunked prefix runs once through the existing native vLLM adapter; FLA
supplies its backward. For the suffix, `_two_precision.py` creates one native
replay cache and calls the shared ModelOpt replay kernel once per token. It saves immutable
anchor/read snapshots at window boundaries. Backward reconstructs one window
at a time, differentiating both the read path and the raw-input triangular solve.
It does not build a second Torch forward graph during the native forward.
Quantization and BF16/FP16 rounding use identity straight-through gradients.

Complete 16-token backward windows automatically use CUDA graphs through
PyTorch's `make_graphed_callables`. It captures the original Torch forward and
backward operators without compiler rewrites, preserving their rounding. Capture warms up on first
use and is shared across layers with matching shapes. Partial windows remain
eager, and packed-sequence boundaries can change between batches. If a surrounding
training CUDA graph is capturing, it records the window operators directly.
This captures the window adjoint, not the complete training iteration; MLP graph
capture is configured separately through Megatron's training settings.

Policy and named quantizers use ModelOpt's existing save/restore lifecycle.
The native backend owns rounding and runtime rings; these per-batch tensors are
not model checkpoint metadata. HF weights alone do not describe replay policy.

## Experimental vLLM serving

Select `general/ptq/gdn_replayssm_nvfp4_read` as `RECIPE_PATH` for NVFP4 reads.
`RECIPE_PATH` is mandatory: the worker never guesses a precision and does not
infer replay policy from HF weights. Use the exact resolved recipe saved with
the training/export artifact.

`examples/vllm_serve/replayssm_worker.py` applies the dynamic recipe immediately
after loading the model, before vLLM plans or allocates cache pages. The ordinary
state-QDQ worker runs its setup later and cannot reserve the additional replay
buffers. This opt-in worker supports GDN with K=128 on vLLM 0.30, one GPU and one
active request. Checkpoint precision is fixed at startup; restart the worker to
change it.

```bash
export PYTHONPATH="$PWD/examples/vllm_serve:${PYTHONPATH:-}"
export RECIPE_PATH=general/ptq/gdn_replayssm_fp8_read
VLLM_GDN_DECODE_KERNEL=triton vllm serve /path/to/hf-checkpoint \
  --worker-cls replayssm_worker.ReplaySSMWorker \
  --dtype bfloat16 --no-async-scheduling \
  --compilation-config '{"cudagraph_mode":"FULL_AND_PIECEWISE"}' \
  --max-num-seqs 1 --no-enable-prefix-caching --mamba-cache-mode none \
  --mamba-cache-dtype float32 \
  --additional-config '{"gdn_prefill_backend":"triton"}'
```

The vLLM 0.30 V2 runner supports compiled execution with full decode CUDA
graphs and piecewise prefill graphs. Its scheduler keeps prompt tokens, including
one-token prompt tails, out of the full decode graph. Replay positions and refresh
flags come from GPU counters, so they advance on every graph replay. Warmup records
the ReplaySSM recurrence using dummy slots rather than the ordinary GDN recurrence.
The default attention splitting operations must remain enabled. Use
`--enforce-eager` for debugging or with the older runner; full graphs that include
prefill are not supported. Single-request and synchronous-scheduling restrictions
still apply.

A BF16 checkpoint plus this recipe enables dynamic state PTQ without calibration.
For QAT/QAD, provide the exported HF weights and the exact resolved recipe used
in training; HF weights alone do not carry the replay policy. Automatic
policy-sidecar loading and compatibility with weight-quantized exports are not
qualified here. Omitting `RECIPE_PATH` is an error rather than an implicit FP8
fallback.

The cache specification adds a read checkpoint, raw key/value/gate/beta rings,
a residual-update ring and a per-slot token counter to the existing convolution
and recurrence state. vLLM allocates and releases all of these in its cache pages.
`replayssm_cache.py` calls the internal kernel with request-slot indices;
no recurrent state is kept in a separate Python request cache. Prefill writes the
FP32 anchor and resets the counter. Decode refreshes the read copy only at the
native handoff and flush boundaries, without applying plain per-token state QDQ.

The worker uses the runner's prompt position to identify one-token prompts and
final prompt chunks, which vLLM's GDN metadata otherwise labels as decode.
Request reuse resets the replay window; old ring entries are masked until replaced.
The two checkpoint tensors still contain floating-storage emulation values.
This path consumes extra memory and does not claim packed FP8/NVFP4 storage or a speedup.

Mixed prefill/decode batches, continuation prefill after decode, speculative
decoding, prefix caching, cache transfer and distributed serving remain outside
this initial engine integration. Keep ordinary state-QDQ serving separate.

## Validation and remaining work

### Internal backend

The standalone GDN forward is implemented in ModelOpt and shares the serving
cache adapter. Regular GPU tests run without `emulation`, checking the window
math against an independent Torch recurrence, gradients, padding, request-slot
reuse, per-head read scales and CUDA graph replay.

On GB300 with vLLM 0.30.0 and Triton 3.7.1, all 19 kernel/training tests and four
real-engine tests passed with `emulation` absent. Engine coverage includes eager
FP8 and CUDA-graph FP8, NVFP4 and stochastic NVFP4 execution. The engine checks
use a tiny offline Qwen3-Next model with dummy weights and one active request;
they do not qualify the separate multi-request adapter changes.

A local comparison over 65 tokens per case against the previous backend found
bit-exact FP32 anchors and FP8/NVFP4 read copies, including stochastic reads.
FP32 outputs differed by at most 1.6e-6 on these inputs; external-backend outputs
are not claimed to be bit-exact. The Q/K/V, gate, beta and initial-state gradients
matched the previous window adjoint bit-exactly in all three quantized profiles.
These checks do not establish full-model accuracy or throughput equivalence.

### Previous external-backend qualification

Before the per-head outer-scale change, NVFP4 qualification on JHB GB300 passed all six indexed-cache cases
(FP8/NVFP4/FP32 reads, eager/captured execution) and the NVFP4 full-decode-graph
engine case. Cache outputs, anchors and read copies were bit-exact against
upstream through flushes and slot reuse, including a partial value tile. The
engine case verified repeated generation, prompt boundaries, counters and CUDA
graph launches. No full-model NVFP4 accuracy result is claimed.

The serving follow-up passed two indexed-cache GPU cases and one real vLLM 0.30
V2-engine test on DFW. The cache cases cover FP8 and FP32 reads, reordered and
paused requests, different window positions, masked slots and unfinished-window
reuse. The tiny offline Qwen3-Next engine test used prompt lengths 73, 5, 65 and 1,
64-token scheduler prefill chunks, and 34 generated tokens per request. Repeating
all four requests exercised 264 decode steps and 16 flushes; every decode output
and FP32 checkpoint matched the upstream runner bit-for-bit, and repeated token
IDs and log probabilities were identical. Four local GPU checks also passed,
including the existing training-adjoint checks.

CUDA graph qualification on JHB GB300 also passed four indexed-cache cases
(FP8/FP32 reads, eager/captured execution) and both eager and compiled engine cases.
The captured cache retained bit-exact output, anchor, read-copy and counter parity
with upstream ReplaySSM through refreshes and slot reuse. The compiled engine test
confirmed CUDA graph launches, repeated generation, prompt-boundary handling and
GPU counter advancement.

This is functional coverage with dummy model weights, not a pretrained quality
result or a full-model QAT/QAD export-to-serving qualification. Those results
used the external backend and the V2 runner.

Focused CPU tests covered policy validation, both read recipes, read-only enablement,
and ModelOpt checkpoint save/restore (41 tests passed). Four local GPU cases passed
on RTX A6000. All six training GPU cases passed on JHB GB300, including packed
sequences, with PyTorch 2.13.0+cu130, Triton 3.7.1, vLLM 0.30.0 and fla-core 0.5.1.
The GPU cases exercise both FP8 and NVFP4 reads and check:

- Bit-exact output parity with the pinned upstream native runner across two replay
  windows, and a separate Torch recurrence for the window adjoint.
- Prefix/handoff gradients, zero padding gradients, and the disabled-read baseline.
- Packed versus separate sequence outputs and gradients, including sequence isolation.

These are numerical and workflow checks, not full-model NVFP4 QAT/QAD or accuracy
results. Regular GPU tests now run without the upstream `emulation` package,
using an independent Torch window reference for numerical and gradient checks.
The existing fork-only tests remain separate qualification.

K=256 failed the predefined numerical gate in the earlier FP8 qualification and
remains unsupported. No tolerance was relaxed. KDA two-precision support,
lower-precision anchors, full-model accuracy and broader scheduler
qualification remain follow-ups.
