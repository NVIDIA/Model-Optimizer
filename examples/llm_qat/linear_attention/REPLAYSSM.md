# ReplaySSM state QAT

This optional extension adds checkpoint-and-update replay to the
[GDN/KDA QAT and QAD example](README.md). The existing INT8/Hadamard profile is
implemented as a comparison baseline. The two-precision GDN profile imports
upstream native forward kernels and adds a window-wise training adjoint.
Integration with vLLM's request cache manager remains a separate follow-up.

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

The default `replay_window=1` refreshes the checkpoint every suffix token. To
enable replay between checkpoints, modify the recipe's execution policy:

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

The reference is upstream ReplaySSM's
[GDN emulation](https://github.com/Johnny-Liou/ReplaySSM/blob/73a9592e9d8a2268620e8883a82f5cec7442ed36/emulation/gdn_replay_unf_vf.py),
pinned at `73a9592e9d8a2268620e8883a82f5cec7442ed36`. ModelOpt imports its
`emulation.GDNReplay` wrapper with mode `tp8_hi32`; no upstream kernels are
copied into ModelOpt. This backend stores quantized values in FP32 tensors.
It qualifies numerical emulation, not compressed cache storage or speed.

Select `--recipe general/ptq/gdn_replayssm_fp8_read` in the main QAT/QAD example.
The recipe sets `precision="replayssm_two_precision"`, `replay_window=16`, and:

| Quantizer | Setting | Checkpoint layout |
| --- | --- | --- |
| `gdn_state_quantizer` | Disabled: recurrence stays FP32 | `[H,V,K]` internally |
| `gdn_read_quantizer` | Dynamic E4M3, `block_sizes={-1: 128}`, identity STE | One scale per 128 key channels of each value row |

Disabling the read quantizer retains the same replay schedule with FP32 reads
inside an explicit training phase. The initial profile requires GDN, normalized
Q/K, K=128, and `scale=K**-0.5`. It accepts `output_final_state=False`
(the default); exporting a dense state from an incomplete window is not supported.
KDA, lower-precision recurrence anchors, NVFP4 reads and stochastic rounding
remain follow-ups.

Install the main example's public vLLM 0.30 and FLA dependencies for the chunked
prefix. Expose only the pinned upstream checkout's `emulation` package, so its
`vllm/` source does not shadow the installed vLLM:

```bash
mkdir -p /path/to/standalone-packages
ln -s /path/to/ReplaySSM/emulation /path/to/standalone-packages/emulation
export PYTHONPATH="/path/to/standalone-packages:${PYTHONPATH:-}"
```

The upstream checkout must be at the pinned revision above. Decode-only calls
need the standalone package without vLLM; a nonempty prefix requires vLLM 0.30.
The existing INT8/Hadamard fork uses a separate environment.

Maintain two checkpoints formed from the same refreshed, unrounded state `S`:

```text
anchor = Q_recurrence(S)
read_copy = Q_read(S)
```

Never form `read_copy` from the rounded anchor, or overwrite the anchor with the
read copy. This implementation uses an FP32 anchor and FP8 read copy. Extending
it to an FP16 anchor or NVFP4 reads requires matching the upstream grouping and
rounding schedule independently.

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
replay cache and calls `GDNReplay.step` once per token. It saves immutable
anchor/read snapshots at window boundaries. Backward reconstructs one window
at a time, differentiating both the read path and the raw-input triangular solve.
It does not build a second Torch forward graph during the native forward.
Quantization and BF16/FP16 rounding use identity straight-through gradients.

Policy and named quantizers use ModelOpt's existing save/restore lifecycle.
The native backend owns rounding and runtime rings; these per-batch tensors are
not model checkpoint metadata. HF weights alone do not describe replay policy.

## Planned vLLM serving integration

Serving with state quantization and ReplaySSM enabled is a required follow-up.
The current plain state-QDQ serving adapter rejects ReplaySSM. Keep that check
until a compatible native backend and its cache lifecycle are validated.

- Resolve the recipe or restored quantizer/policy metadata before engine cache
  allocation. Select the replay window, checkpoint format, scale grouping,
  Hadamard basis, and rounding behavior consistently with training. HF weights
  alone do not supply this policy; require an explicit recipe when it is absent.
- Let the native backend own checkpoint encoding, replay buffers, request-slot
  reset, and window refresh. Skip the plain adapter's per-token state-QDQ hook:
  the stored checkpoint represents the start of the current replay window.
- Execute native prefill and replay decode for inference. The training-only
  Torch gradient graph is not part of serving. Quantize at the selected
  profile's checkpoint boundaries, including prefill handoff and window flush.
- For the two-precision profile, retain separate recurrence/read checkpoints and
  the raw-input solve. A backend implementing only the existing shared-checkpoint
  profile cannot serve the two-precision policy.
- Validate training/export/load/serve with identical token inputs over two replay
  windows, checking outputs, checkpoints, and buffers at handoff and flush. Also
  check continuation prefill, request-slot reuse, and tensor-parallel head shards.

This integration belongs in a serving follow-up after the plain state-QDQ
adapter. Public vLLM and the existing fork require separate qualification; no
ReplaySSM serving support is claimed by this plan.

## Validation and remaining work

Focused CPU tests cover policy validation, recipe loading, read-only enablement,
and ModelOpt checkpoint save/restore (40 tests passed). Two GPU tests passed on
RTX A6000 with PyTorch 2.9.1 and Triton 3.5.1:

- Native token outputs match bit-for-bit across 33 tokens, including handoff and
  two flushes. Switching FP8 reads to FP32 leaves every recurrence anchor and
  both flush outputs bit-identical.
- A separate triangular-solve oracle checks the window adjoint; padded positions
  have zero gradients, incoming state receives gradients, and disabled reads use
  the same replay schedule.

A single-layer smoke test also passed on H100 with PyTorch 2.13.0, Triton 3.7.1,
vLLM 0.30.0 and FLA 0.5.1: 64 prefix tokens, 33 suffix tokens, three padded
positions, and three optimizer steps each for QAT and QAD. Both runs changed
weights, propagated suffix loss into prefix inputs, preserved zero padding
gradients, and matched native inference bit-for-bit after ModelOpt save/reload.
Activation recomputation also reproduced the forward and input gradients.
QAT cross-entropy changed from 2.770705 to 2.770207; QAD KL (computed in FP64 to
avoid cancellation near zero) changed from 1.99646e-7 to 1.98797e-7. These are
synthetic smoke-test losses, not an accuracy-recovery result.

An additional K=256 probe exceeded the reference tolerances (maximum output
difference 4.54e-5, maximum query-gradient difference 3.65e-5). That shape is
not enabled in this initial profile. Further qualification must resolve this reference discrepancy before expanding
support.
Full-model accuracy, Megatron/Bridge training with this new profile, HF export,
and vLLM engine request-lifecycle validation remain separate qualification steps.
The existing fork-only GPU tests continue to cover the INT8/Hadamard baseline.

The new GPU tests require the optional upstream `emulation` package and skip
when it is absent; they are run explicitly in the qualification environment.
