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
`emulation.GDNReplay` wrapper with mode `tp8_hi32` (FP8) or `tp4_hi32` (NVFP4); no upstream kernels are
copied into ModelOpt. This backend stores quantized values in FP32 tensors.
It qualifies numerical emulation, not compressed cache storage or speed.

Select `--recipe general/ptq/gdn_replayssm_fp8_read` or
`--recipe general/ptq/gdn_replayssm_nvfp4_read` in the main QAT/QAD example.
The recipe sets `precision="replayssm_two_precision"`, `replay_window=16`, and:

| Quantizer | Setting | Checkpoint layout |
| --- | --- | --- |
| `gdn_state_quantizer` | Disabled: recurrence stays FP32 | `[H,V,K]` internally |
| `gdn_read_quantizer` (FP8 recipe) | Dynamic E4M3, `block_sizes={-1: 128}`, identity STE | One scale per 128 key channels of each value row |
| `gdn_read_quantizer` (NVFP4 recipe) | Dynamic E2M1, block16 E4M3 scales, identity STE | Groups of 16 key channels; one FP32 global scale per request |

NVFP4 uses the existing TensorQuantizer configuration and upstream replay codec.
A second upstream kernel reduces the request-wide amax and refreshes the read
copy at initialization and each 16-token flush. The FP32 recurrence anchor never
uses the quantized read copy.

Disabling the read quantizer retains the same replay schedule with FP32 reads
inside an explicit training phase. The initial profile requires GDN, normalized
Q/K, K=128, and `scale=K**-0.5`. It accepts `output_final_state=False`
(the default); exporting a dense state from an incomplete window is not supported.
KDA, lower-precision recurrence anchors and stochastic rounding remain follow-ups.

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

Focused CPU tests cover policy validation, both read recipes, read-only enablement,
and ModelOpt checkpoint save/restore (41 tests passed). Four local GPU cases passed
on RTX A6000. All six training GPU cases passed on JHB GB300, including packed
sequences, with PyTorch 2.13.0+cu130, Triton 3.7.1, vLLM 0.30.0 and fla-core 0.5.1.
The GPU cases exercise both FP8 and NVFP4 reads and check:

- Bit-exact output parity with the pinned upstream native runner across two replay
  windows, and a separate Torch recurrence for the window adjoint.
- Prefix/handoff gradients, zero padding gradients, and the disabled-read baseline.
- Packed versus separate sequence outputs and gradients, including sequence isolation.

These are numerical and workflow checks, not full-model NVFP4 QAT/QAD or accuracy
results. The tests require the optional upstream `emulation` package and skip
when it is absent. The existing fork-only tests remain separate qualification.

K=256 failed the predefined numerical gate in the earlier FP8 qualification and
remains unsupported. No tolerance was relaxed. KDA two-precision support,
lower-precision anchors, stochastic rounding, full-model accuracy and engine
integration remain follow-ups.
