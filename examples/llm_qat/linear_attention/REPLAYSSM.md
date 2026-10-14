# ReplaySSM state QAT

This optional extension adds checkpoint-and-update replay to the
[GDN/KDA QAT and QAD example](README.md). The existing INT8/Hadamard profile is
implemented. The two-precision profile described below is a design plan only;
it has no selectable configuration or training implementation yet.

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

## Proposed two-precision profile

The goal is to reduce read precision without feeding the low-precision read
copy back into the recurrence checkpoint. This changes the state machine, not
just its quantizer setting.

The reference is upstream ReplaySSM's
[GDN emulation](https://github.com/Johnny-Liou/ReplaySSM/blob/73a9592e9d8a2268620e8883a82f5cec7442ed36/emulation/gdn_replay_unf_vf.py),
pinned at `73a9592e9d8a2268620e8883a82f5cec7442ed36`. The proposed implementation
will import a qualified upstream backend rather than copy its kernels. Its API
and deployment integration must be qualified separately from the existing fork.

Maintain two checkpoints formed from the same refreshed, unrounded state `S`:

```text
anchor = Q_recurrence(S)
read_copy = Q_read(S)
```

Never form `read_copy` from the rounded anchor, or overwrite the anchor with the
read copy. The first GDN milestone will use an FP32 anchor and FP8 read copy,
then add a stochastic-rounded FP16 anchor and NVFP4 read copy only after matching
the upstream format, scale grouping, and seeded rounding behavior.

The planned forward schedule is:

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

## ModelOpt integration plan

1. Introduce an explicit two-precision profile alongside the existing fork
   profile. Keep precision, window, and rounding policy in the saved linear
   attention policy; keep format/grouping and enablement in named quantizers.
   Resolve proposed recurrence/read quantizer names before exposing a public API.
2. Give the backend one owner for both checkpoints, raw-input rings, update
   rings, and the flush cursor. Separate immutable values saved for backward
   from mutable native caches. Allocate serving buffers once per request group.
3. Reuse native forward values and add a differentiable window-solve adjoint.
   Gradients must follow the anchor refresh and output paths, including the
   prefix handoff. Use straight-through quantization where appropriate, with
   reproducible stochastic-rounding counters during activation recomputation.
   Do not reuse the current shared-checkpoint adjoint without this change.
4. Save resolved policy and quantizer metadata through ModelOpt's existing
   checkpoint lifecycle. Runtime cache contents remain batch state. Require
   the serving adapter to select the same profile and rounding schedule.

## Validation required before enabling the new profile

- Compare token outputs, both checkpoints, and ring contents against the pinned
  native reference at the first token, inside a window, at flush, and across
  two windows, including a nonzero incoming state.
- Verify independently that changing read precision does not change the next
  recurrence anchor when raw inputs and recurrence rounding are held fixed.
- Compare surrogate gradients to an independent differentiable window solve;
  test prefix gradients, padding, and recomputation with fixed rounding seeds.
- Run a small GDN QAT and QAD optimizer step, save/reload, and serving comparison
  with the same profile. Report quality and performance separately.

The moved tests cover the existing Hadamard transform and invalid policy
settings on CPU, plus fork-only native-cache/gradient tests on GPU. Public-vLLM
CI skips the fork-only kernel cases. No two-precision parity, model-quality, or
performance result is claimed by this design plan.
