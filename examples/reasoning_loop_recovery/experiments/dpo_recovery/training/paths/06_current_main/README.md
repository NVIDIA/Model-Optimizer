# Path 6: Pinned NeMo RL main and its dependencies

The numbered items include environment setup and compatibility fixes; they are not a count of
independent bugs. Validation results and unresolved work are listed separately below.

## Setup and fixes

1. **Create a separate pinned-main environment.** Use the release image only as a base. The original
   snapshot is `4d464d5f3359f9ecbfc58a5e33c8ce5bdaf15335`, with its own Bridge/Core/ModelOpt pins;
   the newer snapshot below is separate. See [versions.yaml](../../versions.yaml).
2. **Install the pinned dependencies.** Use uv 0.11.28, Python 3.13.14,
   `uv sync --frozen --extra mcore --extra modelopt`, and initialized submodules. Keep uv cache,
   Python installation, Ray environments, HF import cache, and outputs separate from the release
   environment. Target SM103 with bounded compilation parallelism.
3. **Verify resolved provenance.** Resolve symlink aliases before comparing source paths; the
   original string-prefix checks rejected the same Lustre source through different aliases.
   Check driver and worker imports. Dependency setup passed after correcting uv and these checks.
4. **Configure bounded smoke tests.** Test BF16 HF and round-3 fake-quantized inputs separately,
   each with eight training/four validation pairs, 8,192 tokens, and one step. Disable fused RoPE
   and gradient accumulation fusion. The driver requests `use_fused_linear_logprobs` and chunk
   size 128; verify the pinned worker consumes these settings before any long-sequence run.
5. **Migrate only the obsolete null config field.** Restoration rejected
   `MaxCalibConfig.layerwise.export_dir=None`. Deep-copy and remove only this null field; reject
   non-null values and preserve all other quantization settings. This migration passed.
6. **Resolve output-head extra state — unresolved.** New Core rejects
   `language_model.output_layer._extra_state`. The bounded guard permits omitted legacy TE state
   only for an unquantized output head, FP8/FP4 off, and no active head quantizers. The quantized
   retry still fails. Do not drop quantizer state or relax the guard to obtain a successful load.
7. **Resolve the vanilla worker failure — unresolved in the original snapshot.** The recorded
   worker startup failed with uv exit 141. A successful BF16 smoke is required before promoting
   this branch to full training. Preserve separate logs for later retries.

## Validation and remaining work

Failures and bounded compatibility attempts:

- Dependency/import setup passed after upgrading uv and correcting both provenance path checks.
- Vanilla smoke failed during worker environment startup (uv exited 141); its retry is still failed.
- Quantized restore rejected obsolete `MaxCalibConfig.layerwise.export_dir=None`. A runtime wrapper
  deep-copies the config and removes only this null field; non-null values fail rather than being
  discarded. All other quantization fields are preserved. This migration passed.
- New Core then rejected `language_model.output_layer._extra_state`. An initial Tensor truth-value
  failure led to a guard allowing omitted legacy TE state only for an **unquantized output head**,
  FP8/FP4 off, and no active output-head quantizers. The latest retry still fails the output-layer
  extra-state assertion; the guard does not waive active quantization state. There is no successful
  DPO step for either main smoke in this snapshot.

Use `current_main_20261025/run_inside.sh`, `prepare_and_run.py`, the import hook,
`modelopt_state_compat.py`, and `legacy_output_state_compat.py`. Preserve the original failures
alongside patched attempts. Next, establish the exact output-head extra-state contract between
Core, TE, and ModelOpt, and resolve the vanilla worker environment failure. Do not relax the guard
or drop quantizer extra state merely to pass checkpoint loading.

See [path.yaml](path.yaml) for the configuration record and [shared fixes](../../compatibility/README.md).

## New comparison snapshot

A continuation tests commit `89d8f0c5d217e0270a5718880d811e51f3951e39` with its own lockfile and
submodule pins. Path 7 uses the same NeMo RL commit in the original QAD image. Preserve the earlier
`4d464d5f...` failures above for provenance; they do not establish the outcome of this newer snapshot.
