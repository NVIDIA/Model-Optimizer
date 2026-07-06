# Unified vLLM Attention Worker Design

## Goal

Remove duplicated sparse-attention planning and installation code from the vLLM
example workers while preserving two explicit serving policies:

- `SparseAttnWorker` applies checkpoint-driven sparse attention only.
- `QuantSparseAttnWorker` applies the fixed block-16 NVFP4 Q/K/P/V recipe to
  every supported attention layer and optionally adds checkpoint-driven sparse
  attention.

The refactor must reduce production code, retain atomic validation, and leave
attention numerics and kernel dispatch unchanged.

## Current Problem

`sparse_attn_worker.py` and `quant_sparse_attn_worker.py` independently perform
the same orchestration:

1. Unwrap the loaded vLLM model.
2. Read sparse-attention checkpoint metadata.
3. Traverse attention modules.
4. Match per-layer sparse configuration.
5. Select and clone the backend-matched ModelOpt implementation.
6. Validate the complete plan before mutating the model.
7. Install the replacement immediately after `BaseWorker.load_model`.

The quant worker adds strict runtime and layout validation, Q/K/P/V quantizer
configuration, quantization arguments, cascade disablement, and compilation-
disabled memory profiling. Those are policy extensions of the same attention
installation lifecycle, not a separate serving architecture.

The launcher is not part of this duplication. It owns CLI parsing, worker-module
import setup, and API-server startup, while the worker runs inside each spawned
vLLM GPU process where the model exists.

## Architecture

### One Source Module

`examples/vllm_serve/sparse_attn_worker.py` becomes the single attention-worker
module and declares both public classes in `__all__`:

```python
class _ModelOptAttentionWorker(BaseWorker):
    quantize_attention = False

    def load_model(self, *args, **kwargs):
        super().load_model(*args, **kwargs)
        _install_attention(self, quantize=self.quantize_attention)


class SparseAttnWorker(_ModelOptAttentionWorker):
    pass


class QuantSparseAttnWorker(_ModelOptAttentionWorker):
    quantize_attention = True
```

`QuantSparseAttnWorker` retains its `determine_available_memory` override. The
override remains quant-only because sparse-only serving does not need the
compilation workaround and must not inherit its custom-all-reduce interaction.

`examples/vllm_serve/quant_sparse_attn_worker.py` is deleted. The new quant
worker path is:

```text
sparse_attn_worker.QuantSparseAttnWorker
```

The existing `sparse_attn_worker.SparseAttnWorker` path remains unchanged.

### Shared Planning Pipeline

The common pipeline produces immutable plan records before any module mutation.
Each record contains the layer name, attention module, backend-matched
replacement implementation, sparse keyword arguments, and quant-specific
device/dtype fields when quantization is enabled.

The pipeline performs these shared steps once:

1. Resolve the loaded model and optional sparse checkpoint configuration.
2. Enumerate vLLM attention modules.
3. Match and normalize each layer's sparse configuration.
4. Select and clone the FlashAttention or FlashInfer adapter.
5. Accumulate all errors and reject the complete plan before mutation.

Policy-specific validation is invoked from this pipeline rather than mixed into
the shared traversal.

### Sparse-Only Policy

With `quantize=False`:

- Missing `sparse_attention_config` logs the existing message and leaves the
  model unchanged.
- Only layers with enabled, nonempty sparse configuration enter the plan.
- Unsupported backends are rejected only for layers requesting a sparse
  transform.
- No quantizers, quantization flags, quantization arguments, or profiling
  overrides are installed.
- Existing native fallback behavior for inactive sparse launches is preserved.

### Quant-Plus-Sparse Policy

With `quantize=True`:

- vLLM 0.14 or newer remains required.
- Every regular decoder self-attention layer must enter the plan, even when the
  checkpoint has no sparse metadata.
- Existing global and per-layer quant validation remains mandatory.
- Each layer receives the fixed dynamic block-16 NVFP4 Q/K/P/V recipe.
- Optional sparse metadata augments the same backend-matched implementation.
- P/V quantization arguments, Q/V kernel ownership flags, K/V default scales,
  and cascade disablement remain unchanged.
- The compilation-disabled memory-profile override remains active.

Quant-only vLLM and ModelOpt imports are resolved only when quant mode is used.
Importing or running `SparseAttnWorker` must not acquire the quant worker's
vLLM-version floor. The quant version/API check therefore runs when the quant
policy is selected, not merely when the shared module is imported.

## Startup Data Flow

The vLLM lifecycle remains:

1. The launcher selects a worker class by dotted path.
2. Each GPU process creates that worker and calls `load_model`.
3. The base worker constructs the native attention modules and implementations.
4. `_ModelOptAttentionWorker.load_model` builds and validates the complete
   ModelOpt plan, then installs it.
5. vLLM discovers KV-cache specs and runs memory profiling using the installed
   implementation.
6. vLLM initializes cache/backend metadata and performs warmup and CUDA-graph
   capture.

Installation must not move to `compile_or_warm_up_model`; that hook is too late
for attention profiling, FlashInfer metadata setup, and cache initialization.

## Compatibility

- Preserve the `sparse_attn_worker.SparseAttnWorker` public path and observable
  sparse-only behavior.
- Change the unreleased compact-worker path from
  `quant_sparse_attn_worker.QuantSparseAttnWorker` to
  `sparse_attn_worker.QuantSparseAttnWorker` in tracked examples, documentation,
  and tests.
- Do not add an environment variable or CLI mode flag. The two class names are
  the explicit policy selectors.
- Preserve sparse-only import behavior on vLLM releases older than the quant
  worker's 0.14 minimum.
- Keep `vllm_serve_sparse_attn.py` as a separate launcher and retain its
  sparse-only default.
- Do not merge this path with `FakeQuantWorker`; whole-model fakequant uses a
  later lifecycle hook because calibration requires initialized KV-cache state.
- Preserve the current `MODELOPT_ATTN_QUANT_OFF` isolation behavior if it is
  present when implementation begins. Removing that benchmark-only behavior is
  a separate cleanup decision, not part of worker consolidation.

## Error Handling

- Both policies validate the entire selected layer set before the first model
  mutation.
- Sparse-only mode continues to no-op when no sparse metadata is present.
- Quant mode fails if no regular attention layers are found or if any selected
  layer/runtime feature violates the existing quant contract.
- Backend clone errors and unsupported attention implementations are reported
  with layer names in one aggregated `NotImplementedError`.
- A quant-only import/version failure is raised only when
  `QuantSparseAttnWorker` is selected.

## Testing

Retain the focused sparse and quant worker tests as separate test modules while
pointing both at `sparse_attn_worker.py`.

Required coverage:

- Both public classes install after the base `load_model` call.
- Sparse mode remains a no-op without checkpoint metadata.
- Sparse mode mutates only layers with active sparse configuration.
- Quant mode converts every supported regular attention layer and preserves the
  exact Q/K/P/V NVFP4 recipe for FlashAttention and FlashInfer.
- Both modes reject a multi-layer plan atomically before mutation.
- Quant-only runtime/layout validation does not reject sparse-only serving.
- Sparse-only import remains valid when quant-only vLLM APIs are unavailable.
- Selecting the quant class below vLLM 0.14 produces the existing clear error.
- Quant memory profiling runs under `disable_compilation`; sparse-only profiling
  does not gain that override.
- Full and calibrated-decode CUDA-graph validation remains unchanged.

Run the focused worker and plugin suites, formatting checks, and an import smoke
for both dotted class paths. A single vLLM serving smoke with the quant class is
required after the refactor because the worker path changes.

## Scope And Expected Reduction

The implementation should remove duplicated model traversal, sparse metadata
handling, adapter planning/cloning, load lifecycle, module header, and imports.
The combined production source must be materially smaller than the current 404
lines across the two workers; a net reduction of approximately 40-70 lines is
expected without weakening validation or deleting regression coverage.

If preserving older sparse-only imports requires conditional imports, keep that
logic local to the quant policy. Do not create another helper module solely to
move lines between files.

## Non-Goals

- Changing attention kernels, tile sizes, quantization scales, or numerical
  behavior.
- Replacing the two explicit public worker policies with automatic inference.
- Merging the worker with `vllm_serve_sparse_attn.py`.
- Merging attention-only serving with whole-model `FakeQuantWorker` calibration.
- Adding new sparse algorithms, attention backends, or vLLM feature support.
- Removing focused tests solely to reduce the pull request's line count.
