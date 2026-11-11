# AutoCast Per-Node Calibration Design

## Objective

Add a `calibrate_per_node` option to ONNX AutoCast that computes the same runtime tensor
statistics used by normal calibration without exposing or retaining every intermediate tensor at
once. The option must be available through both `convert_to_mixed_precision()` and
`python -m modelopt.onnx.autocast`.

## Architecture

`ReferenceRunner` remains responsible for loading and validating calibration inputs and preparing
execution providers. Its normal path remains unchanged. A new per-node path executes a sanitized,
type-inferred ONNX graph one node at a time:

1. Construct a runnable one-node model containing the node, its required initializers, graph
   metadata, opset imports, and local functions.
2. Feed live runtime inputs to that model through the same configured ONNX Runtime providers.
3. Immediately reduce each produced tensor to `TensorStats` (`absmax`, minimum, maximum, and
   runtime shape).
4. Keep produced arrays only while downstream nodes still consume them, using tensor consumer
   counts to release dead activations.
5. Close each one-node session before moving to the next node so sessions and provider resources
   do not accumulate.

Statistics are merged online across calibration batches. Per-node mode always returns
`TensorStats`, including for a single batch, because retaining raw tensors would defeat its memory
contract. `NodeClassifier` already accepts both arrays and `TensorStats`, so classification and
conversion behavior remain unchanged.

## Interfaces

- `convert_to_mixed_precision(..., calibrate_per_node: bool = False)` preserves current behavior by
  default.
- `ReferenceRunner.run(inputs=None, calibrate_per_node: bool = False)` selects the execution path.
- `--calibrate_per_node` is a boolean CLI flag with the same default and memory/time trade-off as
  ONNX quantization.

## Compatibility and Error Handling

- Existing calibration remains the default and receives no behavioral changes.
- Per-node models preserve model IR version, opset imports, and local functions.
- Initializers are embedded only in the node that consumes them.
- Existing external-data materialization and file-backed session creation are reused for models
  that cannot be serialized safely as bytes.
- Missing runtime dependencies or missing tensor type metadata fail with a clear `ValueError`
  identifying the node or tensor.
- TensorRT plugin paths and provider order use the existing `ReferenceRunner` configuration.

## Verification

Focused tests will:

- compare per-node statistics with normal all-output calibration on a branched graph;
- cover multiple calibration batches and online extrema aggregation;
- exercise `convert_to_mixed_precision(calibrate_per_node=True)` through real conversion;
- verify CLI parsing and forwarding;
- confirm the normal path remains the default.

The implementation will be validated with the focused AutoCast unit suite and repository
pre-commit checks for changed files. No new dependency is required.
