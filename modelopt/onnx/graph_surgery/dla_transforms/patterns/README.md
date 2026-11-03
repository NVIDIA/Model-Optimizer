# Connected graph transformations

This directory owns decompositions whose meaning depends on a connected group
of operators. Add those rules to [rules.py](rules.py). Use
[operators](../operators/README.md) for individual operator lowering,
[layout](../layout/README.md) for coordinated rank/axis changes, and
[precision](../precision/README.md) for dtype policy. See the
[pipeline README](../README.md) for the common rule and graph-context contracts.

## Rule catalog

| Rule | Recognized graph | Replacement and supported conditions |
| --- | --- | --- |
| `ReshapeTransposeReshape` | `Reshape(to 5D) -> Transpose -> Reshape` | Splits the leading dimension into Slice branches, performs a 4D Reshape/Transpose in each branch, then joins and restores the final shape. Dimensions and controls must be fixed, positive after resolution, and preserve element count. The permutation must keep axis 0, and the first Reshape must retain the source leading dimension. |
| `ReshapeReduceSum` | `Reshape(to 5D) -> ReduceSum` | Merges the reduced dimension with its predecessor, slices that merged dimension, reduces 4D branches, and concatenates them. Supports one fixed non-leading reduction axis. An Unsqueeze restores the original rank when `keepdims=1`. |
| `ReshapeQDQReduceSum` | `Reshape(to 5D) -> QuantizeLinear -> DequantizeLinear -> ReduceSum` | Uses the same sliced reduction while keeping Q/DQ inside every branch. Q/DQ controls must be fixed scalars; per-axis and blocked quantization require another lowering. |
| `TopKVectorGather` | `TopK -> Cast -> Reshape -> Tile -> [integer Casts] -> GatherElements` | Replaces tiled indices with a channel-axis TopK and INT32 vector indices passed to Gather. Requires the precise private index chain, singleton score dimensions before the final axis, and fixed valid K/Tile/Reshape controls. |
| `TerminalProjection` | Terminal `Conv -> Transpose -> Concat -> ReduceSum -> Concat -> Cast(FLOAT) -> Squeeze` with singleton Concats | Splits more than 8192 output channels into Conv chunks of at most 8192, reshapes/casts each chunk to FP32, and concatenates on axis 2. Requires the specific private terminal tail and fixed FP16 or scalar-dequantized UINT8 kernels. |

For the Reshape rules, fixed controls may use copied `0` dimensions and one
inferred `-1`; `allowzero=1`, dynamic controls and nonpositive resolved dimensions
are unsupported. A matched five-dimensional candidate with an unsupported form
raises an error. The branch computations are 4D, while an output adapter may
temporarily restore the original higher rank for later layout processing.

`ReshapeQDQReduceSum` derives from `ReshapeReduceSum` to reuse the same reduction
algorithm. Both are registered concrete rules; private helpers such as
`_resolved_shape()` and `_topk_plan()` are queries, not additional rules.

## Examples and boundaries

- **Reshape/Transpose:** reshape `[2,32]` to `[2,2,2,2,4]`, permute with
  `[0,2,1,3,4]`, then flatten to `[2,32]`. The replacement processes each leading
  slice as a four-dimensional permutation before joining the result.
- **Reduction:** reshape to `[B,A,R,C,D]` and reduce axis 2. The replacement
  views the data as `[B,A*R,C,D]`, reduces one R-wide slice per A, and returns
  `[B,A,C,D]`, or `[B,A,1,C,D]` when keeping dimensions. The quantized variant
  applies its original scalar Q/DQ inside those slices.
- **TopK/Gather:** scores `[1,N]` and data `[1,N,C]` become scores
  `[1,N,1,1]` for TopK on axis 1, followed by an INT32 index vector `[K]` and
  Gather on data axis 1. The result remains `[1,K,C]`. The alternative score
  layout `[1,1,1,N]` uses data axis 2. TopK values must be unused, and the index
  chain must have single-use private intermediates; public/shared indices retain
  the original graph.
- **Projection:** a supported terminal projection with 10240 output channels
  becomes 8192- and 2048-channel Conv chunks, with FP32 results concatenated to
  the original `[1,1,10240]` public output. Its Conv output must be
  `[1,10240,1,1]`, with `group=1`, Transpose permutation `[0,3,2,1]`, reduction
  and squeeze axes `[0]`, and a kept reduction dimension. The original weights
  and controls must not be public or shared outside the matched tail.

Replacement values retain the original terminal interface. Internal shapes and
dtypes are set as operations are emitted; changing metadata alone would not
perform a runtime Reshape or Cast. New constants use the graph's initializer
creator, and unused branches/controls are removed by subsequent cleanup.

## Execution order

[stages.py](../stages.py) applies `ReshapeTransposeReshape`,
`ReshapeQDQReduceSum`, `ReshapeReduceSum`, then `TopKVectorGather`, each as a
separate single-rule SDK application. It removes unused nodes before individual
operator and layout transformations begin. These connected rules run early so
their constituent native operators and shapes remain recognizable.

`TerminalProjection` is defined here because it is a connected pattern, but runs
late in the optional FP16 precision phase, after dtype preparation, precision
boundaries and topological sorting. Adding it to the early pattern sequence would
change which producers and weight dtypes it can see. The full pipeline order is
fixed by `stages.py`.

## Adding a transform

1. Define a documented `RewriteRuleClassBase` subclass in [rules.py](rules.py)
   and export it through `__all__`. `pattern()` captures connectivity;
   `check()` proves eligibility and shared/public ownership; `rewrite()` builds
   replacements. Describe an input/output example and meaningful limitations.
2. Bind per-run state with `NewRule.rule(graph)`. The SDK discovers matching
   nodes; do not add a graph-wide discovery loop for each rule. Pattern arity is
   fixed, so use explicit variants if the supported output counts differ.
3. Register the rule at the correct point in `pattern_rewrites()` in
   [stages.py](../stages.py), or its required later stage, as projection does.
   Preserve dependencies between producer and consumer rewrites. Use a direct
   `InPlacePass` in the owning phase when the work cannot be expressed as a fixed
   pattern; register its invocation through `graph.apply_pass()`.
4. Keep checks read-only, validate before mutation, and set type/shape metadata
   for emitted intermediates. If a replacement still matches the target, make
   its canonical form ineligible or use an appropriate snapshot guard.
5. Extend [test_rules.py](../../../../../tests/unit/onnx/graph_surgery/dla_transforms/patterns/test_rules.py)
   with a real graph and independent expected behavior. Projection's late FP16
   behavior is covered in
   [test_precision.py](../../../../../tests/unit/onnx/graph_surgery/dla_transforms/test_precision.py).
   Include relevant shared/public boundary and unsupported-form cases, and
   update this catalog.

Run focused tests from the repository root:

```bash
pytest tests/unit/onnx/graph_surgery/dla_transforms/patterns tests/unit/onnx/graph_surgery/dla_transforms/test_precision.py
```
