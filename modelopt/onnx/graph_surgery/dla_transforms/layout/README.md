# Layout rules and passes

This directory adapts supported data operations to a shared 4D layout while
preserving their logical results and the public model interface. It handles shape
controls, indexing, axis remapping and operators that require a native layout.
Operator decompositions belong in [operators](../operators/README.md); connected
operator patterns belong in [patterns](../patterns/README.md).

The definitions are in [rules.py](rules.py), shared analysis and direct IR passes
are in [passes.py](passes.py), and [stages.py](../stages.py) owns execution order.
The [package guide](../README.md) describes the full fixed pipeline.

## Shared layout metadata

[DlaGraph](../context.py) calls `begin_layout()` after preceding operator lowering.
`snapshot()` records each current node's native input/output shapes and traces
supported live data paths backward from graph outputs. These snapshots include nodes
created by earlier lowering, rather than only nodes from the input model.

| Context field | Meaning during layout |
| --- | --- |
| `native_shapes` | Layout-entry shapes keyed by node, as `(input_shapes, output_shapes)`. This snapshot stays unchanged through the phase. |
| `planned_shapes` | Active target shapes keyed by tensor name. Rules can coordinate a necessary change to this plan. |
| `public_outputs` | Saved public output names, shapes and types for boundary restoration. |
| `batch_nodes` | Nodes present before the current SDK rule batch; used where a rule needs a narrower eligibility boundary. |

The usual mapping adds leading singleton dimensions: `[N,C]` becomes
`[1,1,N,C]`, and `[N,C,L]` becomes `[1,N,C,L]`. A supported 5D shape can collapse
one singleton dimension. Individual rules impose additional restrictions, such as
an unambiguous removed axis for Transpose. Scalar data is generally left native;
scalar reduction results can have a planned 4D view.

The plan distinguishes data ports from shape, axis, quantization and weight
controls. Fixed Conv weights and control-only paths retain native layouts; a value
used in both roles can need separate views. Missing static data shapes and
unsupported required layout forms raise errors.

The IR holds current `ir.Value` shape/type metadata, not shape history. Rules read
the saved layout metadata even after earlier replacements. Metadata changes alone cannot
reshape runtime data: the boundary pass and individual rules emit real adapters.
`end_layout()` releases the snapshots after boundary commit.

## Rule groups

Each public class below is a fixed `RewriteRuleClassBase` definition in
[rules.py](rules.py). Its `pattern()` captures candidates, `check()` validates the
supported case, and `rewrite()` emits replacement values. A class handles all
matching nodes; there is no separate rule definition for every graph node.

### Shape controls: `SHAPE_RULES`

These rules change controls or attributes to match the data layout. Shape-control
initializers are shared within the batch when their source and contents agree.

| Class | What it does | Example or scope |
| --- | --- | --- |
| `ArgMaxLayout` | Remaps the reduction axis and restores the planned output view when `keepdims=0`. | Rank-3 axis `1` becomes `2`; reduction keeps its dimension internally before a Reshape. |
| `ExpandLayout` | Pads a fixed target shape with leading ones. | `[N,C]` becomes `[1,1,N,C]`. |
| `FlattenLayout` | Replaces Flatten with a fixed 4D Reshape. | `[N,C,H]`, `axis=2` becomes target `[1,1,N*C,-1]`. |
| `ReshapeLayout` | Pads fixed targets, resolves copied zero dimensions, and supports one singleton removal from a 5D target. | Native target `[0,-1]` becomes `[1,1,N,-1]`. Runtime target controls require additional support. |
| `SliceLayout` | Shifts nonnegative axes, making implicit axes explicit; preserves starts, ends and steps. | Rank-3 axis `0` becomes `1`; negative trailing axes remain valid. |
| `TileLayout` | Pads fixed repetitions with leading ones. | Repetitions `[2,3]` become `[1,1,2,3]`. |
| `TransposeLayout` | Remaps the permutation for promotion or an unambiguous 5D singleton collapse. | Rank-2 permutation `[1,0]` becomes `[0,1,3,2]`. |

### Indexing and reductions: `INDEXING_RULES`

Data rank must be static and at most four. Index views and fixed controls are
shared within the batch rather than recreated for each consumer.

| Class | What it does | Example or scope |
| --- | --- | --- |
| `GatherLayout` | Remaps the data axis and gives indices a flat vector view. | Scalar or multidimensional indices are flattened; a noncanonical private result is allowed only when fixed Reshapes restore it. Such a result cannot be public. |
| `GatherElementsLayout` | Promotes indices to the same 4D rank as data and remaps the axis. | Constant indices get an initializer view; runtime indices get Unsqueeze unless already planned as data. Native data/index ranks must match. |
| `ReduceMaxLayout` | Remaps fixed axes and restores the planned result after dimension removal. | `keepdims=0` becomes `keepdims=1` plus Reshape. |
| `ReduceMeanLayout` | Applies the same layout adaptation while preserving the supported axes representation. | Handles fixed axes supplied as an attribute or an input, as applicable to the schema. |
| `ReduceMinLayout` | Preserves the logical minimum result under data promotion. | Rank-3 nonnegative axes shift by one. |
| `ReduceSumLayout` | Remaps fixed axes and restores removed dimensions through Reshape. | Empty axes with `noop_with_empty_axes=1` retain their no-op behavior. |

Runtime reduction axes are unsupported when conversion is required. Gather's
index-flattening restriction prevents accepting a result whose dimension order
would disagree with the shared plan.

### Axis attributes: `AXIS_RULES`

These rules retain the operator, remap its axis for the planned input layout,
and adapt outputs whose planned view differs from the physical result. They use
the applicable schema default when no axis is supplied.
For promotion, rank-2 axis `1` becomes `3`; negative trailing axes remain valid.
A singleton collapse requires an unambiguous axis mapping.

| Class | What it does |
| --- | --- |
| `ConcatAxisLayout` | Remaps the shared axis while retaining every variadic data operand. |
| `DequantizeLinearAxisLayout` | Remaps standard per-axis dequantization without changing scale or zero point. |
| `ContribDequantizeLinearAxisLayout` | Does the same for `com.microsoft` version 1. |
| `LogSoftmaxAxisLayout` | Preserves the logical LogSoftmax axis. |
| `QuantizeLinearAxisLayout` | Remaps standard per-axis quantization without changing scale or zero point. |
| `ContribQuantizeLinearAxisLayout` | Does the same for `com.microsoft` version 1. |
| `SoftmaxAxisLayout` | Preserves the logical Softmax axis. |
| `TopKAxisLayout` | Remaps TopK and preserves both values and indices outputs and their dtypes. |

QDQ with scalar scale and no explicit axis needs no axis rewrite. These rules do
not convert opset versions; FP16 QDQ schema preparation occurs before the IR
pipeline in [schemas.py](../schemas.py).

### Normalization: `NORMALIZATION_RULES`

| Class | What it does | Supported special case |
| --- | --- | --- |
| `LpNormalizationLayout` | Maps the normalization axis; optionally emits explicit L2 arithmetic for FP16 activations. | For last-axis `p=2` followed by one constant-scaled Mul, emits `x / Max(Sqrt(ReduceSum(x*x)), 2**-14)`. |

The explicit L2 form requires `force_fp16_activations`. Its square/sum can overflow
or underflow, and the denominator floor can affect small values. Other eligible
cases retain LpNormalization with a remapped axis.

### Native operator adapters: `NATIVE_RULES`

These rules keep operators whose semantics require a particular native layout,
placing Reshapes around them as needed.

| Class | What it does | Example or scope |
| --- | --- | --- |
| `GroupQueryAttentionLayout1` | Restores promoted inputs to native shapes, runs one-output GQA, and restores its planned output view. | Query `[1,B,S,H]` is viewed as `[B,S,H]` inside GQA. |
| `GroupQueryAttentionLayout2` | Applies the same adaptation to exactly two declared outputs. | Each output uses its own native and planned shapes. |
| `GroupQueryAttentionLayout3` | Handles attention plus present-key and present-value outputs independently. | Already-4D cache outputs retain their native shape. |
| `InstanceNormalizationLayout` | Preserves rank-3 channel semantics by normalizing through `[N,C,L,1]`, then restoring the planned output. | Leading promotion to `[1,N,C,L]` alone would move the channel axis. Rank-4 input needs no adapter; other ranks are unsupported. |

GQA supports `com.microsoft` version 1, preserving optional operands, attributes
and exact declared output count, including unused outputs. Replacement GQA nodes
are absent from `native_shapes`, so they cannot repeatedly acquire adapters.

### Rank adapters: `RANK_ADAPTER_RULES`

| Class | What it does |
| --- | --- |
| `SqueezeLayout` | Removes a redundant internal view or replaces it with Reshape. Public Squeeze axes can absorb leading promotion to preserve the original output rank. |
| `UnsqueezeLayout` | Removes an internal Unsqueeze when planned input/output views agree, or emits the required planned Reshape. |

A public result may need Identity to retain its name. Squeeze's SDK setup/cleanup
hooks preserve tensor names and source metadata across the whole rank-rule batch.
These rules use `batch_nodes` to avoid adapting their own generated nodes again.

## Direct passes and execution order

| Definition in [passes.py](passes.py) | Responsibility |
| --- | --- |
| `snapshot()` | Captures native shapes and computes the live data plan using operator data-port classification. It analyzes the graph without inserting adapters. |
| `SplitAxisLayout` | Remaps Split's axis and adapts output views while preserving arbitrary output arity. A direct pass avoids fixed pattern output counts; rank-2 axis `1` becomes `3`. |
| `LayoutBoundaryCommit` | Materializes shared input/initializer views, updates planned output metadata, restores native control users, and restores public output shapes, types and names. Prunes unused nodes and topologically sorts. |

For example, a public `[N,C]` input keeps its interface and feeds an Unsqueeze for
4D data consumers. A public output produced in `[1,1,N,C]` gets Squeeze back to
`[N,C]`. Native control consumers continue to receive their original views.

Axis and Split rewrites require a planned data output; control-only nodes retain
native layouts. Selecting a singleton in a 5D tensor can change which singleton
the output plan removes, requiring a real output Reshape even when the axis
number stays unchanged. Replacements retain native input-shape requirements for
shared controls without joining the original eligibility snapshot.

[stages.layout()](../stages.py) executes this fixed sequence:

1. `begin_layout()` / `snapshot()`.
2. `SHAPE_RULES`.
3. `INDEXING_RULES`.
4. `AXIS_RULES`, then `SplitAxisLayout`.
5. `NORMALIZATION_RULES`.
6. `NATIVE_RULES`.
7. `RANK_ADAPTER_RULES`.
8. `LayoutBoundaryCommit`, then `end_layout()`.

The tables list classes in their registered group order. Each group is one
ordered SDK batch. Replacing a batch with sequential whole-graph applications can
change matching behavior. Layout runs inside `individual_op_rewrites`, after
indexing/recurrent/matrix lowering and before logical lowering. The public
pipeline has a fixed stage order.

## Adding a transform

- Add a local shape-control, indexing, axis, normalization, native-view or rank
  adaptation as a documented class in [rules.py](rules.py), using the appropriate
  existing helper where its contract fits. Register it in `__all__` and the
  relevant ordered group tuple.
- Use a direct `ir.passes.InPlacePass` in [passes.py](passes.py) for shared graph
  analysis or work that cannot be represented suitably by a fixed pattern, such
  as variadic output handling. Export it and register its position in
  [stages.py](../stages.py).
- If a new operator participates in layout planning, update `data_ports()` and
  the prepared-operator classification in [passes.py](passes.py). Identify its
  data operands explicitly so shape, index and weight controls keep valid views.
- Validate supported shapes, axes, domains and opsets before mutation. For a
  same-operator rewrite, define why the emitted node will be ineligible on the
  next SDK visit; use the appropriate snapshot only when needed.
- Extend [test_rules.py](../../../../../tests/unit/onnx/graph_surgery/dla_transforms/layout/test_rules.py)
  for pattern behavior or [test_passes.py](../../../../../tests/unit/onnx/graph_surgery/dla_transforms/layout/test_passes.py)
  for direct-pass and boundary behavior. Use small real graphs that exercise an
  actual rewrite, numerical equivalence and public-interface preservation where
  relevant; extend existing parameterization for additional forms.

From the repository root, run the focused tests with:

```bash
pytest tests/unit/onnx/graph_surgery/dla_transforms/layout
```
