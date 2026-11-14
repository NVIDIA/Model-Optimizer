# Operator transformations

This directory lowers individual operators into supported primitives or gives
their operands private dtype views. Put a new operator lowering in
[rules.py](rules.py). Put coordinated layout changes in [layout](../layout/README.md)
and connected multi-operator decompositions in [patterns](../patterns/README.md).
The [pipeline README](../README.md) describes the shared rule contract and graph
context.

## Execution order

[stages.py](../stages.py) applies `INDEXING_RULES`, `RECURRENT_RULES` and
`MATRIX_RULES` as three ordered SDK batches. It then removes unused nodes, runs
the layout phase, applies `LOGICAL_RULES`, and runs `FloatConsumerViews`.
This order lets layout processing see the emitted matrix and recurrent graphs,
and lets logical rules consume the resulting layouts.

Each batch receives one shared `views` dictionary. Related rules reuse converted
indices, weights, activations or masks without changing the original owner used
by other consumers. These caches are local to the batch. The public input/output
names, shapes and dtypes remain part of the pipeline's boundary contract.

## Indexing rules

| Rule | Transformation | Supported conditions |
| --- | --- | --- |
| `GatherIndices` | `Gather(data, indices64)` becomes `Gather(data, indices32)`. Runtime indices get a Cast; fixed indices get an INT32 initializer. | INT64 indices, a static valid data axis, and an axis extent that fits INT32. Existing INT32 indices are unchanged. |
| `GatherElementsIndices` | Applies the same private INT32 conversion to `GatherElements`, retaining index rank. | Same checks as `GatherIndices`; both rules reuse one converted view for a shared index owner. |

The conversion does not change the indexing axis or the shared source tensor.
Unsupported index dtypes or required dynamic/oversized indexing forms raise an
error rather than silently narrowing them.

## Matrix rules

| Rule | Transformation | Supported conditions |
| --- | --- | --- |
| `MatMulToConv` | `MatMul(x, W)` becomes rank adapters and Transpose, a pointwise Conv with an `OI11` kernel, then Transpose and optional Squeeze. | Fixed weights of rank 2–4 with singleton leading batch dimensions; static positive activation shapes of rank 2–4, with leading dimension 1 for rank 4. |
| `GemmToConv` | Lowers two-input `Gemm` to the same pointwise Conv form, honoring `transA` and `transB`. | Rank-two activations and weights; `alpha == beta == 1`. |
| `GemmBiasToConv` | Lowers three-input `Gemm`, keeping its channel-vector bias as the Conv bias. | The Gemm checks above and bias shape equal to the output-channel vector. |

For example, `MatMul([M,K], [K,N])` becomes a Conv with activation
`[1,K,1,M]` and kernel `[N,K,1,1]`, then restores `[M,N]`. Activation dtypes
are FLOAT, FLOAT16 or DOUBLE. Runtime weights stay unchanged. Fixed weights may
come through standard or `com.microsoft` DequantizeLinear with scalar or vector
scales; blocked quantization is excluded. Kernel transposition also remaps a
per-axis quantization axis.

`weight_eligible()` is a shared read-only predicate, also used by preprocessing
to retain suitable quantized weight chains. It is not a registered rewrite rule.

## Recurrent rules

| Rule | Transformation | Outputs retained |
| --- | --- | --- |
| `Lstm1` | Unrolls a fixed sequence into Conv gate projections, Sigmoid/Tanh and elementwise state updates. | Sequence `Y`. |
| `Lstm2` | Uses the same unrolling. | Sequence `Y` and final hidden state `Y_h`. |
| `Lstm3` | Uses the same unrolling. | Sequence `Y`, final hidden state `Y_h`, and final cell state `Y_c`. |
| `LstmSequenceLayout` | Replaces an eligible sequence `Transpose -> Reshape` with an identity Transpose and a Reshape whose copied dimensions are resolved. | The original consumer shape and shape-control ownership. |

LSTM inputs must be static FP32 `[steps,1,features]` with `layout=0`, fixed W/R,
and an optional fixed bias. Forward, reverse and bidirectional directions are
supported. Initial states may be fixed or runtime tensors with native
`[directions,1,hidden]` shapes. Sequence lengths, peepholes, custom activations,
clipping and coupled input/forget gates require additional support.

The three LSTM definitions have fixed output counts because SDK patterns fix
arity. The count must match all declared outputs, including unused states.
`LstmSequenceLayout` only handles consumers marked during LSTM unrolling: their
permutation moves singleton dimensions without reordering non-singleton axes,
and the final Reshape has rank below four and fixed controls.

## Logical rules

| Rule | Transformation | Supported conditions |
| --- | --- | --- |
| `WhereZero` | `Where(c,x,0)` becomes `x * Cast(c,FLOAT)`; `Where(c,0,y)` uses `y * (1-mask)`. | One fixed all-zero branch, finite FP32 data and known shapes; the zero branch must not expand the broadcast result. Signed zero and NaN/Inf equivalence are outside this contract. |
| `LogicalNot` | `Not(Cast(x,BOOL))` becomes `1 - Clip(Cast(x,FLOAT),0,1)`. | Input is BOOL or a fixed 0/1 tensor; result is private and used only by non-string standard Casts. Public boolean outputs stay unchanged. |
| `GreaterFloat` | `Greater(x,y)` uses private FLOAT operand views. | Narrow numeric operands convert exactly; INT32/INT64 cases require a fixed operand strictly inside `(-2**24, 2**24)`. Existing FLOAT/FLOAT comparisons stay unchanged. |

## Direct pass

[passes.py](passes.py) contains `FloatConsumerViews`. After logical lowering it
finds integer Unsqueeze outputs used at standard-operator ports whose ONNX schema
requires FLOAT, or whose shared type variable is established as FLOAT by other
inputs/outputs. It inserts one reusable FLOAT Cast per source and rewires only
those ports. Other integer data/control consumers keep their original view.
For example, an INT32 Unsqueeze used by a FLOAT Add gets a private Cast before
that Add. This is a direct `InPlacePass` because consumers can have arbitrary
operator types and input/output counts; it is not a fixed pattern family.

## Adding a transform

1. Add a documented `RewriteRuleClassBase` subclass to [rules.py](rules.py), with
   `pattern()`, eligibility checks in `check()`, and replacement construction in
   `rewrite()`. Reuse a private base only when its contract fits. `_OperatorRule`,
   `_IndexRule`, `_MatrixRule`, `_LogicalRule` and `_LstmRule` are implementation
   helpers, not standalone registrations.
2. Export the class through `__all__` and add it to the appropriate fixed
   `INDEXING_RULES`, `RECURRENT_RULES`, `MATRIX_RULES` or `LOGICAL_RULES` tuple.
   If it needs a different dependency position, register it explicitly in
   [stages.py](../stages.py). Keep fixed arity variants when output counts differ.
3. For graph-wide work, define an `InPlacePass` in [passes.py](passes.py), export
   it, and call it through `graph.apply_pass()` at the required stage position.
4. Preserve shared owners, set new intermediate type/shape metadata, and use
   `graph.add_initializer()` for new constants. Same-operator rewrites must
   become ineligible or use the appropriate batch snapshot to terminate.
5. Extend [test_rules.py](../../../../../tests/unit/onnx/graph_surgery/dla_transforms/operators/test_rules.py)
   or [test_passes.py](../../../../../tests/unit/onnx/graph_surgery/dla_transforms/operators/test_passes.py)
   with a small real graph covering the behavior and relevant shared/public
   ownership cases. Update this catalog when adding a rule.

Run focused tests from the repository root:

```bash
pytest tests/unit/onnx/graph_surgery/dla_transforms/operators
```
