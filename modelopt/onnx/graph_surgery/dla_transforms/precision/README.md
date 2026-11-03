# Precision rules and passes

This optional phase converts eligible internal FLOAT computation and initializer
storage to FLOAT16 while keeping public FLOAT interfaces and required FP32
regions. It runs after layout and cleanup only when
`force_fp16_activations=True`. See the [pipeline guide](../README.md) for the fixed
phase order.

## Rule and direct passes

The single local rewrite is in [rules.py](rules.py); graph-wide precision analysis
and boundary work are in [passes.py](passes.py).

| Definition | What it does and when it applies | Example |
| --- | --- | --- |
| `PrepareFp16` pass | Finds FP32 constraints, converts eligible FLOAT initializer payloads and internal value metadata to FLOAT16, and protects public FLOAT outputs with named boundaries. Preserves overridable initializers, retained/native values, and initializers used at required FP32 ports. Records the original consumers and shares boundary Casts with the following pass. | FLOAT weights become FLOAT16 storage; internal arithmetic feeds a FLOAT output Cast. |
| `Fp16Cast` rule | Retargets internal `Cast(to=FLOAT)` to `Cast(to=FLOAT16)` unless the result is public or belongs to a retained FP32 region. The new target attribute makes the replacement ineligible for another match. | An existing FLOAT input Cast becomes the FP16 boundary instead of adding another Cast. |
| `CastFp16Boundaries` pass | Reconnects the recorded consumers to FLOAT or FLOAT16 operands according to the preparation analysis. Reuses Casts for the same source and destination dtype. Integer controls remain integer. | A FLOAT producer shared by NonMaxSuppression and Mul remains FLOAT; Mul receives its own FLOAT16 view. |

The `fp32_regions` helper in `passes.py` keeps schema-mandated FLOAT ports and
their producer dependencies in FP32. It also retains unknown/overloaded native
operations and regions with non-tensor values. Resize ROI/scales stay FP32.
Known Microsoft Q/DQ and Gelu use the corresponding standard schema for precision
constraints. Explicitly generated FLOAT outputs are retained where changing only
their metadata would disagree with the operation.

Public input declarations stay unchanged. Rank-promotion Unsqueeze adapters on
FLOAT inputs keep their FLOAT view; eligible arithmetic receives a Cast after
that view. Public Squeeze/Cast endpoints and the supported integer-fed terminal
Concat boundary retain their output policy. Output names stay attached to their
public values; newly needed internal names avoid existing owners.

For example, ordinary FLOAT arithmetic becomes:

```text
Before: x:FLOAT ---------> Add(weight:FLOAT) ----------------------> y:FLOAT
After:  x:FLOAT -> Cast16 -> Add(weight:FLOAT16) -> Cast32 ---------> y:FLOAT
```

This changes storage precision and arithmetic rounding. ONNX validation and
unchanged public shapes/types do not establish numerical equivalence or task
accuracy; evaluate FP16 outputs on representative inputs.

## Execution order and related rules

[stages.precision](../stages.py) executes:

1. `PrepareFp16`.
2. `Fp16Cast`, using the preparation's retained FP32 values.
3. `CastFp16Boundaries`, using the same analysis and Cast cache.
4. Topological sorting.
5. `TerminalProjection`, defined in [patterns/rules.py](../patterns/rules.py).
6. Removal of unused nodes.

`TerminalProjection` executes here because its recognized terminal
Conv/Transpose/Concat/ReduceSum/Cast/Squeeze tail needs the established FP16
types. It splits private projections exceeding 8192 output channels into bounded
Conv chunks, reshapes and casts the chunk outputs to FLOAT, and concatenates them
on axis 2. It accepts the supported FP16 or scalar-dequantized UINT8 fixed kernels
with singleton spatial output. Shared weights/controls or other tails remain
unchanged. See the [pattern guide](../patterns/README.md).

The flag also affects the earlier `LpNormalizationLayout` rule in
[layout](../layout/README.md): its supported scaled last-axis L2 case becomes
`x / Max(Sqrt(ReduceSum(x*x)), 2**-14)`. The denominator floor and FP16 square/sum
overflow or underflow are part of that rule's numerical behavior. Its ownership
remains in layout because it needs layout shapes and axis planning.

Standard Q/DQ on supported input opsets 17/18 is upgraded to opset 19 at the
[public entry point](../../make_dla_compatible.py), before inference and IR
rewrites, when this flag is enabled. [schemas.py](../schemas.py) calls the ONNX
version converter directly. This enables the standard FP16 Q/DQ schema; it is not
an opset change performed by precision rules. See
[preprocessing](../preprocessing/README.md) for the entry conditions.

After this phase, final [initializer deduplication](../cleanup.py) merges equal
fixed storage, including duplicates created by FP16 rounding, and the pipeline
sorts the graph again.

## Adding a transform

1. Add a documented `RewriteRuleClassBase` subclass to `rules.py` when it rewrites
   a local precision pattern. Add an `InPlacePass` to `passes.py` when it needs
   graph-wide storage conversion, retained-region analysis or shared boundary
   coordination. Keep schema constraints in the existing analysis where possible.
2. Export the class through the owning module's `__all__` and register it at its
   required position in `stages.precision`. The pipeline is fixed; there is no
   transform-selection list.
3. Set changed tensor metadata and emit the corresponding runtime conversions;
   changing `ir.Value.type` alone does not cast tensor data. Preserve public
   endpoints, fixed controls and shared consumers requiring different dtypes.
4. Extend [test_precision.py](../../../../../tests/unit/onnx/graph_surgery/dla_transforms/test_precision.py)
   with a real graph exercising the changed dtype boundary and numerical result.
   Existing tests cover public I/O, shared Casts, required FP32 regions, promoted
   input adapters and bounded terminal projections.

Place a connected structural rewrite in [patterns](../patterns/README.md) even
if it must execute late here; place axis/layout-sensitive work in
[layout](../layout/README.md). Register the execution dependency explicitly in
[stages.py](../stages.py).
