# Preprocessing rules and passes

This phase prepares quantization, constants and connected FP32 regions before
connected-pattern and individual-operator rewrites. Put transformations here when
they prepare the original graph for later lowering. See the
[pipeline guide](../README.md) for the fixed phase order.

## Rules

The SDK matches the definitions in [rules.py](rules.py). Both standard ONNX and
`com.microsoft` Q/DQ domains have fixed rule instances in this phase.

| Rule | What it does and when it applies | Example |
| --- | --- | --- |
| `SelectedQdq` | Bypasses a DQ branch fed by a selected QuantizeLinear storage type. The default selection is INT16/UINT16; `qdq_quantized_dtypes` changes that selection and `remove_all_qdq` selects every storage type. Supports omitted zero points and explicit storage dtype attributes. Retains other users of a shared quantizer and preserves endpoint names/types with name restoration, Identity or Cast. | `x -> Q(INT16) -> DQ -> y` becomes a direct floating connection, with an adapter if required. |
| `ConstantInitializer` | Materializes dense tensor, numeric scalar and numeric list Constant attributes as initializers. Preserves dtype and scalar/vector rank. Sparse and string attribute forms need additional support and raise an error. | `Constant(value_float=1.25)` becomes an FP32 scalar initializer. |
| `Fp32Cast` | Changes non-FLOAT Cast targets approved by `AnalyzeFp32Casts` to FLOAT. A companion pass updates the connected values' metadata. Does not promote integer shape controls, TopK indices, unknown-schema regions or regions tied to fixed non-FLOAT operands. | `Cast(INT32) -> Abs` becomes `Cast(FLOAT) -> Abs` when the connected region allows FLOAT. |
| `StaticDequantization` | Evaluates immutable scalar/per-axis DQ into an FP32 initializer. Supports INT8/UINT8/INT16/UINT16/INT32 data, FP32 scale/output and optional zero points. Leaves runtime controls, blocked/packed forms and non-FP32 outputs untouched. Shares identical fixed DQ expressions within the batch. Preserves INT8/UINT8 Conv/ConvTranspose weights and eligible MatMul/Gemm weights for later operator lowering. | `DQ(fixed_int8, fixed_scale, fixed_zero) -> Add` becomes an FP32 initializer feeding Add. |

Q/DQ bypass intentionally removes quantization rounding and clipping; it is not
an exact numerical simplification for arbitrary inputs. Likewise, promoting an
integer Cast to FLOAT changes truncation semantics. These are preparation
policies with eligibility constraints, rather than general ONNX optimizations.

Static DQ folding uses [weight_eligible](../operators/rules.py) to share ownership
with matrix lowering. That predicate checks fixed weight layout, activation
rank/type, DQ controls and supported Gemm attributes. Extend that shared predicate
when matrix support changes instead of duplicating its checks here.

## Direct passes and execution order

[passes.py](passes.py) contains operations that need connected type analysis:

| Pass | Responsibility |
| --- | --- |
| `AnalyzeFp32Casts` | Builds schema type-parameter connections, excludes regions that cannot use FLOAT, and records eligible Cast outputs plus propagated values. It does not mutate the model. Non-FLOAT inputs/initializers and schema-mandated integer controls block their connected region. |
| `PromoteFp32Values` | Applies FLOAT metadata to the approved connected values after the Cast rule has changed the actual operations. Metadata alone does not insert a runtime conversion. |

[stages.preprocessing](../stages.py) executes these steps in order:

1. Apply standard/contrib `SelectedQdq` in one batch with shared endpoint-name state.
2. Run `RestoreValueNames` and remove unused nodes.
3. Apply `ConstantInitializer`.
4. Run `AnalyzeFp32Casts`; when it finds candidates, apply `Fp32Cast`, then run
   `PromoteFp32Values`.
5. Apply standard/contrib `StaticDequantization` in one batch with a shared fold cache.
6. Remove unused nodes.

Before the IR phase, the [public entry point](../../make_dla_compatible.py)
validates and infers the model. With `force_fp16_activations=True`, it first uses
the ONNX version converter through [schemas.py](../schemas.py) to upgrade supported
opset-17/18 models containing standard Q/DQ to opset 19. Standard Q/DQ gained FP16
floating operands/results at opset 19. Newer supported opsets and graphs without
standard Q/DQ do not need that upgrade. Individual rules do not upgrade the graph
opset. The later [precision phase](../precision/README.md) performs FP16 conversion.

## Adding a transform

1. Add a documented `RewriteRuleClassBase` subclass to `rules.py` for a local
   match/check/replacement. Use `passes.py` for shared analysis or a direct IR pass
   that cannot be expressed naturally as a fixed pattern.
2. Export it through the owning module's `__all__`; package initializers re-export
   the modules. Bind per-run state through [DlaGraph](../context.py).
3. Register it in the appropriate position in `stages.preprocessing`. The pipeline
   is fixed; adding a rule does not add a user-selectable transform option.
4. Extend [test_preprocessing.py](../../../../../tests/unit/onnx/graph_surgery/dla_transforms/test_preprocessing.py)
   with a real matching graph, relevant shared/public consumers and dtype/control
   cases. Extend [the API tests](../../../../../tests/unit/onnx/graph_surgery/test_make_dla_compatible.py)
   when changing entry-point or opset behavior.

Put connected multi-operator lowering in [patterns](../patterns/README.md),
operator replacement in [operators](../operators/README.md), layout adaptation in
[layout](../layout/README.md), and optional FP16 storage/boundaries in
[precision](../precision/README.md).
