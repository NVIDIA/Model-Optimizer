# DLA transformation rules

This package prepares supported ONNX graphs for DLA deployment through the public
`make_dla_compatible` operation. Operator handlers are internal implementation
details. Target-hardware execution remains a separate deployment check.

The [rules-and-passes design](DESIGN.md) describes the shared graph context and
migration decisions. The implementation guide below describes the current code.

## Public model example: MobileGaze (opset 20)

The pretrained [MobileGaze MobileNetV2](https://github.com/yakhyo/gaze-estimation)
ONNX release uses opset 20 and is published under the upstream MIT license.
Its input is `input: FLOAT[1, 3, 448, 448]`; its outputs are
`yaw` and `pitch`, each `FLOAT[1, 90]`.

From the repository root, install the ONNX dependencies:

```bash
python -m pip install -e ".[onnx]"
```

Download the model:

```python
from urllib.request import urlretrieve

urlretrieve(
    "https://github.com/yakhyo/gaze-estimation/releases/download/weights/mobilenetv2_gaze.onnx",
    "mobilenetv2_gaze.onnx",
)
```

Run the fixed transformation pipeline:

```bash
python -m modelopt.onnx.graph_surgery make-dla-compatible --input mobilenetv2_gaze.onnx --output mobilenetv2_dla.onnx --no-external-data
```

`--no-external-data` keeps this small example in one ONNX file. To enable FP16
internal computation, append `--force-fp16-activations` and choose a different
output filename. Both modes preserve opset 20, the 4D input, and the original
2D public outputs; internal activation tensors use 4D layouts.

Check the default FP32 result with ONNX Runtime on a synthetic input:

```python
import numpy as np
import onnxruntime as ort

options = ort.SessionOptions()
options.intra_op_num_threads = options.inter_op_num_threads = 1
original = ort.InferenceSession(
    "mobilenetv2_gaze.onnx", options, providers=["CPUExecutionProvider"]
)
transformed = ort.InferenceSession(
    "mobilenetv2_dla.onnx", options, providers=["CPUExecutionProvider"]
)
inputs = {"input": np.random.default_rng(42).standard_normal((1, 3, 448, 448)).astype(np.float32)}
expected = original.run(None, inputs)
yaw, pitch = transformed.run(None, inputs)
for actual, reference in zip((yaw, pitch), expected, strict=True):
    np.testing.assert_allclose(actual, reference, rtol=1e-4, atol=1e-5)
print(yaw.shape, pitch.shape)  # (1, 90) (1, 90)
```

A four-input CPU smoke check passed for FP32 with maximum absolute error
`2.86e-6`. FP16 conversion also passed ONNX validation and execution, but did not
meet the same numerical tolerance. A plain FP16 conversion showed similar drift;
FP16 accuracy needs evaluation on representative inputs. These checks do not
measure task accuracy or DLA hardware execution.

## Structure and ownership

Each directory guide catalogs its rules and direct passes, explains their supported
forms and execution order, and identifies where to register and test a new transform.

| Guide | Add a transform here when it handles... |
| --- | --- |
| [Preprocessing](preprocessing/README.md) | Quantization removal/folding, constants or FP32 preparation |
| [Connected patterns](patterns/README.md) | A connected sequence of operators rather than one operator |
| [Operators](operators/README.md) | Operator lowering, index types, recurrent or logical computation |
| [Layout](layout/README.md) | Shape, indexing, axis or normalization changes needed for internal layouts |
| [Precision](precision/README.md) | Optional FP16 storage, activation casts or dtype boundaries |

The pipeline holds one live ONNXScript IR model in `DlaGraph`. Rule definitions are
fixed SDK `RewriteRuleClassBase` subclasses; the SDK finds candidate nodes and
calls their eligibility and replacement methods. Direct operations use SDK
`InPlacePass` classes. Package initializers only export their rules and passes.

| Area | Responsibility |
| --- | --- |
| `context.py` | Live model/initializer access, phase snapshots and SDK execution |
| `stages.py` | Ordered fixed rule batches and direct pass invocations |
| `preprocessing/rules.py` | QDQ removal/folding, constants and eligible Cast rewrites |
| `preprocessing/passes.py` | Connected FP32-region analysis and type preparation |
| `patterns/rules.py` | Connected reshape/reduction, TopK/Gather and projection patterns |
| `operators/rules.py` | Indexing, matrix, logical and LSTM transformations |
| `operators/passes.py` | Schema-driven conversion of native integer views at float consumers |
| `layout/rules.py` | Shape, indexing, axis, normalization and native-layout adapters |
| `layout/passes.py` | Shared layout analysis, variadic Split axes and boundary commit |
| `precision/rules.py` | Eligible activation Cast rewriting |
| `precision/passes.py` | FP32 constraints, FP16 tensor storage and precision boundaries |
| `cleanup.py` | Fixed cleanup rules, name restoration and final initializer deduplication |
| `common.py` | Small IR queries and initializer naming/creation |
| `schemas.py`, `validation.py` | Supported ModelProto opset preparation and input contract |

```python
from modelopt.onnx.graph_surgery import dla_make_dla_compatible

dla_make_dla_compatible("model.onnx", "model_dla.onnx")
```

`run_graph_surgery("make-dla-compatible", ...)` and the CLI use the same operation.
The pipeline always runs preprocessing, connected-pattern rewrites, individual
operator rewrites and cleanup, followed by optional FP16 conversion and final
initializer deduplication. The API and CLI do not accept a transform-selection list.

## Rule contract

Each named rule implements `pattern()`, `check()` and `rewrite()` through the SDK.
Its class docstring explains the transformation, supported forms and numerical or
layout limitations, with an example when useful. Definitions remain static; bind
per-run state with `RuleClass.rule(graph, ...)`.

- `pattern()` describes connectivity, operator/domain and fixed input/output forms.
- `check()` validates the matched node and determines whether a rewrite is needed.
  It leaves the graph unchanged. Simple shape information can be recomputed in `rewrite()`.
- `rewrite()` emits replacement IR values; the SDK replaces nodes and reconnects
  consumers. Keep inline operators rather than creating model-local functions.

No graph-wide operator-discovery loop is needed for each rule. Genuine shared
analyses, such as connected dtype regions and data/control layout planning, remain
explicit passes or phase setup. Avoid generic case-dispatch frameworks and caches
of every node's eligibility when a small condition is sufficient.

Multiple nodes share the same rule. Static GQA/LSTM output variants preserve exact
arity, including declared but unused outputs. The SDK fixes pattern output count;
variadic Split axis handling uses a direct pass to retain arbitrary output arity.

The SDK can visit inserted nodes during a rewrite. Each same-operator rule must
become ineligible, or explicitly restrict itself to the proper snapshot.
`batch_nodes` identifies nodes present before the current SDK batch; `native_shapes`
identifies those present at layout entry. These guards are rule-specific: later
rules must still be able to consume nodes generated by earlier rules.

## Graph context and layout state

`DlaGraph.model` is the source of truth. Its `initializers`, `inputs` and `outputs`
properties return live graph collections; they do not copy tensor payloads.
The shared initializer creator reserves fresh names. Existing local view/control
caches are shared only across the rule variants that require that ownership.

`begin_layout()` captures public output names/types/shapes and the original node
input/output shapes, then computes the active layout plan. It runs after earlier
operator lowering, so generated matrix/recurrent data is included. Original shapes
stay unchanged; layout operations may update the active plan. `end_layout()` frees
these snapshots before later precision work. Batch snapshots are released after each
SDK application.

The IR stores current tensor metadata rather than shape history. Replacement
terminals may inherit old metadata, so set new intermediate types/shapes explicitly
when later rules need them. A scalar shape `()` differs from an unknown shape.
Metadata changes alone do not reshape/cast runtime data; emit the appropriate
Reshape, Unsqueeze, Squeeze or Cast as well.

Boundary commit preserves public names, types and shapes. A tensor shared by data
and shape/index/weight controls can require separate views; native control users
must remain valid. An initializer exposed as a graph input is overridable and is
not a fixed constant. Public output ownership may require an Identity or adapter.

Final cleanup merges fixed initializers by exact dtype, shape and contents after
FP16 conversion, with bounded-memory hashing and collision checks. Public input
and output initializers retain their names and independent ownership.

## Cleanup rules and passes

Cleanup lives in [cleanup.py](cleanup.py), outside the phase directories. Its
rewrite rules run together after operator/layout lowering; final initializer
deduplication runs after optional precision conversion.

| Rule or pass | Behavior and eligibility |
| --- | --- |
| `ReshapeChain` | Collapses adjacent fixed Reshapes with a private intermediate. Concrete positive endpoint dimensions and immutable shape controls are required; equal endpoint shapes bypass the views. |
| `IdentityReshape` | Bypasses a fixed Reshape with identical positive input/output dimensions. Retains an Identity when a public output needs its own name. |
| `InternalIdentity` | Bypasses internal Identity nodes except those present at pipeline entry. Public output Identities retain their interface names. |
| `RestoreValueNames` | Restores endpoint names saved by the rewrite batch after SDK replacement aliasing. |
| `DeduplicateInitializers` | Redirects consumers of duplicate fixed initializers to one retained value. Matches dtype, shape and exact contents regardless of name; protects public input/output initializers. |

Add local cleanup rules or passes to this module and register them in the relevant
position in [stages.py](stages.py). Extend
[test_cleanup.py](../../../../tests/unit/onnx/graph_surgery/dla_transforms/test_cleanup.py)
with the graph behavior the cleanup must preserve. Unused-node removal and
topological sorting use existing SDK passes rather than local equivalents.

## Execution and adding a transform

`stages.py` preserves explicit phase and family order. `apply_patterns()` executes
one ordered SDK rule batch; `apply_pattern()` is its single-rule convenience.
Do not replace a batch with sequential single-rule calls casually: node-first
matching differs from running one entire rule before the next.

`apply_pass()` invokes an in-place SDK pass through its contract checks. Pattern
applications return integer rewrite counts; passes return `PassResult.modified`
and expose operation-specific counts where useful. Do not treat a Boolean pass
result as a rewrite count.

To add a transform:

1. Choose its existing phase and owning `rules.py` or `passes.py`.
2. Define a documented rule class or direct pass using the SDK interfaces.
3. Validate supported shapes, dtypes, domains and opsets before changing the graph.
4. Register the definition in the appropriate ordered batch/pass position in
   `stages.py`; preserve any proven producer/consumer dependency.
5. Extend the owning tests with a small real matching graph, expected behavior,
   public-interface checks and meaningful supported variants.
6. Update scope documentation. Keep experiments and expensive model checks out of
   public unit tests.

Rule and pass helpers stay private to their module unless shared by other groups.
The wrapper handles graph access and execution; individual transforms remain in
their modules. Terminal projection is defined with patterns but executes late in
precision, after its producers have been emitted.

## Scope and validation

Only top-level graphs are processed; nested graphs and model-local functions are
rejected. Standard input opsets are 17, 18, 20 and 21. FP16 standard QDQ can require
migration to opset 19 at entry; this does not expand the accepted input versions.

Where lowering assumes finite FP32 values and may change signed zero. Optional
FP16 L2 decomposition uses a denominator floor of `2**-14`, with possible overflow
or underflow. Unsupported forms need targeted support and an explicit refusal
when conversion is required. Unknown operators are not automatically DLA-supported.

Run the synthetic suite and public pipeline cases with:

```bash
pytest tests/unit/onnx/graph_surgery/dla_transforms tests/unit/onnx/graph_surgery/test_make_dla_compatible.py
```

Tests use synthetic graphs without proprietary data, network downloads or target
hardware. Expected graphs must be specified independently of the transform under
test. Each rule needs an eligible case that actually rewrites, and same-operator
termination checks must fail promptly rather than hanging CI.

Use ONNX checking, numerical execution and public-interface checks for their
different purposes. DLA target execution and task accuracy are separate checks.
FP32 and optional FP16 inference outputs need not be bit-identical.
