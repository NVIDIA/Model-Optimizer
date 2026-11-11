# AutoCast Per-Node Calibration Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development
> (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use
> checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add opt-in, memory-bounded per-node calibration to ONNX AutoCast without changing its
default calibration behavior.

**Architecture:** `ReferenceRunner` will retain input loading and provider setup while delegating
per-node execution to focused helpers in the same module. The per-node path builds and runs one
node model at a time, updates `TensorStats` online, and drops activations after their final consumer;
`convert_to_mixed_precision` and the CLI only select this path.

**Tech Stack:** Python, ONNX, ONNX Runtime through Polygraphy, NumPy, pytest.

## Global Constraints

- Preserve existing AutoCast behavior when `calibrate_per_node=False`.
- Do not add dependencies or expose test-only production APIs.
- Use real ONNX Runtime execution in behavioral tests.
- Follow test-first development and keep unit tests well under five seconds.
- Commit with `git commit -s -S`.

---

### Task 1: Per-node reference execution

**Files:**
- Modify: `tests/unit/onnx/autocast/test_referencerunner.py`
- Modify: `modelopt/onnx/autocast/referencerunner.py`

**Interfaces:**
- Consumes: existing `ReferenceRunner`, `TensorStats`, input loaders, and provider preparation.
- Produces: `ReferenceRunner.run(inputs=None, calibrate_per_node: bool = False)`.

- [ ] **Step 1: Write a failing parity test**

Add a branched model test that invokes both execution paths and compares every tensor's runtime
shape, minimum, maximum, and absolute maximum:

```python
def test_per_node_run_matches_all_outputs(reference_runner):
    inputs = {
        "X1": np.array([[1.0, -2.0, 3.0]], dtype=np.float32),
        "X2": np.array([[4.0, 5.0, -6.0]], dtype=np.float32),
    }
    expected = reference_runner.run(inputs)
    actual = reference_runner.run(inputs, calibrate_per_node=True)

    assert actual.keys() == expected.keys()
    for name, expected_value in expected.items():
        assert isinstance(actual[name], TensorStats)
        np.testing.assert_allclose(actual[name].absmax, np.max(np.abs(expected_value)))
        np.testing.assert_allclose(actual[name].min_val, np.min(expected_value))
        np.testing.assert_allclose(actual[name].max_val, np.max(expected_value))
        assert actual[name].shape == expected_value.shape
```

- [ ] **Step 2: Run the parity test and verify RED**

Run:

```bash
pytest -q tests/unit/onnx/autocast/test_referencerunner.py::test_per_node_run_matches_all_outputs
```

Expected: fail because `ReferenceRunner.run()` does not accept `calibrate_per_node`.

- [ ] **Step 3: Implement minimal single-batch per-node execution**

Add helpers that:

```python
def _update_tensor_stats(
    stats: OrderedDict[str, TensorStats], name: str, value: np.ndarray
) -> None:
    batch_absmax = np.max(np.abs(value)) if value.size else 0
    batch_min = np.min(value) if value.size else 0
    batch_max = np.max(value) if value.size else 0
    previous = stats.get(name)
    stats[name] = TensorStats(
        absmax=max(previous.absmax, batch_absmax) if previous else batch_absmax,
        min_val=min(previous.min_val, batch_min) if previous else batch_min,
        max_val=max(previous.max_val, batch_max) if previous else batch_max,
        shape=value.shape,
    )
```

Build each one-node model with its non-initializer `ValueInfoProto` inputs, outputs, consumed
initializers, opset imports, functions, and IR version. Execute it through `_get_ort_runner`, close
the runner immediately, and retain output arrays only until all consumers have run.

- [ ] **Step 4: Run the parity test and verify GREEN**

Run the command from Step 2. Expected: pass.

- [ ] **Step 5: Write a failing multi-batch aggregation test**

Extend the existing directory-based calibration test:

```python
results = reference_runner.run(temp_dir, calibrate_per_node=True)
assert results["Y1"] == TensorStats(
    absmax=17.0, min_val=3.0, max_val=17.0, shape=(1, 3)
)
assert results["Y2"] == TensorStats(
    absmax=70.0, min_val=2.0, max_val=70.0, shape=(1, 3)
)
```

- [ ] **Step 6: Run the multi-batch test and verify RED**

Run:

```bash
pytest -q tests/unit/onnx/autocast/test_referencerunner.py -k per_node
```

Expected: the new aggregation assertion fails until stats merge across batches.

- [ ] **Step 7: Implement online aggregation and dependency validation**

Iterate each calibration batch directly, update the same ordered statistics mapping, compute tensor
consumer counts once, and raise a `ValueError` naming the node and missing runtime dependency if a
node cannot be executed.

- [ ] **Step 8: Run focused runner tests**

Run:

```bash
pytest -q tests/unit/onnx/autocast/test_referencerunner.py --durations=10
```

Expected: all tests pass and new tests remain under five seconds.

- [ ] **Step 9: Commit**

```bash
git add modelopt/onnx/autocast/referencerunner.py \
  tests/unit/onnx/autocast/test_referencerunner.py
git commit -s -S -m "feat: add AutoCast per-node reference execution"
```

### Task 2: Public API and CLI

**Files:**
- Modify: `tests/unit/onnx/autocast/test_autocast.py`
- Modify: `modelopt/onnx/autocast/convert.py`
- Modify: `modelopt/onnx/autocast/__main__.py`

**Interfaces:**
- Consumes: `ReferenceRunner.run(..., calibrate_per_node=True)` from Task 1.
- Produces: `convert_to_mixed_precision(..., calibrate_per_node: bool = False)` and
  `--calibrate_per_node`.

- [ ] **Step 1: Write failing API and CLI tests**

Add one real conversion test and extend parser coverage:

```python
def test_convert_with_per_node_calibration(temp_model_path):
    converted = convert_to_mixed_precision(temp_model_path, calibrate_per_node=True)
    assert isinstance(converted, onnx.ModelProto)
    onnx.checker.check_model(converted)


def test_calibrate_per_node_parser_argument():
    parser = get_parser()
    assert parser.parse_args(
        ["--onnx_path", "test.onnx", "--calibrate_per_node"]
    ).calibrate_per_node
    assert not parser.parse_args(["--onnx_path", "test.onnx"]).calibrate_per_node
```

- [ ] **Step 2: Run tests and verify RED**

Run:

```bash
pytest -q tests/unit/onnx/autocast/test_autocast.py \
  -k "per_node or parser_argument"
```

Expected: fail because the API and parser option do not exist.

- [ ] **Step 3: Implement API and CLI forwarding**

Add the defaulted API argument and docstring, pass it to `ReferenceRunner.run`, define the
`store_true` parser option, and forward the parsed value from `main`.

- [ ] **Step 4: Run tests and verify GREEN**

Run the command from Step 2. Expected: pass.

- [ ] **Step 5: Commit**

```bash
git add modelopt/onnx/autocast/convert.py modelopt/onnx/autocast/__main__.py \
  tests/unit/onnx/autocast/test_autocast.py
git commit -s -S -m "feat: expose AutoCast per-node calibration"
```

### Task 3: Documentation and verification

**Files:**
- Modify: `docs/source/guides/8_autocast.rst`

**Interfaces:**
- Consumes: public API and CLI from Task 2.
- Produces: documented behavior and final verification evidence.

- [ ] **Step 1: Document the option**

Add `calibrate_per_node=False` to the Python example and explain that enabling it lowers peak
calibration memory by executing nodes separately while increasing runtime.

- [ ] **Step 2: Run focused AutoCast tests**

Run:

```bash
pytest -q tests/unit/onnx/autocast --durations=20
```

Expected: all tests pass.

- [ ] **Step 3: Run pre-commit on changed files**

Run:

```bash
pre-commit run --files modelopt/onnx/autocast/referencerunner.py \
  modelopt/onnx/autocast/convert.py modelopt/onnx/autocast/__main__.py \
  tests/unit/onnx/autocast/test_referencerunner.py \
  tests/unit/onnx/autocast/test_autocast.py docs/source/guides/8_autocast.rst
```

Expected: all available hooks pass; report any environment-only unavailable hook explicitly.

- [ ] **Step 4: Commit documentation and hook fixes**

```bash
git add docs/source/guides/8_autocast.rst
git commit -s -S -m "docs: document AutoCast per-node calibration"
```

- [ ] **Step 5: Audit the final branch**

Verify:

```bash
git status --short
git log --show-signature --format=full -3
git diff --check upstream/main...HEAD
git diff --shortstat upstream/main...HEAD -- . ':!tests' ':!docs'
```

Confirm the only untracked files are the user's pre-existing `.idea/` files, all new commits contain
`Signed-off-by`, signatures are valid, the source insertion budget is below 500 lines, and every
objective requirement has direct test or repository evidence.
