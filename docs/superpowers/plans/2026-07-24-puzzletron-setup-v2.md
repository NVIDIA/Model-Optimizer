# Puzzletron Setup v2 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a new navigable, schema-driven Puzzletron setup wizard with explicit defaults, per-stage resources and batches, multiple vLLM measurements, complete guided MIP controls, and editable post-MIP flows.

**Architecture:** Keep the existing setup wizard unchanged and add `examples/puzzletron/puzzletron_setup_v2.py` backed by focused modules under `puzzletron_setup/v2/`. Persist a versioned authoring state with field provenance and dependency-aware staleness, then render it into the existing experiment/runner/execution runtime contracts. Extend vLLM statistics with backward-compatible named measurements whose shards, topology, cache identity, and aggregate outputs remain independent.

**Tech Stack:** Python, dataclasses, `questionary`/`prompt_toolkit`, PyYAML, OmegaConf at runtime, Puzzletron orchestration/MIP/post-MIP compilers, pytest.

## Global Constraints

- Preserve `examples/puzzletron/puzzletron_setup.py` and its current CLI behavior.
- The new entry point is `examples/puzzletron/puzzletron_setup_v2.py`.
- The new CLI supports `--resume PATH` and explicit `--defaults PATH`; it has no `--detailed`.
- Never discover or implicitly load `nv-internal/sepehr_defaults.yaml`.
- Provide Back at every prompt: visible `← Back` choices and `:back` for text/numeric input.
- Preserve downstream answers and selectively revalidate only dependent fields.
- Ask for independent instances/workers and derive scheduler task count.
- Round invalid batch requests upward visibly and record requested and effective values.
- Create parallel profiles at first use; later stages may reuse or copy them.
- Provide guided controls only; do not add a raw-YAML editor.
- One named vLLM measurement is one exact ISL/OSL/batch/concurrency/topology point.
- Use one default post-MIP flow per MIP run, combining selected variants and objectives.
- The recommended post-MIP flow must not contain Initial Filter.
- Display PTQ and downstream evaluation as unavailable; never emit them.
- `nv-internal/sepehr_defaults.yaml` must set dataset, `.venv_new`, container, mount, pre-run command, Slurm account, and CPU partition `cpu_interactive`.
- The wizard validates and writes bundles but never launches the orchestrator.
- Keep the dependency-light setup path importable without PyTorch and compatible with the repository's lightweight Python environment.
- Use relative repository paths in commands.
- Prefix shell commands with `rtk proxy`.
- Create signed local commits with `git commit -s -S`; never push without explicit approval.

---

## File Structure

### New public and wizard files

- `examples/puzzletron/puzzletron_setup_v2.py`: dependency-light public entry point.
- `puzzletron_setup/v2/__init__.py`: public v2 package exports.
- `puzzletron_setup/v2/cli.py`: v2 argument parsing and process exit behavior.
- `puzzletron_setup/v2/state.py`: authoring-state persistence, provenance, dependency graph, and stale-field revalidation.
- `puzzletron_setup/v2/defaults.py`: defaults schema, file loading, and precedence.
- `puzzletron_setup/v2/session.py`: prompt-frame stack and nested-editor navigation.
- `puzzletron_setup/v2/prompts.py`: interactive and scripted prompt backends.
- `puzzletron_setup/v2/resources.py`: reusable profiles, batch resolution, mesh checks, and allocation summaries.
- `puzzletron_setup/v2/vllm.py`: named-measurement builder/editor and workload projection.
- `puzzletron_setup/v2/mip.py`: guided MIP run/variant/matrix builder/editor.
- `puzzletron_setup/v2/post_mip.py`: guided flow/node builder/editor and recommended template.
- `puzzletron_setup/v2/bundle.py`: canonical experiment/runner/execution rendering and atomic publication.
- `puzzletron_setup/v2/validation.py`: cross-section validation and navigable issues.
- `puzzletron_setup/v2/wizard.py`: ordered top-level interaction.
- `nv-internal/sepehr_defaults.yaml`: explicitly selected personal defaults.

### Runtime files changed for named vLLM measurements

- `modelopt/torch/puzzletron/subblock_stats/measurements.py`: normalize legacy and named measurement contracts.
- `modelopt/torch/puzzletron/stages/pipeline.py`: execute/finalize measurement-specific runtime statistics.
- `modelopt/torch/puzzletron/orchestration/adapters/sharded.py`: create measurement-aware work items and topology-aware attempts.
- `modelopt/torch/puzzletron/orchestration/adapters/stage_compat.py`: validate aggregate measurement completion.
- `examples/puzzletron/run_runtime_stats_shard.py`: select one named measurement per shard attempt.
- `examples/puzzletron/configs/base.yaml`: document the empty named-measurement mapping while preserving legacy fields.

### Tests

- `tests/unit/torch/puzzletron/test_setup_v2_defaults_state.py`
- `tests/unit/torch/puzzletron/test_setup_v2_session.py`
- `tests/unit/torch/puzzletron/test_setup_v2_resources.py`
- `tests/unit/torch/puzzletron/test_vllm_measurements.py`
- `tests/unit/torch/puzzletron/test_setup_v2_mip.py`
- `tests/unit/torch/puzzletron/test_setup_v2_post_mip.py`
- `tests/unit/torch/puzzletron/test_setup_v2_bundle.py`
- `tests/unit/torch/puzzletron/test_setup_v2_wizard.py`

---

### Task 1: Versioned State and Explicit Defaults

**Files:**
- Create: `puzzletron_setup/v2/__init__.py`
- Create: `puzzletron_setup/v2/defaults.py`
- Create: `puzzletron_setup/v2/state.py`
- Create: `nv-internal/sepehr_defaults.yaml`
- Test: `tests/unit/torch/puzzletron/test_setup_v2_defaults_state.py`

**Interfaces:**
- Produces: `ResolvedDefault(value: Any, source: str)`.
- Produces: `load_defaults(path: Optional[Path]) -> Mapping[str, Any]`.
- Produces: `DefaultsResolver.resolve(path: str, fallback: Any = None) -> ResolvedDefault`.
- Produces: `FieldRecord(value, source, dependencies, stale, requested, effective)`.
- Produces: `WizardState.start(campaign_dir, defaults_path)` and `WizardState.resume(path)`.
- Produces: `WizardState.set_field`, `get_field`, `mark_dependents_stale`, `revalidate`, `push_frame`, and `pop_frame`.
- Consumes: only PyYAML and dependency-light standard-library modules.

- [ ] **Step 1: Write failing defaults and state tests**

```python
def test_explicit_defaults_have_lower_precedence_than_preserved_answers(tmp_path):
    defaults_path = tmp_path / "defaults.yaml"
    defaults_path.write_text("schema_version: 1\ndata:\n  source: /default/data\n")
    resolver = DefaultsResolver(
        builtins={"data": {"source": "/builtin/data"}},
        model_derived={},
        file_defaults=load_defaults(defaults_path),
        preserved={"data": {"source": "/saved/data"}},
    )
    assert resolver.resolve("data.source") == ResolvedDefault("/saved/data", "preserved")


def test_changing_field_marks_only_transitive_dependents_stale(tmp_path):
    state = WizardState.start(tmp_path / "campaign", defaults_path=None)
    state.set_field("model.hidden_size", 4096, dependencies=())
    state.set_field("resources.width.tp", 2, dependencies=("model.hidden_size",))
    state.set_field("output.result_root", "/results", dependencies=())
    state.set_field("model.hidden_size", 2048, dependencies=())
    assert state.field("resources.width.tp").stale is True
    assert state.field("output.result_root").stale is False
```

- [ ] **Step 2: Run the tests and verify the new package is missing**

Run:

```bash
rtk proxy .venv-orchestrator/bin/python -m pytest tests/unit/torch/puzzletron/test_setup_v2_defaults_state.py -q
```

Expected: collection fails with `ModuleNotFoundError: No module named 'puzzletron_setup.v2'`.

- [ ] **Step 3: Implement defaults loading and precedence**

Implement the exact public shape:

```python
@dataclass(frozen=True)
class ResolvedDefault:
    value: Any
    source: str


class DefaultsResolver:
    def __init__(self, *, builtins, model_derived, file_defaults, preserved):
        self._layers = (
            ("builtin", builtins),
            ("model", model_derived),
            ("defaults_file", file_defaults),
            ("preserved", preserved),
        )

    def resolve(self, path: str, fallback: Any = None) -> ResolvedDefault:
        resolved = ResolvedDefault(fallback, "fallback")
        for source, layer in self._layers:
            found, value = _lookup(layer, path)
            if found:
                resolved = ResolvedDefault(deepcopy(value), source)
        return resolved
```

Validate `schema_version == 1`, reject unknown dotted paths with the exact offending path, and
return `{}` when `path is None`.

- [ ] **Step 4: Implement atomic state, field provenance, and dependency invalidation**

Use this persisted shape:

```yaml
schema_version: 1
wizard_version: 2
defaults_path: null
fields: {}
navigation:
  frames: []
  cursor: null
collections: {}
updated_at: ""
```

Implement `FieldRecord.to_dict/from_dict`, atomic sibling-temp-file replacement, reverse
dependency traversal, and validator callbacks:

```python
Validator = Callable[[Any, "WizardState"], Optional[str]]

def revalidate(self, validators: Mapping[str, Validator]) -> Mapping[str, str]:
    issues = {}
    for path, record in self._fields.items():
        if not record.stale:
            continue
        error = validators[path](record.effective, self)
        record.stale = error is not None
        if error is not None:
            issues[path] = error
    self.save()
    return issues
```

- [ ] **Step 5: Add the exact personal defaults file**

```yaml
schema_version: 1
data:
  source: /lustre/fs1/portfolios/coreai/projects/coreai_dlalgo_llm/users/ssameni/engineering/Puzzle-KD-Nemotron-Post-Training-Dataset-v2/
infrastructure:
  execution_contract:
    venv: .venv_new
    container: /lustre/fs1/portfolios/coreai/projects/coreai_dlalgo_llm/users/ssameni/enroot/pytorch_25p05.sqsh
    container_mounts: /lustre:/lustre
    prerun_commands:
      - source /lustre/fs1/portfolios/coreai/projects/coreai_dlalgo_llm/users/ssameni/setup-envs.sh
  runner:
    kind: slurm
    slurm:
      account: coreai_dlalgo_llm
      partition_cpu: cpu_interactive
```

- [ ] **Step 6: Run focused tests**

Run:

```bash
rtk proxy .venv-orchestrator/bin/python -m pytest tests/unit/torch/puzzletron/test_setup_v2_defaults_state.py -q
```

Expected: all tests pass.

- [ ] **Step 7: Commit**

```bash
rtk proxy git add puzzletron_setup/v2/__init__.py puzzletron_setup/v2/defaults.py puzzletron_setup/v2/state.py nv-internal/sepehr_defaults.yaml tests/unit/torch/puzzletron/test_setup_v2_defaults_state.py
rtk proxy git commit -s -S -m "feat: add Puzzletron v2 setup state"
```

### Task 2: Universal Back Navigation and Scriptable Prompt Sessions

**Files:**
- Create: `puzzletron_setup/v2/session.py`
- Create: `puzzletron_setup/v2/prompts.py`
- Test: `tests/unit/torch/puzzletron/test_setup_v2_session.py`

**Interfaces:**
- Consumes: `WizardState.push_frame` and `WizardState.pop_frame`.
- Produces: singleton `BACK`, `PromptChoice`, `PromptBackend` protocol, `InteractiveBackend`,
  `ScriptedBackend`, `PromptFrame`, and `WizardSession`.
- Produces: `WizardSession.text`, `integer`, `confirm`, `select`, and `checkbox`.
- Guarantees: all methods may return `BACK`; no method records `:back` as an answer.

- [ ] **Step 1: Write failing one-step-back and nested-editor tests**

```python
def test_text_back_returns_to_exact_previous_frame(state):
    backend = ScriptedBackend(["alpha", ":back"])
    session = WizardSession(state, backend)
    assert session.text("model.source", "Model source") == "alpha"
    assert session.text("model.revision", "Revision") is BACK
    assert session.current_frame.prompt_id == "model.source"


def test_nested_editor_back_restores_item_cursor(state):
    session = WizardSession(state, ScriptedBackend(["second", ":back"]))
    session.enter_collection("mip.runs", item_id="memory-075", cursor=1)
    assert session.text("mip.runs.memory-075.name", "Run name") == "second"
    assert session.text("mip.runs.memory-075.goal", "Goal") is BACK
    assert session.collection_cursor("mip.runs") == 1
```

- [ ] **Step 2: Verify failure**

Run:

```bash
rtk proxy .venv-orchestrator/bin/python -m pytest tests/unit/torch/puzzletron/test_setup_v2_session.py -q
```

Expected: collection fails because `session.py` and `prompts.py` do not exist.

- [ ] **Step 3: Implement backend-neutral prompt primitives**

```python
class _Back:
    pass


BACK = _Back()


@dataclass(frozen=True)
class PromptChoice:
    title: str
    value: Any


class PromptBackend(Protocol):
    def text(self, message: str, default: str) -> Any:
        raise NotImplementedError

    def select(self, message: str, choices: Sequence[PromptChoice], default: Any) -> Any:
        raise NotImplementedError

    def checkbox(
        self, message: str, choices: Sequence[PromptChoice], defaults: Sequence[Any]
    ) -> Any:
        raise NotImplementedError
```

`InteractiveBackend` appends a `← Back` choice to select/checkbox prompts and prints the
`:back` hint for text/numeric prompts. `ScriptedBackend` consumes deterministic answers for
tests.

- [ ] **Step 4: Implement prompt frames and exact back behavior**

```python
@dataclass(frozen=True)
class PromptFrame:
    section: str
    prompt_id: str
    collection: Optional[str] = None
    item_id: Optional[str] = None
    cursor: Optional[int] = None


class WizardSession:
    def _ask(self, frame: PromptFrame, invoke: Callable[[], Any]) -> Any:
        self.state.push_frame(frame)
        value = invoke()
        if value is BACK or value == ":back":
            self.state.pop_frame()
            self.state.pop_frame()
            return BACK
        return value
```

Persist frames after every movement. Replayed scripted answers must use prompt IDs, not display
text, so copy edits do not invalidate resume state.

- [ ] **Step 5: Run focused tests**

Run:

```bash
rtk proxy .venv-orchestrator/bin/python -m pytest tests/unit/torch/puzzletron/test_setup_v2_session.py -q
```

Expected: all tests pass.

- [ ] **Step 6: Commit**

```bash
rtk proxy git add puzzletron_setup/v2/session.py puzzletron_setup/v2/prompts.py tests/unit/torch/puzzletron/test_setup_v2_session.py
rtk proxy git commit -s -S -m "feat: add navigable Puzzletron prompts"
```

### Task 3: Reusable Stage Resources and Visible Batch Resolution

**Files:**
- Create: `puzzletron_setup/v2/resources.py`
- Test: `tests/unit/torch/puzzletron/test_setup_v2_resources.py`

**Interfaces:**
- Consumes: `ParallelMesh`, `pack_gpu_allocation`, and `validate_mesh`.
- Produces: `ParallelProfile`, `BatchResolution`, `StageResources`, `AllocationSummary`.
- Produces: `resolve_batch(requested, profile) -> BatchResolution`.
- Produces: `validate_parallel_profile(profile, inventory) -> None`.
- Produces: `allocation_summary(resources, gpus_per_node) -> AllocationSummary`.
- Produces: `ResourceProfileRegistry.create`, `reuse`, `copy`, `consumers`, and `update`.

- [ ] **Step 1: Write failing batch/profile/allocation tests**

```python
def test_batch_rounds_to_pp_times_data_parallel_unit():
    profile = ParallelProfile("width", tp=1, cp=1, pp=2, dp_shard=2, dp_replicate=2, ep=2)
    assert resolve_batch(3, profile) == BatchResolution(
        requested=3, effective=8, unit=8, adjusted=True
    )


def test_profile_reuse_and_copy_are_distinct():
    registry = ResourceProfileRegistry()
    registry.create(ParallelProfile("width", dp_shard=2))
    assert registry.reuse("width", consumer="bypass").name == "width"
    copied = registry.copy("width", "global_kd", consumer="global_kd")
    assert copied.name == "global_kd"
    assert copied is not registry.get("width")


def test_allocation_summary_derives_tasks_not_user_task_count():
    resources = StageResources(
        stage_id="depth_importance",
        strategy="persistent_pool",
        instances=4,
        profile=ParallelProfile("depth", pp=2, dp_shard=4),
    )
    summary = allocation_summary(resources, gpus_per_node=8)
    assert (summary.nodes, summary.task_count, summary.gpus_per_task) == (4, 4, 8)
```

- [ ] **Step 2: Verify failure**

Run:

```bash
rtk proxy .venv-orchestrator/bin/python -m pytest tests/unit/torch/puzzletron/test_setup_v2_resources.py -q
```

Expected: collection fails because `resources.py` does not exist.

- [ ] **Step 3: Implement immutable profiles and batch resolution**

```python
@dataclass(frozen=True)
class ParallelProfile:
    name: str
    tp: int = 1
    cp: int = 1
    pp: int = 1
    dp_shard: int = 1
    dp_replicate: int = 1
    ep: int = 1
    sequence_parallel: bool = False

    @property
    def batch_unit(self) -> int:
        return self.pp * self.dp_shard * self.dp_replicate


def resolve_batch(requested: int, profile: ParallelProfile) -> BatchResolution:
    unit = profile.batch_unit
    effective = max(unit, ((max(1, int(requested)) + unit - 1) // unit) * unit)
    return BatchResolution(requested, effective, unit, effective != requested)
```

- [ ] **Step 4: Implement model-aware validation and derived packing**

Call canonical `validate_mesh`, then enforce descriptor/model facts:

```python
if profile.dp_shard % profile.ep:
    raise SetupError("DP shard must be divisible by EP")
experts = inventory.facts.get("num_experts")
if experts is not None and int(experts) % profile.ep:
    raise SetupError(f"EP={profile.ep} does not divide {experts} experts")
```

Derive task topology with the same `tasks_per_instance` rule as
`orchestration.adapters.packing.packed_allocation`; never accept raw task count.

- [ ] **Step 5: Implement registry consumer tracking**

When updating a shared profile, return the exact consumer paths that must be marked stale:

```python
def update(self, profile: ParallelProfile) -> Tuple[str, ...]:
    if profile.name not in self._profiles:
        raise KeyError(profile.name)
    self._profiles[profile.name] = profile
    return tuple(sorted(self._consumers.get(profile.name, ())))
```

- [ ] **Step 6: Run focused tests**

Run:

```bash
rtk proxy .venv-orchestrator/bin/python -m pytest tests/unit/torch/puzzletron/test_setup_v2_resources.py tests/unit/torch/puzzletron/test_orchestration_task_topology.py -q
```

Expected: all tests pass.

- [ ] **Step 7: Commit**

```bash
rtk proxy git add puzzletron_setup/v2/resources.py tests/unit/torch/puzzletron/test_setup_v2_resources.py
rtk proxy git commit -s -S -m "feat: add per-stage Puzzletron resources"
```

### Task 4: Backward-Compatible Named vLLM Measurements

**Files:**
- Create: `modelopt/torch/puzzletron/subblock_stats/measurements.py`
- Create: `puzzletron_setup/v2/vllm.py`
- Modify: `modelopt/torch/puzzletron/stages/pipeline.py`
- Modify: `modelopt/torch/puzzletron/orchestration/adapters/sharded.py`
- Modify: `modelopt/torch/puzzletron/orchestration/adapters/stage_compat.py`
- Modify: `examples/puzzletron/run_runtime_stats_shard.py`
- Modify: `examples/puzzletron/configs/base.yaml`
- Test: `tests/unit/torch/puzzletron/test_vllm_measurements.py`

**Interfaces:**
- Consumes: `WizardSession`, `ParallelProfile`, runtime `RuntimeTopology`, and legacy
  `vllm_stats` fields.
- Produces: `VllmMeasurement`, `normalize_vllm_measurements(config)`, and
  `apply_vllm_measurement(config, measurement)`.
- Produces: `VllmMeasurementEditor` CRUD/clone/review methods.
- Produces: `measurement_workload(measurement) -> Mapping[str, int]`.
- Runtime artifacts: `artifacts/vllm_stats/measurements/<id>/subblock_stats.json`,
  `artifacts/vllm_stats/measurements/index.json`, and merged legacy
  `<puzzle_dir>/subblock_stats.json`.

- [ ] **Step 1: Write failing legacy-normalization and multi-measurement tests**

```python
def test_legacy_vllm_config_normalizes_to_implicit_default():
    config = {
        "vllm_stats": {
            "prefill_seq_len": 4096,
            "generation_seq_len": 1024,
            "batch_sizes": [1],
            "runtime_stats": {"granularity": "subblock"},
        }
    }
    measurements = normalize_vllm_measurements(config)
    assert tuple(measurements) == ("default",)
    assert measurements["default"].prefill_seq_len == 4096


def test_named_measurements_have_distinct_identity_and_paths(tmp_path):
    config = {
        "vllm_stats": {
            "measurements": {
                "serving-4k": {"prefill_seq_len": 4096, "generation_seq_len": 512},
                "serving-8k": {"prefill_seq_len": 8192, "generation_seq_len": 1024},
            }
        }
    }
    measurements = normalize_vllm_measurements(config)
    assert measurements["serving-4k"].identity != measurements["serving-8k"].identity
    assert measurements["serving-8k"].relative_stats_path == (
        Path("artifacts/vllm_stats/measurements/serving-8k/subblock_stats.json")
    )
```

- [ ] **Step 2: Write failing adapter test for different measurement topologies**

```python
def test_vllm_work_items_carry_measurement_specific_gpu_groups(compiled_plan):
    work = ShardedAdapter().plan(plan=compiled_plan, node=_vllm_node(compiled_plan))
    by_measurement = {
        item.metadata["measurement_id"]: item.gpus_per_instance for item in work.items
    }
    assert by_measurement == {"one-gpu": 1, "two-gpu": 2}
```

- [ ] **Step 3: Verify failures**

Run:

```bash
rtk proxy .venv-orchestrator/bin/python -m pytest tests/unit/torch/puzzletron/test_vllm_measurements.py -q
```

Expected: collection fails because `measurements.py` does not exist.

- [ ] **Step 4: Implement normalization and stable identities**

```python
@dataclass(frozen=True)
class VllmMeasurement:
    measurement_id: str
    prefill_seq_len: int
    generation_seq_len: int
    batch_size: int
    max_num_seqs: int
    granularity: str
    runtime_stats: Mapping[str, Any]

    @property
    def identity(self) -> str:
        payload = json.dumps(self.to_dict(), sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(payload.encode()).hexdigest()

    @property
    def relative_stats_path(self) -> Path:
        return (
            Path("artifacts")
            / "vllm_stats"
            / "measurements"
            / self.measurement_id
            / "subblock_stats.json"
        )
```

Named settings inherit legacy root defaults before their overrides. Legacy-only input produces
one `default` measurement and unchanged legacy output locations.

- [ ] **Step 5: Implement measurement-specific stage execution and aggregation**

`apply_vllm_measurement` deep-copies the config, overlays the selected workload/runtime fields,
sets a measurement-specific stats/output path, and records the ID/identity.

Change `vllm_stats_stage` and shard worker behavior so one attempt selects exactly one
measurement via `--measurement-id`. Final aggregation:

```python
def merge_vllm_measurements(puzzle_dir: Path, measurements) -> Path:
    rows = []
    for measurement in measurements.values():
        payload = json.loads((puzzle_dir / measurement.relative_stats_path).read_text())
        for row in payload:
            row.setdefault("args", {})["workload_id"] = measurement.measurement_id
            rows.append(row)
    output = puzzle_dir / "subblock_stats.json"
    _atomic_json(output, rows)
    return output
```

Write `measurements/index.json` only after every selected measurement has a non-empty result.

- [ ] **Step 6: Make sharded work and completion measurement-aware**

For `vllm_stats`, create `instances` shards per measurement. Each `WorkItem` metadata contains
`measurement_id`, `measurement_identity`, and topology-derived GPUs. Submission uses the
item's GPU count rather than the stage's default count. Completion requires the index, every
measurement output, the merged aggregate, and the normal report summary.

- [ ] **Step 7: Implement the guided measurement editor**

```python
class VllmMeasurementEditor:
    def __init__(self) -> None:
        self._measurements: OrderedDict[str, VllmMeasurement] = OrderedDict()

    def add(self, measurement: VllmMeasurement) -> None:
        if measurement.measurement_id in self._measurements:
            raise ValueError(f"duplicate vLLM measurement {measurement.measurement_id!r}")
        self._measurements[measurement.measurement_id] = measurement

    def clone(self, source_id: str, target_id: str) -> VllmMeasurement:
        clone = replace(self._measurements[source_id], measurement_id=target_id)
        self.add(clone)
        return clone

    def edit(self, measurement_id: str, **changes: Any) -> VllmMeasurement:
        updated = replace(self._measurements[measurement_id], **changes)
        self._measurements[measurement_id] = updated
        return updated

    def delete(self, measurement_id: str, referenced_by: Sequence[str]) -> None:
        if referenced_by:
            raise ValueError(
                f"measurement {measurement_id!r} is referenced by {sorted(referenced_by)}"
            )
        del self._measurements[measurement_id]

    def workloads(self) -> Mapping[str, Mapping[str, int]]:
        return {
            name: measurement_workload(measurement)
            for name, measurement in self._measurements.items()
        }

    def work_estimate(self, candidate_count: int) -> int:
        return int(candidate_count) * len(self._measurements)
```

Validate IDs, positive workload values, `max_num_seqs >= batch_size`, supported granularity,
`RuntimeTopology`, and duplicate exact settings.

- [ ] **Step 8: Run focused runtime and editor tests**

Run:

```bash
rtk proxy .venv-orchestrator/bin/python -m pytest tests/unit/torch/puzzletron/test_vllm_measurements.py tests/unit/torch/puzzletron/test_sparse_runtime_stats.py tests/unit/torch/puzzletron/test_orchestration_controller.py -q
```

Expected: all tests pass, including legacy single-setting coverage.

- [ ] **Step 9: Commit**

```bash
rtk proxy git add modelopt/torch/puzzletron/subblock_stats/measurements.py puzzletron_setup/v2/vllm.py modelopt/torch/puzzletron/stages/pipeline.py modelopt/torch/puzzletron/orchestration/adapters/sharded.py modelopt/torch/puzzletron/orchestration/adapters/stage_compat.py examples/puzzletron/run_runtime_stats_shard.py examples/puzzletron/configs/base.yaml tests/unit/torch/puzzletron/test_vllm_measurements.py
rtk proxy git commit -s -S -m "feat: support named vLLM measurements"
```

### Task 5: Complete Guided MIP Run Editor

**Files:**
- Create: `puzzletron_setup/v2/mip.py`
- Test: `tests/unit/torch/puzzletron/test_setup_v2_mip.py`

**Interfaces:**
- Consumes: `WizardSession`, inspected axis/depth domains, and named vLLM workloads.
- Consumes: canonical `normalize_mip_profiles`.
- Produces: `MIPRunEditor`, `MIPRunDraft`, `ConstraintDraft`, `VariantDraft`, and
  `ConcreteExpansion`.
- Produces: `MIPRunEditor.to_config() -> Mapping[str, Any]`.
- Produces: `MIPRunEditor.validate(available_depths, available_embeddings,
  available_depth_counts, depth_granularity) -> Sequence[MIPProfile]`.

- [ ] **Step 1: Write failing full-schema and expansion tests**

```python
def test_mip_editor_separates_goal_and_internal_constraints(domains):
    editor = MIPRunEditor(domains, workloads={"serving-8k": _workload(8192, 1024, 4)})
    editor.add_run(
        MIPRunDraft(
            run_id="memory-075",
            goal=ConstraintDraft("memory", "max", "75%", workload="serving-8k"),
            objectives=(
                {"metric": "metrics.lm_loss", "direction": "minimize"},
                {"metric": "metrics.cosine_embedding_loss_hidden_states", "direction": "minimize"},
            ),
            constraints=(ConstraintDraft("experts", "range", [64, 96]),),
        )
    )
    config = editor.to_config()["runs"]["memory-075"]
    assert config["constraints"]["memory"] == {"at": {"serving-8k": {"max": "75%"}}}
    assert config["constraints"]["experts"] == {"range": [64, 96]}


def test_variant_matrix_expansion_is_reported(domains):
    editor = _editor_with_two_variants_three_rows_two_objectives(domains)
    assert editor.expansion("memory-075") == ConcreteExpansion(
        variants=2, matrix_rows=3, objectives=2, concrete_solves=12
    )
```

- [ ] **Step 2: Verify failure**

Run:

```bash
rtk proxy .venv-orchestrator/bin/python -m pytest tests/unit/torch/puzzletron/test_setup_v2_mip.py -q
```

Expected: collection fails because `mip.py` does not exist.

- [ ] **Step 3: Implement typed guided drafts**

```python
@dataclass(frozen=True)
class ConstraintDraft:
    metric: str
    mode: str
    value: Any
    workload: Optional[str] = None

    def to_config(self) -> Any:
        bound = self.value if self.mode == "directional" else {self.mode: self.value}
        return {"at": {self.workload: bound}} if self.workload else bound


@dataclass(frozen=True)
class ConcreteExpansion:
    variants: int
    matrix_rows: int
    objectives: int
    concrete_solves: int
```

Accept all documented friendly metrics plus validated `stats.*`, every bound mode, all
documented units, typed depth, axis selectors, solver controls, homogeneous ranking, variants,
and supported matrix paths.

- [ ] **Step 4: Implement CRUD/clone and inherited review**

`clone` deep-copies a run under a unique ID. Variant review returns explicit inherited and
overridden mappings. Deleting a run returns all post-MIP flow references rather than silently
removing them.

- [ ] **Step 5: Validate exclusively through the canonical compiler**

Render `mip.defaults`, `mip.workloads`, and `mip.runs`, then call:

```python
profiles = normalize_mip_profiles(
    mip_config,
    available_depths=domains.depths,
    available_embeddings=domains.embeddings,
    available_depth_counts=domains.depth_counts,
    depth_granularity=domains.depth_granularity,
)
```

Convert compiler errors into field paths such as
`mip.runs.memory-075.variants.expert-bands.matrix.constraints.experts`.

- [ ] **Step 6: Run focused MIP tests**

Run:

```bash
rtk proxy .venv-orchestrator/bin/python -m pytest tests/unit/torch/puzzletron/test_setup_v2_mip.py tests/unit/torch/puzzletron/test_mip_profiles.py -q
```

Expected: all tests pass.

- [ ] **Step 7: Commit**

```bash
rtk proxy git add puzzletron_setup/v2/mip.py tests/unit/torch/puzzletron/test_setup_v2_mip.py
rtk proxy git commit -s -S -m "feat: add guided Puzzletron MIP editor"
```

### Task 6: Editable Post-MIP Flow DAGs

**Files:**
- Create: `puzzletron_setup/v2/post_mip.py`
- Test: `tests/unit/torch/puzzletron/test_setup_v2_post_mip.py`

**Interfaces:**
- Consumes: MIP run IDs/variants/objectives, `WizardSession`, stage resources, and
  `compile_post_mip_flows`.
- Produces: `PostMIPFlowEditor`, `FlowDraft`, `NodeDraft`, `FlowReview`.
- Produces: `recommended_flow(run_id, objective_metrics, data, serving) -> FlowDraft`.
- Produces: `PostMIPFlowEditor.to_config() -> Mapping[str, Any]`.

- [ ] **Step 1: Write failing recommended-flow and DAG-edit tests**

```python
def test_recommended_flow_has_no_initial_filter():
    flow = recommended_flow(
        "memory-075",
        objective_metrics=("metrics.lm_loss",),
        data={"sequence_length": 4096},
        serving={"input_tokens": 4096, "output_tokens": 1024, "concurrency": 1},
    )
    assert tuple(flow.nodes) == (
        "online_eval",
        "best_lm",
        "materialized",
        "serving",
        "fastest",
        "short_kd",
        "final_eval",
        "best",
    )
    assert "initial_filter" not in flow.nodes


def test_delete_reports_dependents_before_mutation(editor):
    editor.add_node("flow", NodeDraft("eval", "evaluation", input_id="source"))
    editor.add_node("flow", NodeDraft("best", "filter", input_id="eval"))
    assert editor.delete_node("flow", "eval") == ("best",)
    assert "eval" in editor.flow("flow").nodes
```

- [ ] **Step 2: Write failing branch/model-source/artifact tests**

```python
def test_model_source_transformer_adds_dependency(editor):
    editor.add_node("flow", NodeDraft("mat", "materialize", input_id="source"))
    editor.add_node(
        "flow",
        NodeDraft("eval", "evaluation", input_id="source", model_source="mat"),
    )
    review = editor.review("flow")
    assert review.parents["eval"] == ("mip", "mat")
```

- [ ] **Step 3: Verify failure**

Run:

```bash
rtk proxy .venv-orchestrator/bin/python -m pytest tests/unit/torch/puzzletron/test_setup_v2_post_mip.py -q
```

Expected: collection fails because `post_mip.py` does not exist.

- [ ] **Step 4: Implement node and flow CRUD**

```python
IMPLEMENTED_NODE_TYPES = (
    "filter",
    "manual_filter",
    "materialize",
    "evaluation",
    "aiperf",
    "global_kd",
)
RESERVED_NODE_TYPES = ("ptq", "downstream_evaluation")


class PostMIPFlowEditor:
    def __init__(self) -> None:
        self._flows: OrderedDict[str, FlowDraft] = OrderedDict()

    def add_flow(self, flow: FlowDraft) -> None:
        if flow.flow_id in self._flows:
            raise ValueError(f"duplicate post-MIP flow {flow.flow_id!r}")
        self._flows[flow.flow_id] = flow

    def clone_flow(self, source_id: str, target_id: str) -> FlowDraft:
        clone = deepcopy(self._flows[source_id])
        clone = replace(clone, flow_id=target_id)
        self.add_flow(clone)
        return clone

    def add_node(self, flow_id: str, node: NodeDraft) -> None:
        flow = self._flows[flow_id]
        if node.node_id in flow.nodes:
            raise ValueError(f"duplicate post-MIP node {node.node_id!r}")
        flow.nodes[node.node_id] = node

    def edit_node(self, flow_id: str, node_id: str, **changes: Any) -> NodeDraft:
        flow = self._flows[flow_id]
        updated = replace(flow.nodes[node_id], **changes)
        flow.nodes[node_id] = updated
        return updated

    def clone_node(self, flow_id: str, node_id: str, target_id: str) -> NodeDraft:
        clone = replace(self._flows[flow_id].nodes[node_id], node_id=target_id)
        self.add_node(flow_id, clone)
        return clone

    def delete_node(self, flow_id: str, node_id: str) -> Tuple[str, ...]:
        dependents = self.dependents(flow_id, node_id)
        if dependents:
            return dependents
        del self._flows[flow_id].nodes[node_id]
        return ()

    def redirect_node(self, flow_id: str, node_id: str, new_input: str) -> None:
        self.edit_node(flow_id, node_id, input_id=new_input)
```

Reject reserved types at selection time. Filter builders expose top-k quotas, threshold,
Pareto, and weighted aggregate rank. All non-selector nodes expose input, model source,
failure policy, node-specific public config, and applicable resource/batch settings.

- [ ] **Step 5: Implement recommended combined flow**

Source one run with selected variants/objectives. Build exactly the eight-node default tested
above, use LM loss for both evaluation filters, set AIPerf timeout to 900 seconds, and use no
Initial Filter.

- [ ] **Step 6: Compile every edit through the canonical DAG compiler**

Create a minimal experiment wrapper:

```python
experiment = {
    "mip": {"runs": mip_runs},
    "post_mip": {"flows": editor.to_config()},
}
compiled = compile_post_mip_flows(experiment)
```

Map unknown IDs, cycles, forward metrics, artifact-kind errors, and invalid model sources back
to exact node fields.

- [ ] **Step 7: Run post-MIP tests**

Run:

```bash
rtk proxy .venv-orchestrator/bin/python -m pytest tests/unit/torch/puzzletron/test_setup_v2_post_mip.py tests/unit/torch/puzzletron/test_post_mip_runner.py tests/unit/torch/puzzletron/test_post_mip_reporting.py -q
```

Expected: all tests pass.

- [ ] **Step 8: Commit**

```bash
rtk proxy git add puzzletron_setup/v2/post_mip.py tests/unit/torch/puzzletron/test_setup_v2_post_mip.py
rtk proxy git commit -s -S -m "feat: add editable post-MIP flows"
```

### Task 7: Canonical V2 Bundle Rendering and Validation

**Files:**
- Create: `puzzletron_setup/v2/bundle.py`
- Create: `puzzletron_setup/v2/validation.py`
- Test: `tests/unit/torch/puzzletron/test_setup_v2_bundle.py`

**Interfaces:**
- Consumes: complete `WizardState`, resource registry, vLLM measurements, MIP config, and
  post-MIP flow config.
- Produces: `ValidationIssue(path, message)`.
- Produces: `validate_state(state) -> Sequence[ValidationIssue]`.
- Produces: `render_experiment_v2`, `render_runner_v2`, `render_execution_v2`.
- Produces: `build_bundles_v2(campaign_dir, state) -> BundleResult`.
- Guarantees: invalid bundles never replace the last valid bundle.

- [ ] **Step 1: Write failing rendering tests for independent stage controls**

```python
def test_v2_execution_renders_per_stage_instances_and_profiles(v2_state):
    execution = render_execution_v2(v2_state, budget="production")
    width = execution["execution"]["stages"]["width_importance"]
    bypass = execution["execution"]["stages"]["bypass"]
    assert width["instances"] == 1
    assert width["parallel"]["dp_shard"] == 2
    assert bypass["instances"] == 4
    assert bypass["parallel"]["pp"] == 2


def test_requested_and_effective_batches_render_effective_value(v2_state):
    experiment = render_experiment_v2(v2_state, budget="production")
    assert v2_state.field("stages.width_importance.batch").requested == 3
    assert experiment["pruning"]["micro_batch_size"] == 8
```

- [ ] **Step 2: Write failing atomic validation test**

```python
def test_invalid_new_bundle_preserves_previous_bundle(tmp_path, valid_state, invalid_state):
    build_bundles_v2(tmp_path, valid_state)
    previous = (tmp_path / "production/experiment.yaml").read_text()
    with pytest.raises(SetupError, match="mip.runs.memory-075"):
        build_bundles_v2(tmp_path, invalid_state)
    assert (tmp_path / "production/experiment.yaml").read_text() == previous
```

- [ ] **Step 3: Verify failures**

Run:

```bash
rtk proxy .venv-orchestrator/bin/python -m pytest tests/unit/torch/puzzletron/test_setup_v2_bundle.py -q
```

Expected: collection fails because the v2 bundle modules do not exist.

- [ ] **Step 4: Render canonical experiment sections**

Start from the existing base/family fragments, then overlay normalized v2 state:

```python
def render_experiment_v2(state: WizardState, budget: str) -> dict[str, Any]:
    rendered = _base_family_config(state)
    _render_model_data_pruning(rendered, state, budget)
    _render_stage_algorithms_and_batches(rendered, state, budget)
    rendered["vllm_stats"] = render_vllm_config(state)
    rendered["mip"] = render_mip_config(state)
    rendered["post_mip"] = {"flows": render_post_mip_config(state)}
    return rendered
```

Do not route through the old normal/detailed state adapter; render all v2 per-stage values
explicitly.

- [ ] **Step 5: Render runner and per-stage execution**

Each enabled static/dynamic stage gets its own strategy, instances, resource, partition,
`gpus_per_node`, and parallel mapping. AIPerf uses its node Serving topology; named vLLM
measurements retain per-measurement topology metadata used by the adapter.

- [ ] **Step 6: Implement cross-section validation**

Return path-addressed issues after these concrete calls:

```python
profiles = normalize_mip_profiles(
    experiment["mip"],
    available_depths=domains.depths,
    available_embeddings=domains.embeddings,
    available_depth_counts=domains.depth_counts,
    depth_granularity=domains.depth_granularity,
)
post_mip_nodes = compile_post_mip_flows(experiment)
plan = compile_campaign_plan(
    experiment_config_path=experiment_path,
    runner=runner,
    execution=execution["execution"],
    stage_filter="full",
)
```

Also validate defaults provenance, model/dataset sources, profile references, batches,
workloads, metrics, and post-MIP model sources. Sort issues by top-level question order.

- [ ] **Step 7: Publish both bundles transactionally**

Render and compile smoke/production into a temporary sibling directory. Call the existing
`dry_run_plan` for both. Only after both pass, atomically replace generated files and write:

- `resolved_defaults.yaml`
- `README.md` with exact dry-run/launch commands.

- [ ] **Step 8: Run bundle tests**

Run:

```bash
rtk proxy .venv-orchestrator/bin/python -m pytest tests/unit/torch/puzzletron/test_setup_v2_bundle.py tests/unit/torch/puzzletron/test_setup_bundle.py -q
```

Expected: all tests pass and old wizard bundle tests remain green.

- [ ] **Step 9: Commit**

```bash
rtk proxy git add puzzletron_setup/v2/bundle.py puzzletron_setup/v2/validation.py tests/unit/torch/puzzletron/test_setup_v2_bundle.py
rtk proxy git commit -s -S -m "feat: render Puzzletron v2 setup bundles"
```

### Task 8: Ordered Wizard, CLI, End-to-End Review, and Documentation

**Files:**
- Create: `puzzletron_setup/v2/wizard.py`
- Create: `puzzletron_setup/v2/cli.py`
- Create: `examples/puzzletron/puzzletron_setup_v2.py`
- Modify: `examples/puzzletron/README.md`
- Test: `tests/unit/torch/puzzletron/test_setup_v2_wizard.py`

**Interfaces:**
- Consumes: every interface produced by Tasks 1–7.
- Produces: `run_wizard_v2(resume, defaults_path, backend=None) -> Path`.
- Produces: `puzzletron_setup.v2.cli.main(argv=None) -> int`.
- Guarantees: no import of PyTorch on `--help`; no job launch path.

- [ ] **Step 1: Write failing CLI/help/defaults tests**

```python
def test_v2_help_has_defaults_and_no_detailed(capsys):
    with pytest.raises(SystemExit) as exited:
        main(["--help"])
    assert exited.value.code == 0
    output = capsys.readouterr().out
    assert "--defaults" in output
    assert "--resume" in output
    assert "--detailed" not in output


def test_defaults_are_not_loaded_implicitly(tmp_path, scripted_backend):
    campaign = run_wizard_v2(
        resume=None,
        defaults_path=None,
        backend=scripted_backend.for_minimal_campaign(tmp_path),
    )
    state = WizardState.resume(campaign / "answers_v2.yaml")
    assert state.get_field("data.source").source != "defaults_file"
```

- [ ] **Step 2: Write failing end-to-end scripted wizard test**

Script one campaign that:

- Uses `sepehr_defaults.yaml`.
- Goes back from a numeric prompt.
- Changes width parallelism and preserves dataset answers.
- Reuses width resources for bypass.
- Adds two named vLLM settings.
- Adds one MIP run with an extra constraint and a variant matrix.
- Uses the recommended combined post-MIP flow.
- Generates smoke and production bundles.

Assert both bundles compile and the default post-MIP flow has no `initial_filter`.

- [ ] **Step 3: Verify failures**

Run:

```bash
rtk proxy .venv-orchestrator/bin/python -m pytest tests/unit/torch/puzzletron/test_setup_v2_wizard.py -q
```

Expected: collection fails because `wizard.py`, `cli.py`, and the entry point do not exist.

- [ ] **Step 4: Implement the exact top-level section order**

```python
SECTION_BUILDERS = (
    campaign_section,
    model_section,
    data_section,
    infrastructure_section,
    pruning_section,
    pre_mip_stages_section,
    vllm_section,
    mip_section,
    post_mip_section,
    output_review_section,
)
```

Every section starts with a resolved-default summary and Use defaults/Customize/Review/Back.
Stage resources are configured inline when their stage is encountered. Validation issues from
the final review navigate to their owning prompt.

- [ ] **Step 5: Implement dependency-light CLI and public entry point**

```python
def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Create a Puzzletron pruning campaign.")
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--defaults", type=Path)
    return parser
```

Defer importing `wizard.py` until after argument parsing. Handle `KeyboardInterrupt` with an
exact resume command and `SetupError` with exit code 2.

- [ ] **Step 6: Add user documentation**

Document:

```bash
python examples/puzzletron/puzzletron_setup_v2.py \
  --defaults nv-internal/sepehr_defaults.yaml
```

Explain `:back`, visible Back choices, resume, requested/effective batches, inline reusable
profiles, named vLLM settings, combined MIP flows, output files, and the fact that the wizard
does not launch jobs.

- [ ] **Step 7: Run all setup and named-vLLM tests**

Run:

```bash
rtk proxy .venv-orchestrator/bin/python -m pytest tests/unit/torch/puzzletron/test_setup_v2_defaults_state.py tests/unit/torch/puzzletron/test_setup_v2_session.py tests/unit/torch/puzzletron/test_setup_v2_resources.py tests/unit/torch/puzzletron/test_vllm_measurements.py tests/unit/torch/puzzletron/test_setup_v2_mip.py tests/unit/torch/puzzletron/test_setup_v2_post_mip.py tests/unit/torch/puzzletron/test_setup_v2_bundle.py tests/unit/torch/puzzletron/test_setup_v2_wizard.py tests/unit/torch/puzzletron/test_setup_bundle.py tests/unit/torch/puzzletron/test_setup_inspection.py tests/unit/torch/puzzletron/test_setup_candidate_counts.py -q
```

Expected: all tests pass.

- [ ] **Step 8: Run static verification**

Run:

```bash
rtk proxy .venv-orchestrator/bin/python -m compileall -q puzzletron_setup/v2 examples/puzzletron/puzzletron_setup_v2.py modelopt/torch/puzzletron/subblock_stats/measurements.py
rtk proxy git diff --check
```

Expected: both commands exit 0 with no output.

- [ ] **Step 9: Exercise dependency-light help**

Run:

```bash
rtk proxy .venv-orchestrator/bin/python examples/puzzletron/puzzletron_setup_v2.py --help
```

Expected: usage lists `--resume` and `--defaults`, omits `--detailed`, and does not print a
PyTorch import error.

- [ ] **Step 10: Commit**

```bash
rtk proxy git add puzzletron_setup/v2/wizard.py puzzletron_setup/v2/cli.py examples/puzzletron/puzzletron_setup_v2.py examples/puzzletron/README.md tests/unit/torch/puzzletron/test_setup_v2_wizard.py
rtk proxy git commit -s -S -m "feat: add Puzzletron setup v2 wizard"
```

### Task 9: Final Regression Verification

**Files:**
- Modify only if a verification failure identifies a defect in files already listed above.

**Interfaces:**
- Consumes: the complete implementation.
- Produces: evidence that both setup entry points and orchestration compilation remain valid.

- [ ] **Step 1: Run the focused feature suite**

```bash
rtk proxy .venv-orchestrator/bin/python -m pytest tests/unit/torch/puzzletron/test_setup_v2_defaults_state.py tests/unit/torch/puzzletron/test_setup_v2_session.py tests/unit/torch/puzzletron/test_setup_v2_resources.py tests/unit/torch/puzzletron/test_vllm_measurements.py tests/unit/torch/puzzletron/test_setup_v2_mip.py tests/unit/torch/puzzletron/test_setup_v2_post_mip.py tests/unit/torch/puzzletron/test_setup_v2_bundle.py tests/unit/torch/puzzletron/test_setup_v2_wizard.py -q
```

Expected: all tests pass.

- [ ] **Step 2: Run affected existing regression suites**

```bash
rtk proxy .venv-orchestrator/bin/python -m pytest tests/unit/torch/puzzletron/test_setup_bundle.py tests/unit/torch/puzzletron/test_setup_inspection.py tests/unit/torch/puzzletron/test_setup_candidate_counts.py tests/unit/torch/puzzletron/test_mip_profiles.py tests/unit/torch/puzzletron/test_post_mip_runner.py tests/unit/torch/puzzletron/test_post_mip_reporting.py tests/unit/torch/puzzletron/test_sparse_runtime_stats.py tests/unit/torch/puzzletron/test_orchestration_controller.py tests/unit/torch/puzzletron/test_orchestration_task_topology.py -q
```

Expected: all tests pass.

- [ ] **Step 3: Inspect the final diff**

```bash
rtk proxy git status --short
rtk proxy git diff --check
rtk proxy git diff --stat HEAD~8..HEAD
```

Expected: only scoped implementation/test/docs/default files are changed, no whitespace errors,
and no user-owned untracked files are staged.

- [ ] **Step 4: Record final verification without launching Puzzletron**

Do not run `orchestrate.py` or submit Slurm jobs. Report:

- Test commands and pass counts.
- The new entry point.
- The explicit defaults command.
- Any environment-dependent test that could not run.
- Confirmation that no jobs were launched and no remote was pushed.
