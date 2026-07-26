# Puzzletron AIPerf and Parallelism Validation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Correct Puzzletron's AIPerf/vLLM topology contract and reject every
stage parallel profile that is incompatible with the teacher or any selected
candidate geometry that the stage can encounter.

**Architecture:** Put vLLM topology normalization in the dependency-light
orchestration mesh module, and put model/stage compatibility in a new pure
setup-v2 module. The wizard, persisted-state validator, bundle renderer, and
AIPerf runner consume those shared contracts so invalid values cannot bypass
validation through defaults, profile reuse, or edited YAML.

**Tech Stack:** Python 3.12/3.9, dataclasses, PyYAML/OmegaConf configuration,
pytest, vLLM CLI, AIPerf CLI, Puzzletron's scheduler-neutral orchestrator.

## Global Constraints

- New vLLM configs use boolean `enable_expert_parallel`; no independent numeric
  EP is generated.
- For an MoE model with expert parallelism enabled, effective EP is exactly
  `TP * DP`, and the wizard explains this before asking.
- A legacy numeric `expert_parallel_size` is accepted only as `1` (disabled) or
  exactly `TP * DP` (enabled); every other legacy value is rejected.
- Invalid settings are rejected explicitly and the user is asked for a
  different setting; settings are never silently rewritten.
- Bypass and bypass-sanity validate only teacher tensor geometry because they
  mask the resident teacher.
- PP may partition layers unevenly and has no layer-count compatibility rule.
- Candidate-aware stages validate the teacher plus every selected pruning-axis
  value that they can encounter.
- Changes remain model-name-independent and preserve dense/Qwen behavior.
- Preserve all pre-existing dirty-worktree edits and the existing AIPerf
  topology environment override.
- Use relative repository paths in commands and do not push.

---

### Task 1: Normalize vLLM topology and allocation semantics

**Files:**
- Modify: `modelopt/torch/puzzletron/orchestration/mesh.py:11-20,179-215`
- Modify: `modelopt/torch/puzzletron/orchestration/__init__.py:18-48`
- Modify: `puzzletron_orchestrator/__init__.py:31-72`
- Modify: `tests/unit/torch/puzzletron/test_orchestration_mesh.py:1-85`
- Modify: `tests/unit/torch/puzzletron/test_orchestration_lightweight.py:20-43`

**Interfaces:**
- Produces:
  `normalize_vllm_topology(topology: Mapping[str, Any]) -> dict[str, Any]`.
- The normalized mapping contains `tp`, `pp`, `dp`, `prefill_cp`,
  `decode_cp`, `enable_expert_parallel`, `effective_ep`, `gpu_count`, and
  `distributed_executor_backend`.
- Produces:
  `vllm_topology_to_mesh(topology: Mapping[str, Any]) -> ParallelMesh`.
- The returned mesh represents allocation only:
  `ParallelMesh(tp=TP, pp=PP, cp=prefill_CP, dp_replicate=DP,
  dp_shard=1, ep=1)`.
- Consumed by Tasks 2, 3, and 5.

- [ ] **Step 1: Write failing topology tests**

Add literal behavior tests:

```python
from modelopt.torch.puzzletron.orchestration.mesh import (
    normalize_vllm_topology,
    vllm_topology_to_mesh,
)


def test_vllm_expert_parallel_is_tp_times_dp_without_extra_allocation_axis():
    topology = normalize_vllm_topology(
        {
            "tensor_parallel_size": 2,
            "pipeline_parallel_size": 2,
            "data_parallel_size": 4,
            "prefill_context_parallel_size": 1,
            "decode_context_parallel_size": 1,
            "enable_expert_parallel": True,
            "gpu_group_size": 16,
        }
    )
    assert topology["effective_ep"] == 8
    assert topology["gpu_count"] == 16
    assert vllm_topology_to_mesh(topology) == ParallelMesh(
        tp=2, pp=2, cp=1, dp_replicate=4, dp_shard=1, ep=1
    )


def test_vllm_legacy_expert_parallel_rejects_an_independent_size():
    with pytest.raises(
        ValueError,
        match=r"expert_parallel_size=4.*expected 1 or TP \\* DP=8",
    ):
        normalize_vllm_topology(
            {
                "tensor_parallel_size": 2,
                "data_parallel_size": 4,
                "expert_parallel_size": 4,
            }
        )
```

Also mutate the lightweight subprocess test so it imports
`normalize_vllm_topology` through `puzzletron_orchestrator` and still asserts
that `torch` is absent from `sys.modules`.

- [ ] **Step 2: Run the topology tests and verify RED**

Run:

```bash
rtk python -m pytest -o addopts= \
  tests/unit/torch/puzzletron/test_orchestration_mesh.py \
  tests/unit/torch/puzzletron/test_orchestration_lightweight.py \
  -q
```

Expected: collection or assertion failure because
`normalize_vllm_topology` does not exist and the existing mesh overlays a
numeric EP onto DP.

- [ ] **Step 3: Implement the canonical topology function**

Implement positive dimension parsing, DCP divisibility, exact allocation, and
legacy migration in `orchestration/mesh.py`. The central branch is:

```python
explicit = topology.get("enable_expert_parallel")
legacy = topology.get("expert_parallel_size", topology.get("ep"))
effective_ep = tp * dp
if explicit is None:
    if legacy is None or int(legacy) == 1:
        enabled = False
    elif int(legacy) == effective_ep:
        enabled = True
    else:
        raise ValueError(
            f"expert_parallel_size={legacy} is not an independent vLLM degree; "
            f"expected 1 or TP * DP={effective_ep}"
        )
else:
    enabled = bool(explicit)
    expected_legacy = effective_ep if enabled else 1
    if legacy is not None and int(legacy) != expected_legacy:
        raise ValueError(
            f"expert_parallel_size={legacy} conflicts with "
            f"enable_expert_parallel={enabled}; expected {expected_legacy}"
        )
```

Calculate `gpu_count = tp * pp * prefill_cp * dp`. Do not multiply by
effective EP or decode CP. Export the function through both orchestration
entrypoints.

- [ ] **Step 4: Run topology tests and verify GREEN**

Run the command from Step 2. Expected: both files pass and the lightweight
subprocess imports no PyTorch modules.

- [ ] **Step 5: Commit the isolated orchestration contract**

If no pre-existing user edits overlap these files:

```bash
rtk git add modelopt/torch/puzzletron/orchestration/mesh.py \
  modelopt/torch/puzzletron/orchestration/__init__.py \
  puzzletron_orchestrator/__init__.py \
  tests/unit/torch/puzzletron/test_orchestration_mesh.py \
  tests/unit/torch/puzzletron/test_orchestration_lightweight.py
rtk git commit -s -S -m "Fix vLLM expert-parallel topology semantics"
```

If a listed file already has user edits, leave that file unstaged and record
the verified working-tree result instead of committing unrelated hunks.

---

### Task 2: Correct AIPerf workload and vLLM command mapping

**Files:**
- Modify: `modelopt/torch/puzzletron/benchmarks/aiperf.py:94-166,272-322`
- Modify: `modelopt/torch/puzzletron/post_mip/runner.py:482-515`
- Modify: `puzzletron_setup/v2/post_mip.py:262-348`
- Modify: `examples/puzzletron/run_profile_aiperf_worker.py:17-78`
- Modify: `tests/unit/torch/puzzletron/test_aiperf_context_capacity.py:78-150`
- Modify: `tests/unit/torch/puzzletron/test_post_mip_runner.py`
- Modify: `tests/unit/torch/puzzletron/test_profile_aiperf_worker.py`

**Interfaces:**
- Consumes `normalize_vllm_topology` from Task 1.
- `_topology_vllm_args(topology)` emits `--enable-expert-parallel` from the
  boolean normalized setting.
- `_aiperf(...)` consumes `request_count`,
  `minimum_request_count`, and `requests_per_concurrency`; none are forwarded
  as unsupported keyword arguments.
- `recommended_flow(...)` carries a literal `request_count` into the AIPerf
  node.

- [ ] **Step 1: Write failing exact-command tests**

Replace numeric-EP expectations with:

```python
def test_vllm_topology_args_enable_expert_parallel_from_boolean():
    args = _topology_vllm_args(
        {
            "tensor_parallel_size": 2,
            "data_parallel_size": 4,
            "enable_expert_parallel": True,
            "gpu_group_size": 8,
        }
    )
    assert args[args.index("--tensor-parallel-size") + 1] == "2"
    assert args[args.index("--data-parallel-size") + 1] == "4"
    assert "--enable-expert-parallel" in args
    assert "--expert-parallel-size" not in args
```

Add a `_profile_command` assertion with hand-written literals proving:

```python
assert command[command.index("--concurrency") + 1] == "8"
assert command[command.index("--request-count") + 1] == "23"
assert command[command.index("--synthetic-input-tokens-mean") + 1] == "1024"
assert command[command.index("--synthetic-input-tokens-stddev") + 1] == "0"
assert command[command.index("--output-tokens-mean") + 1] == "128"
assert command[command.index("--output-tokens-stddev") + 1] == "0"
```

Add a post-MIP runner test that replaces `run_aiperf_sweep` at the external
GPU/server boundary, calls `_aiperf` with `request_count=23`, and asserts the
real runner adapter passes:

```python
assert captured["concurrencies"] == (8,)
assert captured["request_counts"] == {8: 23}
assert "request_count" not in captured
assert "minimum_request_count" not in captured
assert "requests_per_concurrency" not in captured
```

The fake return should contain one complete `BenchmarkResult`-shaped object
with `concurrency=8`, `metrics={}`, and `raw_artifacts=[]`.

- [ ] **Step 2: Run AIPerf adapter tests and verify RED**

Run:

```bash
rtk python -m pytest -o addopts= \
  tests/unit/torch/puzzletron/test_aiperf_context_capacity.py \
  tests/unit/torch/puzzletron/test_post_mip_runner.py \
  tests/unit/torch/puzzletron/test_profile_aiperf_worker.py \
  -q
```

Expected: failures showing numeric EP still controls the vLLM flag and
post-MIP request-count settings leak into `run_aiperf_sweep`.

- [ ] **Step 3: Implement exact AIPerf and vLLM argument translation**

Delegate `_canonical_topology` to `normalize_vllm_topology`. Emit DP and expert
parallel arguments from the normalized result:

```python
if canonical["dp"] > 1:
    args.extend(
        (
            "--data-parallel-size",
            str(canonical["dp"]),
            "--data-parallel-size-local",
            str(canonical["dp"]),
        )
    )
if canonical["enable_expert_parallel"]:
    args.append("--enable-expert-parallel")
```

Preserve the existing `topology["env"]` propagation exactly.

In the post-MIP runner, accept scalar or sequence concurrency and derive request
counts before `**settings`:

```python
raw_concurrency = settings.pop("concurrency", [1])
if isinstance(raw_concurrency, (int, str)):
    concurrencies = (int(raw_concurrency),)
else:
    concurrencies = tuple(int(value) for value in raw_concurrency)
request_count = settings.pop("request_count", None)
minimum = int(settings.pop("minimum_request_count", 4))
per_concurrency = int(settings.pop("requests_per_concurrency", 2))
request_counts = {
    concurrency: (
        int(request_count)
        if request_count is not None
        else max(minimum, per_concurrency * concurrency)
    )
    for concurrency in concurrencies
}
```

Add `request_count` to the recommended-flow serving config and convert the
profile worker examples to boolean expert parallelism with explicit DP.

- [ ] **Step 4: Run AIPerf adapter tests and verify GREEN**

Run the command from Step 2. Expected: all selected tests pass.

- [ ] **Step 5: Commit isolated adapter changes**

Stage only files without pre-existing user hunks. The intended commit message
is:

```bash
rtk git commit -s -S -m "Correct Puzzletron AIPerf command mapping"
```

Leave `modelopt/torch/puzzletron/benchmarks/aiperf.py` unstaged if staging it
would capture the user's existing topology-environment change.

---

### Task 3: Build the stage-aware geometry compatibility engine

**Files:**
- Create: `puzzletron_setup/v2/parallel_validation.py`
- Create: `tests/unit/torch/puzzletron/test_setup_v2_parallel_validation.py`
- Modify: `puzzletron_setup/v2/resources.py:13-26,155-175`

**Interfaces:**
- Produces:
  `ParallelCompatibilityIssue(path: str, message: str)`.
- Produces:
  `geometry_scope(stage_id: str, node_type: str | None = None) ->
  Literal["teacher", "candidate", "none"]`.
- Produces:
  `validate_automodel_parallelism(profile, inventory, pruning, *,
  stage_id, sequence_length) -> tuple[ParallelCompatibilityIssue, ...]`.
- Produces:
  `validate_vllm_parallelism(topology, inventory, pruning, *,
  stage_id) -> tuple[ParallelCompatibilityIssue, ...]`.
- `inventory` accepts either `ModelInventory` or its persisted mapping;
  `pruning` accepts the persisted setup-v2 pruning mapping.
- Task 4 consumes these functions interactively. Task 5 consumes them from
  persisted state.

- [ ] **Step 1: Write failing literal geometry-domain tests**

Build a small persisted MoE inventory with:

```python
MOE_INVENTORY = {
    "moe": True,
    "facts": {
        "hidden_size": 4096,
        "num_attention_heads": 32,
        "num_key_value_heads": 8,
        "intermediate_size": 11008,
        "num_experts": 64,
    },
    "axes": [
        {
            "axis_id": "hidden_width",
            "teacher_value": 4096,
            "values": [4096, 3072],
            "alignment": 256,
            "label": "Residual width",
        },
        {
            "axis_id": "kv_groups",
            "teacher_value": 8,
            "values": [8, 6],
            "alignment": 1,
            "label": "KV groups",
        },
        {
            "axis_id": "q_heads_per_group",
            "teacher_value": 4,
            "values": [4, 3],
            "alignment": 1,
            "label": "Q heads per group",
        },
        {
            "axis_id": "moe_experts",
            "teacher_value": 64,
            "values": [64, 48],
            "alignment": 16,
            "label": "Experts",
        },
    ],
}
SELECTED = {
    "axes": {
        "hidden_width": {"enabled": True, "values": [3072]},
        "kv_groups": {"enabled": True, "values": [6]},
        "q_heads_per_group": {"enabled": True, "values": [3]},
        "moe_experts": {"enabled": True, "values": [48]},
    }
}
```

Add independent tests proving:

- width-sanity rejects TP=8 because candidate query heads include 18;
- bypass accepts TP=8 because teacher query/KV heads are 32/8;
- replacement scoring rejects EP=64 because candidate experts include 48;
- depth and width importance use teacher geometry;
- CP=3 rejects sequence length 1024;
- dense inventory rejects EP greater than one;
- PP=7 is accepted regardless of layer count;
- valid-divisor messages contain a literal common-divisor list.

Add GDN and Mamba fixtures proving TP checks derived GDN value heads and Mamba
head counts, without treating their head dimensions as TP degrees.

- [ ] **Step 2: Run geometry tests and verify RED**

Run:

```bash
rtk python -m pytest -o addopts= \
  tests/unit/torch/puzzletron/test_setup_v2_parallel_validation.py -q
```

Expected: collection failure because the compatibility module does not exist.

- [ ] **Step 3: Implement geometry domains and stage scopes**

Use explicit stage sets:

```python
TEACHER_STAGE_IDS = frozenset(
    {
        "depth_importance",
        "width_importance",
        "sort_sanity",
        "bypass",
        "bypass_sanity",
    }
)
CANDIDATE_STAGE_IDS = frozenset(
    {
        "width_sanity",
        "slicing_sanity",
        "replacement_scoring",
        "vllm_stats",
    }
)
CANDIDATE_POST_MIP_TYPES = frozenset({"evaluation", "global_kd", "aiperf"})
```

For teacher scope use only teacher values. For candidate scope include the
teacher and selected axis values. Compute query-head domains as the Cartesian
products of KV-group and query-heads-per-group domains, and GDN value heads as
the products of key-group and value-heads-per-group domains.

Implement a hand-derived common-divisor helper using `math.gcd` over all
encountered values. Do not apply any layer-count rule to PP.

- [ ] **Step 4: Implement AutoModel and vLLM rules**

AutoModel checks:

```text
TP: hidden width, query heads, KV heads, dense/expert/shared/latent FFN widths,
    GDN key/value heads, Mamba heads
EP: expert counts
CP: sequence length
mesh: positive dimensions and DP-shard divisible by EP
sequence parallel: TP > 1
dense model: EP == 1
```

vLLM checks:

```text
TP divides every query-head count
DCP divides TP
non-MLA DCP constraints hold for every query/KV combination
expert parallel may be enabled only for MoE
effective EP = TP * DP divides every encountered expert count
```

The non-MLA DCP branch must reject when DCP is greater than one and:

```python
tp <= kv_heads
or dcp > tp // kv_heads
or (query_heads // kv_heads) % dcp
```

Return all issues in deterministic path/message order.

- [ ] **Step 5: Route legacy profile validation through the engine's generic checks**

Keep `validate_parallel_profile(profile, inventory=None)` backward compatible.
It continues to validate mesh and sequence parallelism; when inventory is
provided it calls the teacher-scope AutoModel validator. Do not add candidate
validation here because stage identity and selected axes are unavailable.

- [ ] **Step 6: Run geometry tests and verify GREEN**

Run the command from Step 2. Expected: all literal domain and stage-scope tests
pass.

- [ ] **Step 7: Commit the new compatibility engine**

`puzzletron_setup/v2/parallel_validation.py` and its new test file are isolated
from existing user edits and may be committed with:

```bash
rtk git add puzzletron_setup/v2/parallel_validation.py \
  tests/unit/torch/puzzletron/test_setup_v2_parallel_validation.py
rtk git commit -s -S -m "Add stage-aware Puzzletron parallel validation"
```

Leave `resources.py` unstaged if its baseline changes during execution.

---

### Task 4: Reject and reprompt in the setup-v2 wizard

**Files:**
- Modify: `puzzletron_setup/v2/wizard.py:802-876,969-1045,1810-2018,2285-2735`
- Create: `tests/unit/torch/puzzletron/test_setup_v2_parallel_prompts.py`

**Interfaces:**
- Consumes Task 3 validation functions.
- `_profile_prompt(...)` validates both reused and newly entered profiles using
  the stage scope and current pruning state.
- `_serving_setting_prompt(...)` returns:

```python
{
    "input_tokens": int,
    "output_tokens": int,
    "concurrency": int,
    "request_count": int,
    "topology": {
        "tensor_parallel_size": int,
        "pipeline_parallel_size": int,
        "data_parallel_size": int,
        "prefill_context_parallel_size": int,
        "decode_context_parallel_size": int,
        "enable_expert_parallel": bool,
        "gpu_group_size": int,
        "distributed_executor_backend": "mp",
    },
}
```

- `_configure_dynamic_resources(...)` uses `vllm_topology_to_mesh` for AIPerf
  allocation and never constructs a fake AutoModel EP overlay.

- [ ] **Step 1: Write failing scripted-prompt tests**

Use real `WizardState`, `WizardSession`, `ResourceProfileRegistry`, and
`ScriptedBackend` objects. Add tests proving:

- choosing an incompatible reused profile prints its conflict and consumes a
  second profile choice;
- entering an incompatible TP rejects the profile before confirmation and
  consumes another TP answer;
- a default profile is not accepted when it is incompatible;
- serving asks DP independently and asks a boolean expert-parallel question;
- the captured MoE prompt contains `effective EP is TP * DP`;
- invalid serving TP/DP is rejected and asks for another topology;
- returned topology has no numeric `expert_parallel_size`.

Use a recording backend that stores prompt messages while returning scripted
answers. Assert on accepted profile/topology behavior, not private wizard
state.

- [ ] **Step 2: Run prompt tests and verify RED**

Run:

```bash
rtk python -m pytest -o addopts= \
  tests/unit/torch/puzzletron/test_setup_v2_parallel_prompts.py -q
```

Expected: failures because reused profiles return before validation, serving
has no DP question, and serving asks for numeric EP.

- [ ] **Step 3: Add one reusable issue renderer**

Add:

```python
def _print_parallel_issues(issues) -> None:
    for issue in issues:
        print(f"  {issue.path}: {issue.message}")
    print("  Choose a different parallel setting.")
```

Use it after every failed compatibility check.

- [ ] **Step 4: Loop profile create and reuse until compatible**

Move the reuse menu inside `_profile_prompt`'s outer loop. Validate a reused
profile before returning it. Validate a new profile after all numeric questions
and before confirmation. On an issue, print all conflicts and restart the
profile action.

When `_configure_stage_resource` encounters an invalid built-in/default
profile, call `_profile_prompt` so the user can replace it instead of accepting
or rewriting it.

- [ ] **Step 5: Ask the corrected AIPerf and serving questions**

Add `request_count` after concurrency. Add an independent
`data_parallel_size` question. Replace the numeric EP question with:

```python
enable_ep = session.confirm(
    f"{prefix}.topology.enable_expert_parallel",
    (
        "Enable vLLM expert parallelism? vLLM has no separate EP size; "
        "for this MoE model, effective EP is TP * DP."
    ),
    default=False,
)
```

For dense models store `False` without asking. Calculate GPU group size from
TP, PP, DP, and prefill CP. Validate before returning; on failure, print issues
and restart the serving topology questions.

- [ ] **Step 6: Use the normalized allocation mesh for dynamic AIPerf nodes**

Convert the AIPerf topology with `vllm_topology_to_mesh` and render the returned
allocation mesh. This deliberately emits `ep=1`, `dp_shard=1`, and
`dp_replicate=DP` because scheduler allocation must not model effective vLLM EP
as an extra axis.

- [ ] **Step 7: Run prompt tests and verify GREEN**

Run the command from Step 2. Expected: all scripted prompt tests pass and no
invalid value is silently changed.

- [ ] **Step 8: Commit isolated prompt tests**

Commit the new prompt test file independently:

```bash
rtk git add tests/unit/torch/puzzletron/test_setup_v2_parallel_prompts.py
rtk git commit -s -S -m "Test Puzzletron parallel setting reprompts"
```

Keep `wizard.py` unstaged because it contains substantial pre-existing user
work.

---

### Task 5: Enforce compatibility for persisted state and bundle rendering

**Files:**
- Modify: `puzzletron_setup/v2/validation.py:1-105`
- Modify: `puzzletron_setup/v2/bundle.py:83-145`
- Modify: `puzzletron_setup/bundle.py:136-175,350-375,450-475`
- Modify: `tests/unit/torch/puzzletron/test_setup_bundle.py`
- Extend: `tests/unit/torch/puzzletron/test_setup_v2_parallel_validation.py`

**Interfaces:**
- `validate_state(state)` calls Task 3 validators for every applicable static
  resource, vLLM measurement topology, and dynamic post-MIP node.
- `build_bundles_v2` already treats returned issues as fatal, making this the
  defensive bundle boundary.
- Legacy bundle rendering uses `normalize_vllm_topology` and emits boolean
  expert parallelism for new serving topology.

- [ ] **Step 1: Write failing persisted-state tests**

Create a real temporary `WizardState`, store the literal inventory/pruning data
from Task 3, and add profile/resource collections. Assert:

```python
issues = validate_state(state)
assert any(
    issue.path == "stage_resources.width_sanity.tp"
    and "query-head counts" in issue.message
    for issue in issues
)
```

Add equivalent cases for:

- teacher-only bypass accepting a TP that is compatible with teacher geometry;
- dynamic post-MIP evaluation rejecting a selected candidate FFN size;
- dynamic AIPerf rejecting an invalid candidate TP/effective EP;
- a hand-edited vLLM measurement topology;
- a reused profile referenced by multiple stages being checked independently
  under each stage's scope.

Add bundle tests proving legacy numeric EP=4 with TP=2/DP=4 is rejected, while
boolean expert parallelism allocates eight GPUs and emits no fake AutoModel EP.

- [ ] **Step 2: Run state and bundle tests and verify RED**

Run:

```bash
rtk python -m pytest -o addopts= \
  tests/unit/torch/puzzletron/test_setup_v2_parallel_validation.py \
  tests/unit/torch/puzzletron/test_setup_bundle.py \
  -q
```

Expected: state validation misses the geometry conflicts and legacy bundle
rendering still treats numeric EP as an allocation overlay.

- [ ] **Step 3: Validate all persisted static and dynamic consumers**

In `validate_state`, read:

```python
inventory = _mapping(state.payload.get("inventory"))
pruning = _mapping(state.collection("pruning"))
sequence_length = int(state.get_field("data.sequence_length", 4096))
```

For each static `stage_resources` entry, resolve `profile_name` and validate
with the exact stage ID. For post-MIP resources, resolve the node type from
`post_mip_flows[flow]["nodes"][node]["type"]`; validate evaluation/global-KD
profiles and AIPerf topology with candidate scope. Validate each
`vllm_measurements.*.runtime_stats.topology` as `vllm_stats`.

Convert every compatibility issue to the existing `ValidationIssue` type while
preserving path and message.

- [ ] **Step 4: Replace legacy serving allocation conversion**

Make `_serving_parallel` consume `normalize_vllm_topology`, then render an
allocation-only mesh. Change built-in new topology defaults from numeric
`expert_parallel_size` to boolean `enable_expert_parallel`.

Preserve old configuration reading through Task 1's migration rule; do not
rewrite checked-in historical experiment YAML files as part of this fix.

- [ ] **Step 5: Run state and bundle tests and verify GREEN**

Run the command from Step 2. Expected: all tests pass.

- [ ] **Step 6: Commit isolated validation changes where safe**

Commit files without pre-existing user edits:

```bash
rtk git add puzzletron_setup/v2/validation.py
rtk git commit -s -S -m "Reject incompatible Puzzletron stage profiles"
```

Leave dirty bundle files unstaged if a full-file add would capture unrelated
user work.

---

### Task 6: Focused regression, bundle, and dry-run verification

**Files:**
- Modify only if tests expose a defect in Tasks 1-5.
- Verify:
  `nano/production/experiment.yaml`,
  `nano/production/runner.yaml`,
  `nano/production/execution.yaml`.

**Interfaces:**
- Consumes all preceding tasks.
- Produces a verified setup/bundle/orchestrator contract before GPU use.

- [ ] **Step 1: Run focused regression tests**

Run:

```bash
rtk python -m pytest -o addopts= \
  tests/unit/torch/puzzletron/test_aiperf_context_capacity.py \
  tests/unit/torch/puzzletron/test_orchestration_mesh.py \
  tests/unit/torch/puzzletron/test_orchestration_lightweight.py \
  tests/unit/torch/puzzletron/test_post_mip_runner.py \
  tests/unit/torch/puzzletron/test_profile_aiperf_worker.py \
  tests/unit/torch/puzzletron/test_setup_bundle.py \
  tests/unit/torch/puzzletron/test_setup_v2_parallel_validation.py \
  tests/unit/torch/puzzletron/test_setup_v2_parallel_prompts.py \
  -q
```

Expected: zero failures.

- [ ] **Step 2: Run Puzzletron pre-commit hooks on changed paths**

Run the repository-configured hook command from `CONTRIBUTING.md` against the
changed files. Review any automatic rewrites and rerun focused tests afterward.

- [ ] **Step 3: Compile the existing nano production plan**

Run:

```bash
rtk python examples/puzzletron/orchestrate.py \
  --experiment nano/production/experiment.yaml \
  --runner nano/production/runner.yaml \
  --execution nano/production/execution.yaml \
  --stage full --dry-run
```

Expected: the plan compiles, AIPerf GPU allocation equals TP × PP × DP ×
prefill CP, and no numeric EP is passed to vLLM.

- [ ] **Step 4: Inspect dry-run allocation and command evidence**

Verify every AIPerf work item shows:

```text
vLLM example: --tensor-parallel-size 1 --data-parallel-size 8
MoE only: --enable-expert-parallel
AIPerf example: --concurrency 2 --request-count 8
GPU group formula: TP * PP * DP * prefill_CP
```

Reject the dry run if it includes `--expert-parallel-size`, unsupported
`minimum_request_count`/`requests_per_concurrency` keyword forwarding, or a
world size multiplied by effective EP.

- [ ] **Step 5: Run the broader setup/Puzzletron unit slice**

Run:

```bash
rtk python -m pytest -o addopts= tests/unit/torch/puzzletron \
  -k "aiperf or setup or orchestration_mesh or post_mip_runner" -q
```

Expected: zero failures attributable to this change. Record unrelated baseline
failures verbatim rather than modifying unrelated code.

---

### Task 7: Verify representative saved models on one interactive 8-GPU node

**Files:**
- Read: `nv-internal/CLUSTER_GUIDE_NV.md`
- Read: saved candidate registries and checkpoints under `nano/results/production/`
- Write runtime logs only beneath `nano/results/production/` or `/tmp`.

**Interfaces:**
- Consumes the corrected AIPerf/vLLM contract and existing nano execution
  environment.
- Produces health/AIPerf evidence for at least three representative saved
  models from the set of 32, including the teacher or widest candidate, a
  middle candidate, and a smallest candidate.

- [ ] **Step 1: Verify current scheduler state**

Run:

```bash
rtk sinfo
rtk scontrol show partition interactive
rtk squeue --me
```

Record the current partition limit and availability instead of relying on
remembered cluster state.

- [ ] **Step 2: Resolve exact checkpoint and topology inputs**

Inspect the canonical candidate registry and each selected checkpoint's
`config.json`. Select three existing, complete checkpoints spanning the saved
geometry range. For each, compute valid TP, DP, and effective EP from the actual
query heads, KV heads, expert count, and configured serving workload.

- [ ] **Step 3: Launch one 8-GPU interactive job**

Use the nano runner's account, container, mount, repository, setup command, and
`.venv_new`:

```bash
rtk srun -p interactive -t 4:00:00 -A coreai_dlalgo_llm \
  --nodes=1 --ntasks=1 --gpus-per-node=8 --exclusive \
  --container-image=/lustre/fs1/portfolios/coreai/projects/coreai_dlalgo_llm/users/ssameni/enroot/pytorch_25p05.sqsh \
  --container-mounts=/lustre:/lustre \
  --container-workdir=/lustre/fs1/portfolios/coreai/projects/coreai_dlalgo_llm/users/ssameni/engineering/modelopt_qwen \
  /bin/bash -lc '
    set -Eeuo pipefail
    source /lustre/fs1/portfolios/coreai/projects/coreai_dlalgo_llm/users/ssameni/setup-envs.sh
    source .venv_new/bin/activate
    export PYTHONPATH="$PWD:${PYTHONPATH:-}"
    checkpoint_root=nano/results/production/artifacts/post_mip/nodes/materialized/executions/post_mip_execution_98fbd79cd3523fed/checkpoints
    mapfile -t candidates < <(find "$checkpoint_root" -mindepth 2 -maxdepth 2 \
      -name config.json -printf "%h\n" | sort)
    test "${#candidates[@]}" -ge 3
    selected_indices=(0 $((${#candidates[@]} / 2)) $((${#candidates[@]} - 1)))
    for index in "${selected_indices[@]}"; do
      checkpoint="${candidates[$index]}"
      artifact="/tmp/puzzletron-aiperf-${SLURM_JOB_ID}-$(basename "$checkpoint")"
      python - "$checkpoint" "$artifact" <<'"'"'PY'"'"'
import sys
from modelopt.torch.puzzletron.benchmarks.aiperf import run_aiperf_sweep

checkpoint, artifact = sys.argv[1:]
results = run_aiperf_sweep(
    checkpoint,
    artifact_dir=artifact,
    concurrencies=(1,),
    input_tokens=128,
    output_tokens=32,
    gpu_ids="0,1,2,3,4,5,6,7",
    topology={
        "tensor_parallel_size": 1,
        "pipeline_parallel_size": 1,
        "data_parallel_size": 8,
        "prefill_context_parallel_size": 1,
        "decode_context_parallel_size": 1,
        "enable_expert_parallel": True,
        "gpu_group_size": 8,
    },
    request_counts={1: 4},
    executable="aiperf",
    benchmark_timeout=900,
)
assert len(results) == 1
assert not results[0].failures
assert results[0].metrics["request_throughput"] > 0
print(checkpoint, results[0].metrics["request_throughput"])
PY
    done
  '
```

The sequential script uses `set -Eeuo pipefail`, sources
`/lustre/fs1/portfolios/coreai/projects/coreai_dlalgo_llm/users/ssameni/setup-envs.sh`,
activates `.venv_new`, records package/CUDA versions, and serves one checkpoint
at a time. It terminates each server and verifies no orphan remains before
starting the next checkpoint.

- [ ] **Step 4: Exercise each server with a small real AIPerf request**

For every selected model:

1. launch vLLM with the validated TP/DP topology;
2. wait for `/health`;
3. run AIPerf with fixed small ISL/OSL, concurrency, and request count;
4. verify successful responses and a nonzero request/token throughput;
5. retain vLLM and AIPerf logs with the checkpoint identity; and
6. terminate the server process group.

The command must use `--enable-expert-parallel` for MoE serving and must never
pass a numeric EP flag.

- [ ] **Step 5: Re-run the corrected post-MIP AIPerf path**

Use the canonical orchestrator/config path for the smallest safe AIPerf work
item or a smoke copy of the nano configuration. Verify the same corrected
command is produced through Puzzletron rather than only through a direct vLLM
launch.

- [ ] **Step 6: Run final verification before reporting completion**

Use `superpowers:verification-before-completion`. Re-run the focused tests,
inspect the final diff, confirm no user file was overwritten, and report:

- exact passing test counts;
- dry-run result;
- Slurm job ID/node;
- checkpoint identities tested;
- TP/DP/effective-EP settings;
- vLLM health and AIPerf outcomes; and
- any remaining external or environmental blocker.
