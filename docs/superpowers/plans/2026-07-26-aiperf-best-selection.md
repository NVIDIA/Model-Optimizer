# AIPerf Best-Selection Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add one fixed-topology AIPerf concurrency sweep with global-individual-best and top-K-per-concurrency selection modes, expose both setup v2 questions, and correct two stale tests.

**Architecture:** Keep AIPerf execution unchanged: one server is reused while Puzzletron profiles the configured concurrency list. Add sweep-aware metric resolution to the candidate ledger and extend the existing `top_k` filter with `best_selection_mode`. Setup v2 owns parsing the concurrency list and transfers the selected policy to the downstream fastest filter.

**Tech Stack:** Python 3.12, dataclasses, pytest, Puzzletron post-MIP candidate ledger/filter framework, setup v2 scripted wizard tests.

## Global Constraints

- Use one vLLM topology for all models and concurrency points.
- Accept one or more positive, unique concurrency integers.
- Preserve existing scalar `top_k` behavior when `best_selection_mode` is omitted.
- `individual_best` retains global top K after reducing each model to its own best concurrency.
- `best_per_concurrency` retains top K at each concurrency, then unions and deduplicates models.
- Do not change AIPerf itself or add topology search.
- Keep all existing user worktree changes and stage only task-specific files.

---

### Task 1: Sweep-Aware Metric Resolution and Selection

**Files:**
- Create: `tests/unit/torch/puzzletron/test_post_mip_filters.py`
- Modify: `modelopt/torch/puzzletron/post_mip/records.py`
- Modify: `modelopt/torch/puzzletron/post_mip/filters.py`

**Interfaces:**
- Produces: `CandidateLedger.resolve_concurrency_metrics(revision_id: str, reference: str) -> dict[int, float]`
- Extends: `top_k` filter config with `best_selection_mode: Literal["individual_best", "best_per_concurrency"]`
- Preserves: `apply_filter(...) -> tuple[tuple[str, ...], dict[str, str], dict[str, float]]`

- [ ] **Step 1: Write failing filter tests**

Create ledger fixtures with `serving` observations containing keys such as
`concurrency_1.output_token_throughput`. Add tests asserting:

```python
selected, excluded, scores = apply_filter(
    ledger,
    ("revision-a", "revision-b", "revision-c"),
    {
        "mode": "top_k",
        "metric": "serving.output_token_throughput",
        "direction": "maximize",
        "top_k": 2,
        "best_selection_mode": "individual_best",
    },
)
assert selected == ("revision-b", "revision-a")
assert scores == {"revision-a": 20.0, "revision-b": 30.0, "revision-c": 15.0}
```

Add a separate `best_per_concurrency` case where different models occupy the top two at
concurrency 1 and concurrency 8; assert that the deterministic union contains every winner
once. Add validation cases for an unknown mode, a mode on `threshold`, an unqualified metric,
missing concurrency points, minimize direction, ties, and non-finite values.

- [ ] **Step 2: Run the new tests and verify RED**

Run:

```bash
pytest -o addopts='' -q tests/unit/torch/puzzletron/test_post_mip_filters.py
```

Expected: failures because `best_selection_mode` is rejected and
`resolve_concurrency_metrics` does not exist.

- [ ] **Step 3: Implement ledger sweep resolution**

In `records.py`, parse the metric reference into owner and leaf metric. Reject `mip` and
unqualified references. Resolve the owner's lineage observation, match only the regular
expression `^concurrency_([1-9][0-9]*)\.<escaped metric>$`, accept finite non-boolean numeric
values, and return them keyed by integer concurrency.

- [ ] **Step 4: Implement the two filter reductions**

In `filters.py`:

- allow `best_selection_mode` only on `top_k`;
- validate the two exact values;
- retain the current branch when it is omitted;
- for `individual_best`, require a complete shared concurrency set, reduce by direction,
  populate `scores` with the reduced value, and use existing deterministic top-K ordering;
- for `best_per_concurrency`, require a complete shared set, rank each concurrency
  independently, union top K, set each selected candidate's score to its best one-based rank,
  and order by `(best_rank, revision_id)`;
- exclude partial/non-finite sweeps with explicit reasons.

- [ ] **Step 5: Run filter tests and verify GREEN**

Run the Task 1 pytest command. Expected: all tests pass.

- [ ] **Step 6: Commit Task 1**

Stage only the three Task 1 files and create a signed-off, signed local commit:

```bash
git commit -s -S -m "Add concurrency-aware AIPerf model selection"
```

---

### Task 2: Setup v2 Questions and Recommended Flow

**Files:**
- Modify: `puzzletron_setup/v2/wizard.py`
- Modify: `puzzletron_setup/v2/post_mip.py`
- Modify: `modelopt/torch/puzzletron/post_mip/runner.py`
- Modify: `tests/unit/torch/puzzletron/test_setup_v2_parallel_prompts.py`
- Create: `tests/unit/torch/puzzletron/test_setup_v2_post_mip.py`
- Modify: `tests/unit/torch/puzzletron/test_post_mip_runner.py`

**Interfaces:**
- Produces: `_parse_positive_int_list(value: str) -> list[int]`
- Changes: `_serving_setting_prompt(...)` returns `concurrency: list[int]` and
  `best_selection_mode: str`
- Changes: `recommended_flow(...)` copies the policy to its fastest filter
- Preserves: `_aiperf(...)` never forwards `best_selection_mode` to AIPerf

- [ ] **Step 1: Write failing wizard and flow tests**

Update the scripted serving prompt to answer `"1, 4, 8"` and
`"best_per_concurrency"`. Assert:

```python
assert result["concurrency"] == [1, 4, 8]
assert result["best_selection_mode"] == "best_per_concurrency"
assert "Serving concurrency sweep" in backend.messages
assert "How should the best models be selected?" in backend.messages
```

Add a reprompt test using `"1, 1"` followed by `"1, 2"` and verify duplicates are rejected.
Update the recommended-flow test to assert that the AIPerf node contains the list and its
fastest filter contains:

```python
{
    "metric": "serving.output_token_throughput",
    "best_selection_mode": "best_per_concurrency",
}
```

Add a runner test proving `best_selection_mode` is consumed as setup metadata and not passed
to `run_aiperf_sweep`.

- [ ] **Step 2: Run the focused tests and verify RED**

Run:

```bash
pytest -o addopts='' -q \
  tests/unit/torch/puzzletron/test_setup_v2_parallel_prompts.py \
  tests/unit/torch/puzzletron/test_setup_v2_post_mip.py \
  tests/unit/torch/puzzletron/test_post_mip_runner.py
```

Expected: failures because concurrency is still an integer and the policy is not asked or
propagated.

- [ ] **Step 3: Implement wizard parsing and questions**

Add `_parse_positive_int_list` beside `_serving_setting_prompt`. Use `session.text` for the
comma-separated sweep, reject empty/non-integer/non-positive/duplicate input through the
session validator/reprompt pattern, and use `session.select` for the two policy choices.
Continue asking topology exactly once.

- [ ] **Step 4: Propagate configuration**

Update `recommended_flow` to normalize a scalar or sequence into a list, place the list in
the AIPerf node, use `serving.output_token_throughput`, and copy the selected policy into the
fastest filter. Default to `individual_best`. In `_aiperf`, pop `best_selection_mode` before
calling `run_aiperf_sweep`.

- [ ] **Step 5: Run focused tests and verify GREEN**

Run the Task 2 pytest command. Expected: all tests pass except the separately tracked stale
CPU-stage test until Task 4.

- [ ] **Step 6: Commit Task 2**

Stage only the six Task 2 files and create:

```bash
git commit -s -S -m "Ask for AIPerf concurrency selection policy"
```

---

### Task 3: Candidate-Limit Accounting

**Files:**
- Modify: `puzzletron_setup/bundle.py`
- Modify: `tests/unit/torch/puzzletron/test_setup_bundle.py`

**Interfaces:**
- Extends: `_post_mip_candidate_limits(experiment) -> dict[str, int | None]`
- Guarantees: `best_per_concurrency` upper bound is
  `min(input_limit, top_k * len(concurrency))` when the input is an AIPerf node

- [ ] **Step 1: Write the failing candidate-limit test**

Construct a flow with 10 materialized candidates, an AIPerf sweep `[1, 4, 8]`, and
`top_k: 2`, `best_selection_mode: best_per_concurrency`. Assert the fastest-node bound is 6
and the next sharded node may allocate up to 6 instances. Add an `individual_best` assertion
that the corresponding bound remains 2.

- [ ] **Step 2: Run the candidate-limit test and verify RED**

Run only the new test by node ID. Expected: current code reports 2 for both policies.

- [ ] **Step 3: Implement the upper-bound calculation**

When processing a `top_k` filter with `best_per_concurrency`, find its input node in the same
flow, read its normalized concurrency list, and multiply `top_k` by its length before
intersecting with the input bound. Retain existing logic for every other filter.

- [ ] **Step 4: Run setup-bundle tests and verify GREEN except stale test**

Run:

```bash
pytest -o addopts='' -q tests/unit/torch/puzzletron/test_setup_bundle.py \
  -k 'not test_render_execution_uses_cpu_partition_for_io_bound_stages'
```

Expected: all selected tests pass.

- [ ] **Step 5: Commit Task 3**

Stage only the two Task 3 files and create:

```bash
git commit -s -S -m "Account for per-concurrency selection unions"
```

---

### Task 4: Correct Stale Test Expectations

**Files:**
- Modify: `tests/unit/torch/puzzletron/test_orchestration_mesh.py`
- Modify: `tests/unit/torch/puzzletron/test_setup_bundle.py`

**Interfaces:**
- No production interfaces change.

- [ ] **Step 1: Correct the mesh test**

Rename the test to describe a 16-GPU mesh and assert:

```python
assert allocation.gpus_per_instance == 16
assert allocation.nodes == 2
assert allocation.exclusive is False
```

- [ ] **Step 2: Correct the CPU-stage test**

Remove `sort` from the CPU-only loop and separately assert:

The current scheduler contract represents a GPU stage by omitting `resource: cpu`, so use:

```python
assert "resource" not in stages["sort"]
assert stages["sort"]["strategy"] == "single"
assert stages["sort"]["parallel"] == {
    "tp": 1,
    "cp": 1,
    "pp": 1,
    "ep": 1,
    "dp_shard": 1,
    "dp_replicate": 1,
    "sequence_parallel": False,
}
```

- [ ] **Step 3: Run both corrected tests**

Run the two tests by node ID. Expected: both pass against current production behavior.

- [ ] **Step 4: Commit Task 4**

Stage only the two test files and create:

```bash
git commit -s -S -m "Correct Puzzletron allocation test expectations"
```

---

### Task 5: Integrated Verification

**Files:**
- No new files.

**Interfaces:**
- Verifies the complete approved design.

- [ ] **Step 1: Run the complete focused regression set**

Run the post-MIP filter, runner, setup v2 prompt, setup bundle, mesh, AIPerf mapping, and
state-validation tests. Expected: zero failures.

- [ ] **Step 2: Run Nano's exact full dry-run**

Run:

```bash
python examples/puzzletron/orchestrate.py \
  --experiment nano/production/experiment.yaml \
  --runner nano/production/runner.yaml \
  --execution nano/production/execution.yaml \
  --stage full --dry-run
```

Expected: 17 stages and 18 submissions, with no jobs submitted.

- [ ] **Step 3: Run repository hygiene checks**

Run `git diff --check` and the applicable focused pre-commit hooks. Inspect any hook edits
before staging them. Expected: no whitespace or focused lint failures.

- [ ] **Step 4: Review final diff and commits**

Confirm only task-specific hunks were staged or committed, all unrelated user changes remain
untouched, and no remote push occurred.
