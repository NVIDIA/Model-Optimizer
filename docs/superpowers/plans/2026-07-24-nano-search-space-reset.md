# Nano Search-Space Reset Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Regenerate the Nano Puzzletron campaign with exactly two selected values per pruning axis and remove its complete generated results tree for a clean rerun.

**Architecture:** Treat `../puzzle_runs/nano/answers.yaml` as canonical state, mutate only `answers.pruning.axes.*.values` through `AnswerState`, and regenerate both bundles through `build_bundles`. Validate the semantic diff and compiled dry-run plans before checking Slurm ownership and deleting the exact canonical result root.

**Tech Stack:** Python 3, PyYAML, Puzzletron setup bundle renderer, Puzzletron dependency-light orchestrator, Slurm CLI.

## Global Constraints

- Preserve the Nano model, dataset, topology, MIP, post-MIP, runner, and execution settings.
- Keep exactly `[2688, 2304]`, `[2, 1]`, `[16, 12]`, `[128, 96]`, `[1856, 1344]`, `[3712, 2176]`, `[6, 5]`, `[64, 56]`, and `[64, 56]` for the nine enabled axes in canonical order.
- Preserve the full legal domains in the model inventory; change only the selected campaign search space.
- Delete only `/lustre/fs1/portfolios/coreai/projects/coreai_dlalgo_llm/users/ssameni/engineering/puzzle_runs/nano/results`.
- Do not submit the orchestrator.

---

### Task 1: Update Canonical Search Values and Regenerate Bundles

**Files:**
- Modify: `../puzzle_runs/nano/answers.yaml`
- Regenerate: `../puzzle_runs/nano/smoke/experiment.yaml`
- Regenerate: `../puzzle_runs/nano/smoke/runner.yaml`
- Regenerate: `../puzzle_runs/nano/smoke/execution.yaml`
- Regenerate: `../puzzle_runs/nano/smoke/dry-run-plan.txt`
- Regenerate: `../puzzle_runs/nano/production/experiment.yaml`
- Regenerate: `../puzzle_runs/nano/production/runner.yaml`
- Regenerate: `../puzzle_runs/nano/production/execution.yaml`
- Regenerate: `../puzzle_runs/nano/production/dry-run-plan.txt`
- Regenerate: `../puzzle_runs/nano/README.md`

**Interfaces:**
- Consumes: `AnswerState.resume(path: Path) -> AnswerState`
- Consumes: `build_bundles(campaign_dir: Path, state: Mapping[str, Any]) -> BundleResult`
- Produces: two validated, dry-run-compiled campaign bundles with identical search values

- [ ] **Step 1: Save the pre-change canonical state**

Run:

```bash
cp ../puzzle_runs/nano/answers.yaml /tmp/nano-answers.before.yaml
```

Expected: `/tmp/nano-answers.before.yaml` is a readable copy of the canonical
state.

- [ ] **Step 2: Atomically update only the selected axis values**

Run this Python program through `.venv-orchestrator/bin/python -c`:

```python
from pathlib import Path

from puzzletron_setup.state import AnswerState

expected = {
    "hidden_width": [2688, 2304],
    "kv_groups": [2, 1],
    "q_heads_per_group": [16, 12],
    "moe_experts": [128, 96],
    "moe_expert_intermediate": [1856, 1344],
    "moe_shared_expert_intermediate": [3712, 2176],
    "moe_top_k": [6, 5],
    "mamba_heads": [64, 56],
    "mamba_head_dim": [64, 56],
}
state = AnswerState.resume(Path("../puzzle_runs/nano/answers.yaml"))
axes = state.payload["answers"]["pruning"]["axes"]
assert set(axes) == set(expected)
for name, values in expected.items():
    assert axes[name]["enabled"] is True
    assert axes[name]["teacher_value"] == values[0]
    axes[name]["values"] = values
state.save()
```

Expected: the command exits zero and writes `answers.yaml` atomically.

- [ ] **Step 3: Prove unrelated canonical answers are unchanged**

Load `/tmp/nano-answers.before.yaml` and the new `answers.yaml`, remove
`updated_at` and each `answers.pruning.axes.<name>.values` field from both
mappings, and assert the remaining mappings are equal.

Expected: the comparison exits zero and prints
`unrelated canonical answers unchanged`.

- [ ] **Step 4: Regenerate and dry-run both bundles**

Run this Python program through `.venv-orchestrator/bin/python -c`:

```python
from pathlib import Path

from puzzletron_setup.bundle import build_bundles
from puzzletron_setup.state import AnswerState

campaign = Path("../puzzle_runs/nano")
state = AnswerState.resume(campaign / "answers.yaml")
result = build_bundles(campaign, state.payload)
assert result.smoke.valid, result.smoke.error
assert result.production.valid, result.production.error
```

Expected: both bundles report valid and their dry-run plans are regenerated
without submitting jobs.

- [ ] **Step 5: Verify all three search-space representations**

Load:

- `answers.yaml` at `answers.pruning.axes`;
- `smoke/experiment.yaml` at `search_space.axes`; and
- `production/experiment.yaml` at `search_space.axes`.

Assert that every mapping has the exact nine axes and exact values declared in
Global Constraints, that every value list has length two, and that
`moe_top_k.values == [6, 5]`.

Expected: the command exits zero and prints
`all Nano search spaces contain exactly two values per axis`.

### Task 2: Safely Remove the Generated Nano Results

**Files:**
- Delete: `../puzzle_runs/nano/results/`
- Preserve: `../puzzle_runs/nano/answers.yaml`
- Preserve: `../puzzle_runs/nano/README.md`
- Preserve: `../puzzle_runs/nano/smoke/`
- Preserve: `../puzzle_runs/nano/production/`

**Interfaces:**
- Consumes: `answers.output.result_root` from canonical wizard state
- Produces: an absent Nano results tree and intact validated bundle inputs

- [ ] **Step 1: Resolve and validate the exact deletion target**

Load `answers.output.result_root`, resolve it, and assert equality with:

```text
/lustre/fs1/portfolios/coreai/projects/coreai_dlalgo_llm/users/ssameni/engineering/puzzle_runs/nano/results
```

Expected: the command exits zero and prints the exact absolute result root.

- [ ] **Step 2: Refuse deletion if an active Slurm job references Nano**

Run:

```bash
squeue -u ssameni -h -o %A
```

For every returned job ID, inspect `scontrol show job -o <job-id>` and search
its command, working directory, standard output, and standard error fields for
`/puzzle_runs/nano/`.

Expected: no pending or running Slurm job references the Nano campaign. If one
does, stop without deleting anything.

- [ ] **Step 3: Delete the validated result root**

Run only after Steps 1 and 2 succeed:

```bash
rm -rf /lustre/fs1/portfolios/coreai/projects/coreai_dlalgo_llm/users/ssameni/engineering/puzzle_runs/nano/results
```

Expected: the exact validated 60 GB generated result tree is removed.

- [ ] **Step 4: Verify clean-start state and preserved inputs**

Assert that the deleted result path does not exist and that these files exist:

```text
../puzzle_runs/nano/answers.yaml
../puzzle_runs/nano/README.md
../puzzle_runs/nano/smoke/experiment.yaml
../puzzle_runs/nano/smoke/runner.yaml
../puzzle_runs/nano/smoke/execution.yaml
../puzzle_runs/nano/production/experiment.yaml
../puzzle_runs/nano/production/runner.yaml
../puzzle_runs/nano/production/execution.yaml
```

Expected: the results path is absent, all eight preserved inputs exist, and no
orchestrator process has been launched.
