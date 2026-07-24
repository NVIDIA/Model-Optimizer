# vLLM Repeat-Count Default Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make `4` the global default paired vLLM runtime-estimator repeat count, regenerate the Nano bundles, and remove only its invalidated vLLM-statistics artifacts.

**Architecture:** Declare the public default in `examples/puzzletron/configs/base.yaml` so all composed and wizard-generated configurations inherit it. Align the two defensive Python fallbacks with that public default while retaining explicit overrides and existing cache identity behavior.

**Tech Stack:** Python 3, PyYAML/OmegaConf, pytest, Puzzletron setup bundle renderer, Slurm CLI.

## Global Constraints

- The global default is exactly `repeat_block_n_times: 4`.
- Explicit per-run repeat counts continue to override the default.
- Paired estimation remains exact `N` and `2N` physical construction.
- Do not add model-name conditionals or runtime proxy dimensions.
- Do not submit the orchestrator.
- Preserve Nano non-vLLM artifacts, orchestration history, and the failed vLLM log.

---

### Task 1: Change and Verify the Global Default

**Files:**
- Modify: `examples/puzzletron/configs/base.yaml`
- Modify: `modelopt/torch/puzzletron/subblock_stats/calc_runtime_stats.py`
- Modify: `tests/unit/torch/puzzletron/test_setup_bundle.py`
- Modify: `tests/unit/torch/puzzletron/test_sparse_runtime_stats.py`

**Interfaces:**
- Consumes: `render_experiment(state: Mapping[str, Any], budget: str) -> dict[str, Any]`
- Consumes: `calc_runtime_for_blocks(...)` and `calc_runtime_for_subblocks(...)`
- Produces: a public and defensive default repeat count of four

- [ ] **Step 1: Add a failing wizard-bundle regression test**

Add:

```python
def test_render_experiment_uses_global_runtime_repeat_default() -> None:
    experiment = render_experiment(
        _nemotron_render_state(latent_moe=False),
        "production",
    )

    assert experiment["vllm_stats"]["runtime_stats"]["repeat_block_n_times"] == 4
```

- [ ] **Step 2: Add a failing runtime fallback regression test**

Use a descriptor and monkeypatched `_run_benchmarks` to call both
`calc_runtime_for_blocks` and `calc_runtime_for_subblocks` with an empty
OmegaConf runtime mapping. Capture the generated layouts and assert that each
function produces layout lengths `[4, 8]`.

- [ ] **Step 3: Run both tests and verify the red state**

Run:

```bash
python -m pytest -o addopts='' --confcutdir=tests/unit/torch/puzzletron -q \
  tests/unit/torch/puzzletron/test_setup_bundle.py::test_render_experiment_uses_global_runtime_repeat_default \
  tests/unit/torch/puzzletron/test_sparse_runtime_stats.py::test_block_and_subblock_runtime_default_to_four_repeats
```

Expected: the bundle assertion lacks the key and the runtime layouts are
`[10, 20]`.

- [ ] **Step 4: Implement the minimal global default**

Add this field under `vllm_stats.runtime_stats` in
`examples/puzzletron/configs/base.yaml`:

```yaml
repeat_block_n_times: 4
```

Change both absent-setting fallbacks in `calc_runtime_stats.py`:

```python
runtime_stats_config.get("repeat_block_n_times", 4)
```

- [ ] **Step 5: Run focused and related tests**

Run:

```bash
python -m pytest -o addopts='' --confcutdir=tests/unit/torch/puzzletron -q \
  tests/unit/torch/puzzletron/test_setup_bundle.py \
  tests/unit/torch/puzzletron/test_sparse_runtime_stats.py
```

Expected: all tests pass.

- [ ] **Step 6: Commit the source fix**

Create a signed local commit containing only the four Task 1 files.

### Task 2: Regenerate Nano and Invalidate Old vLLM Outputs

**Files:**
- Regenerate: `../puzzle_runs/nano/smoke/{experiment,runner,execution}.yaml`
- Regenerate: `../puzzle_runs/nano/production/{experiment,runner,execution}.yaml`
- Regenerate: both bundle dry-run plans and campaign README
- Delete: `../puzzle_runs/nano/results/production/runtime_cache/`
- Delete: `../puzzle_runs/nano/results/production/artifacts/vllm_stats/`
- Delete: `../puzzle_runs/nano/results/production/artifacts/subblock_stats/parameter_inventory/`
- Delete if present: `../puzzle_runs/nano/results/production/subblock_stats.json`
- Delete if present: `../puzzle_runs/nano/results/production/manifests/vllm_stats.json`

**Interfaces:**
- Consumes: `build_bundles(campaign_dir: Path, state: Mapping[str, Any]) -> BundleResult`
- Produces: validated Nano YAMLs with repeat count four and no stale vLLM output

- [ ] **Step 1: Regenerate both bundles from canonical answers**

Resume `../puzzle_runs/nano/answers.yaml`, call `build_bundles`, and assert that
both returned validations are valid.

- [ ] **Step 2: Verify the regenerated semantic values**

Load both experiment YAMLs and assert:

```python
experiment["vllm_stats"]["runtime_stats"]["repeat_block_n_times"] == 4
```

Also verify each dry-run plan exists and identifies 18 stages.

- [ ] **Step 3: Check Slurm before deletion**

Query all active jobs for `ssameni` and inspect their job metadata. Stop if any
active job references `/puzzle_runs/nano/`.

- [ ] **Step 4: Resolve and delete only the approved artifacts**

Resolve the production result root from canonical answers, assert it equals
`/lustre/fs1/portfolios/coreai/projects/coreai_dlalgo_llm/users/ssameni/engineering/puzzle_runs/nano/results/production`,
then delete only the five paths listed under Task 2 Files.

- [ ] **Step 5: Verify rerun readiness**

Assert that every invalidated path is absent; the failed vLLM log,
`orchestration/`, conversion outputs, dataset caches, width/depth outputs, and
both regenerated bundle triplets remain present. Run `git diff --check` and
rerun the two focused regression tests.
