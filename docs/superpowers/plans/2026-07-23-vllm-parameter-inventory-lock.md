# vLLM Parameter-Inventory Lock Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Prevent clean sharded vLLM-statistics runs from racing while building shared parameter inventories.

**Architecture:** Keep homogeneous candidate parent `-1` unchanged. Wrap the existing per-width inventory calculation with Puzzletron's existing process-safe `file_lock`, so the cache completeness check and all writes for one width have a single owner.

**Tech Stack:** Python, existing Puzzletron filesystem locking, Ruff.

## Global Constraints

- Do not run the campaign orchestrator or submit Slurm jobs.
- Do not add or run automated tests, per the user's explicit instruction.
- Preserve the existing synthetic parent-layer value `-1`.
- Preserve unrelated dirty-worktree changes.
- Do not change generated campaign configuration or runtime shard allocation.

---

### Task 1: Serialize Same-Width Parameter-Inventory Writers

**Files:**
- Modify: `modelopt/torch/puzzletron/subblock_stats/calc_subblock_stats.py`

**Interfaces:**
- Consumes: `distributed_eval.storage.file_lock(path: Path)`.
- Preserves: `_calculate_parameter_inventory_for_width(...) -> dict`.
- Adds: `_calculate_parameter_inventory_for_width_unlocked(...) -> dict` as the existing calculation body.

- [ ] **Step 1: Import the existing lock**

Add beside the other Puzzletron imports:

```python
from ..distributed_eval.storage import file_lock
```

- [ ] **Step 2: Rename the current calculation body**

Rename the existing function without changing its parameters or body:

```python
def _calculate_parameter_inventory_for_width_unlocked(
    *,
    master_puzzle_dir: Path,
    checkpoint_dir: Path,
    descriptor: Type[ModelDescriptor],
    width: int,
    subblock_configs: list[immutabledict[str, SubblockConfig]],
    cache_root: Path,
    progress_every: int,
) -> dict:
```

- [ ] **Step 3: Add the locking entry point**

Add immediately before the unlocked implementation:

```python
def _calculate_parameter_inventory_for_width(
    *,
    master_puzzle_dir: Path,
    checkpoint_dir: Path,
    descriptor: Type[ModelDescriptor],
    width: int,
    subblock_configs: list[immutabledict[str, SubblockConfig]],
    cache_root: Path,
    progress_every: int,
) -> dict:
    """Build or reuse one width inventory under single-writer ownership."""

    with file_lock(cache_root / f".width-{int(width):04d}.lock"):
        return _calculate_parameter_inventory_for_width_unlocked(
            master_puzzle_dir=master_puzzle_dir,
            checkpoint_dir=checkpoint_dir,
            descriptor=descriptor,
            width=width,
            subblock_configs=subblock_configs,
            cache_root=cache_root,
            progress_every=progress_every,
        )
```

Because the unlocked implementation reads the cache only after the wrapper
acquires the lock, waiting shards observe and reuse the first shard's complete
inventory.

- [ ] **Step 4: Perform non-executing verification**

Run only static checks:

```bash
python -m ruff check modelopt/torch/puzzletron/subblock_stats/calc_subblock_stats.py
python -m ruff format --check modelopt/torch/puzzletron/subblock_stats/calc_subblock_stats.py
git diff --check -- modelopt/torch/puzzletron/subblock_stats/calc_subblock_stats.py
```

Expected: no lint, formatting, or whitespace errors. Do not run Pytest, the
orchestrator, or any Slurm command.

- [ ] **Step 5: Report the manual rerun command without executing it**

Provide:

```bash
source .venv-orchestrator/bin/activate
python examples/puzzletron/orchestrate.py \
  --experiment ../puzzle_runs/qwen/production/experiment.yaml \
  --runner ../puzzle_runs/qwen/production/runner.yaml \
  --execution ../puzzle_runs/qwen/production/execution.yaml \
  --stage vllm_stats
```

Do not delete the now-complete parameter-inventory caches. On manual rerun,
all eight shards should report cache reuse and proceed to their assigned
runtime specs.
