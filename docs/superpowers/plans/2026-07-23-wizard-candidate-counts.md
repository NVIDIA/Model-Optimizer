# Puzzletron Wizard Candidate Counts Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Show exact unique vLLM configuration counts and exact replace-one solution counts in the wizard's block/subblock choices.

**Architecture:** Add one dependency-light counting engine beside the existing setup profiles. It resolves the supported model's physical block families from its Hugging Face config, computes subblock Cartesian domains from the selected axes, and returns immutable counts consumed by two wizard choice-formatting helpers. The generated experiment schema remains unchanged.

**Tech Stack:** Python 3.9+, dataclasses, existing `puzzletron_setup` model profiles, Questionary choice tuples, Pytest, Ruff.

## Global Constraints

- The setup path must remain configuration-only and must not import `modelopt.torch`, PyTorch, converted checkpoints, or model weights.
- vLLM counts are unique active configurations, including teacher configurations and deduplicating across layers.
- Replace-one counts exclude teacher/no-change configurations, are layer-specific, and show per-width plus total counts.
- Selected values are deduplicated before counting; hidden width controls scenario multiplicity and is not a subblock axis.
- Dense Qwen, MoE Qwen, and Nemotron hybrid layouts must use their exact physical layer patterns.
- Missing or unsupported layer topology raises a setup error; approximate counts are not allowed.

---

### Task 1: Dependency-Light Candidate Counting Engine

**Files:**
- Modify: `puzzletron_setup/profiles.py`
- Create: `tests/unit/torch/puzzletron/test_setup_candidate_counts.py`

**Interfaces:**
- Consumes: `InspectedModel.config`, `InspectedModel.inventory`, and the selected pruning-axis mapping created by `_ask_pruning`.
- Produces:
  - `CandidateCounts` with `vllm_subblock`, `vllm_block`, `replacement_subblock_per_width`, `replacement_block_per_width`, and `width_count`.
  - `count_candidate_options(config: Mapping[str, Any], inventory: ModelInventory, axes: Mapping[str, Any]) -> CandidateCounts`.

- [ ] **Step 1: Write failing dense-Qwen tests**

Create `tests/unit/torch/puzzletron/test_setup_candidate_counts.py` with a compact axis helper and the approved dense hybrid example:

```python
from puzzletron_setup.profiles import ModelInventory, count_candidate_options


def _axis(teacher, *values):
    return {
        "enabled": True,
        "teacher_value": teacher,
        "values": list(values),
    }


def test_counts_dense_qwen_hybrid_candidates_exactly():
    inventory = ModelInventory(
        family="qwen3_5",
        descriptor="qwen3_5",
        family_config="family.yaml",
        model_type="qwen3_5",
        architectures=(),
        multimodal=False,
        moe=False,
        num_layers=24,
        num_sublayers=48,
        layer_counts={"linear_attention": 18, "full_attention": 6},
        facts={},
        axes=(),
    )
    config = {
        "text_config": {
            "num_hidden_layers": 24,
            "layer_types": ["linear_attention"] * 18 + ["full_attention"] * 6,
        }
    }
    axes = {
        "hidden_width": _axis(1024, 1024, 768),
        "kv_groups": _axis(2, 2, 1),
        "q_heads_per_group": _axis(4, 4, 2),
        "ffn_intermediate": _axis(3584, 3584, 3072),
        "gdn_key_groups": _axis(16, 16, 8),
        "gdn_value_heads_per_group": _axis(1, 1),
        "gdn_key_head_dim": _axis(128, 128, 96),
        "gdn_value_head_dim": _axis(128, 128, 96),
    }

    counts = count_candidate_options(config, inventory, axes)

    assert counts.vllm_subblock == 14
    assert counts.vllm_block == 24
    assert counts.replacement_subblock_per_width == 168
    assert counts.replacement_block_per_width == 312
    assert counts.width_count == 2
    assert counts.replacement_subblock_total == 336
    assert counts.replacement_block_total == 624
```

Add a second test proving duplicate values do not inflate counts and a
teacher-only block yields one block configuration, its two active subblock
configurations, and zero replace-one solutions:

```python
def test_deduplicates_teacher_only_axis_selections():
    inventory = ModelInventory(
        family="qwen3_5",
        descriptor="qwen3_5",
        family_config="family.yaml",
        model_type="qwen3_5",
        architectures=(),
        multimodal=False,
        moe=False,
        num_layers=1,
        num_sublayers=2,
        layer_counts={"full_attention": 1},
        facts={},
        axes=(),
    )
    config = {
        "text_config": {
            "num_hidden_layers": 1,
            "layer_types": ["full_attention"],
        }
    }
    axes = {
        "hidden_width": _axis(1024, 1024, 1024),
        "kv_groups": _axis(2, 2, 2),
        "q_heads_per_group": _axis(4, 4),
        "ffn_intermediate": _axis(3584, 3584, 3584),
    }

    counts = count_candidate_options(config, inventory, axes)

    assert counts.vllm_subblock == 2
    assert counts.vllm_block == 1
    assert counts.replacement_subblock_per_width == 0
    assert counts.replacement_block_per_width == 0
    assert counts.width_count == 1
```

- [ ] **Step 2: Run the dense-Qwen tests and verify RED**

Run:

```bash
python -m pytest --noconftest -o addopts='' -q \
  tests/unit/torch/puzzletron/test_setup_candidate_counts.py
```

Expected: collection fails because `CandidateCounts` and `count_candidate_options` do not exist.

- [ ] **Step 3: Implement the generic counting representation**

In `puzzletron_setup/profiles.py`, add `import math` with the standard-library
imports, then add these public types and constants:

```python
@dataclass(frozen=True)
class CandidateCounts:
    vllm_subblock: int
    vllm_block: int
    replacement_subblock_per_width: int
    replacement_block_per_width: int
    width_count: int

    @property
    def replacement_subblock_total(self) -> int:
        return self.replacement_subblock_per_width * self.width_count

    @property
    def replacement_block_total(self) -> int:
        return self.replacement_block_per_width * self.width_count


_SUBBLOCK_AXES = {
    "attention": ("kv_groups", "q_heads_per_group"),
    "gdn": (
        "gdn_key_groups",
        "gdn_value_heads_per_group",
        "gdn_key_head_dim",
        "gdn_value_head_dim",
    ),
    "ffn": ("ffn_intermediate",),
    "moe": (
        "moe_experts",
        "moe_expert_intermediate",
        "moe_shared_expert_intermediate",
        "moe_top_k",
        "moe_latent_dim",
    ),
    "mamba": ("mamba_heads", "mamba_head_dim"),
}
```

Implement:

```python
def _axis_domain_size(axes, axis_id):
    axis = _mapping(axes.get(axis_id))
    if not axis or not bool(axis.get("enabled", True)):
        return 1
    values = tuple(dict.fromkeys(int(value) for value in axis.get("values") or ()))
    return len(values) or 1


def _qwen_block_families(config, *, moe):
    language = _language_config(config)
    layer_types = tuple(str(value) for value in language.get("layer_types") or ())
    num_layers = int(language.get("num_hidden_layers", 0))
    if not layer_types:
        interval = int(language.get("full_attention_interval", 0))
        if interval <= 0 or num_layers <= 0:
            raise SetupError("Qwen candidate counting requires layer_types or full_attention_interval.")
        layer_types = tuple(
            "full_attention" if (index + 1) % interval == 0 else "linear_attention"
            for index in range(num_layers)
        )
    if len(layer_types) != num_layers:
        raise SetupError("Qwen layer_types does not match num_hidden_layers.")
    feed_forward = "moe" if moe else "ffn"
    mapping = {
        "full_attention": ("attention", feed_forward),
        "linear_attention": ("gdn", feed_forward),
    }
    try:
        return tuple(mapping[layer_type] for layer_type in layer_types)
    except KeyError as error:
        raise SetupError(f"Unsupported Qwen layer type: {error.args[0]}") from error


```

Then implement `count_candidate_options` for dense and MoE Qwen:

```python
def count_candidate_options(
    config: Mapping[str, Any],
    inventory: ModelInventory,
    axes: Mapping[str, Any],
) -> CandidateCounts:
    """Count exact configuration-only vLLM and replace-one candidates."""

    if inventory.family == "qwen3_5":
        layer_families = _qwen_block_families(config, moe=inventory.moe)
    else:
        raise SetupError(
            f"Candidate counting is not implemented for {inventory.family}"
        )

    active_subblocks = {
        subblock for family in layer_families for subblock in family
    }
    domains = {
        subblock: math.prod(
            _axis_domain_size(axes, axis_id)
            for axis_id in _SUBBLOCK_AXES[subblock]
        )
        for subblock in active_subblocks
    }
    unique_block_families = tuple(dict.fromkeys(layer_families))
    vllm_subblock = sum(domains[subblock] for subblock in active_subblocks)
    vllm_block = sum(
        math.prod(domains[subblock] for subblock in family)
        for family in unique_block_families
    )
    replacement_subblock = sum(
        sum(domains[subblock] - 1 for subblock in family)
        for family in layer_families
    )
    replacement_block = sum(
        math.prod(domains[subblock] for subblock in family) - 1
        for family in layer_families
    )
    hidden = _mapping(axes.get("hidden_width"))
    widths = tuple(
        dict.fromkeys(int(value) for value in hidden.get("values") or ())
    )
    return CandidateCounts(
        vllm_subblock=vllm_subblock,
        vllm_block=vllm_block,
        replacement_subblock_per_width=replacement_subblock,
        replacement_block_per_width=replacement_block,
        width_count=len(widths) or 1,
    )
```

Export both public symbols through `profiles.__all__`.

- [ ] **Step 4: Run dense-Qwen tests and verify GREEN**

Run the Task 1 Step 2 command.

Expected: all tests pass.

- [ ] **Step 5: Add MoE-Qwen regression coverage and failing Nemotron coverage**

Extend the same test file:

```python
def test_counts_qwen_moe_domain_once_for_vllm_and_per_layer_for_scoring():
    # Two full-attention layers and one linear-attention layer.
    # attention=4, gdn=2, moe=4, so:
    # vLLM subblock=4+2+4=10; block=4*4+2*4=24.
    # replace subblock=(2*((4-1)+(4-1)))+(1*((2-1)+(4-1)))=16.
    # replace block=(2*(16-1))+(1*(8-1))=37.
    inventory = ModelInventory(
        family="qwen3_5",
        descriptor="qwen3_5_moe",
        family_config="family.yaml",
        model_type="qwen3_5_moe",
        architectures=(),
        multimodal=False,
        moe=True,
        num_layers=3,
        num_sublayers=6,
        layer_counts={"full_attention": 2, "linear_attention": 1},
        facts={},
        axes=(),
    )
    config = {
        "text_config": {
            "num_hidden_layers": 3,
            "layer_types": [
                "full_attention",
                "full_attention",
                "linear_attention",
            ],
        }
    }
    axes = {
        "hidden_width": _axis(2048, 2048),
        "kv_groups": _axis(2, 2, 1),
        "q_heads_per_group": _axis(8, 8, 4),
        "gdn_key_groups": _axis(16, 16, 8),
        "gdn_value_heads_per_group": _axis(2, 2),
        "gdn_key_head_dim": _axis(128, 128),
        "gdn_value_head_dim": _axis(128, 128),
        "moe_experts": _axis(256, 256, 128),
        "moe_expert_intermediate": _axis(512, 512, 256),
        "moe_shared_expert_intermediate": _axis(512, 512),
        "moe_top_k": _axis(8, 8),
    }

    counts = count_candidate_options(config, inventory, axes)

    assert counts.vllm_subblock == 10
    assert counts.vllm_block == 24
    assert counts.replacement_subblock_per_width == 16
    assert counts.replacement_block_per_width == 37


def test_counts_nemotron_mutually_exclusive_hybrid_pattern():
    # Pattern "*ME-" with attention=4, mamba=4, moe=4, ffn=2.
    # Block and subblock counts are identical because every layer has one
    # active subblock family.
    inventory = ModelInventory(
        family="nemotron3",
        descriptor="nemotron_h",
        family_config="family.yaml",
        model_type="nemotron_h",
        architectures=(),
        multimodal=False,
        moe=True,
        num_layers=4,
        num_sublayers=4,
        layer_counts={"attention": 1, "mamba": 1, "moe": 1, "ffn": 1},
        facts={},
        axes=(),
    )
    config = {
        "num_hidden_layers": 4,
        "hybrid_override_pattern": "*ME-",
    }
    axes = {
        "hidden_width": _axis(2688, 2688),
        "kv_groups": _axis(2, 2, 1),
        "q_heads_per_group": _axis(16, 16, 8),
        "mamba_heads": _axis(64, 64, 48),
        "mamba_head_dim": _axis(64, 64, 48),
        "moe_experts": _axis(128, 128, 96),
        "moe_expert_intermediate": _axis(1856, 1856, 1600),
        "moe_shared_expert_intermediate": _axis(3712, 3712),
        "moe_top_k": _axis(6, 6),
        "ffn_intermediate": _axis(1856, 1856, 1600),
    }

    counts = count_candidate_options(config, inventory, axes)

    assert counts.vllm_subblock == 14
    assert counts.vllm_block == 14
    assert counts.replacement_subblock_per_width == 10
    assert counts.replacement_block_per_width == 10
```

- [ ] **Step 6: Run the new tests and verify Nemotron RED**

Run the Task 1 Step 2 command.

Expected: MoE Qwen passes through the shared Qwen layout; Nemotron fails with
`Candidate counting is not implemented for nemotron3`.

- [ ] **Step 7: Implement Nemotron layout support and verify GREEN**

Add:

```python
def _nemotron_block_families(config):
    language = _language_config(config)
    pattern = str(language.get("hybrid_override_pattern") or "")
    mapping = {
        "*": ("attention",),
        "M": ("mamba",),
        "E": ("moe",),
        "-": ("ffn",),
    }
    if len(pattern) != int(language.get("num_hidden_layers", 0)):
        raise SetupError("Nemotron hybrid pattern does not match num_hidden_layers.")
    try:
        return tuple(mapping[character] for character in pattern)
    except KeyError as error:
        raise SetupError(f"Unsupported Nemotron hybrid marker: {error.args[0]}") from error
```

Dispatch `inventory.family == "nemotron3"` to this resolver, run the Task 1
Step 2 command, and expect all candidate-count tests to pass.

- [ ] **Step 8: Commit Task 1**

```bash
git add puzzletron_setup/profiles.py \
  tests/unit/torch/puzzletron/test_setup_candidate_counts.py
git commit -s -S -m "Add Puzzletron candidate count model"
```

---

### Task 2: Display Counts in Both Wizard Granularity Questions

**Files:**
- Modify: `puzzletron_setup/wizard.py`
- Modify: `tests/unit/torch/puzzletron/test_setup_candidate_counts.py`

**Interfaces:**
- Consumes: `count_candidate_options(model.config, model.inventory, axes)`.
- Produces:
  - `_vllm_granularity_choices(counts: CandidateCounts) -> list[tuple[str, str]]`
  - `_replacement_granularity_choices(counts: CandidateCounts) -> list[tuple[str, str]]`
  - `_ask_runtime(prompts, state, model)` using the same derived counts as `_ask_pruning`.

- [ ] **Step 1: Write failing choice-formatting tests**

Add:

```python
from puzzletron_setup.profiles import CandidateCounts
from puzzletron_setup.wizard import (
    _replacement_granularity_choices,
    _vllm_granularity_choices,
)


def test_formats_exact_candidate_counts_in_granularity_choices():
    counts = CandidateCounts(
        vllm_subblock=14,
        vllm_block=24,
        replacement_subblock_per_width=168,
        replacement_block_per_width=312,
        width_count=2,
    )

    assert _vllm_granularity_choices(counts) == [
        ("Sublayer — 14 unique configurations", "subblock"),
        ("Whole block — 24 unique configurations", "block"),
    ]
    assert _replacement_granularity_choices(counts) == [
        ("Subblock — 168 solutions/width, 336 total across 2 widths", "subblock"),
        ("Whole block — 312 solutions/width, 624 total across 2 widths", "block"),
    ]
```

Add a one-width test expecting:

```python
[
    ("Subblock — 168 solutions", "subblock"),
    ("Whole block — 312 solutions", "block"),
]
```

- [ ] **Step 2: Run formatting tests and verify RED**

Run:

```bash
python -m pytest --noconftest -o addopts='' -q \
  tests/unit/torch/puzzletron/test_setup_candidate_counts.py \
  -k 'formats'
```

Expected: import failure because the formatting helpers do not exist.

- [ ] **Step 3: Implement formatting helpers and wire `_ask_pruning`**

Import `CandidateCounts` and `count_candidate_options` at module scope.

Implement:

```python
def _vllm_granularity_choices(counts: CandidateCounts) -> list[tuple[str, str]]:
    return [
        (f"Sublayer — {counts.vllm_subblock} unique configurations", "subblock"),
        (f"Whole block — {counts.vllm_block} unique configurations", "block"),
    ]


def _replacement_count_label(label, per_width, total, width_count):
    if width_count == 1:
        return f"{label} — {total} solutions"
    return (
        f"{label} — {per_width} solutions/width, "
        f"{total} total across {width_count} widths"
    )


def _replacement_granularity_choices(counts: CandidateCounts) -> list[tuple[str, str]]:
    return [
        (
            _replacement_count_label(
                "Subblock",
                counts.replacement_subblock_per_width,
                counts.replacement_subblock_total,
                counts.width_count,
            ),
            "subblock",
        ),
        (
            _replacement_count_label(
                "Whole block",
                counts.replacement_block_per_width,
                counts.replacement_block_total,
                counts.width_count,
            ),
            "block",
        ),
    ]
```

In `_ask_pruning`, calculate counts immediately after all axis checkboxes and pass `_replacement_granularity_choices(counts)` to the existing prompt. Keep the default value `subblock`.

- [ ] **Step 4: Wire `_ask_runtime` without persisting derived counts**

Change:

```python
def _ask_runtime(
    prompts: PromptSession,
    state: AnswerState,
    model: InspectedModel,
) -> None:
```

Read `axes = state.section("pruning")["axes"]`, calculate the counts from the model, and use `_vllm_granularity_choices(counts)` when vLLM is enabled. Update `run_wizard` to call `_ask_runtime(prompts, state, model)`.

Derived counts are deliberately not saved in `answers.yaml`; resuming recomputes them from the authoritative model config and selected axes.

- [ ] **Step 5: Run formatting and setup tests and verify GREEN**

Run:

```bash
python -m pytest --noconftest -o addopts='' -q \
  tests/unit/torch/puzzletron/test_setup_candidate_counts.py \
  tests/unit/torch/puzzletron/test_setup_bundle.py
```

Expected: all tests pass.

- [ ] **Step 6: Commit Task 2**

```bash
git add puzzletron_setup/wizard.py \
  tests/unit/torch/puzzletron/test_setup_candidate_counts.py
git commit -s -S -m "Show exact candidate counts in Puzzletron setup"
```

---

### Task 3: Verify the Real Qwen Wizard Inputs

**Files:**
- Verify: `puzzletron_setup/profiles.py`
- Verify: `puzzletron_setup/wizard.py`
- Verify: `tests/unit/torch/puzzletron/test_setup_candidate_counts.py`

**Interfaces:**
- Consumes: the existing `../puzzle_runs/qwen/answers.yaml` model config and selected pruning axes.
- Produces: verification evidence only; no campaign artifacts or jobs.

- [ ] **Step 1: Run the focused test suite**

```bash
python -m pytest --noconftest -o addopts='' -q \
  tests/unit/torch/puzzletron/test_setup_candidate_counts.py \
  tests/unit/torch/puzzletron/test_setup_bundle.py
```

Expected: all tests pass.

- [ ] **Step 2: Verify style and syntax**

```bash
python -m ruff check \
  puzzletron_setup/profiles.py \
  puzzletron_setup/wizard.py \
  tests/unit/torch/puzzletron/test_setup_candidate_counts.py \
  tests/unit/torch/puzzletron/test_setup_bundle.py
python -m ruff format --check \
  puzzletron_setup/profiles.py \
  puzzletron_setup/wizard.py \
  tests/unit/torch/puzzletron/test_setup_candidate_counts.py \
  tests/unit/torch/puzzletron/test_setup_bundle.py
python -m compileall -q puzzletron_setup
```

Expected: Ruff reports no errors and no files requiring formatting; compileall exits zero.

- [ ] **Step 3: Run a dependency-light real-answer probe**

Run:

```bash
python -c '
from pathlib import Path
import yaml
from puzzletron_setup.profiles import (
    CandidateCounts,
    count_candidate_options,
    resolve_profile,
)
from puzzletron_setup.wizard import (
    _replacement_granularity_choices,
    _vllm_granularity_choices,
)

state = yaml.safe_load(Path("../puzzle_runs/qwen/answers.yaml").read_text())
config = state["model"]["config"]
inventory = resolve_profile(config).inventory(config)
axes = state["answers"]["pruning"]["axes"]
counts = count_candidate_options(config, inventory, axes)
assert counts == CandidateCounts(
    vllm_subblock=14,
    vllm_block=24,
    replacement_subblock_per_width=168,
    replacement_block_per_width=312,
    width_count=2,
)
print(_vllm_granularity_choices(counts))
print(_replacement_granularity_choices(counts))
'
```

Expected: the assertions pass and both exact wizard choice lists are printed.

- [ ] **Step 4: Check the final diff**

```bash
git diff --check
git status --short
```

Expected: no whitespace errors; only intended files from this feature are newly changed beyond the repository's pre-existing dirty worktree.
