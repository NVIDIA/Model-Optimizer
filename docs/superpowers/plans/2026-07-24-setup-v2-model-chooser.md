# Setup V2 Model Chooser Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Remove redundant campaign and model action menus and add a hierarchical chooser for the eleven explicitly supported Hugging Face models while preserving custom local paths and URLs.

**Architecture:** Define the supported-model catalog and a two-level selection flow in the v2 wizard. The top-level Model prompt selects Custom or a family; a family prompt selects a checkpoint. Keep source normalization, revision selection, model inspection, resume state, and custom input behavior unchanged.

**Tech Stack:** Python dataclasses, questionary, Puzzletron setup-v2 prompt/session APIs.

## Global Constraints

- The campaign directory remains the first prompt and immediately precedes model selection.
- Model groups are exactly `Nemotron 3`, `Qwen 3.5/3.6 Dense`, and `Qwen 3.5/3.6 MoE`.
- Supported choices use the eleven canonical Hugging Face URLs approved in the design.
- `Custom local path or Hugging Face model` remains available.
- Revision selection and immutable Hugging Face resolution remain unchanged.
- Existing `answers_v2.yaml` files remain resumable.
- Do not run the orchestrator.

---

### Task 1: Grouped model chooser

**Files:**
- Modify: `puzzletron_setup/v2/prompts.py`
- Modify: `puzzletron_setup/v2/session.py`
- Modify: `puzzletron_setup/v2/wizard.py`

**Interfaces:**
- Produces: `SUPPORTED_MODEL_GROUPS`, an ordered tuple of group labels and `(short_name, url)` entries.
- Produces: `_model_family_choices()` and `_model_choices_for_family()`.
- Preserves: `run_wizard_v2(...) -> Path`.

- [ ] **Step 1: Verify the current behavior fails the approved assertions**

Run:

```bash
rtk proxy python -c "from puzzletron_setup.v2 import wizard; assert wizard.SECTION_BUILDERS[0].__name__ == 'model_section'"
```

Expected: assertion failure because `campaign_section` is first.

Run:

```bash
rtk proxy python -c "from puzzletron_setup.v2 import wizard; assert hasattr(wizard, 'SUPPORTED_MODEL_GROUPS')"
```

Expected: assertion failure because the catalog is not defined.

- [ ] **Step 2: Define the exact supported-model catalog**

Add `SUPPORTED_MODEL_GROUPS` in `wizard.py` with:

```python
(
    ("Nemotron 3", (...three Nemotron models...)),
    ("Qwen 3.5/3.6 Dense", (...five dense Qwen models...)),
    ("Qwen 3.5/3.6 MoE", (...three MoE Qwen models...)),
)
```

Each entry contains its concise display name and exact canonical URL.

- [ ] **Step 3: Remove the redundant campaign and model action menus**

Delete `campaign_section` and remove it from `SECTION_BUILDERS`. Fresh and
resumed sessions begin section processing at `model_section`; campaign
directory creation remains in `_fresh_state`.

Do not call `_section_action()` from `model_section`.

- [ ] **Step 4: Route model selection through the hierarchical chooser**

Ask `model.source_family` with Custom first and the three family names after
it. A family selection opens a family-specific checkpoint prompt; Back returns
to the family menu. Custom invokes the existing free-form `Local model path or
Hugging Face URL` prompt. Continue through existing source normalization,
revision, and `inspect_model` logic.

- [ ] **Step 5: Verify the focused behavior**

Run:

```bash
rtk proxy python -c "from puzzletron_setup.v2 import wizard; assert wizard.SECTION_BUILDERS[0].__name__ == 'model_section'; assert [group for group, _ in wizard.SUPPORTED_MODEL_GROUPS] == ['Nemotron 3', 'Qwen 3.5/3.6 Dense', 'Qwen 3.5/3.6 MoE']; assert sum(len(models) for _, models in wizard.SUPPORTED_MODEL_GROUPS) == 11"
```

Expected: exit 0.

Run scripted probes that confirm the top level contains Custom followed by the
three families, each submenu contains only its approved models, and Back from
a submenu returns to the top level. Run `rtk proxy git diff --check`.

- [ ] **Step 6: Commit**

```bash
rtk proxy git add puzzletron_setup/v2/prompts.py puzzletron_setup/v2/session.py puzzletron_setup/v2/wizard.py
rtk proxy git commit -s -S -m "feat: add grouped setup model chooser"
```
