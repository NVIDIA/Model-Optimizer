# Setup V2 Model Chooser Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Remove the redundant campaign section menu and add a grouped chooser for the eleven explicitly supported Hugging Face models while preserving custom local paths and URLs.

**Architecture:** Extend the backend-neutral prompt item model with non-selectable separators, then define the supported-model catalog and selection flow in the v2 wizard. Keep source normalization, revision selection, model inspection, resume state, and custom input behavior unchanged.

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
- Produces: `PromptSeparator(title: str)`.
- Produces: `SUPPORTED_MODEL_GROUPS`, an ordered tuple of group labels and `(short_name, url)` entries.
- Produces: `_model_source_choices()`, returning separators, supported model choices, and the custom-source choice.
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

- [ ] **Step 2: Add backend-neutral prompt separators**

Add:

```python
@dataclass(frozen=True)
class PromptSeparator:
    title: str
```

Preserve `PromptSeparator` values in `WizardSession._choices`. Render them with
`questionary.Separator` in interactive select and checkbox prompts. Scripted
backends continue consuming values without rendering choices.

- [ ] **Step 3: Define the exact supported-model catalog**

Add `SUPPORTED_MODEL_GROUPS` in `wizard.py` with:

```python
(
    ("Nemotron 3", (...three Nemotron models...)),
    ("Qwen 3.5/3.6 Dense", (...five dense Qwen models...)),
    ("Qwen 3.5/3.6 MoE", (...three MoE Qwen models...)),
)
```

Each entry contains its concise display name and exact canonical URL.

- [ ] **Step 4: Remove the redundant campaign section**

Delete `campaign_section` and remove it from `SECTION_BUILDERS`. Fresh and
resumed sessions begin section processing at `model_section`; campaign
directory creation remains in `_fresh_state`.

- [ ] **Step 5: Route model customization through the grouped chooser**

When no explicit model default is accepted, ask `model.source_choice` using
`_model_source_choices()`. A supported selection records the URL and concise
name. The custom selection invokes the existing free-form
`Local model path or Hugging Face URL` prompt. Continue through existing source
normalization, revision, and `inspect_model` logic.

- [ ] **Step 6: Verify the focused behavior**

Run:

```bash
rtk proxy python -c "from puzzletron_setup.v2 import wizard; assert wizard.SECTION_BUILDERS[0].__name__ == 'model_section'; assert [group for group, _ in wizard.SUPPORTED_MODEL_GROUPS] == ['Nemotron 3', 'Qwen 3.5/3.6 Dense', 'Qwen 3.5/3.6 MoE']; assert sum(len(models) for _, models in wizard.SUPPORTED_MODEL_GROUPS) == 11"
```

Expected: exit 0.

Run a scripted prompt rendering probe that confirms all three separators,
eleven canonical URLs, and the custom option are present. Run
`rtk proxy git diff --check`.

- [ ] **Step 7: Commit**

```bash
rtk proxy git add puzzletron_setup/v2/prompts.py puzzletron_setup/v2/session.py puzzletron_setup/v2/wizard.py
rtk proxy git commit -s -S -m "feat: add grouped setup model chooser"
```
