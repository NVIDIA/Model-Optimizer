---
name: day0-release
description: Deterministic end-to-end driver for day-0 quantized-checkpoint releases — chains PTQ → deployment → evaluation → comparison with enforced gates between stages, and returns a publish decision (ACCEPT / REGRESSION / ANOMALOUS / INFEASIBLE). Use when the user asks to "release a model at day-0", "quantize and validate model X is within N% of baseline and tell me if it's publishable", or "run the full day-0 workflow". Do NOT use for single-stage requests — quantizing only (use ptq), serving only (use deployment), evaluating only (use evaluation), or comparing two existing runs (use compare-results).
license: Apache-2.0
---

# Day-0 Release

Drive a model from a pretrained checkpoint to a publish decision for a quantized
checkpoint, in a fixed sequence with a gate after every stage. This skill is a
**conductor**: it sequences the existing domain skills and enforces the gates —
it does not re-implement quantization, serving, evaluation, or comparison.

**Goal (the default day-0 criterion):** a quantized checkpoint smaller than the
source, with accuracy drop within the threshold (default <1%) on the standard
benchmark set versus the matching baseline, plus a publish recommendation.

## When to use

Use only for the full goal-driven release. For a single stage, route to the
domain skill directly: quantize → **ptq**, serve → **deployment**, evaluate →
**evaluation**, compare two existing runs → **compare-results**.

## Inputs

Resolve these before starting (ask the user for anything missing):

- **Model** — HF handle or checkpoint path.
- **Recipe / qformat** — e.g. `nvfp4`, `fp8`, or a recipe path. One candidate for v1.
- **Cluster / launcher** — from `clusters.yaml` (see `skills/common/environment-setup.md`).
- **Eval set** — defaults to the AA suite (`evaluation/recipes/tasks/aa/`).
- **Threshold** — max accuracy drop; default `0.01` (1%).

## The chain

```text
setup ─▶ PTQ ─▶ deploy ─▶ baseline-eval ─▶ quantized-eval ─▶ compare ─▶ closeout
          │       │            │                │              │
       gate_ptq  health     gate_run         gate_run      gate_compare
```

Run each stage by invoking the domain skill, then run its gate before
proceeding. **Do not advance past a failed gate.** Copy this checklist and track
progress:

```text
- [ ] Step 0: Resolve inputs; confirm threshold and eval set
- [ ] Step 1: Setup gate — creds present, cluster reachable
- [ ] Step 2: PTQ (ptq skill) → gate_ptq.py
- [ ] Step 3: Deploy (deployment skill) → health gate
- [ ] Step 4: Baseline eval (evaluation skill) → gate_run.py   [skip if cached, see below]
- [ ] Step 5: Quantized eval (evaluation skill) → gate_run.py
- [ ] Step 6: Compare (compare-results skill) → gate_compare.py → decision
- [ ] Step 7: Closeout — report + publish recommendation
```

### Step 1 — Setup gate

Confirm credentials (`skills/common/credentials.md`) and cluster reachability
(`skills/common/remote-execution.md`). If either fails, stop with
`SYSTEMIC` — do not start PTQ.

### Step 2 — PTQ

Invoke the **ptq** skill to produce the quantized checkpoint. Then gate:

```bash
# The ptq skill's post-PTQ validation produces a validation-summary JSON (size
# ratio + layer-precision counts + metadata diffs; see
# ptq/references/checkpoint-validation.md). v1 gates on that summary:
python .claude/skills/day0-release/scripts/gate_ptq.py --summary <validation-summary.json>
#   add `--recipe <qformat>` to override the recipe recorded in the summary
```

`gate_ptq.py` returns JSON `{pass, failure_class, detail}`. On `pass: false`,
branch on `failure_class` (see **Triage** below). Do not deploy an
unvalidated checkpoint.

### Step 3 — Deploy

Invoke the **deployment** skill to serve the checkpoint. Gate on the skill's own
health check: the endpoint must be up and return one successful generation. On
failure, triage (PATCH serving flags / image / TP, or POINT_INFEASIBLE).

### Step 4 — Baseline eval

The baseline is the **source** (pre-quantization) model on the same task set and
sampling params. **Look it up first** — if a matching baseline run already
exists in MLflow (same model, task set, sampling params), reuse it and skip this
stage. Otherwise run it via the **evaluation** skill. Gate with `gate_run.py`.

### Step 5 — Quantized eval

Invoke the **evaluation** skill on the deployed quantized checkpoint, matching
the baseline's task set and sampling params. Gate:

```bash
python .claude/skills/day0-release/scripts/gate_run.py --run <run-summary.json>
```

A `pass: false` here means the run is incomplete or invalid (judge/parse error,
dropped samples) — do **not** compare scores from it.

### Step 6 — Compare

Invoke the **compare-results** skill to produce per-task deltas, then gate:

```bash
python .claude/skills/day0-release/scripts/gate_compare.py \
    --baseline <baseline_scores.json> --candidate <candidate_scores.json> \
    --threshold 0.01
```

Decision from `gate_compare.py`:

- **ACCEPT** — every task within threshold → go to Step 7.
- **REGRESSION** — one or more tasks exceed threshold. **v1 stops here and
  reports** which tasks regressed by how much. (Picking the next recipe and
  re-running is deferred — see Scope.)
- **ANOMALOUS** — scores present but implausible (e.g. baseline lower than
  candidate by a large margin, or a task score outside its valid range) →
  surface to the user.

### Step 7 — Closeout

Report the decision with: source vs output size + ratio, per-task baseline /
candidate / delta / within-threshold, MLflow run IDs, and a publish
recommendation (publish / do-not-publish / needs-human). Archive artifacts to
the workspace.

## Triage (gate failure → decision)

Map a gate's `failure_class` to the next action:

| `failure_class` | Action |
| --- | --- |
| `INFRA_TRANSIENT` | Retry the stage once; if it recurs, `SYSTEMIC`. |
| `MODEL_UNSUPPORTED` | PATCH: fix the recipe pattern / add model support (ptq skill owns the patch loop), then retry. If unpatchable, `POINT_INFEASIBLE`. |
| `QUANT_COVERAGE_FAILURE` | PATCH: fix the recipe wildcard so intended layers are covered; re-run PTQ. |
| `DEPLOYMENT_HEALTH_FAILED` | PATCH: serving flags / image / TP; if the checkpoint can't serve, `POINT_INFEASIBLE`. |
| `EVAL_JUDGE_FAILED` | Usually transient (auth / rate limit) — wait and retry. |
| `SAMPLE_ACCOUNTING_FAILED` | Investigate dropped/failed samples before trusting scores. |
| `USER_CONFIG_ERROR` | Stop and ask the user. |
| `UNKNOWN` | Stop and surface to the user (`NEEDS_HUMAN`). |

`SYSTEMIC` (cluster down, dataset unavailable) aborts the whole run.
`POINT_INFEASIBLE` means this (model, recipe) can't work as configured.

## Output

Return a decision, not a raw artifact:

- `ACCEPT` + report + publish recommendation
- `REGRESSION` + which tasks failed the threshold and by how much
- `ANOMALOUS` / `INFEASIBLE` / `NEEDS_HUMAN` + reason
- Always: workspace path + MLflow run IDs for traceability

## Scope (v1)

In v1: the linear chain + gates + report. On `REGRESSION`, v1 reports and stops.
Deferred to a follow-up: the evaluator-optimizer recipe loop (compare → pick the
next recipe → re-run PTQ), which needs the bigpareto integration and a shared
config/result schema.
