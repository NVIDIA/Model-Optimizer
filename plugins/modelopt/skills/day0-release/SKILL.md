---
name: day0-release
description: Deterministic end-to-end driver for day-0 quantized-checkpoint releases — chains PTQ → evaluation → comparison with enforced gates between stages (the evaluation stage deploys the checkpoint itself), and returns a publish decision (ACCEPT / REGRESSION / ANOMALOUS / INFEASIBLE). Use when the user asks to "release a model at day-0", "quantize and validate model X is within N% of baseline and tell me if it's publishable", or "run the full day-0 workflow". Do NOT use for single-stage requests — quantizing only (use ptq), serving only (use deployment), evaluating only (use evaluation), or comparing two existing runs (use compare-results).
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
- **Cluster / launcher** — from `clusters.yaml` (see the common skill's
  `environment-setup.md`).
- **Eval set** — defaults to the evaluation skill's AA suite
  (`recipes/tasks/aa/`).
- **Threshold** — max accuracy drop; default `0.01` (1%).

## The chain

```text
setup ─▶ PTQ ─▶ canary ─▶ baseline-eval ─▶ quantized-eval ─▶ compare ─▶ verbosity ─▶ closeout
          │        │           │                │              │           │
       gate_ptq  /health    gate_run         gate_run    gate_compare  gate_verbosity
                 + 1 gen
```

The **evaluation** skill deploys the model it evaluates (it stands up its own
endpoint per run), so there is no separate deploy *stage* — a serving failure
during evaluation surfaces through the eval gate (`DEPLOYMENT_HEALTH_FAILED`) and
triages to the **deployment** skill (see Step 4). The Step 2b **canary** is not
that: it runs *before* any evaluation precisely so an unservable checkpoint is
caught in ~15 min rather than after a multi-hour eval.

Accuracy (Step 5) and verbosity (Step 5b) are **independent gates**; closeout
requires both.

Run each stage by invoking the domain skill, then run its gate before
proceeding. **Do not advance past a failed gate.** Copy this checklist and track
progress:

```text
- [ ] Step 0: Resolve inputs; confirm threshold and eval set
- [ ] Step 1: Setup gate — creds present, cluster reachable
- [ ] Step 2: PTQ (ptq skill) → gate_ptq.py
- [ ] Step 2b: Serving canary — /health + one generation (deployment skill)
- [ ] Step 3: Baseline eval (evaluation skill, deploys source) → gate_run.py
- [ ] Step 4: Quantized eval (evaluation skill, deploys candidate) → gate_run.py
- [ ] Step 5: Compare (compare-results skill) → external sanity → gate_compare.py → decision
- [ ] Step 5b: Verbosity gate → gate_verbosity.py
- [ ] Step 6: Closeout — report + publish recommendation
```

### Step 1 — Setup gate

Use the common skill's `credentials.md` and `remote-execution.md` to confirm
credentials and cluster reachability. If either fails, stop with
`SYSTEMIC` — do not start PTQ.

### Step 2 — PTQ

Invoke the **ptq** skill to produce the quantized checkpoint. Then gate:

```bash
# The ptq skill's post-PTQ validation produces a validation-summary JSON (size
# ratio + layer-precision counts + metadata diffs; see the ptq skill's
# references/checkpoint-validation.md). v1 gates on that summary:
python "$SKILL_DIR/scripts/gate_ptq.py" --summary <validation-summary.json>
#   add `--recipe <qformat>` to override the recipe recorded in the summary
```

`gate_ptq.py` returns JSON `{pass, failure_class, detail}`. On `pass: false`,
branch on `failure_class` (see **Triage** below). Do not evaluate an
unvalidated checkpoint.

#### Step 2b — serving canary (MANDATORY before Step 3)

The canary itself is **already specified** by the ptq skill: see
`ptq/references/checkpoint-validation.md` (required gate, canary query and the
`Serving canary` row of its report table) and `ptq/SKILL.md`. Run it there rather
than re-deriving it here — `gate_ptq.py` checks size, coverage and metadata, not
whether the checkpoint *loads*, and skipping the canary has cost a full baseline
eval against a checkpoint the serving stack could never load.

On failure use `failure_class: CHECKPOINT_NOT_SERVABLE` and drop to the
**deployment** skill; do not proceed to Step 3.

**What that spec does not cover: writing a canary that cannot lie.** Both of these
produced a wrong verdict on a large MoE, in opposite directions:

- **Poll ceiling > load time**, with headroom. A 50 min poll against a 51 min load
  reported failure for a checkpoint that serves fine. Large MoE loads are
  CPU-bound fp8 dequant (~50 min); 0% GPU during *load* is normal.
- **Print an explicit `RESULT:` on every path** and exit non-zero on failure. A
  poll loop that falls through to the generation test exits 0 and reads as PASS.
- Server log on shared storage, not node-local `/tmp`.
- Canary the **as-exported** artifact, not a copy you modified to make it work.

### Step 3 — Baseline eval

The baseline is the **source** (pre-quantization) model on the same task set and
sampling params. Always run a fresh baseline via the **evaluation** skill,
which deploys the source model itself. Gate with `gate_run.py`.

### Step 4 — Quantized eval

Invoke the **evaluation** skill on the quantized checkpoint, matching the
baseline's task set and sampling params. The evaluation skill stands up the
serving endpoint itself (it builds the `deployment.command`, e.g. a
`vllm serve …`), so a serving failure surfaces here as a failed `gate_run.py`
with `DEPLOYMENT_HEALTH_FAILED`. When that happens, **drop to the deployment
skill** to reproduce and debug serving in isolation (serve the checkpoint
standalone, confirm `/health` + one generation, iterate on flags / TP / image /
env vars) rather than burning full eval cycles on a broken endpoint — then carry
the working command back into NEL's `deployment.command` and resume the eval. If
the checkpoint genuinely can't serve, `POINT_INFEASIBLE`.

**Before submitting: assert baseline/candidate config parity.** The candidate
config must differ from the baseline's in nothing but checkpoint path and served
model name. Diff mechanically — an eyeball pass misses this:

```bash
diff <(grep -vE 'checkpoint_path|served_model_name' baseline.yaml) \
     <(grep -vE 'checkpoint_path|served_model_name' candidate.yaml)
```

Any other difference biases the comparison and invalidates the gate: a mismatched
`parallelism` between the two sides was worth ~2 pp, enough to invert the sign of
the delta. If the model card splits sampling params per scenario, apply the same
split to both sides. Never set an unbounded `request_timeout` (`1e9`) — it turns a
transient stall into a job that holds its GPUs until the wall clock kills it.

Gate:

```bash
python "$SKILL_DIR/scripts/gate_run.py" --run <run-summary.json>
```

A `pass: false` here means the run is incomplete or invalid (judge/parse error,
dropped samples) — do **not** compare scores from it.

### Step 5 — Compare

Invoke the **compare-results** skill. It must perform the shared external
baseline sanity check before the candidate-delta gate. A failed check is
`ANOMALOUS` with failure class `EXTERNAL_BASELINE_MISMATCH`: investigate and
rerun the baseline. If no credible comparable external score exists, record the
baseline as externally unverified and continue using the validated measured
baseline.

#### Statistical power — check BEFORE trusting any per-task verdict

A task whose measurement noise rivals the threshold cannot decide a gate. Confirm
each task's repeat count gives a standard error below the threshold; otherwise
mark it `INDETERMINATE` rather than reporting a pass/fail.

For example, on a 1 % gate on DeepSeek-V4-Pro, both tasks that originally failed passed once measured properly:

| task | runs pooled | drop | verdict |
| --- | --- | --- | --- |
| SciCode | 1 | 2.96 pp | REGRESSION |
| SciCode | 8 | **-0.96 pp** | PASS |
| IFBench | 5 | 2.73 pp | REGRESSION |
| IFBench | 16 | **0.63 pp** | PASS |

Add precision by submitting the benchmark **more times**, not by raising
`num_repeats` within a run — see `recipes/tasks/aa/scicode.md` for why.

**Re-running does not guarantee fresh samples.** With a warm NEL response cache a "re-run" can
replay cached responses — two runs came back bit-identical to 16 digits. Confirm the score
actually moved before counting a run as an independent repeat.

After recording the external status, produce per-task deltas and run:

```bash
python "$SKILL_DIR/scripts/gate_compare.py" \
    --baseline <baseline_scores.json> --candidate <candidate_scores.json> \
    --threshold 0.01
```

The threshold is a fraction of each task's score scale. Most AA tasks report
0-100, but some (e.g. `tau2_bench_telecom` `Result`) report 0-1; the gate infers
each task's scale (0-1 if both scores are within [0, 1], else 0-100) and
normalizes the drop accordingly, so `--threshold 0.01` means "≤1 pt on a 0-100
task / ≤0.01 on a 0-1 task" uniformly. Pass `--scales '{"task": max}'` to
override inference if a task's scores happen to fall in an ambiguous range.

`gate_compare.py` checks only the candidate delta; it cannot override a failed
external baseline check. Combined decision:

- **ACCEPT (accuracy only)** — no external check failed and every task is within
  the candidate threshold → continue to **Step 5b**. This verdict covers accuracy
  alone; advance to Step 6 only once the verbosity gate also passes. A missing
  comparable external score is not a failure; report it as externally unverified.
- **REGRESSION** — one or more tasks exceed threshold. **v1 stops here and
  reports** which tasks regressed by how much. (Picking the next recipe and
  re-running is deferred — see Scope.)
- **ANOMALOUS** — external baseline sanity failed, or scores are otherwise
  implausible (e.g. baseline lower than candidate by a large margin, or a task
  score is outside its valid range) → correct the baseline or surface it.

#### Step 5b — Verbosity gate (MANDATORY; independent of accuracy)

`gate_compare.py` does not measure verbosity, so stopping after Step 5 leaves a
hard gate unmeasured. Step 5's ACCEPT is accuracy-only — Step 6 requires both.

```bash
python "$SKILL_DIR/scripts/gate_verbosity.py" \
    --baseline <baseline_eval_root> --candidate <candidate_eval_root> \
    --glob 'eval_*' --exclude _high --threshold 0.05
```

Exit codes follow the other gates: `0` pass, `1` the gate failed, `2` it could not read its input.
A `2` means fix the invocation, not the checkpoint — check `--baseline`/`--candidate`, `--glob`
(`no_files_matched` names the pattern that missed), and `--exclude`. If the detail names
`collapsed_keys`, the artifact tree gave several tasks one name: point the gate at a tree whose
task dirs are `<harness>.<task>` rather than `<invocation_id>.<job_index>`, or re-sync so each
`artifacts/` carries its own `config.yml`.

Read `response_stats.avg_completion_tokens` from each task's
`artifacts/eval_factory_metrics.json`. Do **not** use the `reasoning.*` fields:
`reasoning.*_tokens` are always `0` (only the reasoning/content split is missing)
and the `*_words` siblings are a proxy that can disagree with the gate — one task
read +6.10% FAIL in words and +1.32% PASS in tokens.

The gate is two-sided; a large drop in output length is also a change.

Two filters are mandatory: same reasoning effort (pass `--exclude` explicitly — it is empty by
default, since which tier is canonical is run-specific) and complete runs only —
runs within 1% of the largest `successful_count` that **both** sides can match (exact equality
would make one dropped sample unmeasurable). A task with no such count on both sides is
`not_comparable`. When the matched count is below a run
one side has, the task carries `truncated_comparison`.

Tasks sharing no sample count are `not_comparable`, not a delta. Means under
~1000 tokens carry a `short_output_warning` — read absolute counts there.

### Step 6 — Closeout

Report the decision with: source vs output size + ratio, per-task baseline /
candidate / delta / within-threshold, **the verbosity verdict from Step 5b**,
external source and sanity status, MLflow run IDs, and a publish recommendation
(publish / do-not-publish). Archive artifacts to the workspace.

**Publish exactly what was evaluated.** Verify mechanically, not by path
convention: inode-compare a shard in the evaluated directory against the one being
published. A day-0 run leaves near-identical sibling exports
that differ by one calibration suffix — prefix rejected ones `REJECTED-` so the
artifact cannot be picked by autocomplete, and re-run the Step 2b canary against
the final path after any move.

## Triage (gate failure → decision)

Map a gate's `failure_class` to the next action:

| `failure_class` | Action |
| --- | --- |
| `INFRA_TRANSIENT` | Retry the stage once; if it recurs, `SYSTEMIC`. |
| `MODEL_UNSUPPORTED` | PATCH: fix the recipe pattern / add model support (ptq skill owns the patch loop), then retry. If unpatchable, `POINT_INFEASIBLE`. |
| `QUANT_COVERAGE_FAILURE` | PATCH: fix the recipe wildcard so intended layers are covered; re-run PTQ. |
| `SIZE_NOT_REDUCED` | The output is not smaller than the source. If the recipe cannot shrink the source (e.g. mxfp4 under nvfp4, or fp8 under fp8), record `source_precision` in the validation summary and re-run the gate — that states *why* the growth is expected. Otherwise treat it as a real compression failure: check that the recipe matched the intended parameter mass and that the exporter did not retain the original tensors. `accept_size_growth: true` waives it unconditionally (no growth bound) and is a last resort, not the fix — it records no reason, so prefer `source_precision` where the claim is checkable. |
| `CHECKPOINT_NOT_SERVABLE` | The Step 2b canary could not load/generate. Usually a tensor-naming or config-schema mismatch between the exporter and the serving stack, or missing/dangling auxiliary files (tokenizer). Fix the export; do not evaluate. |
| `VERBOSITY_EXCEEDED` | Re-check run hygiene first (mixed reasoning effort, partial runs, unequal sample counts) — that has explained every false positive so far. If the delta survives matched, complete runs, it is a real behavioural change; do not publish on accuracy alone. |
| `DEPLOYMENT_HEALTH_FAILED` | Drop to the **deployment** skill: reproduce serving standalone (`/health` + one generation), debug flags / image / TP / env, then carry the working command into NEL's `deployment.command` and retry the eval. If it can't serve, `POINT_INFEASIBLE`. |
| `EVAL_JUDGE_FAILED` | Usually transient (auth / rate limit) — wait and retry. |
| `SAMPLE_ACCOUNTING_FAILED` | Investigate dropped/failed samples before trusting scores. |
| `EXTERNAL_BASELINE_MISMATCH` | Investigate baseline configuration, correct it, rerun the baseline, and repeat external sanity before comparison. |
| `USER_CONFIG_ERROR` | Correct it from the request, workspace, or model/config metadata and retry; if irrecoverable, return `ANOMALOUS` with evidence. |
| `UNKNOWN` | Investigate with the owning domain skill; if unresolved, return `ANOMALOUS` with the evidence and next automated retry or patch action. |

`gate_ptq.py` also emits non-blocking `notes` (present on every result). Size growth is
**blocking by default**, waived to a note only when the validation summary's `source_precision`
is already at or below the recipe's target bits and the growth is within what that explains. Recording
`source_precision` is part of the ptq skill's validation table, so the waiver is reachable from the
normal pipeline. A BF16 source that failed to compress still fails, which is the case this check
exists for.

`SYSTEMIC` (cluster down, dataset unavailable) aborts the whole run.
`POINT_INFEASIBLE` means this (model, recipe) can't work as configured.

## Output

Return a decision, not a raw artifact:

- `ACCEPT` + report + publish recommendation
- `REGRESSION` + which tasks failed the threshold and by how much
- `ANOMALOUS` / `INFEASIBLE` + reason and next automated action
- Always: workspace path + MLflow run IDs for traceability

## Scope (v1)

In v1: the linear chain + gates + report. On `REGRESSION`, v1 reports and stops.
Deferred to a follow-up: the evaluator-optimizer recipe loop (compare → pick the
next recipe → re-run PTQ), which needs the bigpareto integration and a shared
config/result schema.
