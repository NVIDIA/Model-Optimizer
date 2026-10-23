---
name: specdec-release
description: Deterministic end-to-end driver that takes a target model and a speculative-decoding algorithm (DFlash, DFlash2; not yet EAGLE3, DSpark, Domino or LiLiCorr) to a drafter readiness verdict — chains draft training → export → serving smoke test → freeze → matched A|B benchmark with enforced gates between stages, resumes from a state file, and returns DRAFTER_READY / INFEASIBLE / ANOMALOUS. Use when the user asks to "train a drafter for model X and tell me if it's ready", "run the spec-dec release flow", "resume the spec-dec run", or to gate the artifacts of a draft-training pipeline run. Do NOT use for single-stage requests — configuring, debugging or triaging a draft-training run (use speculative-decoding), serving only (use deployment), or quantizing (use ptq).
license: Apache-2.0
---

# Spec-Dec Release

Drive a target model from no drafter to a **readiness verdict** for a trained drafter, in a
fixed sequence with a gate after every stage. This skill is a **conductor**: it submits the
existing launcher pipelines and enforces the gates. It does not re-implement training,
serving or benchmarking.

**Goal:** a drafter that trained cleanly, exported to fit its target, serves with speculation
active, and was measured validly against the target alone. `DRAFTER_READY` means ready for
**release review**, not publishable: the gates set no per-model performance bar, and AL,
speedup and TPS are evidence. The release criteria, drafter PTQ and publishing are later
stages and are not implemented here.

## When to use

Use only for the full goal-driven flow, or to gate a pipeline run's artifacts. For a single
stage, route to the domain skill: configure a new model, review or triage a training run →
**speculative-decoding**; serve → **deployment**; quantize → **ptq**.

## Inputs

Resolve these before starting (ask the user for anything missing):

- **Target** — HF handle or `/hf-local/...` path.
- **Algorithm** — one with launcher examples: `tools/launcher/examples/<Org>/<Model>/hf_online_<algo>.yaml`
  (training, export, smoke) and `specdec_ab_<algo>.yaml` (A|B bench). No training example →
  configure one with **speculative-decoding** first; a missing bench example is added in Step 4.
  Training must run through `common/specdec/dflash_online_training.sh`, the only script that
  writes `regression_check.json` (EAGLE3 and DSpark train with `train_eagle*.sh`), and the
  algorithm must serve on vLLM (Domino has no path; LiLiCorr serves only on SGLang). Refuse
  any other algorithm before submitting anything.
- **Cluster / launcher** — from `clusters.yaml` (see the common skill's
  `environment-setup.md`). It fixes the **job host**, which holds the launcher's `<job_dir>`:
  the login node with `$SLURM_JOB_DIR`, or this machine with `tools/launcher/local_experiments`
  for local Docker (submit with `hf_local=<dir>`).
- **Budget** — `dev` (the example as committed) or `full` (the overrides its header lists).
  Never invent training knobs.
- **`existing_drafter`** (optional) — an exported drafter to freeze and benchmark instead of
  training one.
- **`approve_submissions`** — `each` (default): before every launcher submission, retries
  included, show the resolved command, job count, node-hours and reason, and wait for an
  OK. `none`: submit within the retry limits below. Either way, log each submission.
- **Run dir** — the working directory, unless the user names another. Don't create a
  subdirectory for the run.

## The chain

```text
setup ─▶ train pipeline (data → train+export → smoke) ─▶ freeze ─▶ A|B bench ─▶ closeout
         │                                               │         │
         gate_train + gate_smoke                         │         gate_bench + gate_correctness (report)
                                                         freeze_drafter
```

Training, export and the smoke test are one launcher experiment, so the smoke test serves the
unfrozen export; the bench serves the frozen copy, which freeze proves byte-identical to the
export at freeze time. Bench in a second experiment, only after both training gates pass.

Run each stage, then its gate. **Do not advance past a failed gate.** Copy this checklist and
track progress:

```text
- [ ] Step 0: Resolve inputs; create the run dir, specdec_state.json, work_log.md
- [ ] Step 1: Setup gate — creds present, cluster reachable, job host python3 ≥ 3.8
- [ ] Step 2: Train pipeline → gate_train.py → gate_smoke.py
- [ ] Step 3: Freeze → freeze_drafter.py
- [ ] Step 4: A|B bench → gate_bench.py → gate_correctness.py (report)
- [ ] Step 5: Closeout — SPECDEC_REPORT.md + readiness verdict
```

### Running a gate

Gates and `freeze_drafter.py` print JSON `{pass, failure_class, detail, ...}` and exit `0`
pass, `1` failed, `2` bad input (`USER_CONFIG_ERROR`: fix the invocation or the recipe value it
names, not the drafter). Run
them on the job host so artifacts are read in place:

```bash
# local Docker
python "$SKILL_DIR/scripts/<gate>.py" <args> > "$RUN_DIR/gates/<stage>.json"
# remote (scripts synced in Step 1). Every Bash call is a new shell, so load the cluster
# in each one, on resume too:
source "<common-skill-dir>/remote_exec.sh"
remote_load_cluster <inputs.cluster> && remote_check_ssh
remote_run "python3 specdec-gates/<gate>.py <args>" > "$RUN_DIR/gates/<stage>.json"
```

If `remote_load_cluster` or `remote_check_ssh` fails, stop with `SYSTEMIC`. Don't fall back to
raw `ssh`.

Trust the exit code first: any exit other than `0`/`1`/`2`, or output that is not one JSON
object (`remote_run` also captures the remote stderr), means the script crashed — re-run it;
never parse a verdict out of it.

**Never read a verdict from a log or a job state** — without `REGRESSION_STRICT=1`, a
failed regression check still ends `COMPLETED`.

**Paths.** Arguments that name a file the gate reads take job-host paths: the job's
`/scratchspace/<x>` is `<job_dir>/cicd/<exp_id>/<x>`, and `/cicd/<x>` is `<job_dir>/cicd/<x>`.
`--draft`/`--target` are matched as text against what the job recorded, so pass the exact
string it was submitted with (e.g. `/hf-local/Qwen/Qwen3-8B`, not `Qwen/Qwen3-8B`).

### Step 0 — Run dir and state

Create `specdec_state.json` (schema under **State and resume**), `work_log.md` and `gates/`
in the run dir. If the state file exists, this is a **resume**: follow its rules.

### Step 1 — Setup gate

Use the common skill's `credentials.md` and `remote-execution.md` to confirm credentials and
cluster reachability. On a remote job host, `remote_sync_to "$SKILL_DIR/scripts" specdec-gates`
and check its `python3 -V` (≥ 3.8, for the gates). If any fails, stop with `SYSTEMIC` — do not
submit training.

### Step 2 — Train pipeline

If the example has no smoke task (`vllm_smoke_test.sh`), add one modeled on
`examples/Qwen/Qwen3-8B/hf_online_dflash2.yaml` task_2.

**Overrides.** Use `pipeline.task_<i>.slurm_config.<key>=`, and append to `args` or
`environment` with `'pipeline.task_<i>.args+=["key=value"]'` or
`'pipeline.task_<i>.environment+=[{"KEY":"value"}]'` (a later entry wins). Dropping an entry
(e.g. the `full` budget's `training.max_steps`) needs an edited copy of the example in the
run dir, passed to `--yaml`.

Fit every submission, smoke and bench included, to the cluster per **speculative-decoding**'s
`references/stages/configure.md` (Step 3). Record those as overrides, not budget changes:
they keep the global batch.

From `tools/launcher`, preview, then rerun with `--yes detach=true` in place of `--to-yaml`
(asking first if `approve_submissions` is `each`):

```bash
uv run launch.py --yaml examples/<Org>/<Model>/hf_online_<algo>.yaml \
    pipeline.global_vars.output_dir=/scratchspace/<run_id> <overrides> [hf_local=<dir>] \
    --to-yaml <run_dir>/train.yaml
```

`launch.py` exits `0` even when submission fails, so check its output for `Unexpected error`,
not the exit code. Tasks queued before the failure stay queued: on Slurm, find them with
`squeue`/`sacct` and cancel them before resubmitting. Record the job ids as `submitted` only
once all are confirmed. Never resubmit into a run's `output_dir`: a retrain is a new run, with
a new `run_id`, run dir and state file. Gate with
`<out>` = `<job_dir>/cicd/<exp_id>/<run_id>`:

```bash
# gate_train.py
--regression <out>/regression_check.json --export <out>/exported-checkpoint-final \
    --target-config <target host path>/config.json --algorithm <algo>
# gate_smoke.py (after gate_train passes)
--smoke <out>/smoke_check.json --draft /scratchspace/<run_id>/exported-checkpoint-final \
    --target <target as the YAML passes it>
```

**Smoke-only submission.** `<run_dir>/smoke.yaml` is the example's smoke task alone, with
`DRAFT_MODEL: <drafter as the job sees it>` replacing `DRAFT_CKPT_DIR` and
`output_dir=/scratchspace/<run_id>`; gate it with `--draft <that DRAFT_MODEL>`, the string
the smoke JSON records. Use it:

- with `existing_drafter` (skips training, not its gates), after the `gate_train.py` above
  with `--export-only` in place of `--regression` and `--export <drafter host path>`;
- to resubmit a smoke test (Triage), with
  `DRAFT_MODEL=/cicd/<training exp_id>/<run_id>/exported-checkpoint-final`.

### Step 3 — Freeze

```bash
# freeze_drafter.py (mkdir -p the destination's parent first)
--src <out>/exported-checkpoint-final --dst <job_dir>/cicd/drafts/<target>/<algo>/<run_id>
# with existing_drafter: --src <drafter host path>
```

It copies to a `.partial` sibling, re-hashes it, writes `SHA256SUMS`, makes it world-readable
and its files read-only, then renames it into place. Record the frozen path, checksums and
`manifest_sha256` in the state.

### Step 4 — A|B bench

If the target has no `specdec_ab_<algo>.yaml`, add one modeled on
`examples/Qwen/Qwen3-8B/specdec_ab_dflash2.yaml`, taking the target, `NUM_SPEC_TOKENS`,
container and GPU count from the smoke task that passed. Preview and submit as in Step 2:

```bash
uv run launch.py --yaml examples/<Org>/<Model>/specdec_ab_<algo>.yaml \
    pipeline.global_vars.draft_model=/cicd/drafts/<target>/<algo>/<run_id> \
    pipeline.global_vars.output_dir=/scratchspace/ab [hf_local=<dir>] --to-yaml <run_dir>/ab.yaml
```

`REPEATS`, `CONCURRENCIES` and `NUM_SPEC_TOKENS` are `environment` entries: change them as in
**Overrides** (Step 2).

Both arms run in one job on one node. Gate with what the run was **submitted** with, taking
each `<VAR>` from the `environment` in `<run_dir>/ab.yaml` (the last entry, if repeated),
with `<ab>` = `<job_dir>/cicd/<exp_id>/ab`:

```bash
# gate_bench.py
--summary <ab>/ab_summary.json --concurrencies <CONCURRENCIES> --repeats <REPEATS> \
    --algorithm <SPEC_ALGO> --num-spec-tokens <NUM_SPEC_TOKENS> \
    --draft /cicd/drafts/<target>/<algo>/<run_id> --target <hf_model>
# gate_correctness.py — report only, never blocks
--summary <ab>/ab_summary.json
```

`SPEC_ALGO` is `run.py`'s serving name, not the training name (`DFLASH` also serves
DFlash2). An empty summary is
`USER_CONFIG_ERROR`: check `--summary`; if it is right, the job died before writing any arm,
so resubmit as for `BENCH_INCOMPLETE`.

### Step 5 — Closeout

Set `verdict` in the state, then write `SPECDEC_REPORT.md` in the run dir from the state,
`gates/` and `work_log.md`:

- **Setup:** inputs and pins (ModelOpt commit, images, launcher example and overrides),
  experiment and job ids.
- **Training:** metrics against the recipe's floors (not checked with `existing_drafter`),
  smoke-test AL.
- **Drafter:** frozen path and checksums.
- **Benchmark:** AL, TPS A/B and speedup per concurrency from `gate_bench`'s `evidence`, its
  `measurement_setup`, and `gate_correctness`'s greedy exact match.
- **Verdicts:** every gate's, the `HUMAN_INTERVENTION` count with reasons, and the readiness
  verdict.

State it is **not** a publish decision, and end with "Next: release review" (SPEED-Bench
measurement on GB200, its gates, model card, publish).

## Watching jobs

**Start watching in the same turn as every submission**, and re-arm the watcher whenever it
expires, until the result file appears or the jobs end. Never end a turn while a stage is
`submitted`, unless the user asks you to stop. Register Slurm jobs with the **monitor** skill,
but track progress by the result files the gates read (`regression_check.json`,
`smoke_check.json`, `ab_summary.json`), not job states.
Read a task's log (`<job_name>_<task index>/` in the experiment dir) within 10 min of it
starting. A job that ends without its result file is still gated. Poll and run every remote
command through `remote_run`, which runs bash: the login shell may be csh, where a bare
`ssh … '… 2>/dev/null'` breaks. Never `set -x` with secrets in scope.

## State and resume

`specdec_state.json`, one per run:

```json
{"schema": 1, "run_id": "...", "inputs": {...},
 "stages": {"train": {"status": "passed", "experiment_id": "cicd_...", "job_ids": [...],
                      "attempt": 1, "overrides": [...],
                      "gates": {"train": {"pass": true, "failure_class": null, "detail": "...",
                                          "file": "gates/train.json"}, "smoke": {...}},
                      "artifacts": {"export": "<job_dir>/cicd/cicd_.../<run_id>/exported-checkpoint-final"}},
            "freeze": {"status": "passed", "path": "...", "sha256": {...}, "manifest_sha256": "..."},
            "bench": {"status": "submitted", "experiment_id": "...", "job_ids": [...],
                      "attempt": 1, "gates": {"bench": {...}, "correctness": {...}}}},
 "verdict": null}
```

`status` is `pending`, `submitted`, `passed`, `failed` or `not run`. Each gate entry keeps
`pass`, `failure_class`, `detail` and the path of its full output under `gates/`. `attempt`
counts the stage's submissions. Paths are absolute job-host paths, `<job_dir>` expanded. On
resume:

- never re-run a `passed` stage;
- **re-watch a `submitted` stage, never resubmit it**, until its result file appears or its
  jobs are gone; then gate it;
- use the host paths recorded in the state, never `/scratchspace`;
- `freeze_drafter.py --dst <frozen path> --verify --manifest-sha256 <recorded>` before
  benchmarking.

`work_log.md` lines: `<UTC> | <step> | ACTION or HUMAN_INTERVENTION | <what and why>`.
Submissions and approvals are `ACTION`. Any step a human had to take is
`HUMAN_INTERVENTION`, including a nudge to check progress or to start watching.

## Triage (gate failure → decision)

| `failure_class` | Gate | Action |
| --- | --- | --- |
| `USER_CONFIG_ERROR` | any | Fix the invocation (paths, `--algorithm`, unknown family → `--expect-architecture`), or permissions or disk space (freeze); re-run the gate only. From `gate_smoke` with `stage: invalid_config`: fix `MIN_ACCEPTANCE_LENGTH`/`NUM_SPEC_TOKENS` in the YAML and resubmit as a smoke-only submission; training needn't rerun |
| `METRICS_MISSING` | train | No regression verdict (training crashed, or the recipe set no thresholds). Review logs with **speculative-decoding**; set `MAX_FINAL_LOSS`/`MIN_FINAL_ACC` in the YAML |
| `TRAIN_REGRESSION` | train | The recipe's own floors failed. Triage with **speculative-decoding**; never waive the floor |
| `EXPORT_MISSING` | train, freeze | Training died before exporting, or the export is incomplete. Triage the training task; retrain as a new run |
| `DRAFT_CONFIG_MISMATCH` | train | Export does not fit the target (architecture, projector type, hidden/vocab size, mask token, layer ids). Fix the recipe overrides per the algorithm sheet; retrain |
| `CHECKPOINT_NOT_SERVABLE` | smoke | Fix the export or serving flags with **deployment**; do not benchmark. If `stage` is `server_died`/`server_timeout` with no config change, resubmit once as a smoke-only submission |
| `SPEC_NOT_ACTIVE` | smoke | AL ≤ 1.05: drafter not wired up or useless. Check `SPEC_METHOD`, token count, drafter path |
| `ACCEPTANCE_REGRESSION` | smoke | Serves and speculates, but AL < the recipe's `MIN_ACCEPTANCE_LENGTH`. Triage training with **speculative-decoding**; do not benchmark |
| `SPEC_CONFIG_MISMATCH` | smoke, bench | Wrong or unrecorded drafter, target, algorithm or token count, or speculation on in arm A / off in arm B. Fix the submission; re-run |
| `FREEZE_MISMATCH` | freeze | The copy differs from the source (re-run the freeze), or the destination already holds another drafter (freeze to a new `run_id`; never overwrite) |
| `BENCH_INCOMPLETE` | bench | Missing rows or arms, appended results, arms with different response counts. Resubmit into a **fresh** `output_dir` (≤ 2 automatic retries) |
| `ARMS_UNMATCHED` | bench | Arms or rows differ in target, engine, GPU, TP, dataset, length, temperature or concurrency. The comparison is invalid; re-run in one job |
| `ANOMALOUS` | bench | Implausible numbers (AL or speedup above tokens + 1, non-positive TPS, unpaired turns). Do not read the numbers; investigate |

The conductor does not patch and retrain on its own: on a training or serving failure it
stops with the triage finding. `SYSTEMIC` (cluster or credentials down) aborts the run.

## Output

Return a verdict, not raw artifacts:

- `DRAFTER_READY` — every blocking gate passed (`gate_train`, `gate_smoke`, `freeze_drafter`,
  `gate_bench`); ready for release review, not a publish decision.
- `INFEASIBLE` — training, export or serving failed and triage found no fix within the run.
- `ANOMALOUS` — the measurement is invalid after the allowed retries, or freeze failed.
- Always: run dir, experiment ids, frozen path, the `HUMAN_INTERVENTION` count.
