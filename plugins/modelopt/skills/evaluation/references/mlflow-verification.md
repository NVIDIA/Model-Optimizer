# Verify MLflow delivery

Evaluation completion and MLflow delivery are separate outcomes. Apply this gate
per invocation/task after run validation, before handoff or cleanup. Use
`launching-evals` to locate existing results and `accessing-mlflow` to query the
tracking server; those vendored skills remain unchanged.

## Before upload

Before any upload (automatic or manual), secret-scan all outgoing artifacts,
including result bundles, configs and logs; redact credentials, tokens,
authorization headers, and secret-bearing URLs. Never upload `.env`/secret files.
Preserve non-secret methodology/provenance in sanitized copies; if safe upload
cannot be established, report export blocked.

Before submission, if generated artifacts cannot be checked before automatic
upload, disable auto-export (`execution.auto_export.destinations: []` for NEL
0.2.6) for both canary and full runs. Keep the MLflow export settings for checked,
sanitized manual delivery from existing results below. A post-run scan cannot
satisfy this prerequisite: 0.2.6 submits its exporter without an agent-review pause.

## Verify before exporting again

1. Record the invocation ID, task/job identity, evaluation outcome, and result
   paths. Check the separate auto-export job's state and logs: evaluation
   `SUCCESS` does not prove export succeeded. If export is queued/running, monitor
   it or report export pending; do not start a competing export.
2. Search the configured MLflow server/experiment by `invocation_id` and task
   identity, not display name alone. Search other experiments if necessary.
   Distinguish canary/full runs and repeated tasks. For nel-next, use its run ID
   and bundle `job_id` identity instead. An inaccessible server is **unverified**,
   not evidence that no run exists.
3. Inspect each actual run's metrics and artifact contents, not just its status
   or artifact listing. Verify the canonical score field/value against the
   validated results, the evaluated checkpoint and resolved configuration
   (including task overrides), and sample/scoring coverage. Check required
   diagnostics: result file, resolved config, runtime metrics when produced,
   and relevant client/server/SLURM/judge logs or compact diagnostic evidence
   supporting the validation summary. Check benchmark-specific artifacts where
   required. Record genuinely inapplicable diagnostics; missing required
   evidence means incomplete export. Artifact paths vary: discover them.
4. Reuse a usable run. Return its verified run URL and invocation/task mapping;
   never infer a run URL from an invocation ID or accept a same-named run.

## Recover existing results only

If auto-export failed or the run is incomplete, first confirm no exporter is
still active and recheck MLflow immediately before recovery. Inspect the
**installed** launcher's version and help (e.g. `nel --version`, `nel --help`,
then `nel export --help` if that subcommand exists). Use only the destination,
invocation/task selection, config, and artifact/log options that version supports.

Manually export the **existing invocation/results** to MLflow. Do not run or
resume evaluation to repair export. Diagnose the export-specific failure: for
example, a CPU export job's container-import failure needs an available,
compatible export image or supported local export, not another GPU evaluation.
Change only export settings using the installed version's supported mechanism;
retain the original evaluation configuration and record recovery settings
separately. A local summary export is not MLflow delivery or an artifact backup.

If a partial run exists, prefer a supported update/repair of that run. Do not
assume retries are idempotent: establish the exporter's behavior first. If a
replacement is unavoidable, record both run IDs and identify the verified
replacement; do not delete the original without authorization. After recovery,
repeat the full verification above. Export exit code zero alone is insufficient.

**nel-next:** Preserve the explicit `nel-next.sh mlflow-push -r <run_id> -c <cfg>`
workflow in `nel-next.md`; SLURM does not auto-export there. The wrapper stages
only `eval-*.json`, not logs; `copy_logs=true` cannot supply unstaged evidence.
Verify the pushed runs, then attach missing sanitized configs/logs or compact
diagnostic evidence to the identified runs using a supported MLflow artifact
upload. Recheck their contents; if evidence is unavailable, report incomplete or
blocked delivery. Do not substitute the legacy launcher's export command.

## Safe evidence and blocked delivery

If recovery remains blocked, preserve a **small, sanitized** evidence bundle in
an approved durable location before mandated cleanup: invocation/task and job
IDs, scores/results, evaluated config, coverage and diagnostic summary, relevant
log excerpts, export error, launcher version, and attempted recovery command
(with secrets removed). Include any known partial MLflow run IDs and missing
artifacts. Do not copy caches, checkpoints, or all traces as an export fallback.
If evidence cannot be preserved, report that blocker before destructive cleanup;
never silently discard the only results.

In the evaluator handoff, report evaluation and export outcomes independently
per task: e.g. `evaluation: valid; export: verified (auto/manual)`, `pending`,
`incomplete`, or `blocked`. Include invocation IDs, verified MLflow run URLs (or
explicitly none verified), missing evidence, the specific blocker, and absolute
preserved-evidence paths. A valid evaluation may have blocked delivery; neither
outcome overrides the other or authorizes another evaluation submission.
