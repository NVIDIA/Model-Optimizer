# Orchestrator Final Report Design

## Goal

Generate the canonical cumulative Puzzletron HTML report after every clean
orchestrator completion, using the campaign's configured runner environment,
and expose the report outcome and artifact paths to the caller.

## Scope

The orchestrator will gain a runner-backed finalization step. It will not add a
new campaign DAG node, alter model-stage identities, or rerun completed model
work. Existing per-scenario reports remain unchanged; the finalizer publishes
the campaign-wide report at:

```text
<puzzle_dir>/artifacts/campaign_report/campaign_report.html
```

## Architecture

Add a focused orchestration reporting helper that:

1. Builds a single CPU-only `AttemptSpec`.
2. Runs `examples/puzzletron/generate_campaign_progress_report.py` with the
   compiled plan's `puzzle_dir` and model display name.
3. Submits the attempt through the controller's existing executor.
4. Polls until the report job reaches a terminal state.
5. Returns the report status, HTML path, manifest path, and scheduler log paths.

Because submission uses the existing executor, Slurm and bare-metal runs inherit
the runner's repository, container, mounts, setup commands, virtual environment,
and post-run hooks. The report command remains outside the dependency-light
login-node orchestrator process.

## Trigger Semantics

Run final report generation only when the selected campaign plan exits cleanly:

- no failed stages;
- not halted, cancelled, or detached;
- not paused for manual input; and
- not limited by `--once` or `--max-iterations` before completion.

This applies to both a newly completed campaign and a resumed invocation whose
selected stages are already complete. The latter provides a safe report-only
regeneration path without rerunning model stages.

Do not generate a final report after failure or interruption because the
controller already returns failed-stage logs and the invocation did not reach a
clean terminal state.

## Failure Policy

Report generation is read-only with respect to model artifacts. A report job
failure must not turn completed model work into a failed campaign.

The orchestrator will:

- log the report failure clearly;
- return `report_status`, `report_path`, `report_manifest_path`, and
  `report_log_paths`;
- leave `halted` unchanged; and
- continue returning exit code zero when all selected model stages completed.

Successful generation returns `report_status: "completed"` and verifies that
both the HTML and manifest exist before reporting success.

## Resource and Scheduler Behavior

The finalizer uses one direct CPU task, zero GPUs, and one node. On Slurm it uses
`runner.slurm.partition_cpu` when configured; otherwise it follows the runner's
normal one-node partition selection. It never requests `--exclusive` or a GPU.

The report log is stored under the campaign's canonical log directory with a
stable final-report label and an attempt-unique suffix.

## CLI and Result Contract

No new required CLI arguments or execution-YAML entries are introduced.

The JSON returned by `examples/puzzletron/orchestrate.py` gains:

```json
{
  "report_status": "completed",
  "report_path": "<puzzle_dir>/artifacts/campaign_report/campaign_report.html",
  "report_manifest_path": "<puzzle_dir>/artifacts/campaign_report/report_manifest.json",
  "report_log_paths": ["<scheduler log path>"]
}
```

For skipped or failed report generation, `report_status` records the terminal
outcome and unavailable artifact fields are `null`.

## Verification

Focused unit coverage will verify:

- a clean completion submits exactly one CPU report attempt;
- the attempt uses the compiled `puzzle_dir` and runner environment;
- interrupted, failed, detached, manual-waiting, and partial invocations do not
  submit a report;
- report failure is returned and logged without changing campaign success; and
- successful generation requires the HTML and manifest to exist.

For the current production campaign, rerun the completed orchestrator after the
change. It must submit only the report job, wait for it, and produce a report
whose embedded campaign data marks Online Eval, Serving, Short KD, Final Eval,
and Best as completed.
