# SWE-bench

SWE-bench uses the OpenHands harness.

## TL;DR

If you only need the run score:

- `artifacts/results.yml`
- `artifacts/.../swebench_summary.json`

If you need the official per-instance eval result:

- `artifacts/.../output.report.json`

If you need per-instance token usage or rich debug data:

- `artifacts/.../output.jsonl`

If you need the quickest failure triage:

- `artifacts/.../output_errors.jsonl`
- `artifacts/.../logs/instance_<id>.log`

## Retries, Attempts, Resume

- `max_retries`
  Inner retry loop inside one attempt. It is used when running an instance throws an exception, for example sandbox/runtime startup failures, conversation crashes, tunnel problems, polling errors, or other hard execution errors. It is not triggered just because the produced patch was bad or the instance scored unresolved; those are handled by the critic and outer attempts. Each retry creates a fresh workspace/runtime rather than continuing the failed environment. Total executions per attempt are `max_retries + 1`.

- `max_attempts`
  Outer iterative attempts. Attempt 1 runs instances not yet completed in `output.critic_attempt_1.jsonl`. Attempt `N>1` only runs instances that the critic judged failed in attempt `N-1`.

- `critic`
  Controls whether another outer attempt is scheduled. The critic evaluates the conversation history plus the produced git patch.

- Default behavior
  SWE-bench here defaults to `PassCritic`, so `max_attempts` is mostly inert unless you switch critics. In practice, most rerun behavior comes from `max_retries`, not `max_attempts`.

- Resume
  When rerunning into the same output dir, the harness reads existing `output.critic_attempt_N.jsonl` files and uses them as its source of truth. If an instance already has a non-error row for that attempt, it is skipped. If it only has an error row, it is treated as unfinished and is run again. This resume behavior does not depend on the critic.

- Context-window errors
  `ContextWindowExceed` is treated as non-recoverable inside the inner retry loop, so the remaining inner retries are skipped immediately. That only answers the inner `max_retries` question. The instance can still run again later if you rerun/resume into the same output dir, because hard-error rows are treated as unfinished even with `PassCritic`. In short:
  inner retry = exception handling inside one attempt;
  outer attempt = critic says previous output failed;
  resume rerun = this attempt only has an error row so far.
  This can also produce more raw request-level `400`s in metrics than final hard-failed instances, because a run can hit one `400` and still finish as a soft `status=stuck` case with a partial patch.

## What Matters

- `swebench_summary.json`
  Single-run summary: `submitted_instances`, `resolved_instances`, `accuracy`.

- `output.report.json`
  Official eval output. Top-level keys: `dataset`, `evaluation_method`, `model_name_or_path`, `resolved`, `resolved_count`, `results`, `total_instances`. Each `results` row has `instance_id`, `resolved`, `error`, `exit_code`.

- `output.jsonl`
  One final JSON row per benchmark instance. In the inspected run, rows included `instance_id`, `error`, `attempt`, `metrics`, `runtime_runs`, `test_result`, `instruction`, and the full `instance` payload. For verification, the useful part is `metrics.accumulated_token_usage.completion_tokens`.

- `output_errors.jsonl`
  Same row shape as `output.jsonl`, but only for hard failures. Read this first when debugging a bad run.

- `output.swebench.jsonl`
  Minimal prediction file for official SWE-bench eval. Fields: `instance_id`, `model_name_or_path`, `model_patch`.

- `metadata.json`
  Run setup/config snapshot. Includes `dataset`, `dataset_split`, `max_iterations`, `conversation_timeout`, `max_attempts`, `max_retries`, `skip_failed_samples`, `workspace_type`, `llm`, `sandbox_config`, `prompt_path`, and `eval_output_dir`.

- `agent_logs/.../run.json`
  Compact run summary. Fields: `run_id`, `status`, `duration`, `tasks`, `config`, `metadata`, timestamps.

- `agent_logs/.../tasks.jsonl`
  Attempt-level task records. Fields: `task_id`, `attempt_id`, `status`, `reward`, `duration`, `termination`, `error`, `trajectory`, `artifacts`, timestamps. `trajectory.usage` has aggregated `prompt_tokens`, `completion_tokens`, `reasoning_tokens`, `content_tokens`.

- `logs/instance_<id>.log`
  Best per-instance raw text log: sandbox startup, repo/setup steps, tool calls, agent/server messages, and failure traces.

## Live Progress

During a running evaluation, the official result files (`output.report.json`, `swebench_summary.json`, `results.yml`) do not exist yet. Use `tasks.jsonl` for live progress — it is written incrementally as each instance finishes its agent conversation.

### Restart-safe progress tracking

`tasks.jsonl` is **append-only**. When a run is restarted (e.g. after SLURM wall-time kill), errored instances are retried and new entries are appended. The same `task_id` can appear multiple times. Raw line counts will exceed 500 for a 500-task benchmark.

**Always deduplicate by `task_id`** (last entry wins) to get accurate progress. Use the script below for both single-run and multi-restart scenarios.

There are two sources of truth for progress, each useful for different things:

| File | Best for | Notes |
|------|----------|-------|
| `tasks.jsonl` | Live progress with rich detail (status, duration, termination reason) | Append-only, needs dedup by `task_id` |
| `output.critic_attempt_1.jsonl` | What the harness considers "done" for resume | Instance with non-error row = skipped on next restart; error row = retried |

**Quick status count** (run from the cluster where the job is running):

```bash
# Replace TASKS_JSONL with the actual path:
# artifacts/.../agent_logs/.../tasks.jsonl
#
# Deduplicates by task_id (last entry wins), so this works correctly
# even after multiple restarts where tasks.jsonl has >500 lines.
python3 -c "
import json, collections, sys
latest = {}
for line in open(sys.argv[1]):
    line = line.strip()
    if not line: continue
    rec = json.loads(line)
    tid = rec.get('task_id', 'unknown')
    latest[tid] = rec.get('status', 'unknown')
counts = collections.Counter(latest.values())
total = len(latest)
for s, c in sorted(counts.items()): print(f'  {s}: {c}')
print(f'  TOTAL unique: {total}/500')
remaining = 500 - total
print(f'  REMAINING: {remaining}')
" TASKS_JSONL
```

Expected output while running (even after restarts):
```
  error: 3
  success: 120
  TOTAL unique: 123/500
  REMAINING: 377
```

Note: `success` here means the instance was resolved; `error` means a hard runtime failure (context window exceeded, timeout, etc.); `failure` means an evaluable patch was produced but did not resolve the instance. During a run, `failure` counts only appear after the official SWE-bench eval step rewrites `tasks.jsonl`, so mid-run you mostly see `success` and `error`.

After a restart, previously-errored instances that now succeed will show as `success` (the latest entry overwrites the old `error` entry in the deduplication).

**What NOT to use:**
- `Progress: N/T evaluated` in client logs — only emitted at the very end, not useful for in-flight monitoring.
- Raw line count of `tasks.jsonl` — will exceed 500 after restarts due to append-only behavior.
- `output.critic_attempt_1.jsonl` for progress display — also append-only with duplicates, and has less detail (no `status`/`termination`/`duration`). However, it is the file the harness reads to decide what to skip vs retry on restart.

## Instance IDs

- Format
  SWE-bench instance IDs are dataset-defined and use `<org>__<repo>-<number>`, for example `django__django-11333`.

- Meaning
  `django__django` corresponds to repo `django/django`. The trailing number is the benchmark instance number within that repo, not a retry/run suffix added by our harness.

- Canonical key
  The harness loads `row["instance_id"]` directly from the dataset and uses the full string as the canonical task key for inference and evaluation metadata lookup.

- Practical implication
  `django__django-11333` and `django__django-16116` are different SWE-bench tasks from the same repo. They can differ in `problem_statement`, `base_commit`, `test_patch`, and expected test outcomes (`FAIL_TO_PASS`, `PASS_TO_PASS`).

- What is `test_patch`?
  Dataset-provided test-only patch used during evaluation, not scoring input from the model. In `eval_infer.py`, the harness loads `meta["test_patch"]`, applies the model patch first, then applies `test_patch`, then runs the benchmark test script. The prompt template does not include `test_patch`; it only includes `problem_statement` and tells the agent not to modify tests. Practical meaning: the model is expected to change non-test source files, while benchmark-owned test updates/scaffolding are applied afterward during evaluation.

## Failure Modes

SWE-bench does not have a clean TB-style `failure_mode` enum. Also, conversation termination is not the same thing as the final per-instance outcome: SWE-bench can still collect and evaluate a partial patch after `status=stuck` or even some `status=error` terminations, so an instance can still end up officially resolved.

Where to look:

- `tasks.jsonl`
  Best lightweight source for final per-instance status and termination reason.
  Use top-level `status` for the final per-instance outcome (`success` / `failure` / `error`).
  Use `termination.reason` for how the conversation ended (`finish_tool`, `finished_no_finish_tool`, `status=error`, `status=stuck`, etc.).
- `output_errors.jsonl`
  Best source for concrete hard-failure messages.
- `output.report.json`
  Best source for official `resolved` / `unresolved`, but its `error` field is not a reliable failure reason.

What to expect:

- `status=success`
  In top-level `tasks.jsonl.status`, this means the instance was resolved in the final official SWE-bench evaluation. This is assigned after evaluation rewrites `tasks.jsonl`, not merely because the agent called `finish` or the run ended cleanly.
  Separate note: `run.json` can also say run-level `status=success`, but that only means the overall evaluation process finished cleanly.
- `status=failure`
  In top-level `tasks.jsonl.status`, the attempt produced something evaluable, but the final official SWE-bench evaluation did not mark the instance as resolved.
- `status=error`
  In top-level `tasks.jsonl.status`, this means a hard runtime failure. This is where agent timeout and similar non-soft errors land.
  Typical examples:
  `Run timed out after <N> seconds`; `Remote conversation ended with error`; `Remote conversation not found (404). The runtime may have been deleted.`; `Polling failed with HTTP <code>`; `LLMContextWindowExceededError` / `ContextWindowExceededError`.
  Exception: `MaxIterationsReached` still uses conversation execution status `error`, but OpenHands treats that specific error code as a normal stop and SWE-bench continues with patch collection/eval.
  In the inspected Nemotron-Super run, all 5 such cases were context-window exceeded after retries.
- `termination.reason = status=stuck`
  This is a conversation end state, not a final per-instance status. Check it in `tasks.jsonl.termination.reason`.
  It means OpenHands stopped the conversation after detecting a no-progress pattern after the last user message.
  Default triggers:
  4 repeated identical action + observation pairs; 3 repeated identical action + error pairs; 3 consecutive agent-only messages; 6-step alternating repeated action/observation pattern.
  After that, SWE-bench may still collect a patch and later mark the instance as top-level `status=success` or `status=failure`.
