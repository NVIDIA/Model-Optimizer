# NEL

Details of the NEL (NeMo Evaluator Launcher) path. Everything in the guidance that is specific to NEL is here.

## Skills

`modelopt-evaluation` (config, submission, NEL run validation, benchmark recipes), `modelopt-launching-evals` (run,
resume, debug, analyze), and the NEL job type in `modelopt-monitor`. NEL's conventions for looking MLflow runs up are
in `modelopt-accessing-mlflow`.

## Health verdict and faults

NEL has no run verdict of its own, so it needs every check in "Run checks NEL needs" below. Apply the bounded-failure policy in
`modelopt-evaluation/references/run-validation.md` (its response and trial gates, both at 2%) to NEL runs only.

## Settings

- **Model card reading:** `modelopt-evaluation/references/model-card-research.md`.
- **Output budget fallback**, only when the task policy permits one and the card discloses none: 65536 for reasoning
  models, 16384 for non-reasoning models (in the same file).
- **Where per-task sampling goes:** per-task `nemo_evaluator_config`; see
  `modelopt-evaluation/references/launcher-workflow.md`. For the NVFP4 sampling table, `max_num_tokens` is
  `nemo_evaluator_config.config.params.max_new_tokens`.
- **Benchmark recipes:** `modelopt-evaluation/recipes/tasks/`. SciCode is run as several separate submissions of
  one repeat each, averaged.
- **Score fields:** the `Score Extraction` section of the matching task recipe; do not rely on ad hoc `results.yml`
  greps.

## Run checks NEL needs (no verdict of its own)

NEL has no run verdict of its own, so the checks that the evaluation service does for a tool with a verdict are done
here, by the evaluator. They are NEL's until NEL is removed; a tool with a verdict replaces them with that verdict.

### Verify a completed run

Do not accept a run as complete because a result file or summary exists. For each completed baseline,
quantized or single-model run:

1. **Logs.** Inspect client, server or deployment, scheduler, judge and task or code-execution logs as they
   apply. Search for `Traceback`, `Exception`, `ERROR`, `FAILED`, `OOM`, `Killed`, `timeout`, `rate limit`,
   `unauthorized`, `connection refused/reset`, `health check`, `sandbox`, `container`, `judge`, `parse` and
   `scoring`, plus task-specific failure strings.
2. **The intended checkpoint loaded.** Inspect startup failures, crashes and restarts, OOMs, request errors,
   input or context clipping, and quantization load errors. Systemic serving failures are blockers.
3. **Judge-backed tasks.** Inspect authentication, rate limits, malformed responses, parsing and scoring.
   Distinguish a protocol-valid failed score from a missing score or an undocumented fallback score.
4. **Code-execution tasks.** Inspect executor, sandbox and container setup, installation, timeout, resource,
   permission and harness diagnostics. Verify each failed trial's protocol-defined outcome; do not treat every
   error as invalid.
5. **Coverage at every level.** Expected, selected, accounted-for and scored samples, repeats or trajectories
   must match. A protocol-valid failed trial counts as accounted for; a missing or unscored trial, a dropped
   submission, an abort that leaves gaps, or a partial result does not.
6. **Reasoning parsing.** If reasoning traces are present, confirm they are parsed, stripped or ignored before
   scoring, consistently. Look for parser or fallback errors, unmatched reasoning delimiters, reasoning text
   leaked into answers, answers stripped with the reasoning, or reasoning disabled when the config meant it on.
7. **Accounting.** Complete the section below for every task, including successful and non-reasoning runs.

Report a validation summary before any score: log scan status, coverage, reasoning and answer parsing status,
and any errors or warnings. If a check fails, label the result incomplete or invalid and return the findings
and a recommendation to the parent or user; do not resubmit a completed run on your own. Keep score,
validation and MLflow delivery separate, and carry counts, rates and warnings into the handoff.

### Timeout and output-limit accounting

For every benchmark and run, inspect structured response or trial artifacts, logs and the resolved config.
Report counts, percentages, explicit denominators and artifact paths before the score:

| Check | Required evidence |
|---|---|
| Coverage | Expected trials including repeats; completed and scored, failed, skipped, missing; the score denominator and whether failures get zero or are excluded. |
| Timeouts | Timed-out over all request attempts; affected unique trials over expected trials; terminal-timeout trials over expected trials. Separate recovered retries from terminal failures by layer: client or proxy, agent or task, judge, sandbox or verifier. |
| Output limits | Limit-stopped over observed responses; affected unique trials over expected trials. Use termination metadata such as `finish_reason: length` (and [doom-loop](../doom-loops.md) repetition stops, counted separately); separate output caps, context exhaustion, and agent step or total-token limits where possible. |
| Limits | Effective request, agent and judge limits, output-token and context limits, timeout strategy, overrides, concurrency, sharding and serving setup. |

Deduplicate resumed records by sample, trial and repeat id, and keep request attempts separate from trials.
The categories overlap, so do not sum them as disjoint failures. Log matches and token counts near a cap are
clues, not proof of affected samples. Missing termination metadata, sampled-only artifacts or omitted failures
make full-run rates **unknown**, not zero: say what evidence is missing and do not extrapolate.

- **Recovered retries** are reported separately; they do not excuse a terminal unscored failure or missing
  coverage.
- **Provisional or inconclusive:** unknown accounting leaves validation incomplete. Investigate infrastructure
  failures, unexpected exclusions and mismatched limits before a model-quality verdict. A valid run alone does
  not establish quantization feasibility.
- **Comparison:** matching wall-clock limits does not isolate model quality, since speed, queueing, concurrency
  and verbosity change the work completed. Compare both sides' limit-hit rates; unresolved timeout effects make
  attribution to quantization inconclusive.
- **Reruns:** never silently drop timed-out trials or raise only one model's limits. Use matched settings,
  preserve the original results, and label protocol changes.
- **Keep protocol limits fixed.** Verify that prompt, history and output fit the deployed context, and report
  mismatches or unknown reference settings; proposed changes need the parent's or user's approval.

## When the baseline fails the fault gate at the output cap

The 2% response-fault gate counts responses that stopped at the output limit with no final answer. A reasoning
model's long tail can exceed it at the NEL fallback budget (GPQA at 65536: 3.5% to 5.1% in four runs), so the
baseline is invalid and no candidate comparison is possible. Do not stop there, and do not change one side only.
The supported recovery is **one matched correction proposed by the evaluator with the evidence** (the length-stop
and empty-answer counts, that prompt plus output still fits the served context, and the card's own guidance for
long outputs), **approved by the parent or the user**, applied to the baseline and the candidate alike, with the
original invalid run kept and the change labeled as a protocol change. In the earlier end-to-end runs raising the
output budget from 65536 to 131072 on both sides, for GPQA only, brought the fault rate under the gate. If the
approval is refused or the matched rerun is still invalid, report the comparison as blocked, not as a result.

## Doom loops

Where `repetition_detection` goes in the config, and the `finish_reason` remap that the nemo-skills client needs, are
in `modelopt-evaluation/references/doom_loops.md`.

## MLflow

- NEL runs carry the tags `invocation_id`, `task_name`, `benchmark`, `harness`.
- Token averages are metrics such as `<task>_pass_at_1_avg-of-N_answer_tokens_statistics_avg`; use the one whose N
  equals the repeat count (on a checked IFBench run it equalled `response_stats.avg_completion_tokens` in
  `eval_factory_metrics.json`). Sample count is `*_num_prompts` times the repeat count.
- `simple_evals` runs (GPQA) carry no token metrics, only the stored artifact
  `<harness>.<task>/artifacts/eval_factory_metrics.json`.

## Jobs, resuming and cancelling

A wedged NEL job must not be `scancel`-ed to free the GPUs: NEL's `afternotok` continuation refuses a user-cancelled
predecessor and discards the response cache (one `scancel` cost about 59 GPU-hours). Let it hit the wall clock.
Resume behavior is in `modelopt-evaluation/references/run-validation.md`. A warm NEL response cache can make a
"re-run" replay cached responses.
