# NEL

Details of the NEL (NeMo Evaluator Launcher) path. Everything in the guidance that is specific to NEL is here.

## Skills

`modelopt-evaluation` (config, submission, NEL run validation, benchmark recipes), `modelopt-launching-evals` (run,
resume, debug, analyze), and the NEL job type in `modelopt-monitor`. NEL's conventions for looking MLflow runs up are
in `modelopt-accessing-mlflow`.

## Health verdict and faults

NEL has no run verdict of its own, so every check in `run-validity.md` applies. Apply the bounded-failure policy in
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
