# NeMo Gym

Details of the NeMo Gym path. The model onboarding, the run and its recovery are done by the Gym skills; this
file only says where they cover a duty of this guidance and where they do not.

## Skills

`add-model-eval-config` (model onboarding), `run-and-babysit-eval` (submit, babysit, inspect, recover) and
`nemo-gym-score-triage` (a score that does not match a reference, two runs that disagree, or whether a
baseline-versus-candidate difference is real).

## Health verdict and faults

Take the tool's own verdict (`SUCCESS` with health `PASS`; its 2% limit counts evaluation-side faults). A wrong,
truncated or empty answer, or a clean agent timeout, is the model's result and counts as zero; do not add a
response-level invalidation of your own. `UNVERIFIED` means the run finished but some grading was not recorded:
report the score as unverified.

## Settings

- **Model card reading:** `add-model-eval-config` (sampling, the reasoning switch, parsers, context length,
  parallelism, the serving image). It does not cover ARM64 or non-standard GPU images.
- **A quantized variant** is served with its baseline's config except for the fields the quantization requires
  (`add-model-eval-config`, section 9, "A quantized variant"); the same skill accepts a deployment derived elsewhere.
- **Output budget, repeats and the score definition** come from the benchmark's configuration.
- **Investigating a baseline that does not match its published score:** `nemo-gym-score-triage`.

## Doom loops

Gym counts `repetition_stops` (the model's own loops, scored). How to enable the detection per request is not
verified; if the tool cannot pass the field, say so instead of editing the evaluation.

## MLflow

- Runs carry the tags `precision`, `pair_id`, `stage`, `requested_task`.
- The headline score is the metric `<agent>/mean/reward` on a 0 to 1 scale (multiply by 100; on every checked run it
  equalled the run summary score). Token averages are `<agent>/mean/output_tokens`; sample counts are
  `coverage/expected` and `coverage/scored`; evaluation health is under `health/*` and `coverage/*`.
- One run is one benchmark of one submission, so a combined score averages equal-repeat runs, and the task status
  comes from the tool's verdict (a benchmark's run can be `FINISHED` while its submission failed overall).
