# External Baseline Check

For a baseline-vs-candidate comparison, perform this check after run validation
and before applying the candidate-delta gate or issuing a success verdict. This
is additional to, not a replacement for, the apples-to-apples and baseline
precision checks in `modelopt-compare-results`.

For each baseline task:

1. Search for a published score for the exact model in its Hugging Face model
   card and on Artificial Analysis (<https://artificialanalysis.ai/>); use
   either credible source. A score for a sibling size, release, or precision is
   not an exact-model reference.
2. Match model variant, benchmark and version, metric, reasoning/thinking mode,
   prompt and chat template, sampling and token budget, sample count, and
   evaluation protocol as closely as possible. Record the external score,
   source URL, and every known protocol difference. Do not treat a mismatched
   result as directly comparable; find a closer source or mark the task
   externally unverified.
3. Put both scores on a 0-100 scale, then calculate, for higher-is-better
   metrics:

   ```text
   difference (pp) = abs(measured baseline - external)
   ```

   Treat a credible comparable result as verified only when the absolute
   difference is approximately 5 percentage points or less. A difference
   greater than approximately 5 points fails the check even if the candidate is
   within its normal delta gate (for example, `<1pp`). For an external score of
   60, the approximate range is 55-65; a baseline of 54 is 6 points away and
   fails.

Report each task as `verified`, `failed`, or `externally unverified`. A large
upward difference also does not establish a clean match; investigate protocol
differences before marking it verified. If no credible comparable score exists,
state `externally unverified` and do not invent a reference or claim the sanity
check passed. This status does not block comparison or publication: use the
validated measured baseline, apply the candidate-delta gate, and report that no
external corroboration was available. Only a `failed` external check blocks the
comparison.

If any task fails, do not report the quantized evaluation as successful and do
not apply the candidate-delta gate to that baseline. For a NeMo Gym run, use its `nemo-gym-score-triage` skill
for the investigation; this check only decides whether the baseline is credible. Investigate disabled
reasoning/thinking, reasoning parser or adapter handling, prompt/chat-template
differences, sampling or token-budget differences, benchmark version or metric,
incomplete samples, and serving failures. Rerun a corrected baseline, validate
it, and repeat this check before comparing the candidate.

For score harvesting, use the score field the evaluation tool defines for the benchmark: for NEL, the `Score
Extraction` section of the task recipe in `modelopt-evaluation/recipes/tasks/` (do not rely on ad hoc
`results.yml` greps); for NeMo Gym, the headline score of the run summary, which is also the run's MLflow
`<agent>/mean/reward` metric on a 0 to 1 scale (see [run-validity.md](run-validity.md#reading-results-from-mlflow)).

For baseline-versus-candidate deltas, use the `modelopt-compare-results` skill after each run passes validation.
