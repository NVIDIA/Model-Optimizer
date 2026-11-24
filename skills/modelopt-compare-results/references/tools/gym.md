# Comparing NeMo Gym runs

- **Score field:** the run summary score (the run's MLflow `<agent>/mean/reward`, on a 0 to 1 scale).
- **Run validity:** take the tool's own verdict (see `modelopt-eval-guidance`, `references/tools/gym.md` and
  `references/run-validity.md`, which also lists the metrics and tags to read from MLflow).
- **Baseline check and timeout accounting:** `modelopt-eval-guidance`, `references/external-baseline-check.md` and the
  "Timeout and output-limit accounting" section of `references/run-validity.md`.
- **Real difference or noise, and a baseline that misses its published score:** the `nemo-gym-score-triage` skill.
- **SciCode:** in the runs checked so far the eight repeats ran inside one submission, so confirm what a run
  contains before counting runs per side.
