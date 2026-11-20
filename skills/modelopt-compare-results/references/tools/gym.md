# Comparing NeMo Gym runs

- **Score field:** the run summary score (the run's MLflow `<agent>/mean/reward`, on a 0 to 1 scale).
- **Run validity:** take the tool's own verdict (see `modelopt-eval-guidance`, `references/tools/gym.md`).
- **Real difference or noise, and a baseline that misses its published score:** the `nemo-gym-score-triage` skill.
- **SciCode:** in the runs checked so far the eight repeats ran inside one submission, so confirm what a run
  contains before counting runs per side.
