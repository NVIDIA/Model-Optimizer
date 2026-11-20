# Comparing NEL runs

- **Fetching runs from MLflow:** the `modelopt-accessing-mlflow` skill (NEL conventions) fetches configs and
  artifacts for NEL runs.
- **Run validity:** also apply evaluation Step 9, "Verify completed evaluation run", and the NEL policy in
  `modelopt-evaluation/references/run-validation.md`, before comparing scores.
- **Score field:** the matching task recipe's **Score Extraction** section
  (`modelopt-evaluation/recipes/tasks/<task>.md`).
- **SciCode:** see the NEL recipe `modelopt-evaluation/recipes/tasks/aa/scicode.md`; it runs as several separate
  submissions of one repeat each, so count one run per submission.
