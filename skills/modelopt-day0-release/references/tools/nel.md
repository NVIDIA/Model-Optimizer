# NEL in the Day 0 release gates

Use this only when the task names NEL.

- **Serving failure recovery:** carry the working command back into NEL's `deployment.command` and resume the eval.
- **Config parity:** diff the two NEL configs as in Step 4 (`checkpoint_path`, `served_model_name` aside). Never set an
  unbounded `request_timeout` (`1e9`): it turns a transient stall into a job that holds its GPUs until the wall
  clock kills it.
- **Failure policy:** the bounded policy in
  [NEL run validation](../../../modelopt-evaluation/references/run-validation.md) overrides the retry actions in
  Triage, and applies also to the run-summary you build for `gate_run.py` (NEL has no summary of its own).
- **Re-running:** a warm NEL response cache can replay cached responses (two runs came back bit-identical to 16
  digits).
- **Verbosity from result directories** (adapter `scripts/nel_adapter.py`):

  ```bash
  python "$SKILL_DIR/scripts/gate_verbosity.py" \
      --baseline <baseline_eval_root> --candidate <candidate_eval_root> \
      --glob 'eval_*' --exclude _high --threshold 0.05
  ```

  Read `response_stats.avg_completion_tokens` from each task's `artifacts/eval_factory_metrics.json`. Do **not**
  use the `reasoning.*` fields: `reasoning.*_tokens` are always `0` (only the reasoning/content split is missing)
  and the `*_words` siblings are a proxy that can disagree with the gate (one task read +6.10% FAIL in words and
  +1.32% PASS in tokens). If an exit `2` detail names `collapsed_keys`, the artifact tree gave several tasks one
  name: point the gate at a tree whose task dirs are `<harness>.<task>` rather than
  `<invocation_id>.<job_index>`, or re-sync so each `artifacts/` carries its own `config.yml`.
- **Verbosity from MLflow:** `simple_evals` runs (GPQA) have no token metric, so the adapter reads their stored
  `eval_factory_metrics.json` artifact.
