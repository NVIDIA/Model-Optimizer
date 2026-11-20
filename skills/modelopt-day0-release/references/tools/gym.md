# NeMo Gym in the Day 0 release gates

- **Serving failure recovery:** there is no `deployment.command`; carry the working settings back into the serving
  record (`modelopt-deployment`, step 5c) and the model profile, and resume with the tool's recovery guidance.
- **Config parity:** diff the resolved configs the two runs stored (the `global_config.json` artifact of each
  MLflow run) under the same rule. Do not change the evaluation's timeouts at all; its recovery guidance says what
  may be raised.
- **Run summary:** the tool's own run summary has the fields `gate_run.py` reads; pass it through.
- **`UNVERIFIED`:** the run finished but some grading was not recorded (gate class `EVAL_UNVERIFIED`).
- **Re-running:** a run is checked for replay by its run id and output directory.
- **Verbosity:** MLflow mode only (adapter `scripts/gym_adapter.py`).
