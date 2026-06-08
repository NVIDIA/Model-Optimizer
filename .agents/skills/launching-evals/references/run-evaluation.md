# Run the evaluation

Follow the three phases and track your progress in the output.

1. **INPUT** -> EXPLORE -> ACT
2. ~~INPUT~~ -> **EXPLORE** -> ACT
3. ~~INPUT~~ -> ~~EXPLORE~~ -> **ACT**

## 1. INPUT

Gather requirements from the user:

- **Config path**: The YAML config file to run. NOTE: You might already have the config path in your memory from the previous step.
- **Credentials**: Some tasks require environment variables (e.g., `AWS_ACCESS_KEY_ID`, `HF_TOKEN`). Check if there is a `.env` file in the workspace root. If not, ask the user to create one with the credentials exported in it.
- **Task filter** (optional): Specific tasks to run via `-t <task_name>`.
- **Overrides** (optional): Any `-o key=value` overrides.
- **Dry-run first?** (optional): Preview with `--dry-run` before submitting.

## 2. EXPLORE

- Preview the resolved config and the sbatch script by adding `--dry-run` flag to the final command.

## 3. ACT

1. Submit the evaluation: `uv run nemo-evaluator-launcher run --config <path.yaml> ...`
   - NEL automatically reads `.env` from the workspace root — no need to source it manually.
