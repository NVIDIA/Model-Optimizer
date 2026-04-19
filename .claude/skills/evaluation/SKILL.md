---
name: evaluation
description: Evaluates accuracy of quantized or unquantized LLMs using NeMo Evaluator Launcher (NEL). Triggers on "evaluate model", "benchmark accuracy", "run MMLU", "evaluate quantized model", "accuracy drop", "run nel". Handles deployment, config generation, and evaluation execution. Not for quantizing models (use ptq) or deploying/serving models (use deployment).
license: Apache-2.0
# Based on nel-assistant skill from NeMo Evaluator Launcher (commit f1fa073)
# https://github.com/NVIDIA-NeMo/Evaluator/tree/f1fa073/packages/nemo-evaluator-launcher/.claude/skills/nel-assistant
# Modifications: renamed to evaluation, added workspace management (Step 0),
# auto-detect ModelOpt quantization format, quantization-aware benchmark defaults.
---

## NeMo Evaluator Launcher Assistant

You're an expert in NeMo Evaluator Launcher! Guide the user through creating production-ready YAML configurations, running evaluations, and monitoring progress via an interactive workflow specified below.

### Workspace and Pipeline Integration

If `MODELOPT_WORKSPACE_ROOT` is set, read `skills/common/workspace-management.md`. Check for existing workspaces — especially if evaluating a model from a prior PTQ or deployment step. Reuse the existing workspace so you have access to the quantized checkpoint and any code modifications.

This skill is often the final stage of the PTQ → Deploy → Eval pipeline. If the model required runtime patches during deployment (transformers upgrade, framework source fixes), carry those patches into the NEL config via `deployment.command`.

### Workflow

```text
Config Generation Progress:
- [ ] Step 0: Check workspace (if MODELOPT_WORKSPACE_ROOT is set)
- [ ] Step 1: Check if nel is installed and if user has existing config
- [ ] Step 2: Build the base config file
- [ ] Step 3: Configure model path and parameters
- [ ] Step 4: Fill in remaining missing values
- [ ] Step 5: Confirm tasks (iterative)
- [ ] Step 6: Advanced - Multi-node (Data Parallel)
- [ ] Step 7: Advanced - Interceptors
- [ ] Step 7.5: Check container registry auth (SLURM only)
- [ ] Step 8: Run the evaluation
```

**Step 1: Check prerequisites**

Test that `nel` is installed with `nel --version`. If not, instruct the user to `pip install nemo-evaluator-launcher`.

If the user already has a config file (e.g., "run this config", "evaluate with my-config.yaml"), skip to Step 8. Optionally review it for common issues (missing `???` values, quantization flags) before running.

**Step 2: Build the base config file**

Prompt the user with "I'll ask you 5 questions to build the base config we'll adjust in the next steps". Guide the user through the 5 questions using AskUserQuestion:

1. Execution:

- Local
- SLURM

2. Deployment:

- None (External)
- vLLM
- SGLang
- NIM
- TRT-LLM

3. Auto-export:

- None (auto-export disabled)
- MLflow
- wandb

4. Model type

- Base
- Chat
- Reasoning

5. Benchmarks:
  Allow for multiple choices in this question.
1. Standard LLM Benchmarks (like MMLU, IFEval, GSM8K, ...)
2. Code Evaluation (like HumanEval, MBPP, and LiveCodeBench)
3. Math & Reasoning (like AIME, GPQA, MATH-500, ...)
4. Safety & Security (like Garak and Safety Harness)
5. Multilingual (like MMATH, Global MMLU, MMLU-Prox)

Only accept options from the categories listed above (Execution, Deployment, Auto-export, Model type, Benchmarks). YOU HAVE TO GATHER THE ANSWERS for the 5 questions before you can build the base config.

> **Note:** These categories come from NEL's `build-config` CLI. **Always run `nel skills build-config --help` first** to get the current options — they may differ from this list (e.g., `chat_reasoning` instead of separate `chat`/`reasoning`, `general_knowledge` instead of `standard`). When the CLI's current options differ from this list, prefer the CLI's options.

When you have all the answers, run the script to build the base config:

```bash
nel skills build-config --execution <local|slurm> --deployment <none|vllm|sglang|nim|trtllm> --model_type <base|chat|reasoning> --benchmarks <standard|code|math_reasoning|safety|multilingual> [--export <none|mlflow|wandb>] [--output <OUTPUT>]
```

Where `--output` depends on what the user provides:

- Omit: Uses current directory with auto-generated filename
- Directory: Writes to that directory with auto-generated filename
- File path (*.yaml): Writes to that specific file

It never overwrites existing files.

**Step 3: Configure model path and parameters**

Ask for model path. Determine type:

- Checkpoint path (local directory — starts with `/`, `./`, `../`, `~`, or contains no `/` but exists on disk) → set `deployment.checkpoint_path: <path>` and `deployment.hf_model_handle: null`
- HF handle (e.g., `org/model-name` — contains exactly one `/` and does not exist locally) → set `deployment.hf_model_handle: <handle>` and `deployment.checkpoint_path: null`

**Auto-detect ModelOpt quantization format** (checkpoint paths only):

Check for `hf_quant_config.json` in the checkpoint directory:

```bash
cat <checkpoint_path>/hf_quant_config.json 2>/dev/null
```

If found, read `quantization.quant_algo` and set the correct vLLM/SGLang quantization flag in `deployment.extra_args`:

| `quant_algo` | Flag to add |
|-------------|-------------|
| `FP8` | `--quantization modelopt` |
| `W4A8_AWQ` | `--quantization modelopt` |
| `NVFP4`, `NVFP4_AWQ` | `--quantization modelopt_fp4` |
| Other values | Try `--quantization modelopt`; consult vLLM/SGLang docs if unsure |

If no `hf_quant_config.json`, also check `config.json` for a `quantization_config` section with `quant_method: "modelopt"`. If neither is found, the checkpoint is unquantized — no flag needed.

> **Note:** Some models require additional env vars for deployment (e.g., `VLLM_NVFP4_GEMM_BACKEND=marlin` for Nemotron Super). These are not in `hf_quant_config.json` — they are discovered during model card research below.

**Quantization-aware benchmark defaults:**

When a quantized checkpoint is detected, read `references/quantization-benchmarks.md` for benchmark sensitivity rankings and recommended sets. Present recommendations to the user and ask which to include.

Read `references/model-card-research.md` for the full extraction checklist (sampling params, reasoning config, ARM64 compatibility, pre_cmd, etc.). Use WebSearch to research the model card, present findings, and ask the user to confirm.

**Step 4: Fill in remaining missing values**

- Find all remaining `???` missing values in the config.
- Ask the user only for values that couldn't be auto-discovered from the model card (e.g., SLURM hostname, account, output directory, MLflow/wandb tracking URI). Don't propose any defaults here. Let the user give you the values in plain text.
- Ask the user if they want to change any other defaults e.g. execution partition or walltime (if running on SLURM) or add MLflow/wandb tags (if auto-export enabled).

**Step 5: Confirm tasks (iterative)**

Show tasks in the current config. Loop until the user confirms the task list is final:

1. Tell the user: "Run `nel ls tasks` to see all available tasks".
2. Ask if they want to add/remove tasks or add/remove/modify task-specific parameter overrides.
   To add per-task `nemo_evaluator_config` as specified by the user, e.g.:

   ```yaml
   tasks:
     - name: <task>
       nemo_evaluator_config:
         config:
           params:
             temperature: <value>
             max_new_tokens: <value>
             ...
   ```

3. Apply changes.
4. Show updated list and ask: "Is the task list final, or do you want to make more changes?"

**Known Issues**

- NeMo-Skills workaround (self-deployment only): If using `nemo_skills.*` tasks with self-deployment (vLLM/SGLang/NIM), add at top level:

  ```yaml
  target:
    api_endpoint:
      api_key_name: DUMMY_API_KEY
  ```

  For the None (External) deployment the `api_key_name` should be already defined. The `DUMMY_API_KEY` export is handled in Step 8.

**Step 6: Advanced - Multi-node**

If the user needs multi-node evaluation (model >120B, or more throughput), read `references/multi-node.md` for the configuration patterns (HAProxy multi-instance, Ray TP/PP, or combined).

**Step 7: Advanced - Interceptors**

- Tell the user they should see: <https://docs.nvidia.com/nemo/evaluator/latest/libraries/nemo-evaluator/interceptors/index.html> .
- DON'T provide any general information about what interceptors typically do in API frameworks without reading the docs. If the user asks about interceptors, only then read the webpage to provide precise information.
- If the user asks you to configure some interceptor, then read the webpage of this interceptor and configure it according to the `--overrides` syntax but put the values in the YAML config under `evaluation.nemo_evaluator_config.config.target.api_endpoint.adapter_config` (NOT under `target.api_endpoint.adapter_config`) instead of using CLI overrides.
  By defining `interceptors` list you'd override the full chain of interceptors which can have unintended consequences like disabling default interceptors. That's why use the fields specified in the `CLI Configuration` section after the `--overrides` keyword to configure interceptors in the YAML config.

**Documentation Errata**

- The docs may show incorrect parameter names for logging. Use `max_logged_requests` and `max_logged_responses` (NOT `max_saved_*` or `max_*`).

**Step 7.5: Check container registry authentication (SLURM only)**

NEL's default deployment images by framework:

| Framework | Default image | Registry |
| --- | --- | --- |
| vLLM | `vllm/vllm-openai:latest` | DockerHub |
| SGLang | `lmsysorg/sglang:latest` | DockerHub |
| TRT-LLM | `nvcr.io/nvidia/tensorrt-llm/release:...` | NGC |
| Evaluation tasks | `nvcr.io/nvidia/eval-factory/*:26.03` | NGC |

Before submitting, verify the cluster has credentials for the deployment image. See `skills/common/slurm-setup.md` section 6 for the full procedure.

```bash
ssh <host> "grep -E '^\s*machine\s+' ~/.config/enroot/.credentials 2>/dev/null"
```

**Decision flow (check before submitting):**
1. Check if the cluster has credentials for the default DockerHub image (see command above)
2. If DockerHub credentials exist → use the default image and submit
3. If DockerHub credentials are missing but can be added → add them (see `slurm-setup.md` section 6), then submit
4. If DockerHub credentials cannot be added → override `deployment.image` to the NGC alternative and submit:

   ```yaml
   deployment:
     image: nvcr.io/nvidia/vllm:<YY.MM>-py3  # check https://catalog.ngc.nvidia.com/orgs/nvidia/containers/vllm for latest tag
   ```

5. **Do not retry more than once** without fixing the auth issue

**Step 8: Run the evaluation**

Print the following commands to the user. Propose to execute them in order to confirm the config works as expected before the full run.

**Important**: Export required environment variables based on your config. If any tokens or keys are missing (e.g. `HF_TOKEN`, `NGC_API_KEY`, `api_key_name` from the config), ask the user to put them in a `.env` file in the project root so you can run `set -a && source .env && set +a` (or equivalent) before executing `nel run` commands.

```bash
# If using pre_cmd or post_cmd (review pre_cmd content before enabling — it runs arbitrary commands):
export NEMO_EVALUATOR_TRUST_PRE_CMD=1

# If using nemo_skills.* tasks with self-deployment:
export DUMMY_API_KEY=dummy
```

1. **Dry-run** (validates config without running):

   ```bash
   nel run --config <config_path> --dry-run
   ```

2. **Test with limited samples** (quick validation run):

   ```bash
   nel run --config <config_path> -o ++evaluation.nemo_evaluator_config.config.params.limit_samples=10
   ```

3. **Re-run a single task** (useful for debugging or re-testing after config changes):

   ```bash
   nel run --config <config_path> -t <task_name>
   ```

   Combine with `-o` for limited samples: `nel run --config <config_path> -t <task_name> -o ++evaluation.nemo_evaluator_config.config.params.limit_samples=10`

4. **Full evaluation** (production run):

   ```bash
   nel run --config <config_path>
   ```

After the dry-run, check the output from `nel` for any problems with the config. If there are no problems, propose to first execute the test run with limited samples and then execute the full evaluation. If there are problems, resolve them before executing the full evaluation.

**Monitoring Progress**

After job submission, register the job per the **monitor skill** for durable cross-session tracking. For one-off queries (live status, debugging a failed run, analyzing results) use the **launching-evals skill**; for querying past runs in MLflow use **accessing-mlflow**.

**NEL-specific diagnostics** (for debugging failures):

```bash
# Quick status check
nel status <invocation_id>
nel info <invocation_id>

# Get log paths
nel info <invocation_id> --logs

# Inspect logs via SSH
ssh <user>@<host> "tail -100 <log_path>/server-<slurm_job_id>-*.log"   # deployment errors
ssh <user>@<host> "tail -100 <log_path>/client-<slurm_job_id>.log"     # evaluation errors
ssh <user>@<host> "tail -100 <log_path>/slurm-<slurm_job_id>.log"      # scheduling/walltime
ssh <user>@<host> "grep -i 'error\|failed' <log_path>/*.log"           # search all logs
```

---

Direct users with issues to:

- **GitHub Issues:** <https://github.com/NVIDIA-NeMo/Evaluator/issues>
- **GitHub Discussions:** <https://github.com/NVIDIA-NeMo/Evaluator/discussions>

Now, copy this checklist and track your progress:

```text
Config Generation Progress:
- [ ] Step 0: Check workspace (if MODELOPT_WORKSPACE_ROOT is set)
- [ ] Step 1: Check if nel is installed and if user has existing config
- [ ] Step 2: Build the base config file
- [ ] Step 3: Configure model path and parameters
- [ ] Step 4: Fill in remaining missing values
- [ ] Step 5: Confirm tasks (iterative)
- [ ] Step 6: Advanced - Multi-node (Data Parallel)
- [ ] Step 7: Advanced - Interceptors
- [ ] Step 7.5: Check container registry auth (SLURM only)
- [ ] Step 8: Run the evaluation
```
