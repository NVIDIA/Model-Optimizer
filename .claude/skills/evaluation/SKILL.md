---
name: evaluation
description: Evaluate accuracy of quantized or unquantized LLMs using NeMo Evaluator Launcher (NEL). Use when user says "evaluate model", "benchmark accuracy", "run MMLU", "evaluate quantized model", "accuracy drop", "run nel", or needs to measure how quantization affects model quality. Handles model deployment, config generation, and evaluation execution.
license: Apache-2.0
# Based on nel-assistant skill from NeMo Evaluator Launcher (commit f1fa073)
# https://github.com/NVIDIA-NeMo/Evaluator/tree/f1fa073/packages/nemo-evaluator-launcher/.claude/skills/nel-assistant
# Modifications: renamed to evaluation, added workspace management (Step 0),
# auto-detect ModelOpt quantization format, quantization-aware benchmark defaults.
---

## NeMo Evaluator Launcher Assistant

You're an expert in NeMo Evaluator Launcher! Guide the user through creating production-ready YAML configurations, running evaluations, and monitoring progress via an interactive workflow specified below.

### Workspace (multi-user / Slack bot)

If `MODELOPT_WORKSPACE_ROOT` is set, read `skills/common/workspace-management.md`. Check for existing workspaces — especially if evaluating a model from a prior PTQ or deployment step. Reuse the existing workspace so you have access to the quantized checkpoint and any code modifications.

### Workflow

```text
Config Generation Progress:
- [ ] Step 0: Check workspace (if MODELOPT_WORKSPACE_ROOT is set)
- [ ] Step 1: Check if nel is installed
- [ ] Step 2: Build the base config file
- [ ] Step 3: Configure model path and parameters
- [ ] Step 4: Fill in remaining missing values
- [ ] Step 5: Confirm tasks (iterative)
- [ ] Step 6: Advanced - Multi-node (Data Parallel)
- [ ] Step 7: Advanced - Interceptors
- [ ] Step 8: Run the evaluation
```

**Step 1: Check if nel is installed**

Test that `nel` is installed with `nel --version`.

If not, instruct the user to `pip install nemo-evaluator-launcher`.

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

DON'T ALLOW FOR ANY OTHER OPTIONS, only the ones listed above under each category (Execution, Deployment, Auto-export, Model type, Benchmarks). YOU HAVE TO GATHER THE ANSWERS for the 5 questions before you can build the base config.

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

- Checkpoint path (starts with `/` or `./`) → set `deployment.checkpoint_path: <path>` and `deployment.hf_model_handle: null`
- HF handle (e.g., `org/model-name`) → set `deployment.hf_model_handle: <handle>` and `deployment.checkpoint_path: null`

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

If no `hf_quant_config.json`, the checkpoint is unquantized — no flag needed.

**Quantization-aware benchmark defaults:**

When a quantized checkpoint is detected, recommend benchmarks sensitive to quantization accuracy loss:

- **Always include**: MMLU (general knowledge, most affected by quantization)
- **Recommended**: GSM8K (math reasoning — sensitive to precision loss), ARC-Challenge (reasoning)
- **Good to add**: HumanEval (code generation — catches subtle degradation), Winogrande (commonsense)
- **Less useful for quant comparison**: IFEval (instruction following — rarely affected by quantization)

Present these recommendations to the user and ask which to include. If the user already specified benchmarks, keep their choice but mention any accuracy-sensitive benchmarks they may have missed.

Use WebSearch to find model card (HuggingFace, build.nvidia.com). Read it carefully, the FULL text, the devil is in the details. Extract ALL relevant configurations:

- Sampling params (`temperature`, `top_p`)
- Context length (`deployment.extra_args: "--max-model-len <value>"`)
- TP/DP settings (to set them appropriately, AskUserQuestion on how many GPUs the model will be deployed)
- Reasoning config (if applicable):
  - reasoning on/off: use either:
    - `adapter_config.custom_system_prompt` (like `/think`, `/no_think`) and no `adapter_config.params_to_add` (leave `params_to_add` unrelated to reasoning untouched)
    - `adapter_config.params_to_add` for payload modifier (like `"chat_template_kwargs": {"enable_thinking": true/false}`) and no `adapter_config.custom_system_prompt` and `adapter_config.use_system_prompt: false` (leave `custom_system_prompt` and `use_system_prompt` unrelated to reasoning untouched).
  - reasoning effort/budget (if it's configurable, AskUserQuestion what reasoning effort they want)
  - higher `max_new_tokens`
  - etc.
- Deployment-specific `extra_args` for vLLM/SGLang (look for the vLLM/SGLang deployment command)
- Deployment-specific vLLM/SGLang versions (by default we use latest docker images, but you can control it with `deployment.image` e.g. vLLM above `vllm/vllm-openai:v0.11.0` stopped supporting `rope-scaling` arg used by Qwen models)
- ARM64 / non-standard GPU compatibility: The default `vllm/vllm-openai` image only supports common GPU architectures. For ARM64 platforms or GPUs with non-standard compute capabilities (e.g., NVIDIA GB10 with sm_121), use NGC vLLM images instead:
  - Example: `deployment.image: nvcr.io/nvidia/vllm:26.01-py3`
  - AskUserQuestion about their GPU architecture if the model card doesn't specify deployment constraints
- Any preparation requirements (e.g., downloading reasoning parsers, custom plugins):
  - If the model card mentions downloading files (like reasoning parsers, custom plugins) before deployment, add `deployment.pre_cmd` with the download command
  - Use `curl` instead of `wget` as it's more widely available in Docker containers
  - Example: `pre_cmd: curl -L -o reasoning_parser.py https://huggingface.co/.../reasoning_parser.py`
  - When using `pip install` in `pre_cmd`, always use `--no-cache-dir` to avoid cross-device link errors in Docker containers (the pip cache and temp directories may be on different filesystems)
  - Example: `pre_cmd: pip3 install --no-cache-dir flash-attn --no-build-isolation`
- Any other model-specific requirements

Remember to check `evaluation.nemo_evaluator_config` and `evaluation.tasks.*.nemo_evaluator_config` overrides too for parameters to adjust (e.g. disabling reasoning)!

Present findings, explain each setting, ask user to confirm or adjust. If no model card found, ask user directly for the above configurations.

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

There are two multi-node patterns. Ask the user which applies:

**Pattern A: Multi-instance (independent instances with HAProxy)**

Only if model >120B parameters or user wants more throughput. Explain: "Each node runs an independent deployment instance. HAProxy load-balances requests across all instances."

```yaml
execution:
    num_nodes: 4       # Total nodes
    num_instances: 4   # 4 independent instances → HAProxy auto-enabled
```

**Pattern B: Multi-node single instance (Ray TP/PP across nodes)**

When a single model is too large for one node and needs pipeline parallelism across nodes. Use `vllm_ray` deployment config:

```yaml
defaults:
  - deployment: vllm_ray   # Built-in Ray cluster setup (replaces manual pre_cmd)

execution:
    num_nodes: 2           # Single instance spanning 2 nodes

deployment:
    tensor_parallel_size: 8
    pipeline_parallel_size: 2
```

**Pattern A+B combined: Multi-instance with multi-node instances**

For very large models needing both cross-node parallelism AND multiple instances:

```yaml
defaults:
  - deployment: vllm_ray

execution:
    num_nodes: 4       # Total nodes
    num_instances: 2   # 2 instances of 2 nodes each → HAProxy auto-enabled

deployment:
    tensor_parallel_size: 8
    pipeline_parallel_size: 2
```

**Common Confusions**

- **`num_instances`** controls independent deployment instances with HAProxy. **`data_parallel_size`** controls DP replicas *within* a single instance.
- Global data parallelism is `num_instances x data_parallel_size` (e.g., 2 instances x 8 DP each = 16 replicas).
- With multi-instance, `parallelism` in task config is the total concurrent requests across all instances, not per-instance.
- `num_nodes` must be divisible by `num_instances`.

**Step 7: Advanced - Interceptors**

- Tell the user they should see: <https://docs.nvidia.com/nemo/evaluator/latest/libraries/nemo-evaluator/interceptors/index.html> .
- DON'T provide any general information about what interceptors typically do in API frameworks without reading the docs. If the user asks about interceptors, only then read the webpage to provide precise information.
- If the user asks you to configure some interceptor, then read the webpage of this interceptor and configure it according to the `--overrides` syntax but put the values in the YAML config under `evaluation.nemo_evaluator_config.config.target.api_endpoint.adapter_config` (NOT under `target.api_endpoint.adapter_config`) instead of using CLI overrides.
  By defining `interceptors` list you'd override the full chain of interceptors which can have unintended consequences like disabling default interceptors. That's why use the fields specified in the `CLI Configuration` section after the `--overrides` keyword to configure interceptors in the YAML config.

**Documentation Errata**

- The docs may show incorrect parameter names for logging. Use `max_logged_requests` and `max_logged_responses` (NOT `max_saved_*` or `max_*`).

**Step 8: Run the evaluation**

Print the following commands to the user. Propose to execute them in order to confirm the config works as expected before the full run.

**Important**: Export required environment variables based on your config. If any tokens or keys are missing (e.g. `HF_TOKEN`, `NGC_API_KEY`, `api_key_name` from the config), ask the user to put them in a `.env` file in the project root so you can run `set -a && source .env && set +a` (or equivalent) before executing `nel run` commands.

```bash
# If using pre_cmd or post_cmd:
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

After job submission, you can monitor progress using:

1. **Check job status:**

   ```bash
   nel status <invocation_id>
   nel info <invocation_id>
   ```

2. **Stream logs** (Local execution only):

   ```bash
   nel logs <invocation_id>
   ```

   Note: `nel logs` is not supported for SLURM execution.

3. **Inspect logs via SSH** (SLURM workaround):

   When `nel logs` is unavailable (SLURM), use SSH to inspect logs directly:

   First, get log locations:

   ```bash
   nel info <invocation_id> --logs
   ```

   Then, use SSH to view logs:

   **Check server deployment logs:**

   ```bash
   ssh <username>@<hostname> "tail -100 <log path from `nel info <invocation_id> --logs`>/server-<slurm_job_id>-*.log"
   ```

   Shows vLLM server startup, model loading, and deployment errors (e.g., missing wget/curl).

   **Check evaluation client logs:**

   ```bash
   ssh <username>@<hostname> "tail -100 <log path from `nel info <invocation_id> --logs`>/client-<slurm_job_id>.log"
   ```

   Shows evaluation progress, task execution, and results.

   **Check SLURM scheduler logs:**

   ```bash
   ssh <username>@<hostname> "tail -100 <log path from `nel info <invocation_id> --logs`>/slurm-<slurm_job_id>.log"
   ```

   Shows job scheduling, health checks, and overall execution flow.

   **Search for errors:**

   ```bash
   ssh <username>@<hostname> "grep -i 'error\|warning\|failed' <log path from `nel info <invocation_id> --logs`>/*.log"
   ```

---

Direct users with issues to:

- **GitHub Issues:** <https://github.com/NVIDIA-NeMo/Evaluator/issues>
- **GitHub Discussions:** <https://github.com/NVIDIA-NeMo/Evaluator/discussions>

Now, copy this checklist and track your progress:

```text
Config Generation Progress:
- [ ] Step 0: Check workspace (if multi-user)
- [ ] Step 1: Check if nel is installed
- [ ] Step 2: Build the base config file
- [ ] Step 3: Configure model path and parameters
- [ ] Step 4: Fill in remaining missing values
- [ ] Step 5: Confirm tasks (iterative)
- [ ] Step 6: Advanced - Multi-node (Data Parallel)
- [ ] Step 7: Advanced - Interceptors
- [ ] Step 8: Run the evaluation
```
