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
- [ ] Step 7.5: Check container registry auth for private images (SLURM only)
- [ ] Step 8: Run the evaluation
- [ ] Step 9: Verify evaluation comparability
```

**Step 1: Check prerequisites**

Test that `nel` is installed with `nel --version`. If not, instruct the user to `pip install nemo-evaluator-launcher`.

If the user already has a config file (e.g., "run this config", "evaluate with my-config.yaml"), skip to Step 8. Optionally review it for common issues (missing `???` values, quantization flags) before running.

**Shortcut: use pre-built task snippets.** If the user asks for a specific benchmark (e.g., "run MMLU-Pro", "evaluate with AIME"), check `recipes/tasks/` (relative to this skill's directory) for a matching task snippet. Available: mmlu_pro, gpqa, aime2025, livecodebench, ifbench, scicode. Task snippets contain only the task-specific config (name, params, repeats) — not the full NEL config. To use them:

1. Read the task snippet(s) the user wants
2. Use `recipes/examples/example_eval.yaml` as the base config template
3. Replace the `tasks:` section with the selected snippet(s)
4. Do Step 3 (auto-detect model settings from checkpoint) and Step 4 (fill in `???` values)
5. Proceed to Step 7.5/8

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

Prefer vLLM for NEL self-deployment unless the user explicitly asks for another runtime, the model card requires another runtime, or the evaluation targets an already-running endpoint (`deployment: none`).

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

Check `config.json` first for a `quantization_config` section with `quant_method: "modelopt"`. If absent, check the legacy/backward-compatible `hf_quant_config.json`:

```bash
cat <checkpoint_path>/config.json 2>/dev/null
cat <checkpoint_path>/hf_quant_config.json 2>/dev/null
```

If ModelOpt quantization is detected, read the quantization algorithm from `quantization_config.quant_algo` or `quantization.quant_algo`.

- **vLLM:** Do not add a `--quantization` flag by default. Recent vLLM reads `quantization_config` / `hf_quant_config.json` and selects the ModelOpt backend automatically; adding a stale or mismatched flag can cause a config mismatch. Only add an explicit flag if the model card, vLLM version, or dry-run error requires it.
- **SGLang:** Use SGLang-specific docs/model-card guidance. For offline ModelOpt checkpoints, recent SGLang can parse the config in many cases; if an explicit flag is required, common values are `--quantization modelopt_fp8` for FP8 and `--quantization modelopt_fp4` for NVFP4. Some exported ModelOpt flows document `--quantization modelopt`; verify against the installed SGLang version.

If neither file contains a ModelOpt quantization config, treat the checkpoint as unquantized — no quantization flag needed.

> **Note:** Some models require additional env vars for deployment (e.g., `VLLM_NVFP4_GEMM_BACKEND=marlin` for Nemotron Super). These may not be in the quantization config files — they are discovered during model card research below.

**Auto-detect deployment settings from checkpoint:**

Read `config.json` from the checkpoint (or HF model card) and build `deployment.extra_args` dynamically:

```bash
cat <checkpoint_path>/config.json 2>/dev/null
```

| Field in `config.json` | What to set | Example |
| --- | --- | --- |
| `max_position_embeddings` | `--max-model-len <value>` | `131072` → `--max-model-len 131072` |
| `auto_map` exists | `--trust-remote-code` | Only add if model has custom code |

Then use WebSearch to check the model card (HuggingFace page) for deployment-specific settings:

| Model card signal | What to set |
| --- | --- |
| Reasoning model (thinking/CoT) | `--reasoning-parser` and `--reasoning-parser-plugin` if a custom parser is provided |
| Tool-calling support | `--enable-auto-tool-choice --tool-call-parser <parser>` |
| Custom vLLM flags documented | Add as specified (e.g., `--mamba_ssm_cache_dtype float32`) |

Combine all detected flags into a single `deployment.extra_args` override. The recipe's default `--max-model-len 32768` is a fallback — always prefer the value from `config.json`.

**Quantization-aware benchmark defaults:**

When a quantized checkpoint is detected, read `references/quantization-benchmarks.md` for benchmark sensitivity rankings and recommended sets. Present recommendations to the user and ask which to include.

**Baseline comparison preflight:**

When a quantized checkpoint is detected, identify the matching baseline before launching the full quantized run. The baseline is usually the pre-quantization source model/checkpoint for this run, but it may itself be quantized (for example, an FP8 checkpoint used as the baseline for an NVFP4 checkpoint). First infer the baseline from the PTQ source model/checkpoint in the workspace or config used to create the quantized checkpoint. If it cannot be inferred, ask the user for the baseline model/checkpoint or an existing baseline invocation/run path. If no matching baseline exists, prepare a companion baseline config and launch it before or alongside the quantized config. The baseline config should match the quantized config's benchmark versions, task configs, serving args, token limits, dataset setup, credentials, cluster, and container as closely as possible; change only the model/checkpoint and adjust quantization-specific flags to match the baseline checkpoint. Do not treat the quantized score as release-ready until the baseline comparison exists.

Read `references/model-card-research.md` for the full extraction checklist (sampling params, reasoning config, ARM64 compatibility, pre_cmd, etc.). Use WebSearch to research the model card, present findings, and ask the user to confirm.

For reasoning-capable models, prefer reasoning mode for evaluation because it usually produces the highest task scores; configure the model-card-specific on/off control and any reasoning budget or effort setting. If the user wants lower variance/noise, lower latency/cost, or an apples-to-apples comparison against non-reasoning baselines, also consider a non-reasoning companion run.

**Step 4: Fill in remaining missing values**

- Find all remaining `???` missing values in the config.
- Ask the user only for values that couldn't be auto-discovered from the model card (e.g., SLURM hostname, account, output directory, MLflow/wandb tracking URI). Don't propose any defaults here. Let the user give you the values in plain text.
- Ask the user if they want to change any other defaults e.g. execution partition or walltime (if running on SLURM) or add MLflow/wandb tags (if auto-export enabled).

**Step 5: Confirm tasks (iterative)**

Show tasks in the current config. Loop until the user confirms the task list is final:

1. Tell the user: "Run `nel ls tasks` to see all available tasks".
2. If the task list includes a benchmark with a pre-built snippet in `recipes/tasks/`,
   prefer that snippet over hand-written task overrides unless the user explicitly asks
   for different sampling or prompt settings. For reasoning-mode comparisons, keep the
   recipe repeat counts; for tasks without a recipe, use `num_repeats >= 3` when the
   benchmark supports repeats.
3. Ask if they want to add/remove tasks or add/remove/modify task-specific parameter overrides.
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

4. Apply changes.
5. Show updated list and ask: "Is the task list final, or do you want to make more changes?"

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

**Step 7.5: Check container registry authentication for private images (SLURM only)**

NEL's default deployment images by framework:

| Framework | Default image | Registry |
| --- | --- | --- |
| vLLM | `vllm/vllm-openai:latest` | DockerHub |
| SGLang | `lmsysorg/sglang:latest` | DockerHub |
| TRT-LLM | `nvcr.io/nvidia/tensorrt-llm/release:...` | NGC |
| Evaluation tasks | `nvcr.io/nvidia/eval-factory/*:26.03` | NGC |

Before submitting, identify the exact deployment and evaluation-task images that will be pulled. If the images are public, skip the registry-authentication preflight; pyxis/enroot can pull public images without stored credentials. Do not require credentials just because the registry is DockerHub or NGC.

Only verify cluster credentials when an image is private or access-restricted (private DockerHub repo, private NGC repo, internal registry, or user-provided image that is not known to be public). See `skills/common/slurm-setup.md` section 6 for the credential setup procedure.

```bash
ssh <host> "grep -E '^\s*machine\s+' ~/.config/enroot/.credentials 2>/dev/null"
```

**Decision flow (check before submitting):**
1. If the selected images are public → submit without an auth preflight
2. If any selected image is private or access-restricted → check for credentials for that image's registry (see command above)
3. If credentials exist → use the selected image and submit
4. If credentials are missing but can be added → add them (see `slurm-setup.md` section 6), then submit
5. If credentials cannot be added → switch to a public image when a compatible one exists, for example:

   ```yaml
   deployment:
     image: nvcr.io/nvidia/vllm:<YY.MM>-py3  # check https://catalog.ngc.nvidia.com/orgs/nvidia/containers/vllm for latest tag
   ```

6. **Do not retry more than once** after an auth failure without fixing credentials or switching images

**Step 8: Run the evaluation**

Print the following commands to the user. Propose to execute them in order to confirm the config works as expected before the full run.

**Important**: Export required environment variables based on your config. If any tokens or keys are missing, point the user to `recipes/env.example` — it lists all possible keys with notes on which tasks need them. Ask the user to copy it, fill in their keys, and source it:

```bash
cp recipes/env.example .env
# Edit .env with your keys
set -a && source .env && set +a
```

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

**Step 9: Verify evaluation comparability**

Before treating a baseline-vs-quantized delta as a model quality result, verify the two runs are comparable:

1. Confirm the prompt text and chat template/rendered messages match between the baseline and quantized evaluations.
2. Confirm generation settings match, including temperature, top_p, top_k, max tokens, stop strings, reasoning mode/budget, and any task-specific overrides.
3. If reasoning traces are present, confirm they are stripped or ignored before scoring, consistently for both runs.
4. Confirm the number of evaluated samples matches for each task and split.
5. Confirm the same accuracy metric/score field is used for the baseline and quantized comparison.

If any item differs, either rerun with matched settings or label the result as not an apples-to-apples quantization comparison.

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
- [ ] Step 7.5: Check container registry auth for private images (SLURM only)
- [ ] Step 8: Run the evaluation
- [ ] Step 9: Verify evaluation comparability
```
