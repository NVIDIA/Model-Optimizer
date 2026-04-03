---
name: modelopt
description: End-to-end model optimization pipeline that chains quantization with deployment or evaluation. Use when user says "optimize model end-to-end", "quantize and deploy", "quantize and serve", "quantize and evaluate", "quantize and benchmark accuracy", "full optimization loop", "run the full pipeline", "optimize and test accuracy", "find best quantization recipe", or wants to go from a pretrained model to a deployed or accuracy-verified quantized model. Do NOT use for individual tasks like only quantizing (use ptq), only deploying (use deployment), or only evaluating (use evaluation).
license: Apache-2.0
---

# ModelOpt Optimizer — Pipeline Orchestrator

Orchestrates optimization pipelines by chaining skills. Supports two modes:

1. **PTQ + Deploy** — quantize then serve as an API endpoint
2. **PTQ + Evaluate** — quantize then benchmark accuracy (evaluation handles deployment internally)

This skill delegates to sub-skills. **Do not duplicate their logic — invoke them.**

## Workspace Management

If `MODELOPT_WORKSPACE_ROOT` is set (multi-user / Slack bot), read `skills/common/workspace-management.md` first. **All sub-skills in the pipeline must run in the same workspace** so they share the checkpoint and any code modifications. Create or reuse a workspace named after the model (e.g., `qwen3-0.6b`, `llama-3.1-8b-fp8`) before invoking any sub-skill.

## Pipeline Selection

Determine which pipeline the user needs:

| User says | Pipeline |
|-----------|----------|
| "quantize and deploy", "quantize and serve" | PTQ + Deploy |
| "quantize and evaluate", "optimize end-to-end", "find best recipe" | PTQ + Evaluate |

If the user only wants quantization without deploy/eval, the `ptq` skill handles it directly — this skill should not be used.

If unclear, ask: **"After quantization, do you want to (a) deploy the model as a server, (b) evaluate accuracy, or (c) just get the checkpoint?"** If they answer (c), hand off to the `ptq` skill.

## Step 1: Gather Info

Collect from the user (skip what's already provided):

1. **Model path** — local path or HuggingFace model ID (save this for baseline comparison in Step 4)
2. **Quantization format** — e.g., fp8, nvfp4, int4_awq (or "recommend one")
3. **Execution target** — local GPU or remote cluster. Check for `~/.config/modelopt/clusters.yaml` or `.claude/clusters.yaml`. If found, ask which cluster to use. Both sub-skills support remote execution via `remote_exec.sh`.
4. **GPU IDs** — which GPUs to use (default: `0`; skip if remote — sub-skills handle GPU allocation via SLURM)
5. For Deploy pipeline: **Deployment framework** — vLLM, SGLang, or TRT-LLM (default: vLLM)
6. For Evaluate pipeline: **Evaluation tasks** — default: `mmlu`

## Step 2: Quantize

**Invoke the `ptq` skill.** It handles environment detection, model compatibility, format selection, job submission, and checkpoint verification.

Input: model path, quantization format, export path, GPU IDs.
Output: quantized checkpoint at export path.

## Step 3: Deploy or Evaluate

### PTQ + Deploy

**Invoke the `deployment` skill.** It starts an inference server with the quantized checkpoint.

Input: checkpoint path, framework, GPU IDs, port.
Output: running server at `http://localhost:<port>`.

### PTQ + Evaluate

**Invoke the `evaluation` skill.** It handles deploying the quantized model, configuring NEL evaluation, running benchmarks, and collecting results.

Input: quantized checkpoint path, evaluation tasks.
Output: accuracy scores per task.

## Step 4: Baseline Comparison (PTQ + Evaluate only)

After evaluation completes, ask: **"Would you like to compare against the unquantized baseline?"**

If yes:
1. Run the evaluation skill again with the **original model path** (from Step 1) and the same benchmark tasks
2. Present a side-by-side comparison table:

```text
| Benchmark | BF16 (baseline) | FP8 (quantized) | Delta |
|-----------|-----------------|-----------------|-------|
| MMLU      | 67.3%           | 65.2%           | -2.1% |
| GSM8K     | 54.1%           | 52.8%           | -1.3% |
```

3. Flag any benchmark with >2% accuracy drop — suggest trying a lighter quantization format

## Step 5: Present Results and Iterate

Show results and ask: **"Are you satisfied with these results?"**

- **Yes** — Done. Report final model path and summary.
- **No** — Propose a different recipe (lighter or heavier quantization), loop to Step 2.
- **Quit** — Report partial results. Clean up any running servers.
