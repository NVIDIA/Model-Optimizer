---
name: modelopt
description: End-to-end model optimization pipeline that chains quantization with deployment or evaluation. Use when user says "optimize model end-to-end", "quantize and deploy", "quantize and evaluate", "quantize and benchmark accuracy", "full optimization loop", "find best quantization recipe", or wants to go from a pretrained model to a deployed or accuracy-verified quantized model. Do NOT use for individual tasks like only quantizing (use ptq), only deploying (use deployment), or only evaluating (use evaluation).
---

# ModelOpt Optimizer — Pipeline Orchestrator

Orchestrates optimization pipelines by chaining skills. Supports three modes:

1. **PTQ only** — quantize a model
2. **PTQ + Deploy** — quantize then serve as an API endpoint
3. **PTQ + Evaluate** — quantize then benchmark accuracy (evaluation handles deployment internally)

This skill delegates to sub-skills. **Do not duplicate their logic — invoke them.**

## Pipeline Selection

Determine which pipeline the user needs:

| User says | Pipeline |
|-----------|----------|
| "quantize", "run PTQ" (no mention of deploy/eval) | PTQ only → invoke `ptq` skill directly |
| "quantize and deploy", "quantize and serve" | PTQ + Deploy |
| "quantize and evaluate", "optimize end-to-end", "find best recipe" | PTQ + Evaluate |

If unclear, ask: **"After quantization, do you want to (a) deploy the model as a server, (b) evaluate accuracy, or (c) just get the checkpoint?"**

## Step 1: Gather Info

Collect from the user (skip what's already provided):

1. **Model path** — local path or HuggingFace model ID
2. **Quantization format** — e.g., fp8, nvfp4, int4_awq (or "recommend one")
3. **GPU IDs** — which GPUs to use (default: `0`)
4. For Deploy pipeline: **Deployment framework** — vLLM, SGLang, or TRT-LLM (default: vLLM)
5. For Evaluate pipeline: **Evaluation tasks** — default: `mmlu`

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

## Step 4: Present Results and Iterate

Show results and ask: **"Are you satisfied with these results?"**

- **Yes** — Done. Report final model path and summary.
- **No** — Propose a different recipe (lighter or heavier), loop to Step 2.
- **Quit** — Report partial results. Clean up any running servers.
