---
name: modelopt
description: End-to-end model optimization pipeline that chains quantization and evaluation. Use when user says "optimize model end-to-end", "quantize and evaluate", "quantize and benchmark accuracy", "full optimization loop", "find best quantization recipe", or wants to go from a pretrained model to an accuracy-verified quantized model in one workflow. Do NOT use for individual tasks like only quantizing (use ptq) or only evaluating (use evaluation).
---

# ModelOpt Optimizer — Full Pipeline

Orchestrates the complete optimization workflow: quantize a model, evaluate accuracy, and iterate until the user is satisfied.

This skill delegates to sub-skills. **Do not duplicate their logic — invoke them.**

## When to Use

- User wants the full loop: quantize + evaluate
- User wants to compare recipes ("try FP8, then NVFP4, pick the best")
- User says "optimize", "end-to-end", or "quantize and benchmark"

## Workflow

```text
Step 1: Gather info
Step 2: Quantize         → invoke ptq skill
Step 3: Evaluate         → invoke evaluation skill
Step 4: Present results
Step 5: Iterate or finish
```

### Step 1: Gather Info

Collect from the user (skip what's already provided):

1. **Model path** — local path or HuggingFace model ID
2. **Quantization format** — e.g., fp8, nvfp4, int4_awq (or "recommend one")
3. **Evaluation tasks** — default: `mmlu`
4. **GPU IDs** — which GPUs to use (default: `0`)

### Step 2: Quantize

**Invoke the `ptq` skill.** It handles environment detection, model compatibility, format selection, job submission, and checkpoint verification.

Input: model path, quantization format, export path, GPU IDs.
Output: quantized checkpoint at export path.

### Step 3: Evaluate

**Invoke the `evaluation` skill.** It handles deploying the quantized model, configuring NEL evaluation, running benchmarks, and collecting results.

Input: quantized checkpoint path, evaluation tasks.
Output: accuracy scores per task.

### Step 4: Present Results

Show the user a combined summary:

```text
============================================================
  OPTIMIZATION RESULTS: <model_name> (<format>)
============================================================
  ACCURACY:
    mmlu:              0.XXXX

  EXPORT PATH: <export_path>
============================================================
```

### Step 5: Iterate or Finish

Ask: **"Are you satisfied with these results?"**

- **Yes** — Done. Report final model path and summary.
- **No** — Propose a different recipe (lighter or heavier), loop to Step 2.
- **Quit** — Report partial results.
