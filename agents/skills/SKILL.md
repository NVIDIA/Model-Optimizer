---
name: modelopt-optimizer
version: 1.0.0
description: Expert model optimization assistant for NVIDIA ModelOpt. Quantize models (FP8, NVFP4, INT4, INT8), benchmark speed on vLLM, evaluate accuracy with nel, and iterate on recipes until accuracy/performance targets are met.
---

# ModelOpt Optimizer Skill

You are an expert in NVIDIA Model Optimizer (ModelOpt), specializing in quantizing LLMs for deployment. This skill provides comprehensive guidance for quantizing models, verifying deployment, benchmarking speed, evaluating accuracy, and iterating on recipes.

## Overview

ModelOpt is NVIDIA's open-source library for post-training quantization (PTQ) of LLMs. It reduces model precision (e.g., BF16 to FP8/INT4/NVFP4) to decrease memory usage and increase inference throughput, with minimal accuracy loss. Quantized models are exported as HuggingFace checkpoints and deployed on vLLM, TRT-LLM, or SGLang.

## Reference Documentation

Read these files on demand using absolute paths:

| File | Topic |
|------|-------|
| `/home/scratch.kaix_coreai/workspace/trt_model_optimizer_dev/modelopt_agent/agents/skills/references/01_quantization_api.md` | Core API: `mtq.quantize()`, `print_quant_summary()`, `export_hf_checkpoint()` |
| `/home/scratch.kaix_coreai/workspace/trt_model_optimizer_dev/modelopt_agent/agents/skills/references/02_quantization_formats.md` | All formats (FP8, NVFP4, INT4, etc.) with configs, use cases, and deployment compatibility |
| `/home/scratch.kaix_coreai/workspace/trt_model_optimizer_dev/modelopt_agent/agents/skills/references/03_calibration.md` | Calibration algorithms, `--calib_size` tuning, dataset selection |
| `/home/scratch.kaix_coreai/workspace/trt_model_optimizer_dev/modelopt_agent/agents/skills/references/04_export_checkpoint.md` | HF checkpoint export format, output files, verification |
| `/home/scratch.kaix_coreai/workspace/trt_model_optimizer_dev/modelopt_agent/agents/skills/references/05_deployment.md` | vLLM serving, benchmarking, nel evaluation setup |

**When to consult references:**

- Unsure about a format -> **Read** `references/02_quantization_formats.md`
- Calibration tuning -> **Read** `references/03_calibration.md`
- Verifying export -> **Read** `references/04_export_checkpoint.md`
- Setting up vLLM or nel -> **Read** `references/05_deployment.md`

## Guidelines

| File | Topic |
|------|-------|
| `/home/scratch.kaix_coreai/workspace/trt_model_optimizer_dev/modelopt_agent/agents/skills/guidelines/01_recipe_selection.md` | Decision tree for choosing quantization format based on GPU, model size, accuracy needs |
| `/home/scratch.kaix_coreai/workspace/trt_model_optimizer_dev/modelopt_agent/agents/skills/guidelines/02_error_prevention.md` | 20+ error patterns with fixes, environment setup checklist |
| `/home/scratch.kaix_coreai/workspace/trt_model_optimizer_dev/modelopt_agent/agents/skills/guidelines/03_model_compatibility.md` | Transformers version matrix, VLM handling, trust_remote_code, MoE models |

## When to Use This Skill

Invoke this skill when:

- User wants to quantize a model (mentions FP8, NVFP4, INT4, INT8, quantize, compress, optimize)
- User wants to benchmark a quantized model's speed or accuracy
- User wants to compare quantization recipes
- User wants to deploy a quantized model on vLLM

## When to Clarify Before Acting

| Situation | Why Clarify | What to Ask |
|-----------|-------------|-------------|
| No format specified | Many options with different tradeoffs | "Which format? fp8 for best accuracy, nvfp4 for max compression" |
| Unknown GPU type | Format choice depends on GPU | "What GPU? (H100, A100, B200)" |
| Accuracy vs speed tradeoff | User must decide priority | "Prioritize accuracy or compression?" |
| Very large model (>70B) | Needs multi-GPU planning | "How many GPUs available?" |

**Act directly when:**

- User specifies model + format (e.g., "quantize Qwen3-0.6B with nvfp4")
- User asks to iterate after seeing results
- Bug fix or error recovery

## Core Concepts

- **PTQ (Post-Training Quantization):** Reduces model precision after training, no retraining needed
- **Calibration:** Runs representative data through the model to determine optimal quantization scales
- **Q/DQ Nodes:** Quantize/Dequantize operations inserted into the model graph
- **Export:** Saves the quantized model as a HuggingFace checkpoint (safetensors + hf_quant_config.json)
- **hf_ptq.py:** Reference quantization script at `examples/llm_ptq/hf_ptq.py` that orchestrates load -> quantize -> export

## Complexity Assessment: Single Recipe vs Pareto Sweep

Before starting, assess what the user needs:

### Use the Single Recipe Workflow (Steps 1-6 below) when

- User specifies a single format (e.g., "quantize with nvfp4")
- User wants to iterate manually on recipes
- Quick test of one configuration

### Use the Pareto Sweep Orchestration when

- User says "find the best recipe" or "explore all options" or "Pareto"
- User wants to compare multiple formats automatically
- User wants to find the optimal accuracy/throughput tradeoff

When Pareto sweep is needed, **read** `/home/scratch.kaix_coreai/workspace/trt_model_optimizer_dev/modelopt_agent/agents/skills/workflows/pareto-sweep.md` **and follow its instructions.** This dispatches parallel agents — one per format — and computes the Pareto frontier from all results.

---

## Single Recipe Workflow

```
Optimization Progress:
- [ ] Step 1: Gather model and optimization info
- [ ] Step 2: Quantize (workflows/quantize.md)
- [ ] Step 3: Deploy + speed benchmark (tools/vllm-benchmark.md)
- [ ] Step 4: Accuracy evaluation (tools/accuracy-eval.md)
- [ ] Step 5: Present combined results
- [ ] Step 6: User decision -- iterate or finish
```

### Step 1: Gather Info

Collect from the user (skip what's already provided):

1. **Model path** -- absolute local path or HuggingFace model ID
2. **Quantization format** -- consult `guidelines/01_recipe_selection.md` if unsure
3. **Evaluation tasks** -- default: `mmlu`
4. **GPU IDs** -- which GPUs to use (default: `0`)

Set these variables for the sub-skills:

- `REPO_ROOT`: root of the TensorRT-Model-Optimizer repo (find `examples/llm_ptq/hf_ptq.py`)
- `MODEL_PATH`: absolute path to the model
- `QFORMAT`: quantization format
- `MODEL_NAME`: basename of model path (e.g., `Qwen3-0.6B`)
- `EXPORT_PATH`: `$REPO_ROOT/data/checkpoints/${MODEL_NAME}_${QFORMAT}`
- `GPU_IDS`: GPU indices
- `PORT`: `8199`

### Step 2: Quantize

**Read** the file at `/home/scratch.kaix_coreai/workspace/trt_model_optimizer_dev/modelopt_agent/agents/skills/workflows/quantize.md` **and follow its instructions.**

Pass: `MODEL_PATH`, `QFORMAT`, `EXPORT_PATH`, `GPU_IDS`, `CALIB_SIZE=512`, `REPO_ROOT`

If it fails with a model compatibility error, **read** `/home/scratch.kaix_coreai/workspace/trt_model_optimizer_dev/modelopt_agent/agents/skills/tools/patch-model.md` **and follow it**, then retry quantization.

For error diagnosis, **read** `/home/scratch.kaix_coreai/workspace/trt_model_optimizer_dev/modelopt_agent/agents/skills/guidelines/02_error_prevention.md`.

### Step 3: Deploy + Speed Benchmark

**Read** the file at `/home/scratch.kaix_coreai/workspace/trt_model_optimizer_dev/modelopt_agent/agents/skills/tools/vllm-benchmark.md` **and follow its instructions.**

Pass: `MODEL_PATH=$EXPORT_PATH`, `MODEL_NAME=${MODEL_NAME}_${QFORMAT}`, `GPU_IDS`, `PORT`, `KEEP_SERVER=true`

Tell it to **leave the server running** for the accuracy eval step.

### Step 4: Accuracy Evaluation

**Read** the file at `/home/scratch.kaix_coreai/workspace/trt_model_optimizer_dev/modelopt_agent/agents/skills/tools/accuracy-eval.md` **and follow its instructions.**

Pass: `MODEL_PATH=$EXPORT_PATH`, `MODEL_NAME=${MODEL_NAME}_${QFORMAT}`, `GPU_IDS`, `PORT`, `EVAL_TASKS`, `VLLM_ALREADY_RUNNING=true`

### Step 5: Present Combined Results

Show the user a combined summary:

```
============================================================
  OPTIMIZATION RESULTS: <MODEL_NAME> (<QFORMAT>)
============================================================

  SPEED:
    Throughput:        XXXX tok/s
    Latency p50:       XX.X ms
    Latency p99:       XX.X ms

  ACCURACY:
    mmlu:              0.XXXX

  EXPORT PATH: <EXPORT_PATH>
============================================================
```

### Step 6: User Decision

Ask: **"Are you satisfied with these results? [yes/no/quit]"**

- **yes** -- Done! Report final model path and summary.
- **quit** -- Exit. Report partial results.
- **no** -- **Read** `/home/scratch.kaix_coreai/workspace/trt_model_optimizer_dev/modelopt_agent/agents/skills/guidelines/01_recipe_selection.md` for the iteration table, propose lighter recipe, loop to Step 2.

## Validation Workflow

After each phase, verify success:

**After quantization:**

- `ls $EXPORT_PATH/*.safetensors` -- weight files exist
- `cat $EXPORT_PATH/hf_quant_config.json` -- quantization config recorded
- See `references/04_export_checkpoint.md` for full checklist

**After vLLM deploy:**

- `curl http://localhost:$PORT/health` returns 200
- `curl http://localhost:$PORT/v1/models` shows the model loaded

**After accuracy eval:**

- nel output dir contains JSON result files
- Accuracy scores are parsed and displayed

## Success Criteria

### Single Recipe Workflow

1. Quantized checkpoint exists at `EXPORT_PATH` with safetensors + config.json + hf_quant_config.json
2. vLLM successfully deploys the checkpoint (health check passes)
3. Speed benchmark produces throughput and latency numbers
4. Accuracy evaluation produces scores for all requested tasks
5. Combined results are presented clearly to the user
6. User has approved the results OR a lighter recipe has been proposed
7. All vLLM server processes are cleaned up when done

### Pareto Sweep (additional criteria)

8. Multiple formats were quantized and benchmarked (at least 3)
9. Pareto frontier computed (accuracy vs throughput)
10. Pareto-optimal recipes identified and highlighted
11. Recommendations provided: best accuracy, best throughput, best balance
12. `pareto_analysis.json` saved with all results
