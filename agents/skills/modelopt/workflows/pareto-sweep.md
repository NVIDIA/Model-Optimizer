# Workflow: Pareto Sweep — Explore Optimal Quantization Recipes

## Inputs (provided by caller)

- `MODEL_PATH`: Absolute path to the model directory
- `GPU_IDS`: Comma-separated GPU indices (e.g., `0,1,2,3`)
- `REPO_ROOT`: Root of the TensorRT-Model-Optimizer repo
- `EVAL_TASKS`: Comma-separated eval tasks (default: `mmlu`)
- `FORMATS`: (optional) Comma-separated formats to try. If not specified, auto-select based on GPU type.

## Overview

Run multiple quantization formats in parallel, benchmark each, and compute the Pareto frontier (accuracy vs throughput). This finds the optimal recipes without manual iteration.

## Procedure

### 1. Determine formats to sweep

If `FORMATS` is not provided, auto-select based on GPU and model:

**For Blackwell (B100/B200):**

```
FORMATS="fp8,nvfp4,nvfp4_awq,int4_awq,w4a8_awq"
```

**For Hopper (H100/H200):**

```
FORMATS="fp8,int8_sq,int4_awq,w4a8_awq"
```

**For Ampere (A100) or older:**

```
FORMATS="int8_sq,int4_awq,w4a8_awq"
```

Detect GPU type:

```bash
nvidia-smi --query-gpu=name --format=csv,noheader | head -1
```

- Contains "B100" or "B200" or "GB" -> Blackwell
- Contains "H100" or "H200" or "GH" -> Hopper
- Otherwise -> Ampere/older

### 2. Set up variables

```bash
MODEL_NAME=$(basename "$MODEL_PATH")
RESULTS_DIR="$REPO_ROOT/data/pareto_results/${MODEL_NAME}"
mkdir -p "$RESULTS_DIR"
```

### 3. Run BF16 baseline (if not already done)

Before quantized runs, get baseline accuracy and speed:

**Read and follow** `/home/scratch.kaix_coreai/workspace/trt_model_optimizer_dev/modelopt_agent/agents/skills/tools/vllm-benchmark.md` with `MODEL_PATH` (original, not quantized), `MODEL_NAME`, `GPU_IDS`, `PORT=8199`, `KEEP_SERVER=true`.

Then **read and follow** `/home/scratch.kaix_coreai/workspace/trt_model_optimizer_dev/modelopt_agent/agents/skills/tools/accuracy-eval.md` with the same model, `VLLM_ALREADY_RUNNING=true`.

Store baseline results:

```python
import json
baseline = {
    "format": "bf16",
    "throughput_tps": <from benchmark>,
    "latency_p50_ms": <from benchmark>,
    "accuracy": {"mmlu": <from eval>},
}
json.dump(baseline, open("$RESULTS_DIR/bf16_results.json", "w"))
```

Kill the baseline vLLM server before proceeding.

### 4. Dispatch parallel quantization pipelines

**IMPORTANT:** Use the Agent tool to spawn parallel agents. Each agent runs the full pipeline for one format.

For each format in `FORMATS`, spawn an agent with this prompt:

```
You are running a quantization pipeline for format: <FORMAT>.

1. Read and follow /home/scratch.kaix_coreai/workspace/trt_model_optimizer_dev/modelopt_agent/agents/skills/workflows/quantize.md
   - MODEL_PATH=<MODEL_PATH>
   - QFORMAT=<FORMAT>
   - EXPORT_PATH=<REPO_ROOT>/data/checkpoints/<MODEL_NAME>_<FORMAT>
   - GPU_IDS=<assigned GPU>
   - CALIB_SIZE=512
   - REPO_ROOT=<REPO_ROOT>

2. If quantization succeeds, read and follow /home/scratch.kaix_coreai/workspace/trt_model_optimizer_dev/modelopt_agent/agents/skills/tools/vllm-benchmark.md
   - MODEL_PATH=<EXPORT_PATH>
   - MODEL_NAME=<MODEL_NAME>_<FORMAT>
   - GPU_IDS=<assigned GPU>
   - PORT=<unique port per format>
   - KEEP_SERVER=true

3. Read and follow /home/scratch.kaix_coreai/workspace/trt_model_optimizer_dev/modelopt_agent/agents/skills/tools/accuracy-eval.md
   - MODEL_PATH=<EXPORT_PATH>
   - MODEL_NAME=<MODEL_NAME>_<FORMAT>
   - GPU_IDS=<assigned GPU>
   - PORT=<same port>
   - EVAL_TASKS=<EVAL_TASKS>
   - VLLM_ALREADY_RUNNING=true

4. Kill the vLLM server when done.

5. Write results to <RESULTS_DIR>/<FORMAT>_results.json:
   {
     "format": "<FORMAT>",
     "throughput_tps": <number>,
     "latency_p50_ms": <number>,
     "latency_p99_ms": <number>,
     "accuracy": {"mmlu": <number>, ...},
     "export_path": "<EXPORT_PATH>",
     "success": true/false,
     "error": null or "<error message>"
   }
```

**GPU and port assignment:**

If multiple GPUs available, assign one per agent:

- Format 1 -> GPU 0, port 8199
- Format 2 -> GPU 1, port 8200
- Format 3 -> GPU 2, port 8201
- etc.

If fewer GPUs than formats, run sequentially on the same GPU (different ports still).

**Launch agents in parallel** using the Agent tool with multiple tool calls in one message. Use `subagent_type="general-purpose"`.

### 5. Collect results

After all agents complete, read all result files:

```python
import json, glob

results = []
for f in sorted(glob.glob("$RESULTS_DIR/*_results.json")):
    r = json.load(open(f))
    if r.get("success"):
        results.append(r)
    else:
        print(f"SKIPPED {r['format']}: {r.get('error', 'unknown error')}")
```

### 6. Compute Pareto frontier

A recipe is Pareto-optimal if no other recipe has BOTH higher accuracy AND higher throughput.

```python
def compute_pareto(results):
    """Return Pareto-optimal points from results list."""
    sorted_r = sorted(results, key=lambda x: x["accuracy"].get("mmlu", 0), reverse=True)
    pareto = []
    max_throughput = -1
    for r in sorted_r:
        throughput = r.get("throughput_tps", 0)
        if throughput > max_throughput:
            pareto.append(r)
            max_throughput = throughput
    return pareto

pareto_points = compute_pareto(results)
```

### 7. Present results

Display all results with Pareto-optimal points highlighted:

```
============================================================
  PARETO SWEEP RESULTS: <MODEL_NAME>
============================================================

  Format          Accuracy(MMLU)  Throughput(tok/s)  Pareto?
  ------          --------------  -----------------  -------
  bf16            0.6543          234.5
  fp8             0.6521          456.7              *
  int8_sq         0.6489          423.1
  nvfp4           0.6234          789.0              *
  int4_awq        0.6198          678.2
  w4a8_awq        0.6345          567.3              *

  * = Pareto-optimal (no other recipe is better in BOTH metrics)

  RECOMMENDED:
    Best accuracy:    fp8    (MMLU 0.6521, 456.7 tok/s)
    Best throughput:  nvfp4  (MMLU 0.6234, 789.0 tok/s)
    Best balance:     w4a8   (MMLU 0.6345, 567.3 tok/s)

  Checkpoints saved at: $REPO_ROOT/data/checkpoints/
============================================================
```

### 8. Save Pareto analysis

```python
analysis = {
    "model": MODEL_NAME,
    "all_results": results,
    "pareto_frontier": pareto_points,
    "baseline": baseline,
}
json.dump(analysis, open("$RESULTS_DIR/pareto_analysis.json", "w"), indent=2)
```

## Error Handling

- If a format fails to quantize: log the error, skip it, continue with others
- If a format fails to deploy on vLLM: log the error, skip it
- If fewer than 2 formats succeed: warn user, show available results anyway
- If all formats fail: report errors, suggest checking model compatibility

## Output

- `$RESULTS_DIR/pareto_analysis.json` -- full analysis with all results and Pareto frontier
- `$RESULTS_DIR/<format>_results.json` -- individual results per format
- Pareto frontier table presented to user
- Recommendation: best accuracy, best throughput, best balance
