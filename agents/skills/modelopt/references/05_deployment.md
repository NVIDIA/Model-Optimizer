# Deployment Reference

## vLLM Serving

### Start Server

```bash
CUDA_VISIBLE_DEVICES=$GPU_IDS python -m vllm.entrypoints.openai.api_server \
  --model "$MODEL_PATH" \
  --served-model-name "$MODEL_NAME" \
  --port 8199 \
  --max-model-len 4096 \
  --trust-remote-code \
  --tensor-parallel-size $TP_SIZE
```

vLLM **auto-detects** the quantization format from `hf_quant_config.json` in the checkpoint directory. No `--quantization` flag needed for ModelOpt-exported checkpoints.

### Key Server Flags

| Flag | Default | Description |
|------|---------|-------------|
| `--model` | (required) | Path to model checkpoint |
| `--served-model-name` | model path | Name for API responses |
| `--port` | 8000 | Server port |
| `--max-model-len` | auto | Max sequence length (reduce if OOM) |
| `--trust-remote-code` | false | Allow custom model code |
| `--tensor-parallel-size` | 1 | Number of GPUs for tensor parallelism |
| `--gpu-memory-utilization` | 0.9 | Fraction of GPU memory to use (reduce if OOM) |
| `--dtype` | auto | Model dtype (usually auto-detected) |

### Health Check

```bash
curl http://localhost:8199/health
# Returns 200 when server is ready
```

### Model Verification

```bash
curl http://localhost:8199/v1/models
# Returns JSON with loaded model names
```

### Quick Inference Test

```bash
curl http://localhost:8199/v1/completions \
  -H "Content-Type: application/json" \
  -d '{"model": "MODEL_NAME", "prompt": "Hello", "max_tokens": 10}'
```

## vLLM Speed Benchmarking

### Using `vllm bench serve`

```bash
vllm bench serve \
  --model "$MODEL_NAME" \
  --base-url http://localhost:8199/v1 \
  --num-prompts 100 \
  --input-len 128 \
  --output-len 128
```

**Metrics reported:**

- Throughput: requests/s, output tokens/s
- TTFT (time to first token): mean, p50, p99
- ITL (inter-token latency): mean, p50, p99
- End-to-end latency: mean, p50, p99

### Fallback: Python Script

If `vllm bench` is unavailable, use the inline Python benchmark in `/home/scratch.kaix_coreai/workspace/trt_model_optimizer_dev/modelopt_agent/agents/skills/sub-skills/vllm-benchmark.md`.

## nel Accuracy Evaluation

### Config Template

```yaml
target:
  api_endpoint:
    url: http://localhost:8199/v1
    model_id: MODEL_NAME
    api_key_name: DUMMY_API_KEY

evaluation:
  nemo_evaluator_config:
    config:
      params:
        request_timeout: 3600
        parallelism: 16
        extra:
          tokenizer: /absolute/path/to/model
          tokenizer_backend: huggingface
  tasks:
    - name: lm-evaluation-harness.mmlu

execution:
  output_dir: /path/to/results
```

### Run

```bash
export DUMMY_API_KEY=dummy
nel run --config eval_config.yaml
```

### Available Tasks

Common tasks for LLM accuracy:

- `lm-evaluation-harness.mmlu` — MMLU (primary accuracy metric)
- `lm-evaluation-harness.arc_challenge` — ARC Challenge
- `lm-evaluation-harness.hellaswag` — HellaSwag
- `lm-evaluation-harness.gpqa` — GPQA
- `lm-evaluation-harness.commonsense_qa` — CommonsenseQA

Run `nel ls tasks` to see all available tasks.

### Results

nel writes JSON results to the `output_dir`. Parse with:

```python
import json, glob
for f in glob.glob("results/**/*.json", recursive=True):
    data = json.load(open(f))
    if "results" in data:
        for task, metrics in data["results"].items():
            acc = metrics.get("acc,none") or metrics.get("acc_norm,none")
            if acc: print(f"{task}: {acc:.4f}")
```

## TRT-LLM (Future)

ModelOpt HF checkpoints can be converted to TRT-LLM format using `trtllm-build`. This is not yet supported in the skill workflow but the checkpoint format is compatible.
