---
name: deployment
description: Serve a quantized or unquantized LLM checkpoint as an OpenAI-compatible API endpoint using vLLM, SGLang, or TRT-LLM. Use when user says "deploy model", "serve model", "start vLLM server", "launch SGLang", "TRT-LLM deploy", "AutoDeploy", "benchmark throughput", "serve checkpoint", or needs an inference endpoint from a HuggingFace or ModelOpt-quantized checkpoint.
---

# Deployment Skill

Serve a model checkpoint as an OpenAI-compatible inference endpoint. Supports vLLM, SGLang, and TRT-LLM (including AutoDeploy).

## Quick Start

Use the deploy script for the fastest path. It auto-detects quantization format from the checkpoint:

```bash
# Start vLLM server with a ModelOpt checkpoint
scripts/deploy.sh start --model ./qwen3-0.6b-fp8

# Start with SGLang and tensor parallelism
scripts/deploy.sh start --model ./llama-70b-nvfp4 --framework sglang --tp 4

# Start from HuggingFace hub
scripts/deploy.sh start --model nvidia/Llama-3.1-8B-Instruct-FP8

# Test the API
scripts/deploy.sh test

# Check status
scripts/deploy.sh status

# Stop
scripts/deploy.sh stop
```

The script handles: GPU detection, quantization flag auto-detection (FP8 vs FP4), server lifecycle (start/stop/restart/status), health check polling, and API testing.

## Decision Flow

### 0. Identify the checkpoint

Determine what the user wants to deploy:

- **Local quantized checkpoint** (from ptq skill or manual export): look for `hf_quant_config.json` in the directory
- **HuggingFace model hub** (e.g., `nvidia/Llama-3.1-8B-Instruct-FP8`): use directly
- **Unquantized model**: deploy as-is (BF16) or suggest quantizing first with the ptq skill

Check the quantization format if applicable:

```bash
cat <checkpoint_path>/hf_quant_config.json 2>/dev/null || echo "No quant config — unquantized or legacy format"
```

### 1. Choose the framework

If the user hasn't specified a framework, recommend based on this priority:

| Situation | Recommended | Why |
|-----------|-------------|-----|
| General use | **vLLM** | Widest ecosystem, easy setup, OpenAI-compatible |
| Best SGLang model support | **SGLang** | Strong DeepSeek/Llama 4 support |
| Maximum optimization | **TRT-LLM** | Best throughput via engine compilation |
| Mixed-precision / AutoQuant | **TRT-LLM AutoDeploy** | Only option for AutoQuant checkpoints |

Check the support matrix in `references/support-matrix.md` to confirm the model + format + framework combination is supported.

### 2. Check the environment

**GPU availability:**

```bash
python -c "import torch; [print(f'GPU {i}: {torch.cuda.get_device_name(i)}') for i in range(torch.cuda.device_count())] if torch.cuda.is_available() else print('no-gpu')"
```

**Framework installed?**

```bash
# vLLM
python -c "import vllm; print(f'vLLM {vllm.__version__}')" 2>/dev/null || echo "vLLM not installed"

# SGLang
python -c "import sglang; print(f'SGLang {sglang.__version__}')" 2>/dev/null || echo "SGLang not installed"

# TRT-LLM
python -c "import tensorrt_llm; print(f'TRT-LLM {tensorrt_llm.__version__}')" 2>/dev/null || echo "TRT-LLM not installed"
```

If the framework is not installed, consult `references/setup.md` for installation instructions.

**GPU memory estimate:**

- BF16 model: `num_params × 2 bytes` (e.g., 8B model ≈ 16 GB)
- FP8 model: `num_params × 1 byte` (e.g., 8B model ≈ 8 GB)
- FP4 model: `num_params × 0.5 bytes` (e.g., 8B model ≈ 4 GB)
- Add ~2-4 GB for KV cache and framework overhead

If the model exceeds single GPU memory, use tensor parallelism (`-tp <num_gpus>`).

### 3. Deploy

Read the framework-specific reference for detailed instructions:

| Framework | Reference file |
|-----------|---------------|
| vLLM | `references/vllm.md` |
| SGLang | `references/sglang.md` |
| TRT-LLM | `references/trtllm.md` |

**Quick-start commands** (for common cases):

#### vLLM

```bash
# Serve as OpenAI-compatible endpoint
python -m vllm.entrypoints.openai.api_server \
    --model <checkpoint_path> \
    --quantization modelopt \
    --tensor-parallel-size <num_gpus> \
    --host 0.0.0.0 --port 8000
```

For NVFP4 checkpoints, use `--quantization modelopt_fp4`.

#### SGLang

```bash
python -m sglang.launch_server \
    --model-path <checkpoint_path> \
    --quantization modelopt \
    --tp <num_gpus> \
    --host 0.0.0.0 --port 8000
```

#### TRT-LLM (direct)

```python
from tensorrt_llm import LLM, SamplingParams
llm = LLM(model="<checkpoint_path>")
outputs = llm.generate(["Hello, my name is"], SamplingParams(temperature=0.8, top_p=0.95))
```

#### TRT-LLM AutoDeploy

For AutoQuant or mixed-precision checkpoints, see `references/trtllm.md`.

### 4. Verify the deployment

After the server starts, verify it's healthy:

```bash
# Health check
curl -s http://localhost:8000/health

# List models
curl -s http://localhost:8000/v1/models | python -m json.tool

# Test generation
curl -s http://localhost:8000/v1/completions \
    -H "Content-Type: application/json" \
    -d '{
        "model": "<model_name>",
        "prompt": "The capital of France is",
        "max_tokens": 32
    }' | python -m json.tool
```

All checks must pass before reporting success to the user.

### 5. Benchmark (optional)

If the user wants throughput/latency numbers, run a quick benchmark:

```bash
# vLLM benchmark
python -m vllm.entrypoints.openai.api_server ... &  # if not already running

python -m vllm.benchmark_serving \
    --model <model_name> \
    --port 8000 \
    --num-prompts 100 \
    --request-rate 10
```

Report: throughput (tok/s), latency p50/p99, time to first token (TTFT).

## Error Handling

| Error | Cause | Fix |
|-------|-------|-----|
| `CUDA out of memory` | Model too large for GPU(s) | Increase `--tensor-parallel-size` or use a smaller model |
| `quantization="modelopt" not recognized` | vLLM/SGLang version too old | Upgrade: vLLM >= 0.10.1, SGLang >= 0.4.10 |
| `hf_quant_config.json not found` | Not a ModelOpt-exported checkpoint | Re-export with `export_hf_checkpoint()`, or remove `--quantization` flag |
| `Connection refused` on health check | Server still starting | Wait 30-60s for large models; check logs for errors |
| `modelopt_fp4 not supported` | Framework doesn't support FP4 for this model | Check support matrix in `references/support-matrix.md` |

## Success Criteria

1. Server process is running and healthy (`/health` returns 200)
2. Model is listed at `/v1/models`
3. Test generation produces coherent output
4. Server URL and port are reported to the user
5. If benchmarking was requested, throughput/latency numbers are reported
