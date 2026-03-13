# TRT-LLM Deployment Reference

## Requirements

- TensorRT-LLM >= 0.17.0
- Typically installed via NVIDIA container: `nvcr.io/nvidia/tensorrt-llm/release:<version>`
- Or: `pip install tensorrt-llm`

## Direct LLM API (recommended for unified HF checkpoints)

### Python API

```python
from tensorrt_llm import LLM, SamplingParams

llm = LLM(model="<checkpoint_path>")
# Quantization format is auto-detected from hf_quant_config.json

sampling_params = SamplingParams(temperature=0.8, top_p=0.95)
outputs = llm.generate(["Hello, my name is"], sampling_params)

for output in outputs:
    print(f"Prompt: {output.prompt!r}, Generated: {output.outputs[0].text!r}")
```

### From HuggingFace Hub

```python
from tensorrt_llm import LLM

llm = LLM(model="nvidia/Llama-3.1-8B-Instruct-FP8")
print(llm.generate(["What is AI?"]))
```

### With tensor parallelism

```python
from tensorrt_llm import LLM

llm = LLM(model="<checkpoint_path>", tensor_parallel_size=4)
```

## AutoDeploy (for AutoQuant / mixed-precision)

AutoDeploy automates graph transformations for optimized inference. Required for AutoQuant checkpoints.

### End-to-end script

```bash
# Quantize and deploy in one step
./examples/llm_autodeploy/scripts/run_auto_quant_and_deploy.sh \
    --hf_ckpt <model_path> \
    --save_quantized_ckpt <output_path> \
    --quant fp8,nvfp4 \
    --effective_bits 4.5
```

Parameters:

- `--hf_ckpt`: Path to unquantized HuggingFace checkpoint
- `--save_quantized_ckpt`: Output path for quantized checkpoint
- `--quant`: Quantization formats (e.g., `fp8,nvfp4`)
- `--effective_bits`: Target precision (higher = more accuracy for sensitive layers)
- `--world_size`: Number of GPUs for tensor parallelism
- `--calib_batch_size`: Calibration batch size (reduce if OOM, default 8)

### AutoDeploy API server

```python
# examples/llm_autodeploy/api_server.py provides a FastAPI server
# with OpenAI-compatible endpoints using AutoDeploy
```

### Test AutoDeploy

```bash
python examples/llm_autodeploy/api_client.py --prompt "What is AI?" "What is golf?"
```

### Notes

- NVFP4 in AutoDeploy requires Blackwell GPUs
- For Hopper: remove `nvfp4` from `--quant` and set `--effective_bits` above 8.0
- AutoDeploy supports CUDA graphs, torch compile backends, and KV cache optimization

## Legacy TRT-LLM Checkpoint (deprecated)

The legacy export path using `export_tensorrt_llm_checkpoint()` is deprecated. Use the unified HF checkpoint format with `export_hf_checkpoint()` instead.

If you encounter a legacy checkpoint (no `hf_quant_config.json`, has `rank*.safetensors` pattern), it needs the TRT-LLM build API to create an engine before deployment. See `docs/source/deployment/1_tensorrt_llm.rst`.

## Evaluation with TRT-LLM

```python
# examples/llm_eval/lm_eval_tensorrt_llm.py
# Runs lm_evaluation_harness benchmarks with TRT-LLM
python examples/llm_eval/lm_eval_tensorrt_llm.py \
    --model_path <checkpoint_path> \
    --tasks gsm8k,mmlu
```

## Common Issues

| Issue | Fix |
|-------|-----|
| `No module named tensorrt_llm` | Install via container or pip |
| NVFP4 inference fails on Hopper | NVFP4 requires Blackwell GPUs for inference |
| Slow first inference | Engine compilation happens on first run; subsequent runs are cached |
| OOM during engine build | Reduce `--max_batch_size` or increase TP |
