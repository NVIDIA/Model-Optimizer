# vLLM Deployment Reference

## Requirements

- vLLM >= 0.10.1
- `pip install vllm`

## Realquant Deployment (recommended)

Realquant uses dedicated quantized kernels for maximum performance. This is the default path for ModelOpt-exported checkpoints.

### As OpenAI-compatible server

```bash
python -m vllm.entrypoints.openai.api_server \
    --model <checkpoint_path> \
    --quantization modelopt \
    --tensor-parallel-size <num_gpus> \
    --host 0.0.0.0 --port 8000 \
    --served-model-name <model_name>
```

For NVFP4 checkpoints, use `--quantization modelopt_fp4`.

### As Python API

```python
from vllm import LLM, SamplingParams

llm = LLM(model="<checkpoint_path>", quantization="modelopt")
# For FP4: quantization="modelopt_fp4"

sampling_params = SamplingParams(temperature=0.8, top_p=0.95)
outputs = llm.generate(["Hello, my name is"], sampling_params)

for output in outputs:
    print(f"Prompt: {output.prompt!r}, Generated: {output.outputs[0].text!r}")
```

### From HuggingFace Hub

```python
from vllm import LLM, SamplingParams

llm = LLM(model="nvidia/Llama-3.1-8B-Instruct-FP8", quantization="modelopt")
outputs = llm.generate(["What is AI?"], SamplingParams(temperature=0.8))
```

## Fakequant Deployment (research)

Fakequant is 2-5x slower than realquant but doesn't require dedicated kernel support. Useful for research and testing new quantization schemes.

Reference: `examples/vllm_serve/`

```bash
# Environment variables for configuration
export QUANT_CFG=NVFP4_DEFAULT_CFG    # Quantization format
export QUANT_CALIB_SIZE=512            # Calibration samples
export QUANT_DATASET=cnn_dailymail     # Calibration dataset

python examples/vllm_serve/vllm_serve_fakequant.py <model_path> \
    -tp <num_gpus> --host 0.0.0.0 --port 8000
```

## Benchmarking

```bash
# Start server first, then benchmark
python -m vllm.benchmark_serving \
    --model <model_name> \
    --port 8000 \
    --num-prompts 100 \
    --request-rate 10
```

Or use lm_eval for accuracy:

```bash
lm_eval --model local-completions \
    --tasks gsm8k \
    --model_args model=<model_name>,base_url=http://localhost:8000/v1/completions,num_concurrent=1,max_retries=3,tokenized_requests=False,batch_size=128
```

## Common Issues

| Issue | Fix |
|-------|-----|
| `quantization="modelopt"` not recognized | Upgrade vLLM to >= 0.10.1 |
| OOM on startup | Increase `--tensor-parallel-size` or reduce `--max-model-len` |
| AWQ checkpoints not loading | AWQ is not supported in vLLM via modelopt path; use FP8 or NVFP4 |
| Mixed precision not working | Not supported for fakequant |
