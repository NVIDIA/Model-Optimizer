# SGLang Deployment Reference

## Requirements

- SGLang >= 0.4.10
- `pip install sglang[all]`

## Server Deployment

### As OpenAI-compatible server

```bash
python -m sglang.launch_server \
    --model-path <checkpoint_path> \
    --quantization modelopt \
    --tp <num_gpus> \
    --host 0.0.0.0 --port 8000
```

For NVFP4 checkpoints, use `--quantization modelopt_fp4`.

### As Python API

```python
import sglang as sgl

llm = sgl.Engine(model_path="<checkpoint_path>", quantization="modelopt")
# For FP4: quantization="modelopt_fp4"

sampling_params = {"temperature": 0.8, "top_p": 0.95}
outputs = llm.generate(["Hello, my name is"], sampling_params)

for output in outputs:
    print(f"Generated: {output['text']}")
```

### From HuggingFace Hub

```python
import sglang as sgl

llm = sgl.Engine(model_path="nvidia/Llama-3.1-8B-Instruct-FP8", quantization="modelopt")
outputs = llm.generate(["What is AI?"], {"temperature": 0.8})
```

## Speculative Decoding

SGLang supports speculative decoding with EAGLE and EAGLE3 models:

```bash
python -m sglang.launch_server \
    --model-path <target_model> \
    --speculative-algorithm EAGLE \
    --speculative-draft-model-path <draft_model> \
    --speculative-num-steps 3 \
    --speculative-eagle-topk 4 \
    --tp <num_gpus> \
    --host 0.0.0.0 --port 8000
```

Reference: `examples/specdec_bench/specdec_bench/models/sglang.py`

## Key SGLang Flags

| Flag | Description |
|------|-------------|
| `--model-path` | Path to checkpoint or HF model ID |
| `--quantization` | `modelopt` (FP8) or `modelopt_fp4` (FP4) |
| `--tp` | Tensor parallelism size |
| `--ep` | Expert parallelism (for MoE models) |
| `--enable-torch-compile` | Enable torch.compile for better perf |
| `--cuda-graph-max-bs` | Max batch size for CUDA graphs |
| `--attention-backend` | `flashinfer` (default) or `triton` |

## Common Issues

| Issue | Fix |
|-------|-----|
| `quantization="modelopt"` not recognized | Upgrade SGLang to >= 0.4.10 |
| DeepSeek FP4 not working | Check support matrix — SGLang FP4 support varies by model |
| OOM on startup | Increase `--tp` or reduce `--max-total-tokens` |
