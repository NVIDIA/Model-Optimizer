# LLM Export (Windows)

Export LLMs from PyTorch to ONNX with quantization and GQA surgery.

## Supported Precisions

- `nvfp4` — NVIDIA FP4 quantization
- `int4_awq` — INT4 AWQ quantization
- `int8_sq` — INT8 SmoothQuant

## Usage

### NVFP4

```bash
python llm_export.py --hf_model_path "meta-llama/Llama-3.2-3B-Instruct" --dtype nvfp4 --output_dir ./llama3.2-3b-nvfp4
```

### INT4 AWQ

```bash
python llm_export.py --hf_model_path "meta-llama/Llama-3.2-3B-Instruct" --dtype int4_awq --output_dir ./llama3.2-3b-int4
```

### INT8 SmoothQuant

```bash
python llm_export.py --hf_model_path "Qwen/Qwen2.5-3B-Instruct" --dtype int8_sq --output_dir ./qwen-3b-int8
```

## Options

| Argument | Description |
|---|---|
| `--hf_model_path` | HuggingFace model name or local path |
| `--dtype` | Quantization precision (`fp16`, `fp8`, `int4_awq`, `int8_sq`, `nvfp4`) |
| `--output_dir` | Directory to save the exported ONNX model |
| `--calib_size` | Calibration dataset size (default: 512) |
| `--save_original` | Save the pre-surgery ONNX for debugging |
| `--trust_remote_code` | Trust remote code when loading from HuggingFace |
| `--onnx_path` | Skip export, run surgery on an existing ONNX |
| `--config_path` | Path to config.json if not alongside the model |
