# Perplexity Evaluation Tool

## Overview

This tool evaluates the perplexity of ONNX Runtime GenAI models and HuggingFace models using the [WikiText-2](https://huggingface.co/datasets/wikitext) dataset. Perplexity is a standard metric for language models: lower values indicate better predictive performance.

## Attribution

This script is originally based on [perplexity_metrics.py](https://github.com/microsoft/onnxruntime-genai/blob/main/tools/python/model_validation/perplexity_metrics.py) from the Microsoft ONNX Runtime GenAI repository. It has been modified to handle:

- Multiple context lengths
- Configurable chunk sizes
- Enhanced prefill chunking handling
- HuggingFace model evaluation support

## Scripts

- **`perplexity_metrics.py`**: Core evaluation logic for computing perplexity.
- **`run_perplexity.py`**: Command-line utility for evaluating one or more models and saving results to CSV.

## Requirements

- Python 3.8+
- CUDA 12.x (if using GPU acceleration)

- Install dependencies:

  **For CUDA 12.x (recommended for CUDA 12.1-12.9):**
  
  ```bash
  pip install -r requirements.txt
  ```

- Install ONNX Runtime GenAI (required for ONNX model evaluation):

  ```bash
  pip install onnxruntime-genai
  ```

- [HuggingFace CLI](https://huggingface.co/docs/huggingface_hub/main/en/guides/cli) login is required to access the WikiText-2 dataset:

  ```bash
  huggingface-cli login
  ```

## Supported Models

### ONNX Runtime GenAI Models

- Any ONNX Runtime GenAI model exported with a compatible `genai_config.json` and tokenizer.
- Supported architectures include: Gemma, Llama, Mistral, Phi (language + vision), Qwen.
- Supported execution providers: CPU, DirectML, CUDA, NvTensorRtRtx.

### HuggingFace Models

- Any HuggingFace causal language model (e.g., `meta-llama/Llama-2-7b-hf`, `gpt2`, `mistralai/Mistral-7B-v0.1`).
- Models are automatically downloaded from the HuggingFace Hub if not cached locally.
- Supports custom data types (float16, bfloat16, float32) for efficient inference.

## How to Run

### Evaluate ONNX Models

#### Single Model

```bash
python run_perplexity.py --models /path/to/model
```

#### Multiple Models

```bash
python run_perplexity.py --models /path/to/model1 /path/to/model2
```

#### Custom Input Sequence Length(s)

You can specify the input sequence length(s) to evaluate using the `--i` argument:

```bash
python run_perplexity.py --models /path/to/model --i 1024,2048,4096,8192,12288
```

#### Custom Prefill Chunk Size

You can specify the prefill chunk size to evaluate using the `--chunk_size` argument:

```bash
python run_perplexity.py --models /path/to/model --i 1024,2048,4096,8192,12288 --chunk_size=1024
```

### Evaluate HuggingFace Models

#### Basic HuggingFace Model Evaluation

```bash
python run_perplexity.py --hf_model meta-llama/Llama-2-7b-hf --i 1024
```

#### With Custom Data Type (Recommended for Performance)

```bash
python run_perplexity.py --hf_model meta-llama/Llama-2-7b-hf --hf_dtype float16 --i 1024
```

#### With Multiple Input Lengths

```bash
python run_perplexity.py --hf_model meta-llama/Llama-2-7b-hf --hf_dtype float16 --i 1024,2048,4096
```

#### On CPU (if no GPU available)

```bash
python run_perplexity.py --hf_model gpt2 --hf_device cpu --i 1024
```

### Evaluate Both ONNX and HuggingFace Models Together

Compare ONNX and HuggingFace models side-by-side:

```bash
python run_perplexity.py \
  --models /path/to/onnx_model \
  --hf_model meta-llama/Llama-2-7b-hf \
  --hf_dtype float16 \
  --i 1024 \
  --output comparison_results.csv
```

### HuggingFace Model Arguments

- `--hf_model`: HuggingFace model name or local path (e.g., `meta-llama/Llama-2-7b-hf`)
- `--hf_device`: Device to run on (`cuda`, `cpu`, `cuda:0`, etc.) - default: `cuda`
- `--hf_dtype`: Data type for model weights - options: `float16`, `bfloat16`, `float32`, `fp16`, `bf16`, `fp32` - default: model default (usually float32)

### Custom Output File

```bash
python run_perplexity.py --models /path/to/model --output results.csv
```

## Expected Output

Expected scores often fall between 2 and 1000; lower is better. See ranges below.

### Perplexity Configuration Setting (for ONNX models)

- If **kv_chunking** is enabled in the model configuration (i.e., `"chunk_size"` is present in the `"search"` section of `genai_config.json`), then:
  - `max_input_seq_length` is set to **8192**
  - `stride` is set to the value of `chunk_size`
- If **kv_chunking** is not enabled (default):
  - `max_input_seq_length` is **1024**
  - `stride` is **512**

### For HuggingFace Models

- Default `max_length` is **1024**
- Default `stride` is **512** (or `chunk_size` if specified)

### Console Output

```text
============================================================
Evaluating HuggingFace model: meta-llama/Llama-2-7b-hf
============================================================
[INFO] Loading Wikitext-2 'test' split ...
[TOKENIZER] Tokenizing ...

[RESULT] Perplexity of meta-llama/Llama-2-7b-hf: 5.47

HuggingFace perplexity evaluation completed

============================================================
Evaluating perplexity for: /path/to/onnx_model
============================================================
[INFO] Loading Wikitext-2 'test' split ...
[TOKENIZER] Tokenizing ...

[RESULT] Perplexity of /path/to/onnx_model: 5.48

Perplexity evaluation completed successfully
```

### CSV Output

Generated file contains:

- Model Path (model directory or HuggingFace model name)
- Model Type (ONNX or HuggingFace)
- Input Length
- Perplexity score
- Status (Success/Failed)
- Error details (if any)

## Debug Mode

Set `DEBUG = True` in `perplexity_metrics.py` for detailed logs.

## Typical Perplexity Ranges

- Excellent: 2-20
- Good: 20-40
- OK: 40-80
- Poor: 100+

## Common Use Cases

### Compare ONNX vs. HuggingFace Model

Verify that your ONNX exported model has similar perplexity to the original HuggingFace model:

```bash
python run_perplexity.py \
  --models /path/to/exported_onnx_model \
  --hf_model meta-llama/Llama-2-7b-hf \
  --hf_dtype float16 \
  --i 1024 \
  --output validation_results.csv
```

### Evaluate Small Models (for quick testing)

```bash
python run_perplexity.py --hf_model gpt2 --hf_dtype float16 --i 1024
```

### Benchmark Multiple Quantization Variants

```bash
python run_perplexity.py \
  --models /path/to/fp16_model /path/to/int8_model /path/to/int4_model \
  --hf_model original/model-name \
  --hf_dtype float16 \
  --i 2048 \
  --output quantization_comparison.csv
```
