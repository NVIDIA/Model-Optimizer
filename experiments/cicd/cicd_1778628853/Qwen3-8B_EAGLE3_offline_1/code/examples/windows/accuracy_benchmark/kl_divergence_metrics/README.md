# KL Divergence Model Validation Toolkit

This toolkit provides comprehensive model validation capabilities using KL divergence metrics to compare two models. It's designed to evaluate the similarity between model outputs across different optimization techniques, frameworks, and hardware backends.

## Overview

The toolkit measures output similarity between models using KL (Kullback-Leibler) divergence, which quantifies how one probability distribution differs from another. Lower KL divergence values indicate more similar model outputs.

**Primary Use Cases:**

1. **Model Optimization Validation** - Verify that optimized models (quantization, pruning) maintain output quality
2. **Framework Comparison** - Compare Hugging Face models vs ONNX Runtime GenAI models
3. **Precision Analysis** - Evaluate FP16 vs INT4 vs INT8 model outputs
4. **Execution Provider Testing** - Test different EP implementations (CUDA, DirectML, CPU, TensorRT)

## Key Components

### Main Script

| Script | Purpose | Comparison Modes |
|--------|---------|------------------|
| `compute_kl_divergence.py` | **Two-model sequential comparison** | • HF vs GenAI<br>• GenAI vs GenAI (same EP)<br>• GenAI vs HF<br>• HF vs HF |

### Datasets Used

- **Wikitext-2** test split for consistent evaluation across all models
- Automatic dataset loading and preprocessing via HuggingFace datasets

## Installation

### 1. Install Base Requirements

   ```bash
   pip install -r requirements.txt
   ```

   Note: Install torch with CUDA for faster inference:
   "pip install torch torchvision torchaudio --index-url <https://download.pytorch.org/whl/cu129>"

### 2. Install ONNX Runtime GenAI Package

Install **one** of the following based on your hardware:

   ```bash
# For CUDA
   pip install onnxruntime-genai-cuda
   
   # For DirectML support  
   pip install onnxruntime-genai-directml
   
# For CPU
   pip install onnxruntime-genai
   ```

## Usage Examples

### Quick Start

#### Compare HF vs GenAI Model

```bash
python compute_kl_divergence.py \
    --model1 "meta-llama/Llama-3.1-8B-Instruct" --model1_type hf \
    --model2 "G:\models\genai_model" --model2_type genai \
    --device cuda \
    --output results.json
```

#### Compare Two GenAI Models (Same EP)

```bash
python compute_kl_divergence.py \
    --model1 "G:\models\genai_fp16" --model1_type genai \
    --model2 "G:\models\genai_int4" --model2_type genai \
    --output fp16_vs_int4.json
```

### Advanced Options

#### Enable Debug Output

```bash
python compute_kl_divergence.py \
    --model1 "meta-llama/Llama-3.1-8B-Instruct" --model1_type hf \
    --model2 "G:\models\genai_model" --model2_type genai \
    --device cuda \
    --output results.json \
    --debug  # Enables verbose logging
```

## Configuration Parameters

### compute_kl_divergence.py

**Required Parameters:**

| Parameter | Description | Values |
|-----------|-------------|--------|
| `--model1` | Path to first model | Local path or HF Hub identifier |
| `--model1_type` | Type of first model | `hf`, `genai` |
| `--model2` | Path to second model | Local path or HF Hub identifier |
| `--model2_type` | Type of second model | `hf`, `genai` |

**Optional Parameters:**

| Parameter | Description | Default |
|-----------|-------------|---------|
| `--device` | Device for HF model inference | `cuda` |
| `--output` | Output JSON file path | None (prints to console) |
| `--debug` | Enable verbose debug output | False |

**Model Path Formats:**

- **HF models**:
  - Hub identifier: `meta-llama/Llama-3.1-8B-Instruct`
  - Local path: `F:\shared\Llama-3.1-8B-Instruct`
- **GenAI models**:
  - Local path only: `G:\models\genai_model`

### Key Insights

- **Lower is better**: Smaller KL divergence = more similar outputs
- **Relative comparison**: Compare against baseline (e.g., HF FP32)

## Troubleshooting

### Common Issues and Solutions

#### 1. CUDA Out of Memory

**Error:**

```text
RuntimeError: CUDA out of memory
```

**Solutions:**

- Use CPU for HF model: `--device cpu`
- Close other applications using GPU
- Try smaller batch size (modify code if needed)
- Ensure only one model loads at a time (script should handle this)

#### 2. Execution Provider Mismatch

**Error:**

```text
[INFO] Comparing two GenAI models (same execution provider)
```

**Note:** This is informational. GenAI vs GenAI comparisons require same EP.

**Solution:** Ensure both models were created for the same execution provider.
