# Torch Quantization to ONNX Export

This example demonstrates how to quantize PyTorch models followed by export to ONNX format. The scripts leverage the ModelOpt toolkit for quantization and ONNX export.

For **vision models**, the `torch_quant_to_onnx.py` script in this directory handles quantization and ONNX export directly.

For **LLMs and VLMs**, use [TensorRT-Edge-LLM](https://github.com/NVIDIA/TensorRT-Edge-LLM) which provides a complete pipeline for quantizing models with ModelOpt and exporting them to optimized ONNX for deployment on edge platforms (Jetson, DRIVE).

<div align="center">

| **Section** | **Description** | **Link** |
| :------------: | :------------: | :------------: |
| Pre-Requisites | Required packages to use this example | [Link](#pre-requisites) |
| Vision Models | Quantize timm models and export to ONNX | [Link](#vision-models) |
| LLM Quantization and Export | Quantize and export LLMs/VLMs via TensorRT-Edge-LLM | [Link](#llm-quantization-and-export-with-tensorrt-edge-llm) |
| Supported Models | LLM and VLM models supported by TensorRT-Edge-LLM | [Link](#supported-models) |
| Mixed Precision | Auto mode for optimal per-layer quantization | [Link](#mixed-precision-quantization-auto-mode) |
| Resources | Extra links to relevant resources | [Link](#resources) |

</div>

## Pre-Requisites

### Docker

Please use the TensorRT docker image (e.g., `nvcr.io/nvidia/tensorrt:26.02-py3`) or visit our [installation docs](https://nvidia.github.io/Model-Optimizer/getting_started/2_installation.html) for more information.

Set the following environment variables inside the TensorRT docker.

```bash
export CUDNN_LIB_DIR=/usr/lib/x86_64-linux-gnu/
export LD_LIBRARY_PATH="${CUDNN_LIB_DIR}:${LD_LIBRARY_PATH}"
```

### Local Installation

Install Model Optimizer with `onnx` dependencies using `pip` from [PyPI](https://pypi.org/project/nvidia-modelopt/) and install the requirements for the example:

```bash
pip install -U "nvidia-modelopt[onnx]"
pip install -r requirements.txt
```

For TensorRT Compiler framework workloads:

Install the latest [TensorRT](https://developer.nvidia.com/tensorrt) from [here](https://developer.nvidia.com/tensorrt/download).

## Vision Models

The `torch_quant_to_onnx.py` script quantizes [timm](https://github.com/huggingface/pytorch-image-models) vision models and exports them to ONNX.

### What it does

- Loads a pretrained timm torch model (default: ViT-Base).
- Quantizes the torch model to FP8, MXFP8, INT8, NVFP4, or INT4_AWQ using ModelOpt.
- Exports the quantized model to ONNX.
- Postprocesses the ONNX model to be compatible with TensorRT.
- Saves the final ONNX model.

> *Opset 20 is used to export the torch models to ONNX.*

### Usage

```bash
python torch_quant_to_onnx.py \
    --timm_model_name=vit_base_patch16_224 \
    --quantize_mode=<fp8|mxfp8|int8|nvfp4|int4_awq> \
    --onnx_save_path=<path to save the exported ONNX model>
```

### Evaluation

If the input model is of type image classification, use the following script to evaluate it. The script automatically downloads and uses the [ILSVRC/imagenet-1k](https://huggingface.co/datasets/ILSVRC/imagenet-1k) dataset from Hugging Face. This gated repository requires authentication via Hugging Face access token. See <https://huggingface.co/docs/hub/en/security-tokens> for details.

> *Note: TensorRT 10.11 or later is required to evaluate the MXFP8 or NVFP4 ONNX models.*

```bash
python ../onnx_ptq/evaluate.py \
    --onnx_path=<path to the exported ONNX model> \
    --imagenet_path=<HF dataset card or local path to the ImageNet dataset> \
    --engine_precision=stronglyTyped \
    --model_name=vit_base_patch16_224
```

## LLM Quantization and Export with TensorRT-Edge-LLM

[TensorRT-Edge-LLM](https://github.com/NVIDIA/TensorRT-Edge-LLM) provides a complete pipeline for quantizing LLMs and VLMs using NVIDIA ModelOpt and exporting them to optimized ONNX for deployment on edge platforms such as NVIDIA Jetson and DRIVE.

### Overview

The pipeline follows these stages:

1. **Quantize** (x86 host with GPU) — Reduce model precision using ModelOpt (FP8, INT4 AWQ, NVFP4)
2. **Export** (x86 host with GPU) — Convert quantized model to ONNX
3. **Build** (edge device) — Compile ONNX into TensorRT engines
4. **Inference** (edge device) — Run the compiled engines

### Installation

```bash
# Use the PyTorch Docker image (recommended)
docker pull nvcr.io/nvidia/pytorch:25.12-py3
docker run --gpus all -it --rm -v $(pwd):/workspace -w /workspace nvcr.io/nvidia/pytorch:25.12-py3 bash

# Clone and install TensorRT-Edge-LLM
git clone https://github.com/NVIDIA/TensorRT-Edge-LLM.git
cd TensorRT-Edge-LLM
git submodule update --init --recursive
python3 -m venv venv
source venv/bin/activate
pip3 install .

# Verify installation
tensorrt-edgellm-quantize-llm --help
tensorrt-edgellm-export-llm --help
```

**System requirements:**

- x86-64 Linux (Ubuntu 22.04 or 24.04 recommended)
- NVIDIA GPU with Compute Capability 8.0+ (Ampere or newer)
- CUDA 12.x or 13.x, Python 3.10+
- GPU VRAM: 16 GB for models up to 3B, 40 GB for models up to 4B, 80 GB for models up to 8B

### CLI Tools

| Tool | Purpose |
| :--- | :--- |
| `tensorrt-edgellm-quantize-llm` | Quantize LLM models using ModelOpt (FP8, INT4 AWQ, NVFP4) |
| `tensorrt-edgellm-export-llm` | Export LLM to ONNX with precision-specific optimizations |
| `tensorrt-edgellm-export-visual` | Export visual encoders for multimodal VLM models |
| `tensorrt-edgellm-quantize-draft` | Quantize EAGLE draft models for speculative decoding |
| `tensorrt-edgellm-export-draft` | Export EAGLE draft models to ONNX |
| `tensorrt-edgellm-insert-lora` | Insert LoRA patterns into existing ONNX models |
| `tensorrt-edgellm-process-lora` | Process LoRA adapter weights for runtime loading |

### Example: Quantize and Export an LLM

```bash
# Step 1: Quantize with ModelOpt
tensorrt-edgellm-quantize-llm \
    --model_dir Qwen/Qwen2.5-3B-Instruct \
    --quantization fp8 \
    --output_dir quantized/qwen2.5-3b-fp8

# Step 2: Export to ONNX
tensorrt-edgellm-export-llm \
    --model_dir quantized/qwen2.5-3b-fp8 \
    --output_dir onnx_models/qwen2.5-3b
```

### Example: Quantize and Export a VLM

```bash
# Quantize the language model component
tensorrt-edgellm-quantize-llm \
    --model_dir Qwen/Qwen2.5-VL-3B-Instruct \
    --quantization fp8 \
    --output_dir quantized/qwen2.5-vl-3b

# Export the language model
tensorrt-edgellm-export-llm \
    --model_dir quantized/qwen2.5-vl-3b \
    --output_dir onnx_models/qwen2.5-vl-3b/llm

# Export the visual encoder
tensorrt-edgellm-export-visual \
    --model_dir Qwen/Qwen2.5-VL-3B-Instruct \
    --output_dir onnx_models/qwen2.5-vl-3b/visual
```

### Example: EAGLE Speculative Decoding

```bash
# Quantize base model
tensorrt-edgellm-quantize-llm \
    --model_dir meta-llama/Llama-3.1-8B-Instruct \
    --quantization fp8 \
    --output_dir quantized/llama3.1-8b-base

# Export base model with EAGLE flag
tensorrt-edgellm-export-llm \
    --model_dir quantized/llama3.1-8b-base \
    --output_dir onnx_models/llama3.1-8b/base \
    --is_eagle_base

# Quantize EAGLE draft model
tensorrt-edgellm-quantize-draft \
    --base_model_dir meta-llama/Llama-3.1-8B-Instruct \
    --draft_model_dir EAGLE3-LLaMA3.1-Instruct-8B \
    --quantization fp8 \
    --output_dir quantized/llama3.1-8b-draft

# Export draft model
tensorrt-edgellm-export-draft \
    --draft_model_dir quantized/llama3.1-8b-draft \
    --base_model_dir meta-llama/Llama-3.1-8B-Instruct \
    --output_dir onnx_models/llama3.1-8b/draft
```

### Quantization Methods

| Method | Description |
| :--- | :--- |
| FP8 | Best accuracy-to-memory balance on SM89+ hardware (Hopper, Ada) |
| INT4 AWQ | Weight-only quantization; effective for memory-constrained platforms and low-batch inference |
| NVFP4 | 4-bit format for NVIDIA Blackwell and Thor hardware; applies to both weights and activations |
| MXFP8 | Experimental; Microscaling FP8 format for SM89+ hardware |
| INT8 SmoothQuant | Experimental; INT8 weight and activation quantization with SmoothQuant |
| INT4 GPTQ | Can be loaded directly from HuggingFace Hub (no additional quantization needed) |

### Supported Models

For the latest support matrix, see the [TensorRT-Edge-LLM Supported Models](https://nvidia.github.io/TensorRT-Edge-LLM/developer_guide/getting-started/supported-models.html) page.

#### LLMs

| Model | FP16 | FP8 | INT4 | NVFP4 |
| :--- | :---: | :---: | :---: | :---: |
| [Llama-3-8B-Instruct](https://huggingface.co/meta-llama/Meta-Llama-3-8B-Instruct) | ✅ | ✅ | ✅ | ✅ |
| [Llama-3.1-8B-Instruct](https://huggingface.co/meta-llama/Llama-3.1-8B-Instruct) | ✅ | ✅ | ✅ | ✅ |
| [Llama-3.2-3B-Instruct](https://huggingface.co/meta-llama/Llama-3.2-3B-Instruct) | ✅ | ✅ | ✅ | ✅ |
| [Qwen2-0.5B-Instruct](https://huggingface.co/Qwen/Qwen2-0.5B-Instruct) | ✅ | ✅ | ✅ | ✅ |
| [Qwen2-1.5B-Instruct](https://huggingface.co/Qwen/Qwen2-1.5B-Instruct) | ✅ | ✅ | ✅ | ✅ |
| [Qwen2-7B-Instruct](https://huggingface.co/Qwen/Qwen2-7B-Instruct) | ✅ | ✅ | ✅ | ✅ |
| [Qwen2.5-0.5B-Instruct](https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct) | ✅ | ✅ | ✅ | ✅ |
| [Qwen2.5-1.5B-Instruct](https://huggingface.co/Qwen/Qwen2.5-1.5B-Instruct) | ✅ | ✅ | ✅ | ✅ |
| [Qwen2.5-3B-Instruct](https://huggingface.co/Qwen/Qwen2.5-3B-Instruct) | ✅ | ✅ | ✅ | ✅ |
| [Qwen2.5-7B-Instruct](https://huggingface.co/Qwen/Qwen2.5-7B-Instruct) | ✅ | ✅ | ✅ | ✅ |
| [Qwen3-0.6B](https://huggingface.co/Qwen/Qwen3-0.6B) | ✅ | ✅ | ✅ | ✅ |
| [Qwen3-1.7B](https://huggingface.co/Qwen/Qwen3-1.7B) | ✅ | ✅ | ✅ | ✅ |
| [Qwen3-4B-Instruct-2507](https://huggingface.co/Qwen/Qwen3-4B-Instruct-2507) | ✅ | ✅ | ✅ | ✅ |
| [Qwen3-8B](https://huggingface.co/Qwen/Qwen3-8B) | ✅ | ✅ | ✅ | ✅ |
| [DeepSeek-R1-Distill-Qwen-1.5B](https://huggingface.co/deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B) | ✅ | ✅ | ✅ | ✅ |
| [DeepSeek-R1-Distill-Qwen-7B](https://huggingface.co/deepseek-ai/DeepSeek-R1-Distill-Qwen-7B) | ✅ | ✅ | ✅ | ✅ |

#### VLMs

| Model | FP16 | FP8 | INT4 | NVFP4 |
| :--- | :---: | :---: | :---: | :---: |
| [Qwen2-VL-2B-Instruct](https://huggingface.co/Qwen/Qwen2-VL-2B-Instruct) | ✅ | ✅ | ✅ | ✅ |
| [Qwen2-VL-7B-Instruct](https://huggingface.co/Qwen/Qwen2-VL-7B-Instruct) | ✅ | ✅ | ✅ | ✅ |
| [Qwen2.5-VL-3B-Instruct](https://huggingface.co/Qwen/Qwen2.5-VL-3B-Instruct) | ✅ | ✅ | ✅ | ✅ |
| [Qwen2.5-VL-7B-Instruct](https://huggingface.co/Qwen/Qwen2.5-VL-7B-Instruct) | ✅ | ✅ | ✅ | ✅ |
| [Qwen3-VL-2B-Instruct](https://huggingface.co/Qwen/Qwen3-VL-2B-Instruct) | ✅ | ✅ | ✅ | ✅ |
| [Qwen3-VL-4B-Instruct](https://huggingface.co/Qwen/Qwen3-VL-4B-Instruct) | ✅ | ✅ | ✅ | ✅ |
| [Qwen3-VL-8B-Instruct](https://huggingface.co/Qwen/Qwen3-VL-8B-Instruct) | ✅ | ✅ | ✅ | ✅ |
| [InternVL3-1B](https://huggingface.co/OpenGVLab/InternVL3-1B) | ✅ | ✅ | ✅ | ✅ |
| [InternVL3-2B](https://huggingface.co/OpenGVLab/InternVL3-2B) | ✅ | ✅ | ✅ | ✅ |
| [Phi-4-multimodal-instruct](https://huggingface.co/microsoft/Phi-4-multimodal-instruct) | ✅ | ✅ | ✅ | ✅ |

### Troubleshooting

- **GPU out of memory**: Use a larger GPU (40 GB for models up to 4B, 80 GB for models up to 8B) or try `--device cpu` (limited precision support).
- **Calibration dataset issues**: Download the dataset manually and pass the local path with `--calib_dataset ./path/to/dataset`.
- **Accuracy degradation**: Try FP8 instead of INT4/NVFP4, or increase calibration sample size.

For full documentation, see the [TensorRT-Edge-LLM Developer Guide](https://nvidia.github.io/TensorRT-Edge-LLM/).

## Mixed Precision Quantization (Auto Mode)

The `auto` mode enables mixed precision quantization by searching for the optimal quantization format per layer. This approach balances model accuracy and compression by assigning different precision formats (e.g., NVFP4, FP8) to different layers based on their sensitivity.

### How it works

1. **Sensitivity Analysis**: Computes per-layer sensitivity scores using gradient-based analysis
2. **Format Search**: Searches across specified quantization formats for each layer
3. **Constraint Optimization**: Finds the optimal format assignment that satisfies the effective bits constraint while minimizing accuracy loss

### Key Parameters

| Parameter | Default | Description |
| :--- | :---: | :--- |
| `--effective_bits` | 4.8 | Target average bits per weight across the model. Lower values = more compression but potentially lower accuracy. The search algorithm finds the optimal per-layer format assignment that meets this constraint while minimizing accuracy loss. For example, 4.8 means an average of 4.8 bits per weight (mix of FP4 and FP8 layers). |
| `--num_score_steps` | 128 | Number of forward/backward passes used to compute per-layer sensitivity scores via gradient-based analysis. Higher values provide more accurate sensitivity estimates but increase search time. Recommended range: 64-256. |
| `--calibration_data_size` | 512 | Number of calibration samples used for both sensitivity scoring and calibration. For auto mode, labels are required for loss computation. |

### Usage

```bash
python torch_quant_to_onnx.py \
    --timm_model_name=vit_base_patch16_224 \
    --quantize_mode=auto \
    --auto_quantization_formats NVFP4_AWQ_LITE_CFG FP8_DEFAULT_CFG \
    --effective_bits=4.8 \
    --num_score_steps=128 \
    --calibration_data_size=512 \
    --evaluate \
    --onnx_save_path=vit_base_patch16_224.auto_quant.onnx
```

### Results (ViT-Base)

| | Top-1 accuracy (torch) | Top-5 accuracy (torch) |
| :--- | :---: | :---: |
| Torch autocast (FP16) | 85.11% | 97.53% |
| NVFP4 Quantized | 84.558% | 97.36% |
| Auto Quantized (FP8 + NVFP4, 4.78 effective bits) | 84.726% | 97.434% |

## Resources

- 📅 [Roadmap](https://github.com/NVIDIA/Model-Optimizer/issues/146)
- 📖 [Documentation](https://nvidia.github.io/Model-Optimizer)
- 🎯 [Benchmarks](../benchmark.md)
- 💡 [Release Notes](https://nvidia.github.io/Model-Optimizer/reference/0_changelog.html)
- 🐛 [File a bug](https://github.com/NVIDIA/Model-Optimizer/issues/new?template=1_bug_report.md)
- ✨ [File a Feature Request](https://github.com/NVIDIA/Model-Optimizer/issues/new?template=2_feature_request.md)

### Technical Resources

There are many quantization schemes supported in the example scripts:

1. The [FP8 format](https://developer.nvidia.com/blog/nvidia-arm-and-intel-publish-fp8-specification-for-standardization-as-an-interchange-format-for-ai/) is available on the Hopper and Ada GPUs with [CUDA compute capability](https://developer.nvidia.com/cuda-gpus) greater than or equal to 8.9.

1. The [INT4 AWQ](https://arxiv.org/abs/2306.00978) is an INT4 weight only quantization and calibration method. INT4 AWQ is particularly effective for low batch inference where inference latency is dominated by weight loading time rather than the computation time itself. For low batch inference, INT4 AWQ could give lower latency than FP8/INT8 and lower accuracy degradation than INT8.

1. The [NVFP4](https://blogs.nvidia.com/blog/generative-ai-studio-ces-geforce-rtx-50-series/) is one of the new FP4 formats supported by NVIDIA Blackwell GPU and demonstrates good accuracy compared with other 4-bit alternatives. NVFP4 can be applied to both model weights as well as activations, providing the potential for both a significant increase in math throughput and reductions in memory footprint and memory bandwidth usage compared to the FP8 data format on Blackwell.
