# Torch Quantization to ONNX Export

This example demonstrates how to quantize PyTorch models (vision and LLM) followed by export to ONNX format. The scripts leverage the ModelOpt toolkit for both quantization and ONNX export.

<div align="center">

| **Section** | **Description** | **Link** |
| :------------: | :------------: | :------------: |
| Pre-Requisites | Required packages to use this example | [Link](#pre-requisites) |
| Vision Models | Quantize timm models and export to ONNX | [Link](#vision-models) |
| LLM Export | Export LLMs to quantized ONNX | [Link](#llm-export) |
| Mixed Precision | Auto mode for optimal per-layer quantization | [Link](#mixed-precision-quantization-auto-mode) |
| Support Matrix | View the ONNX export supported LLM models | [Link](#onnx-export-supported-llm-models) |
| Resources | Extra links to relevant resources | [Link](#resources) |

</div>

## Pre-Requisites

### Docker

Please use the TensorRT docker image (e.g., `nvcr.io/nvidia/tensorrt:25.08-py3`) or visit our [installation docs](https://nvidia.github.io/Model-Optimizer/getting_started/2_installation.html) for more information.

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

## LLM Export

The `llm_export.py` script exports LLM models to ONNX with optional quantization.

### What it does

- Loads a HuggingFace LLM model (local path or model name).
- Optionally quantizes the model to FP8, INT4_AWQ, or NVFP4.
- Exports the model to ONNX format.
- Post-processes the ONNX graph for TensorRT compatibility.

### Usage

```bash
python llm_export.py \
    --hf_model_path=<HuggingFace model name or local path> \
    --dtype=<fp16|fp8|int4_awq|nvfp4> \
    --output_dir=<directory to save ONNX model>
```

### Examples

Export Qwen2 to FP16 ONNX:

```bash
python llm_export.py \
    --hf_model_path=Qwen/Qwen2-0.5B-Instruct \
    --dtype=fp16 \
    --output_dir=./qwen2_fp16
```

Export Qwen2 to FP8 ONNX with quantization:

```bash
python llm_export.py \
    --hf_model_path=Qwen/Qwen2-0.5B-Instruct \
    --dtype=fp8 \
    --output_dir=./qwen2_fp8
```

Export to NVFP4 with custom calibration:

```bash
python llm_export.py \
    --hf_model_path=Qwen/Qwen3-0.6B \
    --dtype=nvfp4 \
    --calib_size=512 \
    --output_dir=./qwen3_nvfp4
```

### Key Parameters

| Parameter | Description |
| :--- | :--- |
| `--hf_model_path` | HuggingFace model name (e.g., `Qwen/Qwen2-0.5B-Instruct`) or local model path |
| `--dtype` | Export precision: `fp16`, `fp8`, `int4_awq`, or `nvfp4` |
| `--output_dir` | Directory to save the exported ONNX model |
| `--calib_size` | Number of calibration samples for quantization (default: 512) |
| `--lm_head` | Precision of lm_head layer (default: `fp16`) |
| `--save_original` | Save the raw ONNX before post-processing |
| `--trust_remote_code` | Trust remote code when loading from HuggingFace Hub |

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

## ONNX Export Supported LLM Models

| Model | FP16 | INT4 | FP8 | NVFP4 |
| :---: | :---: | :---: | :---: | :---: |
| [Llama-3-8B-Instruct](https://huggingface.co/meta-llama/Meta-Llama-3-8B-Instruct) | ✅ | ✅ | ✅ | ✅ |
| [Llama3.1-8B](https://huggingface.co/meta-llama/Llama-3.1-8B) | ✅ | ✅ | ✅ | ✅ |
| [Llama3.2-3B](https://huggingface.co/meta-llama/Llama-3.2-3B) | ✅ | ✅ | ✅ | ✅ |
| [Qwen2-0.5B-Instruct](https://huggingface.co/Qwen/Qwen2-0.5B-Instruct) | ✅ | ✅ | ✅ | ✅ |
| [Qwen2-1.5B-Instruct](https://huggingface.co/Qwen/Qwen2-1.5B-Instruct) | ✅ | ✅ | ✅ | ✅ |
| [Qwen2-7B-Instruct](https://huggingface.co/Qwen/Qwen2-7B-Instruct) | ✅ | ✅ | ✅ | ✅ |
| [Qwen2.5-0.5B-Instruct](https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct) | ✅ | ✅ | ✅ | ✅ |
| [Qwen2.5-1.5B-Instruct](https://huggingface.co/Qwen/Qwen2.5-1.5B-Instruct) | ✅ | ✅ | ✅ | ✅ |
| [Qwen2.5-3B-Instruct](https://huggingface.co/Qwen/Qwen2.5-3B-Instruct) | ✅ | ✅ | ✅ | ✅ |
| [Qwen2.5-7B-Instruct](https://huggingface.co/Qwen/Qwen2.5-7B-Instruct) | ✅ | ✅ | ✅ | ✅ |

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
