<div align="center">

# NVIDIA Model Optimizer - Windows

A Library to Quantize and Compress Deep Learning Models for Optimized Inference on Native Windows RTX GPUs

[![Documentation](https://img.shields.io/badge/Documentation-latest-brightgreen.svg?style=flat)](https://nvidia.github.io/Model-Optimizer/)
[![version](https://img.shields.io/badge/v0.33.0-orange?label=Release)](https://pypi.org/project/nvidia-modelopt/)
[![license](https://img.shields.io/badge/License-Apache%202.0-blue)](../../LICENSE)

[Examples](#examples) |
[Benchmark Results](#benchmark-results)

</div>

## Latest News

- [2024/11/19] [Microsoft and NVIDIA Supercharge AI Development on RTX AI PCs](https://blogs.nvidia.com/blog/ai-decoded-microsoft-ignite-rtx/)
- [2024/11/18] [Quantized INT4 ONNX models available on Hugging Face for download](https://huggingface.co/collections/nvidia/optimized-onnx-models-for-nvidia-rtx-gpus)

## Table of Contents

- [Overview](#overview)
- [Installation](#installation)
- [Techniques](#techniques)
  - [Quantization](#quantization)
- [Examples](#examples)
- [Support Matrix](#support-matrix)
- [Benchmark Results](#benchmark-results)
- [Collection of Optimized ONNX Models](#collection-of-optimized-onnx-models)
- [Release Notes](#release-notes)

## Overview

The **Model Optimizer - Windows** (**ModelOpt-Windows**) is engineered to deliver advanced model compression techniques, including quantization, to Windows RTX PC systems. Specifically tailored to meet the needs of Windows users, ModelOpt-Windows is optimized for rapid and efficient quantization, featuring local GPU calibration, reduced system and video memory consumption, and swift processing times.
The primary objective of the ModelOpt-Windows is to generate optimized, standards-compliant ONNX-format models. This makes it an ideal solution for seamless integration with ONNX Runtime (ORT) and DirectML (DML) frameworks, ensuring broad compatibility with any inference framework supporting the ONNX standard. Furthermore, ModelOpt-Windows integrates smoothly within the Windows ecosystem, with full support for tools and SDKs such as Olive and ONNX Runtime, enabling deployment of quantized models across various independent hardware vendors (IHVs) through the DML path and TensorRT path.

Model Optimizer is available for free for all developers on [NVIDIA PyPI](https://pypi.org/project/nvidia-modelopt/). This repository is for sharing examples and GPU-optimized recipes as well as collecting feedback from the community.

## Installation

ModelOpt-Windows can be installed either as a standalone toolkit or through Microsoft's Olive.

### Standalone Toolkit Installation (with CUDA 12.x)

To install ModelOpt-Windows as a standalone toolkit on CUDA 12.x systems, run the following commands:

```bash
pip install nvidia-modelopt[onnx]
```

### Installation with Olive

To install ModelOpt-Windows through Microsoft's Olive, use the following commands:

```bash
pip install olive-ai[nvmo]
pip install onnxruntime-genai-directml>=0.4.0
pip install onnxruntime-directml==1.20.0
```

For more details, please refer to the [detailed installation instructions](https://nvidia.github.io/Model-Optimizer/getting_started/windows/_installation_for_Windows.html).

## Techniques

### Quantization

Quantization is an effective model optimization technique for large models. Quantization with ModelOpt-Windows can compress model size by 2x-4x, speeding up inference while preserving model quality. ModelOpt-Window enables highly performant quantization formats including INT4, FP8, INT8, etc. and supports advanced algorithms such as AWQ and SmoothQuant\* focusing on post-training quantization (PTQ) for ONNX and PyTorch\* models with DirectML, CUDA and TensorRT\* inference backends.

For more details, please refer to the [detailed quantization guide](https://nvidia.github.io/Model-Optimizer/guides/windows_guides/_ONNX_PTQ_guide.html).

## Getting Started

The ONNX quantization API requires a model, calibration data, along with quantization settings like algorithm, calibration-EPs etc. Here’s an example snippet to apply INT4 AWQ quantization:

```python
from modelopt.onnx.quantization.int4 import quantize as quantize_int4
# import other packages as needed
calib_inputs = get_calib_inputs(dataset, model_name, cache_dir, calib_size, batch_size,...)
quantized_onnx_model = quantize_int4(
    onnx_path,
    calibration_method="awq_lite",
    calibration_data_reader=None if use_random_calib else calib_inputs,
    calibration_eps=["dml", "cpu"]
)
onnx.save_model(
    quantized_onnx_model,
    output_path,
    save_as_external_data=True,
    location=os.path.basename(output_path) + "_data",
    size_threshold=0,
)
```

Check `modelopt.onnx.quantization.quantize_int4` for details about INT4 quantization API.

Refer to our [Support Matrix](https://nvidia.github.io/Model-Optimizer/guides/0_support_matrix.html#support-matrix) for details about supported features and models.

To learn more about ONNX PTQ, refer to our [docs](https://nvidia.github.io/Model-Optimizer/guides/windows_guides/_ONNX_PTQ_guide.html#onnx-ptq-guide-windows).

### Deployment

The quantized onnx model can be deployed using frameworks like onnxruntime. Ensure that model’s opset is 19+ for FP8 quantization, and it is 21+ for INT4 quantization. This is needed due to different opset requirements of ONNX’s [Q](https://onnx.ai/onnx/operators/onnx__QuantizeLinear.html)/[DQ](https://onnx.ai/onnx/operators/onnx__DequantizeLinear.html) nodes for INT4, FP8 data-types support. Refer to [Apply Post Training Quantization (PTQ)](https://nvidia.github.io/Model-Optimizer/guides/windows_guides/_ONNX_PTQ_guide.html#apply-onnx-ptq) for details.

```python
# write steps (say, upgrade_opset() method) to upgrade or patch opset of the model, if needed
# the opset-upgrade, if needed, can be done on either base ONNX model or on the quantized model
# finally, save the quantized model

quantized_onnx_model = upgrade_opset(quantized_onnx_model)
onnx.save_model(
    quantized_onnx_model,
    output_path,
    save_as_external_data=True,
    location=os.path.basename(output_path) + "_data",
    size_threshold=0,
)
```

For detailed instructions about deployment of quantized models with ONNX Runtime, see the [ONNX Runtime Deployment Guide](https://nvidia.github.io/Model-Optimizer/deployment/2_onnxruntime.html).

> [!Note]
> The ready-to-deploy optimized ONNX models from ModelOpt-Windows are available at HuggingFace [NVIDIA collections](https://huggingface.co/collections/nvidia/optimized-onnx-models-for-nvidia-rtx-gpus).

## Examples

- Examples for Post-Training Quantization (PTQ) of ONNX models:
  - [PTQ for GenAI LLMs](./onnx_ptq/genai_llm/README.md) covers how to use ONNX PTQ with [ONNX Runtime GenAI](https://onnxruntime.ai/docs/genai) built LLM ONNX models, and their deployment with DirectML.
  - [PTQ for Whisper](./onnx_ptq/whisper/README.md) illustrates using ONNX PTQ with a Whisper ONNX model (i.e. an ASR model). It also provides example script for Optimum-ORT based inference of Whisper using CUDA EP.
  - [PTQ for SAM2](./onnx_ptq/sam2/README.md) illustrates using ONNX PTQ with a SAM2 ONNX model (i.e. a segmentation model).
- Examples that demonstrate PTQ of a PyTorch model followed by ONNX export:
  - [Diffusers](./torch_onnx/diffusers/README.md) example demonstrates how to apply PTQ to diffusion models in PyTorch format and then export the quantized models to ONNX.
- [MMLU Benchmark](./accuracy_benchmark/README.md) provides an example script for MMLU benchmarking of LLM models, and demonstrates how to run it with various popular backends like DirectML, TensorRT-LLM\* and model formats like ONNX and PyTorch\*.

## Support Matrix

| Model Type | Support Matrix |
|------------|----------------|
| Large Language Models (LLMs) | [View Support Matrix](./onnx_ptq/genai_llm/README.md#support-matrix) |
| Automatic Speech Recognition | [View Support Matrix](./onnx_ptq/whisper/README.md#support-matrix) |
| Segmentation Models | [View Support Matrix](./onnx_ptq/sam2/README.md#support-matrix) |
| Diffusion Models | [View Support Matrix](./torch_onnx/diffusers/README.md#support-matrix) |

## Benchmark Results

Please refer to [benchmark results](./Benchmark.md) for performance and accuracy comparisons of popular Large Language Models (LLMs).

## Collection Of Optimized ONNX Models

The ready-to-deploy optimized ONNX models from ModelOpt-Windows are available at [HuggingFace NVIDIA collections](https://huggingface.co/collections/nvidia/optimized-onnx-models-for-nvidia-rtx-gpus). These models can be deployed using DirectML backend. Follow the instructions provided along with the published models for deployment.

## Release Notes

Please refer to [changelog](https://nvidia.github.io/Model-Optimizer/reference/0_changelog.html)

\* *Experimental support*
