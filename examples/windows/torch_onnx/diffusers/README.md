# Guide for quantizing diffusion models

This repository provides relevant steps, script, and guidance for quantization of diffusion models.

## Table of Contents

- [Installation and Pre-requisites](#installation-and-pre-requisites)
- [Quantization of Backbone](#quantization-of-backbone)
- [Inference using ONNX runtime](#inference-using-onnxruntime)
- [Quantization Support Matrix](#quantization-support-matrix)
- [Validated Settings](#validated-settings)
- [Troubleshoot](#troubleshoot)

## Installation and Pre-requisites

We recommend using separate python virtual environment for steps like ONNX export, quantization, inference etc. to keep dependencies isolated and avoid version conflicts.

Install ModelOpt toolkit along with its dependencies (e.g. MSVC Compiler / Visual Studio / VC Redistributable, CUDA etc.); refer its [installation guide](https://nvidia.github.io/Model-Optimizer/getting_started/windows/_installation_standalone.html). Then, install the required dependencies from ModelOpt's [diffusers example](https://github.com/NVIDIA/TensorRT-Model-Optimizer/tree/main/examples/diffusers) for quantization. For inference, refer forthcoming [inference section](#inference-using-onnxruntime).

## Quantization of backbone

In diffusion models, the *backbone* refers to the main neural network architecture used in the denoising process. It is the core computational component that predicts the noise at each diffusion step. Some diffusion models use U-Net backbone; on the other hand, recent diffusion models like SD and Flux use transformer based backbone models.

ModelOpt's [diffusers example](https://github.com/NVIDIA/TensorRT-Model-Optimizer/tree/main/examples/diffusers) illustrates quantizing the backbone of diffusion models. In addition, it enables the export of the quantized backbone to ONNX format for further deployment or optimization.

> *The quantized ONNX backbone model is exported using opset 20, as this is the highest opset version supported by torch.onnx.export at the time this example was verified.*

> *In case of FP4/FP8 quantization, the exported ONNX model contain TRT specific custom ops for some quantization steps (like dynamic quantization etc.).*

## Inference using onnxruntime

The optimum-onnxruntime tool provides pipelines for various models including diffusion models, for running them through onnxruntime.

### ONNX Export

Running multi-modal models such as SD or Flux with Optimum ONNX Runtime requires ONNX model files for all components in the pipeline (for example, text encoders, VAE encoders/decoders, and others). The Hugging Face [Optimum-CLI](https://huggingface.co/docs/optimum-onnx/onnx/usage_guides/export_a_model) tool can be used for exporting Hugging Face models to the ONNX format. It generally uses `onnxruntime-gpu` for FP16 ONNX export of models.

Example command-line to obtain FP16 SD3.5-Medium ONNX model:

```bash

optimum-cli export onnx --model stabilityai/stable-diffusion-3.5-medium --dtype fp16 --device cuda --task text-to-image --opset 20 ./sd3.5_medium_fp16

```

This command will generate ONNX files for every model used throughout the pipeline, organizing each component's exported files into its own dedicated subdirectory within a central output directory.

We recommend using separate python virtual environment for ONNX export.

### Post-Processing on quantized ONNX model

For running the quantized backbone ONNX model using onnxruntime, we need to do following steps:

- Opset 23 supports FP4 type in Q/DQ ONNX nodes. So, update opset of the quantized backbone ONNX model to 23. This process involves modifying all affected nodes in the ONNX graph whose definitions have changed between opset 21 and 23 (both inclusive), ensuring compatibility with the new specifications. Finally, set the opset field in the ONNX model to version 23 to complete the transition.

```python

model = onnx.load(onnx_path)
...
# update affected nodes, if any, as per new opset
...
new_opset_imports = [
    helper.make_opsetid("", new opset),     # Default domain
    #
    # Update other domains as needed, for example:
    #
    helper.make_opsetid("com.microsoft", 1) # Microsoft domain for contrib-ops
    helper.make_opsetid("trt", 1)           # TRT domain for TRT specific custom-ops
]

updated_quantized_onnx_model = onnx.helper.make_model(model.graph, opset_imports=new_opset_imports)
...
# save updated quantized onnx model
...
```

- Update the quantized backbone ONNX model by explicitly specifying type information for the outputs of TRT custom operators. Providing accurate output data types in the ONNX model's graph ensures that ONNX Runtime can correctly infer tensor types and avoids type inference errors during session creation or model inference. Refer `type_update_trt_custom_ops.py` sample script provided in this example.

```bash
python type_update_trt_custom_ops.py --input_path="E:\model.onnx" --output_path="E:\output\model.onnx"
```

### Onnxruntime Execution Provider for running FP4 / FP8 model

The [TRTRTX EP](https://onnxruntime.ai/docs/execution-providers/TensorRTRTX-ExecutionProvider.html) in onnxruntime is updated to support TRT specific custom ops used in FP4/FP8 quantized model; you can either build it from source or, if available, install the Python wheel from PyPI. For detailed instructions on installation and usage, refer to the official ONNX Runtime [TRTRTX EP](https://onnxruntime.ai/docs/execution-providers/TensorRTRTX-ExecutionProvider.html) documentation.

### Place quantized ONNX model

Place the quantized ONNX model file for the backbone inside its specific subdirectory within the exported ONNX model structure.

### Inference using optimum-onnxruntime

Optimum-ONNX Runtime provides pipelines such as ORTStableDiffusion3Pipeline and ORTFluxPipeline that can be used to run ONNX-exported diffusion models. These pipelines offer a convenient, high-level interface for loading the exported graph and performing inference. For a practical reference, see the stable diffusion inference [example script](https://github.com/microsoft/onnxruntime-inference-examples/tree/main/python/models/stable_difusion) in the ONNX Runtime inference examples repository.

## Quantization Support Matrix

| Model | fp8 | nvfp4<sup>1</sup> |
| :---: | :---: |
| SD3-Medium-Diffusers | ❌ | ✅ |
| SD3.5-Medium | ✅  | ✅ |
| Flux.1.Dev<sup>2</sup> | ✅  | ✅ |

> *<sup>1.</sup> NVFP4 inference requires Blackwell GPUs for speedup.*

> *<sup>2.</sup> It is recommended to enable cpu-offloading and have 128+ GB of system RAM for quantizing Flux.1.Dev on RTX5090.*

> *The accuracy loss after PTQ may vary depending on the actual model and the quantization method. Different models may have different accuracy loss and usually the accuracy loss is more significant when the base model is small. If the accuracy after PTQ is not meeting the requirement, please try disabling with KV-Cache or MHA quantization, try out different calibration settings (like calibration samples data, samples size, diffusion steps etc.) or perform QAT / QAD (not yet supported / validated on Windows RTX).*

Please refer to [support matrix](https://nvidia.github.io/Model-Optimizer/guides/0_support_matrix.html) for a full list of supported features and models.

## Validated Settings

- Python 3.11.9
- CUDA settings on Host - CUDA 12.9, cuDNN 9.5 (cudnn-windows-x86_64-9.5.0.50_cuda12-archive)
- Windows11 22621
- RTX 5090
- RTX Driver - 581.42
- Visual Studio 2022 - Community edition
- Base PyTorch model - [stabilityai/stable-diffusion-3.5-medium](https://huggingface.co/stabilityai/stable-diffusion-3.5-medium)
- Modality - text-to-image
- ONNX Exporter - HuggingFace Optimum, opset-20, FP16 ONNX models
- Inference EP for functional check - [TRTRTX EP](https://onnxruntime.ai/docs/execution-providers/TensorRTRTX-ExecutionProvider.html)
- Quantization configs - NVFP4 with `max` calibration (FP4 linears, FP8 MHA)

## Troubleshoot

1. FP16 ONNX export with Hugging Face’s optimum-cli can sometimes fail with a “no cuda kernel image is available” (or similar) error. This may be the issue with `onnxruntime-gpu` version installed; refer to this [github issue](https://github.com/microsoft/onnxruntime/issues/26181) for details.

2. Sometimes exporting Flux or Stable Diffusion models with ModelOpt’s  [diffusers example](https://github.com/NVIDIA/TensorRT-Model-Optimizer/tree/main/examples/diffusers) can fail during ONNX export with an error like the one below. Downgrading `diffusers` to version 0.34 (instead of 0.35 or later) can avoid this issue; see this [github issue](https://github.com/NVIDIA/TensorRT-Model-Optimizer/issues/262) for details.

    ```bash
    torch.onnx.errors.UnsupportedOperatorError: Exporting the operator 'aten::rms_norm' to ONNX opset version 20 is not supported
    ```

    Alternatively, you can apply a small patch to the export script similar to the one proposed in [pull-request 642](https://github.com/NVIDIA/Model-Optimizer/pull/642).

3. With recent transformers versions such as 4.53 and 4.56, you may encounter following import error. You may try downgrading to an earlier transformers release, for example 4.51.3 or 4.49.

    ```bash
    ImportError: cannot import name 'CLIPSdpaAttention' from 'transformers.models.clip.modeling_clip' 
    ```
