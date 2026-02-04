## Table of Contents

- [Overview](#overview)
- [Setup](#setup)
- [Prepare ORT-GenAI Compatible Base Model](#prepare-ort-genai-compatible-base-model)
- [Quantization](#quantization)
- [Evaluate the Quantized Model](#evaluate-the-quantized-model)
- [Deployment](#deployment)
- [Support Matrix](#support-matrix)
- [Troubleshoot](#troubleshoot)

## Overview

The example script showcases how to utilize the **ModelOpt-Windows** toolkit for optimizing ONNX (Open Neural Network Exchange) models through quantization. This toolkit is designed for developers looking to enhance model performance, reduce size, and accelerate inference times, while preserving the accuracy of neural networks deployed with backends like TensorRT-RTX, DirectML, CUDA on local RTX GPUs running Windows.

Quantization is a technique that converts models from floating-point to lower-precision formats, such as integers, which are more computationally efficient. This process can significantly speed up execution on supported hardware, while also reducing memory and bandwidth requirements.

This example takes an ONNX model as input, along with the necessary quantization settings, and generates a quantized ONNX model as output. This script can be used for quantizing popular, [ONNX Runtime GenAI](https://onnxruntime.ai/docs/genai) built Large Language Models (LLMs) in the ONNX format.

## Setup

1. Install ModelOpt-Windows. Refer [installation instructions](../../README.md).

1. Install required dependencies

   ```bash
   pip install -r requirements.txt
   ```

## Prepare ORT-GenAI Compatible Base Model

You may generate the base model using the model builder that comes with onnxruntime-genai. The ORT-GenAI's [model-builder](https://github.com/microsoft/onnxruntime-genai/tree/main/src/python/py/models) downloads the original Pytorch model from Hugging Face, and produces an ONNX GenAI-compatible base model in ONNX format. See example command-line below:

```bash
python -m onnxruntime_genai.models.builder -m meta-llama/Meta-Llama-3-8B -p fp16 -e dml -o E:\llama3-8b-fp16-dml-genai
```

## Quantization

To begin quantization, run the script like below:

```bash
python quantize.py --model_name=meta-llama/Meta-Llama-3-8B \
                          --onnx_path="E:\model_store\genai\llama3-8b-fp16-dml-genai\opset_21\model.onnx" \
                          --output_path="E:\model_store\genai\llama3-8b-fp16-dml-genai\opset_21\cnn_32_lite_0.1_16\model.onnx" \
                          --calib_size=32 --algo=awq_lite --dataset=cnn
```

### Command Line Arguments

The table below lists key command-line arguments of the ONNX PTQ example script.

| **Argument** | **Supported Values** | **Description** |
|---------------------------|------------------------------------------------------|-------------------------------------------------------------|
| `--calib_size` | 32 , 64, 128 (default) | Specifies the calibration size. |
| `--dataset` | cnn (default), pilevel | Choose calibration dataset: cnn_dailymail or pile-val. |
| `--algo` | awq_lite (default), awq_clip, rtn, rtn_dq | Select the quantization algorithm. |
| `--onnx_path` | input .onnx file path | Path to the input ONNX model. |
| `--output_path` | output .onnx file path | Path to save the quantized ONNX model. |
| `--use_zero_point` | Default: zero-point is disabled | Use this option to enable zero-point based quantization. |
| `--block-size` | 32, 64, 128 (default) | Block size for AWQ. |
| `--awqlite_alpha_step` | 0.1 (default) | Step-size for AWQ scale search, user-defined |
| `--awqlite_run_per_subgraph` | Default: run_per_subgraph is disabled | Use this option to run AWQ scale search at the subgraph level |
| `--awqlite_disable_fuse_nodes` | Default: fuse_nodes enabled | Use this option to disable fusion of input scales in parent nodes. |
| `--awqclip_alpha_step` | 0.05 (default) | Step-size for AWQ weight clipping, user-defined |
| `--awqclip_alpha_min` | 0.5 (default) | Minimum AWQ weight-clipping threshold, user-defined |
| `--awqclip_bsz_col` | 1024 (default) | Chunk size in columns during weight clipping, user-defined |
| `--calibration_eps` | dml, cuda, cpu, NvTensorRtRtx (default: [cuda,cpu]) | List of execution-providers to use for session run during calibration |
| `--add_position_ids` | Default: position_ids input is disabled | Use this option to enable position_ids input in calibration data|
| `--enable_mixed_quant` | Default: mixed-quant is disabled | Use this option to enable mixed precision quantization|
| `--layers_8bit` | Default: None | Use this option to override default mixed-quant strategy|
| `--gather_quantize_axis` | Default: None | Use this option to enable INT4 quantization of Gather nodes - choose 0 or 1|
| `--gather_block_size` | Default: 32 | Block-size for Gather node's INT4 quantization (when its enabled using gather_quantize_axis option)|
| `--use_column_major` | Default: disabled | Apply column-major storage optimization for execution providers that need it. Only applicable for DQ-only quantization.|

Run the following command to view all available parameters in the script:

```bash
python quantize.py --help
```

Note:

1. For the `algo` argument, we have following options to choose from: awq_lite, awq_clip, rtn, rtn_dq.
   - The 'awq_lite' option does core AWQ scale search and INT4 quantization.
   - The 'awq_clip' option primarily does weight clipping and INT4 quantization.
   - The 'rtn' option does INT4 RTN quantization with Q->DQ nodes for weights.
   - The 'rtn_dq' option does INT4 RTN quantization with only DQ nodes for weights.
1. RTN algorithm doesn't use calibration-data.
1. If needed for the input base model, use `--add_position_ids` command-line option to enable
   generating position_ids calibration input. The GenAI built LLM models produced with DML EP has
   position_ids input but ones produced with CUDA EP, NvTensorRtRtx EP don't have position_ids input.
   Use `--help` or command-line options table above to inspect default values.

Please refer to `quantize.py` for further details on command-line parameters.

### Mixed Precision Quantization (INT4 + INT8)

ModelOpt-Windows supports **mixed precision quantization**, where different layers in the model can be quantized to different bit-widths. This approach combines INT4 quantization for most layers (for maximum compression and speed) with INT8 quantization for important or sensitive layers (to preserve accuracy).

#### Why Use Mixed Precision?

Mixed precision quantization provides an optimal balance between:

- **Model Size**: Primarily INT4 keeps the model small
- **Inference Speed**: INT4 layers run faster and smaller
- **Accuracy Preservation**: Critical layers in INT8 maintain model quality

Based on benchmark results, mixed precision quantization shows significant advantages:

| Model | Metric | INT4 RTN | Mixed RTN (INT4+INT8) | Improvement |
|:------|:-------|:-------------|:---------------------|:-----------|
| DeepSeek R1 1.5B | MMLU | 32.40% | 33.90% | +1.5% |
| | Perplexity | 46.304 | 44.332 | -2.0 (lower is better) |
| Llama 3.2 1B | MMLU | 39.90% | 44.70% | +4.8% |
| | Perplexity | 16.900 | 14.176 | -2.7 (lower is better) |
| Qwen 2.5 1.5B | MMLU | 56.70% | 57.50% | +0.8% |
| | Perplexity | 10.933 | 10.338 | -0.6 (lower is better) |

As shown above, mixed precision significantly improves accuracy with minimal disk size increase (~85-109 MB).

#### How Mixed Precision Works

The quantization strategy selects which layers to quantize to INT8 vs INT4:

1. **INT8 Layers** (Higher Precision): Important layers that significantly impact model quality. Quantized per-channel

2. **INT4 Layers** (Maximum Compression): All other layers. Qunatized blockwise.

This strategy preserves accuracy for the most sensitive layers while maintaining aggressive compression elsewhere.

#### Using Mixed Precision Quantization

##### Method 1: Use the default mixed precision strategy

```bash
python quantize.py --model_name=meta-llama/Meta-Llama-3.2-1B \
                   --onnx_path="E:\models\llama3.2-1b-fp16\model.onnx" \
                   --output_path="E:\models\llama3.2-1b-int4-int8-mixed\model.onnx" \
                   --algo=awq_lite \
                   --calib_size=32 \
                   --enable_mixed_quant
```

The `--enable_mixed_quant` flag automatically applies the default strategy.

##### Method 2: Specify custom layers for INT8

```bash
python quantize.py --model_name=meta-llama/Meta-Llama-3.2-1B \
                   --onnx_path="E:\models\llama3.2-1b-fp16\model.onnx" \
                   --output_path="E:\models\llama3.2-1b-int4-int8-custom\model.onnx" \
                   --algo=awq_lite \
                   --calib_size=32 \
                   --layers_8bit="layers.0,layers.1,layers.15,layers.16"
```

The `--layers_8bit` option allows you to manually specify which layers to quantize to INT8. You can use:

- Layer indices: `layers.0,layers.5,layers.10`
- Layer paths: `model/layers.0/attn/qkv_proj`
- Partial names: `qkv_proj,down_proj`

##### Technical Details

- **Block Size**: INT4 layers use block-wise quantization (default block-size=128), INT8 uses per-channel quantization
- **Quantization Axis**: INT4 (per-block), INT8 (per-channel row-wise)
- **Compatibility**: Works with both `awq_lite` and `rtn_dq` algorithms
- **Automatic Detection**: The `--layers_8bit` option automatically enables mixed quantization

For more benchmark results and detailed accuracy metrics, refer to the [Benchmark Guide](../../Benchmark.md).

## Evaluate the Quantized Model

To evaluate the quantized model, please refer to the [accuracy benchmarking](../../accuracy_benchmark/README.md) and [onnxruntime-genai performance benchmarking](https://github.com/microsoft/onnxruntime-genai/tree/main/benchmark/python).

## Deployment

Once an ONNX FP16 model is quantized using ModelOpt-Windows, the resulting quantized ONNX model can be deployed using [ORT-GenAI](https://onnxruntime.ai/) or [ORT](https://onnxruntime.ai/).

Refer to the following example scripts and tutorials for deployment:

1. [ORT GenAI examples](https://github.com/microsoft/onnxruntime-genai/tree/main/examples/python)
1. [ONNX Runtime documentation](https://onnxruntime.ai/docs/api/python/)

## Support Matrix

| Model | ONNX INT4 AWQ (W4A16) |
| :---: | :---: |
| Llama3.1-8B-Instruct | ✅ |
| Phi3.5-mini-Instruct | ✅ |
| Mistral-7B-Instruct-v0.3 | ✅ |
| Llama3.2-3B-Instruct | ✅ |
| Gemma-2b-it | ✅ |
| Gemma-2-2b | ✅ |
| Gemma-2-9b | ✅ |
| Nemotron Mini 4B Instruct | ✅ |
| Qwen2.5-7B-Instruct | ✅ |
| DeepSeek-R1-Distill-Llama-8B | ✅ |
| DeepSeek-R1-Distil-Qwen-1.5B | ✅ |
| DeepSeek-R1-Distil-Qwen-7B | ✅ |
| DeepSeek-R1-Distill-Qwen-14B | ✅ |
| Mistral-NeMo-Minitron-2B-128k-Instruct | ✅ |
| Mistral-NeMo-Minitron-4B-128k-Instruct | ✅ |
| Mistral-NeMo-Minitron-8B-128k-Instruct | ✅ |

> *All LLMs in the above table are [GenAI](https://github.com/microsoft/onnxruntime-genai/) built LLMs.*

 > *`ONNX INT4 AWQ (W4A16)` means INT4 weights and FP16 activations using AWQ algorithm.*

## Troubleshoot

1. **Configure Directories**

   - Update the `cache_dir` variable in the `main()` function to specify the path where you want to store Hugging Face files (optional).
   - If you're low on space on the C: drive, change the TMP and TEMP environment variable to a different drive (e.g., `D:\temp`).

1. **Authentication for Restricted Models**

   If the model you wish to use is hosted on Hugging Face and requires authentication, log in using the [huggingface-cli](https://huggingface.co/docs/huggingface_hub/main/en/guides/cli#huggingface-cli-login) before running the quantization script.

   ```bash
   huggingface-cli login --token <HF_TOKEN>
   ```

1. **Check Read/Write Permissions**

   Ensure that both the input and output model paths have the necessary read and write permissions to avoid any permission-related errors.

1. **Check Output Path**

   Ensure that output .onnx file doesn't exist already. For example, if the output path is `C:\dir1\dir2\quant\model_quant.onnx` then the path `C:\dir1\dir2\quant` should be valid and the directory `quant` should not already contain `model_quant.onnx` file before quantization. If the output .onnx file already exists, then that can get appended during saving of the quantized model resulting in corrupted or invalid output model.

1. **Check Input Model**

   During INT4 AWQ execution, the input onnx model (one mentioned in `--onnx_path` argument) will be run with onnxruntime (ORT) for calibration (using ORT EP mentioned in `--calibration_eps` argument). So, make sure that input onnx model is running fine with the specified ORT EP.

1. **Config availability for calibration with NvTensorRtRtx EP**

   Note that while using `NvTensorRtRtx` for INT4 AWQ quantization, profile (min/max/opt ranges) of input-shapes of the model is created internally using the details from the model's config (e.g. config.json in HuggingFace model card). This input-shapes-profile is used during onnxruntime session creation. Make sure that config.json is available in the model-directory if `model_name` is a local model path (instead of HuggingFace model-name).

1. **Error - Invalid Position-IDs input to the ONNX model**

   The ONNX models produced using ONNX GenerativeAI (GenAI) have different IO bindings for models produced using different execution-providers (EPs). For instance, model built with DML EP has position-ids input in the ONNX model but models builts using CUDA EP or NvTensorRtRtx EP don't have position-ids inputs. So, if base model requires, use `--add_position_ids` command-line argument for enabling position_ids calibration input or set "add_position_ids" variable to `True` value (hard-code) in the quantize script if required.
