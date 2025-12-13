# Scripts for ONNX Whisper model

This repository contains an example to demontrate 8-bit quantization of Whisper ONNX model.

## Table of Contents

- [ONNX export](#onnx-export)
- [Inference script](#inference-script)
- [Quantization script](#quantization-script)
- [Validated Settings](#validated-settings)
- [Troubleshoot](#troubleshoot)

## ONNX export

The HuggingFace Optimum-CLI tool can be used for export of the HuggingFace Whisper model.

Example command-line to obtain FP32 whisper_large model:

```bash

optimum-cli export onnx --model openai/whisper-large E:\model_store\optimum\whisper_large --task automatic-speech-recognition-with-past --opset 20

```

Example command-line to obtain FP16 whisper_medium model:

```bash

optimum-cli export onnx --model openai/whisper-medium E:\model_store\optimum\whisper_medium --task automatic-speech-recognition-with-past --opset 20 --dtype fp16 --device cuda

```

## Install dependencies

1. Install ModelOpt along with its dependencies (ModelOpt's onnx module installation).
1. Install dependencies mentioned in `requirements.txt` file.

```bash

pip install -r requirements.txt

```

### Optional: Installing latest PyTorch and TorchAudio versions (2.8+)

For users who need the latest versions of PyTorch and TorchAudio (version >=2.8), follow these additional steps:

1. **Install PyTorch and TorchAudio with CUDA support:**

```bash

pip install --upgrade torch torchaudio --index-url https://download.pytorch.org/whl/cu128

```

1. **Install torchcodec:**

```bash

pip install torchcodec

```

1. **Set up FFMPEG dependencies:**
   - Download FFMPEG from: <https://www.gyan.dev/ffmpeg/builds/packages/ffmpeg-7.1.1-full_build-shared.7z>
   - Extract the archive
   - Copy all DLL files from the `bin` folder of the extracted FFMPEG directory and paste them into the torchcodec package folder (typically located at `<Python_Root_Folder>/Lib/site-packages/torchcodec`)

1. **Copy PyTorch DLL files:**
   - Navigate to the `lib` folder in the torch package directory
   - Copy all DLL files from this folder and paste them into the torchcodec package folder

After completing these steps, torchaudio will be ready for use.

## Inference script

The script `whisper_optimum_ort_inference.py` is for Optimum-ORT based inference of an ONNX Whisper model. It takes an audio file (.wav) as input and transcribes its content in english. This script also supports Word Error Rate (WER) accuracy measurement.

To run the inference of Whisper ONNX model, relevant files like encoder_model.onnx, decoder_model.onnx, decoder_with_past_model.onnx, tokenizer files, config.json, generation_config.json, vocab file etc. should be kept together in a directory and should provide that directory path to the inference script.

Useful parameters:

| **Argument** | **Description** |
|---------------------------|------------------------------------------------------------------------------------------------------|
| `--model_name` | Specifies the HuggingFace model ID |
| `--onnx_model_dir` | Specifies the directory contains all relevant files for the ONNX model. |
| `--inference_ep` | Specifies the EP to use for inference. Default is CUDA EP. |
| `--test_samples_count` | Specifies the count of samples to use for inference. Default is 100. |
| `--log_model_output` | Specifies whether to log inference output of the model. Default is off. |
| `--cache_dir` | Specifies the cache directory to use for HuggingFace files |
| `--dtype` | Data-type of the model's tensors. Choose from `fp32`, `fp16`. Default is `fp32`. |
| `--audio_file_path` | Path of the input audio file in .wav format. |
| `--run_wer_test` | If True, runs WER accuracy benchmarking using samples from librispeech_asr dataset. Default is False.|

Please refer the script for more details.

Example command-line:

```bash

python .\whisper_optimum_ort_inference.py --model_name=openai/whisper-large --onnx_model_dir=E:\whisper_large \
                                          --audio_file_path=E:\demo.wav \
                                          --run_wer_test --test_samples_count=50

```

A sample audio file (.wav) is provided with this example (*file*: `demo.wav`).

## Quantization script

The script `whisper_onnx_quantization.py` supports various quantization schemes for the given ONNX whisper model.

Following are some useful parameters of this script. Please refer the script for more details.

| **Argument** | **Description** |
|---------------------------|--------------------------------------------------------------------------------------------------------------|
| `--model_name` | HuggingFace model ID |
| `--base_model_dir` | Directory containing all relevant files for the exported ONNX model |
| `--onnx_path` | Input .onnx file path |
| `--output_path` | Output .onnx file path. |
| `--calib_method` | Calibration method for quantization (`max` or `entropy`). Default is `max`. |
| `--quant_mode` | Quantization mode to be used (`int8` or `fp8`). Default is `int8`. |
| `--calib_size` | Number of input calibration samples. Default is `32`. |
| `--batch_size` | Batch size for calibration samples. Default is `1`. |
| `--use_random_calib` | True when we want to use one randomly generated calibration sample. Default is `False`. |
| `--qdq_for_weights` | If True, Q->DQ nodes will be added for weights, otherwise only DQ nodes will be added. Default is `False`. |
| `--calibration_eps` | Comma-separated list of calibration endpoints. Choose from 'cuda', 'cpu', 'dml'. Default is \[`cuda`, `cpu`\]. |
| `--cache_dir` | Cache directory for HuggingFace files. Change this as needed on your system. |
| `--dtype` | Data-type of the model's tensors. Choose from `fp32`, `fp16`. Default is `fp32`. |

See below for example command-lines.

```bash
python .\whisper_onnx_quantization.py --model_name=openai/whisper-large --base_model_dir=E:\whisper_large\base \
                                      --onnx_path=E:\whisper_large\base\encoder_model.onnx \
                                      --output_path=E:\whisper_large\quant_output\encoder_model.onnx

```

```bash
python .\whisper_onnx_quantization.py --model_name=openai/whisper-large --base_model_dir=E:\whisper_large\base \
                                      --onnx_path=E:\whisper_large\base\decoder_model.onnx \
                                      --output_path=E:\whisper_large\quant_output\decoder_model.onnx
```

- Make sure to use GPU-compatible version of dependencies (torch, torchaudio etc.). For instance, installing cu128 binaries of torch and torchaudio should support RTX 5090 Blackwell GPUs.

- The Whisper quantization script supports quantization of following Whisper ONNX files: `encoder_model.onnx`, `decoder_model.onnx`, `decoder_with_past_model.onnx`.

- In case, ONNX installation unexpectedly throws error, then one can try with other ONNX versions.

## Validated Settings

These scripts are currently validated with following settings:

- Python 3.11.9
- CUDA settings on Host - CUDA 12.4, cuDNN 9.5 (cudnn-windows-x86_64-9.5.0.50_cuda12-archive)
- Windows11 22621
- RTX 4090
- Base PyTorch model - [HuggingFace openai\\whisper-large model](https://huggingface.co/openai/whisper-large)
- ONNX Exporter - HuggingFace Optimum, opset-20, FP32 ONNX model
- Inference EP - CUDA EP
- Test samples count - 100 (for inference / WER-accuracy-benchmarking)
- Quantization algos - INT8 with `Max` calibration (W8A8), FP8 with `Max` calibration (W8A8)
- Calibration size - 32
- Calibration EPs - \[`cuda`, `cpu`\]
- Audio dataset - `librispeech_asr` dataset (32 samples used for calibration, 100+ samples used for WER test)
  - `load_dataset("librispeech_asr", "clean", split="test", trust_remote_code=True)`
- Quantization support for various ONNX files - `encoder_model.onnx`, `decoder_model.onnx`, `decoder_with_past_model.onnx`
- The `use_merged` argument in optimum-ORT's Whisper model API is kept False.

## Troubleshoot

1. This example demonstrates quantization and inference using `librispeech_asr` dataset. In case of any issue with dataset or load-dataset, one can hookup loading any other dataset as needed.

   - Try out streaming option in load_dataset API (see [this](https://github.com/huggingface/datasets/issues/4609) github issue about load-dataset of ASR dataset).
   - Try out different splits for calibration and inference.
   - Try out different ASR datasets etc.
