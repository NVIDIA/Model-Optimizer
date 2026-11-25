# Scripts for ONNX SAM2 model

This repository contains an example to demontrate 8-bit quantization of SAM2 ONNX model.

## Table of Contents

- [ONNX export and Inference tool](#onnx-export-and-inference-tool)
- [Quantization](#quantization)
- [Support Matrix](#support-matrix)
- [Validated Settings](#validated-settings)
- [Troubleshoot](#troubleshoot)

## ONNX export and Inference tool

The [samexporter](https://github.com/vietanhdev/samexporter) tool can be used for producing ONNX SAM2 model and for its inference with image inputs. Refer it for its installation and usage details. It internally uses `torch.onnx.export` for doing ONNX export.

- Opset of the ONNX base model (or exported ONNX model) should be 19+ to support FP8 quantization.

> Use separate python virtual environment for running the samexporter tool. ModelOpt toolkit and samexporter tool can have conflicting or incompatible dependencies.

> Currently, the samexporter installation doesn't seem to work out-of-the-box with python 3.12 as samexporter mentions dependency on onnxruntime 1.16.3 which doesn't support python 3.12. So, while working with python 3.12, one can try installing samexporter from source after updating the onnxruntime version to 1.20.0+ in samexporter project's dependencies list (file: pyproject.toml). Also, samexporter installation with python 3.12 is observed to work with [cmake 3.27.7](https://github.com/Kitware/CMake/releases/tag/v3.27.7), but it produced error with cmake 4.0.0-rc4. Make sure to update `PATH` environment variable for cmake/bin directory and then restart the command-line.

> By default, samexporter tool comes with `onnxruntime` package (i.e. CPU EP). For inference with CUDA EP, one needs to uninstall existing `onnxruntime` package, and then install `onnxruntime-gpu` (1.20.x) package. Make sure that after this update, `python -c "import onnxruntime"` runs successfully.

## Quantization

The script `sam2_onnx_quantization.py` supports INT8 (W8A8) and FP8 (W8A8) quantization schemes for `encoder` of the ONNX exported SAM2 model. To use it, install ModelOpt toolkit along with dependencies.

> Install ModelOpt along with its dependencies (ModelOpt's onnx module installation). Install dependencies mentioned in `requirements.txt` file.

```bash
pip install -r requirements.txt
```

Some useful parameters:

| **Argument** | **Description** |
|---------------------------|--------------------------------------------------------------------------------------------------------------|
| `--onnx_path` | Input .onnx file path |
| `--output_path` | Output .onnx file path. |
| `--calib_method` | Calibration method for quantization (`max` or `entropy`). Default is `max`. |
| `--quant_mode` | Quantization mode to be used (`int8` or `fp8`). Default is `int8`. |
| `--calib_size` | Number of input calibration samples. Default is `32`. |
| `--use_random_calib` | True when we want to use one randomly generated calibration sample. Default is `False`. |
| `--qdq_for_weights` | If True, Q->DQ nodes will be added for weights, otherwise only DQ nodes will be added. Default is `False`. |
| `--calibration_eps` | Comma-separated list of calibration endpoints. Choose from 'cuda', 'cpu', 'dml'. Default is "`cuda`, `cpu`". |
| `--dtype` | Data-type of the model's tensors. Choose from `fp32`, `fp16`. Default is `fp32`. |
| `--image_dir` | Directory containing image files to be used for calibration of sam2's encoder. |
| `--image_file_extension` | Extension of image files to be used for calibration of sam2's encoder. E.g. `jpg` (default), `png`. |
| `--image_input_dimension` | Last 2 dimensions of the image input to encoder, in comma-separated fashion. Default: `1024,1024` |
| `--op_types_to_quantize` | Comma-separated list of op-types to quantize. Choose from 'MatMul', 'Conv'. Default is "`MatMul`". |

Please refer the script for more details.

Example command-line:

```bash

python .\sam2_onnx_quantization.py --onnx_path=E:\base\sam2_hiera_large.encoder.onnx --output_path=E:\quant\sam2_hiera_large.encoder.onnx --image_dir=E:\sam_image_dataset

```

## Support Matrix

| Model | ONNX INT8 Max (W8A8) | ONNX FP8 Max (W8A8) |
| :---: | :---: | :---: |
| sam2-hiera-large | ✅ | ✅ |

> *`ONNX INT8 Max` means INT8 (W8A8) quantization of ONNX model using Max calibration. Similar holds true for the term `ONNX FP8 Max`.*

## Validated Settings

This example is currently validated with following settings:

- Python 3.11.9
- CUDA-12.4
- ONNX exported FP32 `sam2_hiera_large` opset-19 model - exported using [samexporter](https://github.com/vietanhdev/samexporter) tool
- Quantization settings:
  - onnx 1.17.0
  - onnxruntime-gpu 1.20.1 (for ORT-CUDA EP)
  - Quantization algos - INT8 with `Max` calibration (W8A8) and DQ-only mode for weights, FP8 with `Max` calibration (W8A8) with both DQ-only and QDQ models for weights
  - op-types-to-quantize = `MatMul`
  - Calibration size - 32
  - Calibration EPs - \[`cuda`, `cpu`\]
  - Calibration data - [SA-1B](https://ai.meta.com/datasets/segment-anything-downloads/) - download and extract sa_000000.tar file.
  - Quantization support for various ONNX files - `encoder` model is quantized, `decoder` model is not quantized (not needed since decoder is already very small ~20MB).
  - A separate python virtual environment used for quantization (`python -m venv .\venv_quantization`)
- Inference settings:
  - tool used for inference: [samexporter](https://github.com/vietanhdev/samexporter)
  - ORT-CUDA EP (needed a minor patch in `samexporter` to pass \[`cuda`, `cpu`\] as providers, file: sam2_onnx.py)
  - tasks - image prediction or segmentation
  - precision of ONNX models - `encoder` model is quantized, `decoder` model is not quantized.
  - test samples - a few image-inputs examples from [samexporter](https://github.com/vietanhdev/samexporter)

## Troubleshoot

1. If `samexporter` tool gives error about tensor mismatch or issue in load-state-dictionary, then try hardcoding the absolute path of config file in the tool, as a workaround (file: export_sam2.py).

1. Sometimes using `samexporter` with numpy 2.x gives error (e.g. ImportError: numpy.core.multiarray failed to import). Try using Numpy 1.26.2.

   ```bash
   pip install numpy==1.26.2
   ```

1. If inference of the FP8 (dq-only for weights) model results in error about presence of both INT8 and FP8 types in a DQ node, then that would probably be due to error in FP8 type detection by ONNX during quantization. To troubleshoot, we suggest to try out quantization with ONNX latest wheel (1.17+) and/or latest python (e.g. 3.12), in a fresh new virtual environment. Besides, FP8 with QDQ for weights should work.
