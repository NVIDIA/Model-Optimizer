# FAR3D ONNX PTQ and Argoverse 2 evaluation

This example quantizes the FAR3D image encoder and decoder to INT8 or FP8 with Model Optimizer and evaluates the complete pipeline on the Argoverse 2 validation set. It follows the [NVIDIA DL4AGX FAR3D workflow](https://github.com/NVIDIA/DL4AGX/tree/master/AV-Solutions/far3d-trt).

FAR3D uses a legacy PyTorch/MMCV environment that is incompatible with the current Model Optimizer Python dependencies. The provided image uses `nvcr.io/nvidia/pytorch:25.06-py3` for the complete workflow and isolates the legacy FAR3D packages in a Python 3.8 virtual environment.

## 1. Prepare FAR3D and Argoverse 2

Clone DL4AGX, initialize its submodules, and apply its FAR3D patch:

```bash
git clone https://github.com/NVIDIA/DL4AGX.git
cd DL4AGX
git submodule update --init --recursive
cd AV-Solutions/far3d-trt/dependencies/Far3D
git apply ../../patch/far3d.patch
git apply /path/to/Model-Optimizer/examples/onnx_ptq/far3d/far3d_pytorch_25_06.patch
cd ../..
```

The second patch makes the unused CUDA 11-only FlashAttention implementation optional; the reference configuration uses MMCV `MultiheadAttention`.

Download the [Argoverse 2 sensor validation set](https://www.argoverse.org/av2.html), the [reference FAR3D checkpoint](https://github.com/NVIDIA/DL4AGX/tree/master/AV-Solutions/far3d-trt#pytorch-model-to-onnx), and its configuration. The remaining commands assume:

```text
far3d-trt/
├── data/av2/val/
├── dependencies/Far3D/projects/configs/far3d.py
└── weights/iter_82548.pth
```

Build the example image from the Model Optimizer checkout:

```bash
docker build \
  -f /path/to/Model-Optimizer/examples/onnx_ptq/far3d/Dockerfile \
  -t far3d-modelopt \
  /path/to/Model-Optimizer
```

Start the image and mount the FAR3D checkout:

```bash
docker run --rm -it --network=host --gpus=all --shm-size=80G --privileged \
  -v /data/av2:/data/av2 \
  -v /path/to/far3d-trt:/workspace/far3d-trt \
  far3d-modelopt
```

Use `/opt/far3d/bin/python` for data preparation, export, and evaluation. It selects the isolated legacy FAR3D environment:

```bash
export PYTHONPATH=/workspace/far3d-trt/dependencies/Far3D
cd /workspace/far3d-trt
/opt/far3d/bin/python /opt/modelopt/examples/onnx_ptq/far3d/prepare_metadata.py data/av2
```

## 2. Export the ONNX models

```bash
/opt/far3d/bin/python tools/export_onnx.py \
  dependencies/Far3D/projects/configs/far3d.py \
  weights/iter_82548.pth
```

This produces `far3d.encoder.onnx` and `far3d.decoder.onnx`.

## 3. Prepare calibration batches

Build temporary engines from the exported models. They run the reference pipeline while collecting representative encoder and decoder inputs:

```bash
trtexec \
  --onnx=far3d.encoder.onnx \
  --saveEngine=far3d.encoder.fp16.engine \
  --fp16 \
  --skipInference
trtexec \
  --onnx=far3d.decoder.onnx \
  --saveEngine=far3d.decoder.fp16.engine \
  --stronglyTyped \
  --skipInference
```

Extract 512 batches sampled every 20 frames from the Argoverse 2 validation loader:

```bash
/opt/far3d/bin/python /opt/modelopt/examples/onnx_ptq/far3d/prepare_calibration.py \
  dependencies/Far3D/projects/configs/far3d.py \
  data/far3d_calibration \
  --encoder-engine far3d.encoder.fp16.engine \
  --decoder-engine far3d.decoder.fp16.engine \
  --num-samples 512 \
  --sample-skip-interval 20
```

The calibration directory contains separate `encoder/` and `decoder/` batches. Decoder batches include the image features, camera geometry, and temporal state seen by the reference decoder.

## 4. Quantize the models

Use the base Python environment for Model Optimizer:

```bash
python /opt/modelopt/examples/onnx_ptq/far3d/quantize.py \
  far3d.encoder.onnx \
  far3d.decoder.onnx \
  data/far3d_calibration
```

Both models use max calibration. INT8 is the default; use `--quantization-mode fp8` to produce `far3d.encoder.fp8.onnx` and `far3d.decoder.fp8.onnx` instead. FP8 deployment requires an FP8-capable GPU.

The quantizer preserves the accuracy-sensitive exclusions used by the DL4AGX reference: the `OSA4_5` block and nodes downstream of `lateral_convs` remain in high precision.

To keep the decoder in its original mixed FP16/FP32 precision, add `--fp16-decoder`; decoder calibration batches are not required in that mode. This flag can be combined with either quantization mode.

Build both engines in the same container. Serialized TensorRT engines are not portable across TensorRT versions or GPU architectures.

Set the precision to the quantization mode used above:

```bash
precision=int8  # Use fp8 for FP8 models.
trtexec \
  --onnx=far3d.encoder.${precision}.onnx \
  --saveEngine=far3d.encoder.${precision}.engine \
  --stronglyTyped \
  --skipInference
trtexec \
  --onnx=far3d.decoder.${precision}.onnx \
  --saveEngine=far3d.decoder.${precision}.engine \
  --stronglyTyped \
  --skipInference
```

When using `--fp16-decoder`, build `far3d.decoder.onnx` as `far3d.decoder.fp16.engine` instead.

## 5. Evaluate accuracy

```bash
precision=int8  # Use fp8 for FP8 models.
/opt/far3d/bin/python /opt/modelopt/examples/onnx_ptq/far3d/evaluate.py \
  dependencies/Far3D/projects/configs/far3d.py \
  far3d.encoder.${precision}.engine \
  far3d.decoder.${precision}.engine
```

Use `--max-samples N` for an inference smoke test. Dataset metrics are skipped when only part of the validation set is processed.

## Results on Argoverse 2 validation set

The following results use TensorRT 10.11.0.33 on an NVIDIA RTX 6000 Ada Generation GPU. Model quantization uses PyTorch 2.8.0a0 from the 25.06 PyTorch container, while the FAR3D export and evaluation environment uses PyTorch 1.13.1. Accuracy is measured over all 23,522 validation frames after calibration with 512 batches sampled every 20 frames.

| Encoder precision | Decoder precision | Framework | GPU compute time (ms) | Accuracy (mAP) |
| --- | --- | --- | ---: | ---: |
| INT8 | INT8 | TensorRT 10.11 | 25.4 | 0.124 |
| FP8 | FP8 | TensorRT 10.11 | 31.2 | 0.000 |
| FP8 | FP16 | TensorRT 10.11 | 31.5 | 0.241 |

GPU compute time is the sum of the encoder and decoder median times reported by `trtexec`, with host-to-device and device-to-host transfers disabled. Results depend on the TensorRT version and GPU architecture and are not directly comparable with the DRIVE Orin-X measurements in the [DL4AGX reference](https://github.com/NVIDIA/DL4AGX/tree/master/AV-Solutions/far3d-trt#results-on-argoverse2-validation-set).
