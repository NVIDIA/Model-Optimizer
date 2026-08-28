# FAR3D ONNX PTQ and Argoverse 2 evaluation

This example quantizes the FAR3D image encoder and decoder to INT8 or FP8 with Model Optimizer and evaluates the complete pipeline on the Argoverse 2 validation set. It follows the [NVIDIA DL4AGX FAR3D workflow](https://github.com/NVIDIA/DL4AGX/tree/master/AV-Solutions/far3d-trt).

The provided image uses `nvcr.io/nvidia/pytorch:25.06-py3`, CUDA 12.9, and TensorRT 10.11. FAR3D runs in an isolated Python 3.9 environment with PyTorch 2.8 `cu129` and MMCV compiled against the container's CUDA toolkit; Model Optimizer uses the base Python 3.12 environment.

The evaluator omits MMDetection3D's unused Lyft integration and its `plyfile`, TensorBoard, and KITTI-only scikit-image dependencies.

## 1. Prepare FAR3D and Argoverse 2

Clone DL4AGX, initialize its submodules, and apply its FAR3D patch:

```bash
git clone https://github.com/NVIDIA/DL4AGX.git
cd DL4AGX
git checkout 9f7b29104c253d5bc68334e7b83b3eecb72d4572
git submodule update --init --recursive
cd AV-Solutions/far3d-trt/dependencies/Far3D
git apply ../../patch/far3d.patch
git apply /path/to/Model-Optimizer/examples/onnx_ptq/far3d/far3d_optional_flash_attn.patch
cd ../..
```

The second patch makes the unused legacy FlashAttention implementation optional and removes an unused IPython debug import; the reference configuration uses MMCV `MultiheadAttention`.

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

The default MMCV build targets SM 8.9 and includes PTX for newer GPUs. Set `--build-arg TORCH_CUDA_ARCH_LIST=...` when building for an older GPU.

Start the image and mount the FAR3D checkout:

```bash
docker run --rm -it --gpus=all --shm-size=80G \
  -v /data/av2:/data/av2 \
  -v /path/to/far3d-trt:/workspace/far3d-trt \
  far3d-modelopt
```

Use `/opt/far3d/bin/python` for data preparation, export, and evaluation. It selects the isolated FAR3D environment:

```bash
export PYTHONPATH=/workspace/far3d-trt/dependencies/Far3D
cd /workspace/far3d-trt
/opt/far3d/bin/python /opt/Model-Optimizer/examples/onnx_ptq/far3d/prepare_metadata.py data/av2
```

## 2. Export the ONNX models

```bash
/opt/far3d/bin/python tools/export_onnx.py \
  dependencies/Far3D/projects/configs/far3d.py \
  weights/iter_82548.pth
```

This produces `far3d.encoder.onnx` and `far3d.decoder.onnx`.

## 3. Prepare calibration batches

Build temporary engines from the exported models. The FP16 encoder matches the validated inference pipeline used to collect decoder inputs, while a strongly typed decoder preserves the graph's mixed FP16/FP32 tensor types:

```bash
trtexec \
  --onnx=far3d.encoder.onnx \
  --saveEngine=far3d.encoder.calibration.engine \
  --fp16 \
  --skipInference
trtexec \
  --onnx=far3d.decoder.onnx \
  --saveEngine=far3d.decoder.calibration.engine \
  --stronglyTyped \
  --skipInference
```

Extract 512 batches sampled every 20 frames from the Argoverse 2 validation loader:

```bash
/opt/far3d/bin/python /opt/Model-Optimizer/examples/onnx_ptq/far3d/prepare_calibration.py \
  dependencies/Far3D/projects/configs/far3d.py \
  data/far3d_calibration \
  --encoder-engine far3d.encoder.calibration.engine \
  --decoder-engine far3d.decoder.calibration.engine \
  --num-samples 512 \
  --sample-skip-interval 20
```

The calibration directory contains separate `encoder/` and `decoder/` batches. Decoder batches include the image features, camera geometry, and temporal state seen by the reference decoder.

## 4. Quantize the models

Use the base Python environment for Model Optimizer:

```bash
python /opt/Model-Optimizer/examples/onnx_ptq/far3d/quantize.py \
  --encoder-onnx far3d.encoder.onnx \
  --decoder-onnx far3d.decoder.onnx \
  --calibration-dir data/far3d_calibration
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
/opt/far3d/bin/python /opt/Model-Optimizer/examples/onnx_ptq/far3d/evaluate.py \
  dependencies/Far3D/projects/configs/far3d.py \
  far3d.encoder.${precision}.engine \
  far3d.decoder.${precision}.engine
```

Use `--max-samples N` for an inference smoke test. Dataset metrics are skipped when only part of the validation set is processed.

## Results on Argoverse 2 validation set

The historical results below use TensorRT 10.11.0.33 on an NVIDIA RTX 6000 Ada Generation GPU. Model quantization uses PyTorch 2.8.0a0 from the provided 25.06 PyTorch container, while export and evaluation used the previous PyTorch 1.13.1/CUDA 11.7 environment. Accuracy is measured over all 23,522 validation frames after calibration with 512 batches sampled every 20 frames. Rerun the workflow to measure the current CUDA 12.9 environment.

| Encoder precision | Decoder precision | Framework | GPU compute time (ms) | Accuracy (mAP) |
| --- | --- | --- | ---: | ---: |
| FP32 | FP32 | TensorRT 10.11 | 92.5 | 0.241 |
| FP16 | FP32 | TensorRT 10.11 | 47.8 | 0.241 |
| FP16 | FP16 | TensorRT 10.11 | 45.0 | 0.241 |
| INT8 | FP16 | TensorRT 10.11 | 24.6 | 0.236 |
| FP8 | FP16 | TensorRT 10.11 | 31.5 | 0.241 |

Quantizing the decoder to INT8 or FP8 produced severe accuracy degradation in this evaluation and is not recommended. Keep the decoder in its original mixed FP16/FP32 precision.

GPU compute time is the sum of the encoder and decoder median times reported by `trtexec`, with host-to-device and device-to-host transfers disabled. Results depend on the TensorRT version and GPU architecture and are not directly comparable with the DRIVE Orin-X measurements in the [DL4AGX reference](https://github.com/NVIDIA/DL4AGX/tree/master/AV-Solutions/far3d-trt#results-on-argoverse2-validation-set).
