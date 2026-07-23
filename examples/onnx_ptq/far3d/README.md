# FAR3D ONNX PTQ and Argoverse 2 evaluation

This example quantizes the FAR3D image encoder to INT8 with Model Optimizer and evaluates the complete encoder-decoder pipeline on the Argoverse 2 validation set. It follows the [NVIDIA DL4AGX FAR3D workflow](https://github.com/NVIDIA/DL4AGX/tree/master/AV-Solutions/far3d-trt).

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

The second patch makes the unused FlashAttention implementation optional. The reference FAR3D configuration uses MMCV `MultiheadAttention`, so the CUDA 11-only `flash-attn==0.2.8` extension is not required.

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

Extract 500 batches sampled every 20 frames from the Argoverse 2 validation loader:

```bash
/opt/far3d/bin/python /opt/modelopt/examples/onnx_ptq/far3d/prepare_calibration.py \
  dependencies/Far3D/projects/configs/far3d.py \
  data/far3d_calibration \
  --num-samples 500 \
  --sample-skip-interval 20
```

The calibration directory is approximately 25 GiB at the reference model's 960x640 resolution.

## 4. Quantize the encoder

Use the base Python environment for Model Optimizer:

```bash
python /opt/modelopt/examples/onnx_ptq/far3d/quantize.py \
  far3d.encoder.onnx \
  data/far3d_calibration \
  --output-path far3d.encoder.int8.onnx
```

The quantizer preserves the accuracy-sensitive exclusions used by the DL4AGX reference: the `OSA4_5` block and nodes downstream of `lateral_convs` remain in high precision.

Build both engines in the same container. Serialized TensorRT engines are not portable across TensorRT versions or GPU architectures.

```bash
trtexec \
  --onnx=far3d.encoder.int8.onnx \
  --saveEngine=far3d.encoder.int8.engine \
  --stronglyTyped \
  --skipInference
trtexec \
  --onnx=far3d.decoder.onnx \
  --saveEngine=far3d.decoder.fp16.engine \
  --stronglyTyped \
  --skipInference
```

## 5. Evaluate accuracy

```bash
/opt/far3d/bin/python /opt/modelopt/examples/onnx_ptq/far3d/evaluate.py \
  dependencies/Far3D/projects/configs/far3d.py \
  far3d.encoder.int8.engine \
  far3d.decoder.fp16.engine
```

The evaluator runs every Argoverse 2 validation frame and reports the dataset metrics, including mAP. The DL4AGX reference reports 0.230 mAP for its INT8 encoder and FP16 decoder, compared with 0.232 mAP for FP16 encoder and decoder. Exact results can vary with TensorRT version and target GPU.

Use `--max-samples N` for an inference smoke test. Dataset metrics are skipped when only part of the validation set is processed.
