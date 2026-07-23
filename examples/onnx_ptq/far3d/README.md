# FAR3D ONNX PTQ and Argoverse 2 evaluation

This example quantizes the FAR3D image encoder to INT8 with Model Optimizer and evaluates the complete encoder-decoder pipeline on the Argoverse 2 validation set. It follows the [NVIDIA DL4AGX FAR3D workflow](https://github.com/NVIDIA/DL4AGX/tree/master/AV-Solutions/far3d-trt).

FAR3D uses a legacy PyTorch/MMCV environment that is incompatible with the current Model Optimizer Python dependencies. The workflow therefore uses the FAR3D export environment for data preparation, export, and evaluation, and a current TensorRT container for ONNX quantization and engine building.

## 1. Prepare FAR3D and Argoverse 2

Clone DL4AGX, initialize its submodules, and apply its FAR3D patch:

```bash
git clone https://github.com/NVIDIA/DL4AGX.git
cd DL4AGX
git submodule update --init --recursive
cd AV-Solutions/far3d-trt/dependencies/Far3D
git apply ../../patch/far3d.patch
cd ../..
```

Download the [Argoverse 2 sensor validation set](https://www.argoverse.org/av2.html), the [reference FAR3D checkpoint](https://github.com/NVIDIA/DL4AGX/tree/master/AV-Solutions/far3d-trt#pytorch-model-to-onnx), and its configuration. The remaining commands assume:

```text
far3d-trt/
├── data/av2/val/
├── dependencies/Far3D/projects/configs/far3d.py
└── weights/iter_82548.pth
```

Build the DL4AGX container as documented upstream, then extend it with the TensorRT version used by this example:

```bash
cd docker
docker build --network=host -t far3d .
docker build \
  --build-arg FAR3D_IMAGE=far3d \
  -f /path/to/Model-Optimizer/examples/onnx_ptq/far3d/Dockerfile \
  -t far3d-modelopt \
  /path/to/Model-Optimizer
```

Start the extended image and mount this Model Optimizer checkout at `/workspace/modelopt`:

```bash
docker run --rm -it --network=host --gpus=all --shm-size=80G --privileged \
  -v /data/av2:/data/av2 \
  -v "$(pwd)/../:/workspace/far3d-trt" \
  -v /path/to/Model-Optimizer:/workspace/modelopt \
  far3d-modelopt
```

Inside the FAR3D container, prepare the validation metadata:

```bash
export PYTHONPATH=/workspace/far3d-trt/dependencies/Far3D
cd /workspace/far3d-trt
python /workspace/modelopt/examples/onnx_ptq/far3d/prepare_metadata.py data/av2
```

## 2. Export the ONNX models

```bash
python tools/export_onnx.py \
  dependencies/Far3D/projects/configs/far3d.py \
  weights/iter_82548.pth
```

This produces `far3d.encoder.onnx` and `far3d.decoder.onnx`.

## 3. Prepare calibration batches

Extract 500 batches sampled every 20 frames from the Argoverse 2 validation loader:

```bash
python /workspace/modelopt/examples/onnx_ptq/far3d/prepare_calibration.py \
  dependencies/Far3D/projects/configs/far3d.py \
  data/far3d_calibration \
  --num-samples 500 \
  --sample-skip-interval 20
```

The calibration directory is approximately 25 GiB at the reference model's 960x640 resolution.

## 4. Quantize the encoder

Exit the FAR3D container and start a current TensorRT container with the FAR3D workspace and Model Optimizer checkout mounted:

```bash
docker run --rm -it --gpus=all --shm-size=80G \
  -v /path/to/far3d-trt:/workspace/far3d-trt \
  -v /path/to/Model-Optimizer:/workspace/modelopt \
  -w /workspace/far3d-trt \
  nvcr.io/nvidia/tensorrt:25.06-py3
```

Inside the TensorRT container:

```bash
export CUDNN_LIB_DIR=/usr/lib/x86_64-linux-gnu
export LD_LIBRARY_PATH="${CUDNN_LIB_DIR}:${LD_LIBRARY_PATH}"
python -m pip install -e '/workspace/modelopt[onnx]'

python /workspace/modelopt/examples/onnx_ptq/far3d/quantize.py \
  far3d.encoder.onnx \
  data/far3d_calibration \
  --output-path far3d.encoder.int8.onnx
```

The quantizer preserves the accuracy-sensitive exclusions used by the DL4AGX reference: the `OSA4_5` block and nodes downstream of `lateral_convs` remain in high precision.

Build both engines before exiting the TensorRT container. Serialized TensorRT engines are not portable across TensorRT versions or GPU architectures.

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

Return to the `far3d-modelopt` container, which provides the matching TensorRT Python runtime:

```bash
export PYTHONPATH=/workspace/far3d-trt/dependencies/Far3D
cd /workspace/far3d-trt

python /workspace/modelopt/examples/onnx_ptq/far3d/evaluate.py \
  dependencies/Far3D/projects/configs/far3d.py \
  far3d.encoder.int8.engine \
  far3d.decoder.fp16.engine
```

The evaluator runs every Argoverse 2 validation frame and reports the dataset metrics, including mAP. The DL4AGX reference reports 0.230 mAP for its INT8 encoder and FP16 decoder, compared with 0.232 mAP for FP16 encoder and decoder. Exact results can vary with TensorRT version and target GPU.
