# BEVFormer ONNX PTQ and nuScenes evaluation

This example exports BEVFormer-tiny to ONNX, prepares temporal calibration data, quantizes the model to INT8 or FP8 with Model Optimizer, builds TensorRT engines, and evaluates NDS and mAP on the nuScenes validation set. It extends the [NVIDIA DL4AGX BEVFormer INT8 workflow](https://github.com/NVIDIA/DL4AGX/tree/9f7b29104c253d5bc68334e7b83b3eecb72d4572/AV-Solutions/bevformer-int8-eq) with FP8 and reuses the calibration reader shared by the Model Optimizer FAR3D and PETR examples.

The container pins [BEVFormer_tensorrt](https://github.com/DerryHub/BEVFormer_tensorrt/tree/303d3140c14016047c07f9db73312af364f0dd7c) and applies the TensorRT 10 compatibility patch from DL4AGX. The exported model uses custom TensorRT plugins, so export, calibration, engine building, and evaluation must use the plugin library built in the container.

## 1. Build and start the container

Build from the Model Optimizer repository root:

```bash
docker build \
  -f examples/onnx_ptq/bevformer/Dockerfile \
  -t bevformer-modelopt .
```

Download the nuScenes v1.0 trainval set and CAN bus expansion data. Use the dataset, CAN bus data, and checkpoint only under their upstream terms, including the [nuScenes terms of use](https://www.nuscenes.org/terms-of-use). Start the container with the dataset and an artifact directory mounted:

```bash
mkdir -p bevformer_artifacts
docker run --rm -it --gpus=all --network=host --shm-size=20g \
  -v /path/to/nuscenes:/workspace/BEVFormer_tensorrt/data/nuscenes \
  -v /path/to/can_bus:/workspace/BEVFormer_tensorrt/data/can_bus \
  -v "$(pwd)/bevformer_artifacts:/artifacts" \
  bevformer-modelopt
```

The remaining commands run inside the container:

```bash
export BEVFORMER_ROOT=/workspace/BEVFormer_tensorrt
export DL4AGX_ROOT=/workspace/DL4AGX
export MODELOPT_ROOT=/opt/Model-Optimizer
export PLUGIN_PATH=${BEVFORMER_ROOT}/TensorRT/lib/libtensorrt_ops.so
export CONFIG=${BEVFORMER_ROOT}/configs/bevformer/plugin/bevformer_tiny_trt_p2.py
cd ${BEVFORMER_ROOT}
```

Build the custom plugins inside the GPU-enabled container. The build targets the compute capability of its active GPU:

```bash
cmake -S ${BEVFORMER_ROOT}/TensorRT \
  -B ${BEVFORMER_ROOT}/TensorRT/build \
  -DCMAKE_TENSORRT_PATH=/usr
cmake --build ${BEVFORMER_ROOT}/TensorRT/build --parallel
cmake --install ${BEVFORMER_ROOT}/TensorRT/build
```

Generate the temporal train and validation metadata required by BEVFormer:

```bash
bash samples/bevformer/create_data.sh
```

This creates the following files in the mounted nuScenes directory:

```text
nuscenes/
├── nuscenes_infos_temporal_train.pkl
└── nuscenes_infos_temporal_val.pkl
```

## 2. Export BEVFormer to ONNX

Download the published BEVFormer-tiny checkpoint:

```bash
wget --continue \
  -O /artifacts/bevformer_tiny_epoch_24.pth \
  https://github.com/zhiqi-li/storage/releases/download/v1.0/bevformer_tiny_epoch_24.pth

echo "7305046dbaa4fe8b1fa6d6acb9e0e3d605a70a3c473f763e936103428d2b2f12  /artifacts/bevformer_tiny_epoch_24.pth" | \
  sha256sum --check
```

Export the `nv_half2` plugin variant at opset 13 and copy it to the artifact directory:

```bash
python tools/pth2onnx.py \
  ${CONFIG} \
  /artifacts/bevformer_tiny_epoch_24.pth \
  --opset_version=13 \
  --cuda \
  --flag=cp2_op13

cp checkpoints/onnx/bevformer_tiny_epoch_24_cp2_op13.onnx /artifacts/
export ONNX_PATH=/artifacts/bevformer_tiny_epoch_24_cp2_op13.onnx
```

Post-process a copy for ONNX Runtime. This copy is used only while generating calibration data; quantization uses the original export:

```bash
python ${DL4AGX_ROOT}/AV-Solutions/bevformer-int8-eq/tools/onnx_postprocess.py \
  --onnx=${ONNX_PATH} \
  --trt_plugins=${PLUGIN_PATH}

export CALIBRATION_ONNX=/artifacts/bevformer_tiny_epoch_24_cp2_op13_post.onnx
```

## 3. Prepare calibration data

Generate 600 temporal samples from the nuScenes training split:

```bash
PYTHONPATH=${BEVFORMER_ROOT} \
python ${MODELOPT_ROOT}/examples/onnx_ptq/bevformer/prepare_calibration.py \
  ${CONFIG} \
  --onnx=${CALIBRATION_ONNX} \
  --trt-plugin=${PLUGIN_PATH} \
  --output-dir=/artifacts/calibration \
  --num-samples=600
```

The script runs the post-processed model with ONNX Runtime and carries `prev_bev` across frames while resetting it between scenes. It saves one NPZ file per sample instead of building one large archive in memory. The output directory must be empty.

## 4. Quantize to INT8 and FP8

INT8 uses entropy calibration by default. All `MatMul` nodes remain in FP16, matching the DL4AGX recommendation:

```bash
python ${MODELOPT_ROOT}/examples/onnx_ptq/bevformer/quantize.py \
  --onnx=${ONNX_PATH} \
  --calibration-dir=/artifacts/calibration \
  --trt-plugins=${PLUGIN_PATH} \
  --quantization-mode=int8 \
  --output=/artifacts/bevformer_tiny_epoch_24_cp2_op13.int8.onnx
```

FP8 uses max calibration by default and requires a GPU with compute capability 8.9 or later:

```bash
python ${MODELOPT_ROOT}/examples/onnx_ptq/bevformer/quantize.py \
  --onnx=${ONNX_PATH} \
  --calibration-dir=/artifacts/calibration \
  --trt-plugins=${PLUGIN_PATH} \
  --quantization-mode=fp8 \
  --output=/artifacts/bevformer_tiny_epoch_24_cp2_op13.fp8.onnx
```

The custom TensorRT plugins and all `MatMul` nodes remain at higher precision. Model Optimizer automatically upgrades the FP8 model to the required ONNX opset.

## 5. Build TensorRT engines

Build the FP16 baseline from the original ONNX model:

```bash
trtexec \
  --onnx=${ONNX_PATH} \
  --saveEngine=/artifacts/bevformer_tiny_epoch_24_cp2_op13.fp16.engine \
  --staticPlugins=${PLUGIN_PATH} \
  --fp16 \
  --skipInference
```

Build strongly typed engines from the explicitly quantized models:

```bash
for precision in int8 fp8; do
  trtexec \
    --onnx=/artifacts/bevformer_tiny_epoch_24_cp2_op13.${precision}.onnx \
    --saveEngine=/artifacts/bevformer_tiny_epoch_24_cp2_op13.${precision}.engine \
    --staticPlugins=${PLUGIN_PATH} \
    --stronglyTyped \
    --skipInference
done
```

Serialized TensorRT engines are specific to the TensorRT version and GPU architecture used to build them.

## 6. Evaluate accuracy and latency

Evaluate all 6,019 nuScenes validation samples:

```bash
for precision in fp16 int8 fp8; do
  python tools/bevformer/evaluate_trt.py \
    ${CONFIG} \
    /artifacts/bevformer_tiny_epoch_24_cp2_op13.${precision}.engine \
    --trt_plugins=${PLUGIN_PATH} | \
    tee /artifacts/evaluate_${precision}.log
done
```

Measure TensorRT GPU compute time independently of data loading and post-processing:

```bash
for precision in fp16 int8 fp8; do
  trtexec \
    --loadEngine=/artifacts/bevformer_tiny_epoch_24_cp2_op13.${precision}.engine \
    --staticPlugins=${PLUGIN_PATH} \
    --warmUp=1000 \
    --duration=10 \
    --iterations=100 | \
    tee /artifacts/trtexec_${precision}.log
done
```

## Results on nuScenes validation

Measurements use an NVIDIA RTX 6000 Ada Generation GPU (compute capability 8.9), TensorRT 10.14.1.48, CUDA 13.1, and ONNX Runtime 1.24.0.dev20260123002. Quantization uses 600 training samples. NDS and mAP are produced after all 6,019 validation samples complete; GPU compute time is the median reported by the command above and excludes data loading and post-processing.

The INT8 and FP8 rows are mixed-precision graphs. Quantized operators use the listed format, while the custom TensorRT plugins, excluded `MatMul` nodes, and other unsupported paths remain in FP16. External inputs and outputs remain FP32.

| Precision | TensorRT GPU compute time (median, ms) | NDS | mAP |
| --- | ---: | ---: | ---: |
| FP16 | 4.597 | 0.3546 | 0.2515 |
| INT8/FP16 | 3.180 | 0.3512 | 0.2505 |
| FP8/FP16 | 4.106 | 0.3526 | 0.2489 |
