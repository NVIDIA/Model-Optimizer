# BEVFormer ONNX PTQ and nuScenes evaluation

This example extends the [DL4AGX BEVFormer workflow at source revision `9f7b291`](https://github.com/NVIDIA/DL4AGX/tree/9f7b29104c253d5bc68334e7b83b3eecb72d4572/AV-Solutions/bevformer-int8-eq) with temporal calibration and FP8 quantization. Build the TensorRT 10.14 environment with its [Dockerfile](https://github.com/NVIDIA/DL4AGX/blob/9f7b29104c253d5bc68334e7b83b3eecb72d4572/AV-Solutions/bevformer-int8-eq/docker/tensorrt.Dockerfile). The TensorRT plugins are compiled after the container starts so they target the GPU used for deployment.

## 1. Build and run the container

Clone the pinned DL4AGX revision and build its BEVFormer image. Set `MODELOPT_ROOT` to this Model Optimizer checkout so the container can use the example scripts and current Model Optimizer source.

```bash
export MODELOPT_ROOT=$(git rev-parse --show-toplevel)
export DL4AGX_ROOT=/path/to/DL4AGX
git clone https://github.com/NVIDIA/DL4AGX.git ${DL4AGX_ROOT}
git -C ${DL4AGX_ROOT} checkout --detach \
  9f7b29104c253d5bc68334e7b83b3eecb72d4572
export DL4AGX_EXAMPLE=${DL4AGX_ROOT}/AV-Solutions/bevformer-int8-eq
docker build \
  --build-arg TORCH_CUDA_ARCH_LIST=8.9 \
  --file ${DL4AGX_EXAMPLE}/docker/tensorrt.Dockerfile \
  --tag modelopt-onnx-bevformer \
  ${DL4AGX_EXAMPLE}
```

This build targets compute capability 8.9, matching the NVIDIA RTX 6000 Ada Generation validation system. Set `TORCH_CUDA_ARCH_LIST` to the compute capability of the deployment GPU when using another architecture.

Download nuScenes v1.0 trainval and CAN bus expansion data under the [nuScenes terms of use](https://www.nuscenes.org/terms-of-use). Create a host directory outside the Model Optimizer checkout for generated artifacts, then start the container with the dataset and artifact directories mounted:

```bash
export BEVFORMER_ARTIFACTS=/path/to/bevformer_artifacts
mkdir -p "${BEVFORMER_ARTIFACTS}"
docker run --rm -it --gpus=all --shm-size=20g \
  -e PYTHONPATH=/opt/Model-Optimizer \
  -v "${MODELOPT_ROOT}:/opt/Model-Optimizer:ro" \
  -v "${DL4AGX_EXAMPLE}:/mnt/dl4agx:ro" \
  -v "${BEVFORMER_ARTIFACTS}:/artifacts" \
  -v /path/to/nuscenes:/workspace/BEVFormer_tensorrt/data/nuscenes \
  -v /path/to/can_bus:/workspace/BEVFormer_tensorrt/data/can_bus \
  modelopt-onnx-bevformer
```

The DL4AGX image provides and owns the BEVFormer dependency stack. The read-only Model Optimizer mount and `PYTHONPATH` ensure the commands below use this checkout rather than the Model Optimizer release installed by the image.

Inside the container, set the paths used by the remaining steps:

```bash
export BEVFORMER_ROOT=/workspace/BEVFormer_tensorrt
export MODELOPT_ROOT=/opt/Model-Optimizer
export ARTIFACTS=/artifacts
export CONFIG=${BEVFORMER_ROOT}/configs/bevformer/plugin/bevformer_tiny_trt_p2.py
export ONNX_PATH=${ARTIFACTS}/bevformer_tiny_epoch_24_cp2_op13.onnx
export FP16_ONNX=${ARTIFACTS}/bevformer_tiny_epoch_24_cp2_op13.fp16.onnx
export PLUGIN_PATH=${BEVFORMER_ROOT}/TensorRT/lib/libtensorrt_ops.so
export FP16_ENGINE=${ARTIFACTS}/bevformer_tiny_epoch_24_cp2_op13.fp16.engine
```

Treat `${CONFIG}`, `${PLUGIN_PATH}`, and `${ARTIFACTS}/calibration` as trusted local inputs. MMCV executes the Python configuration, TensorRT loads the native plugin, and quantization reads every matching calibration batch. Use only the configuration and locally compiled plugin from the checked-out source revision and the calibration data produced by this workflow.

Compile the TensorRT plugins for the deployment GPU and prepare the nuScenes metadata:

```bash
cd ${BEVFORMER_ROOT}
git apply --check /mnt/dl4agx/bevformer_trt10.patch
git apply /mnt/dl4agx/bevformer_trt10.patch
cmake -S ${BEVFORMER_ROOT}/TensorRT -B ${BEVFORMER_ROOT}/TensorRT/build \
  -DCMAKE_TENSORRT_PATH=/usr
cmake --build ${BEVFORMER_ROOT}/TensorRT/build --parallel
cmake --install ${BEVFORMER_ROOT}/TensorRT/build
bash samples/bevformer/create_data.sh
```

The source repositories, patches, checkpoint, and dataset retain their upstream licenses and terms.

## 2. Export and build the FP16 engine

Download and verify the BEVFormer-tiny checkpoint, then export the original ONNX model:

```bash
wget --continue --directory-prefix="${ARTIFACTS}" \
  https://github.com/zhiqi-li/storage/releases/download/v1.0/bevformer_tiny_epoch_24.pth
echo "7305046dbaa4fe8b1fa6d6acb9e0e3d605a70a3c473f763e936103428d2b2f12  ${ARTIFACTS}/bevformer_tiny_epoch_24.pth" \
  | sha256sum --check
cd ${BEVFORMER_ROOT}
python tools/pth2onnx.py ${CONFIG} ${ARTIFACTS}/bevformer_tiny_epoch_24.pth \
  --opset_version=13 --cuda --flag=cp2_op13
cp checkpoints/onnx/bevformer_tiny_epoch_24_cp2_op13.onnx ${ONNX_PATH}
```

Use ModelOpt AutoCast to create a typed mixed FP16/FP32 graph. Build its strongly typed TensorRT engine for temporal feedback and the FP16 accuracy baseline:

```bash
python -m modelopt.onnx.autocast \
  --onnx_path=${ONNX_PATH} \
  --output_path=${FP16_ONNX} \
  --low_precision_type=fp16 \
  --keep_io_types \
  --providers trt cuda:0 cpu \
  --trt_plugins ${PLUGIN_PATH}
trtexec --onnx=${FP16_ONNX} \
  --saveEngine=${FP16_ENGINE} \
  --staticPlugins=${PLUGIN_PATH} \
  --stronglyTyped \
  --skipInference
```

## 3. Generate temporal calibration data

Generate exactly 600 ordered training samples. Each saved input name, shape, and dtype is checked against the original ONNX model, while the FP16 engine propagates `prev_bev`. The generator resets `prev_bev` and CAN bus deltas at scene boundaries and publishes the calibration directory only after all requested samples complete.

```bash
cd ${BEVFORMER_ROOT}
PYTHONPATH=${MODELOPT_ROOT}:${BEVFORMER_ROOT} python \
  ${MODELOPT_ROOT}/examples/onnx_ptq/bevformer/prepare_calibration.py \
  ${CONFIG} \
  --onnx=${ONNX_PATH} \
  --engine=${FP16_ENGINE} \
  --trt-plugin=${PLUGIN_PATH} \
  --output-dir=${ARTIFACTS}/calibration \
  --num-samples=600
```

Use a smaller `--num-samples` only for a smoke test. Accuracy results should use all 600 calibration samples.

## 4. Quantize and build INT8 and FP8 engines

INT8 uses entropy calibration by default; FP8 uses max calibration. Both modes preserve the source ONNX model and leave custom plugin operations and `MatMul` nodes in FP16.

```bash
for precision in int8 fp8; do
  python ${MODELOPT_ROOT}/examples/onnx_ptq/bevformer/quantize.py \
    --onnx=${ONNX_PATH} \
    --calibration-dir=${ARTIFACTS}/calibration \
    --trt-plugins=${PLUGIN_PATH} \
    --quantization-mode=${precision} \
    --output=${ARTIFACTS}/bevformer_tiny_epoch_24_cp2_op13.${precision}.onnx
  trtexec --onnx=${ARTIFACTS}/bevformer_tiny_epoch_24_cp2_op13.${precision}.onnx \
    --saveEngine=${ARTIFACTS}/bevformer_tiny_epoch_24_cp2_op13.${precision}.engine \
    --staticPlugins=${PLUGIN_PATH} \
    --stronglyTyped \
    --skipInference
done
```

## 5. Evaluate

Evaluate each engine inside the same image and on the same GPU architecture used for the build:

```bash
cd ${BEVFORMER_ROOT}
for precision in fp16 int8 fp8; do
  python tools/bevformer/evaluate_trt.py ${CONFIG} \
    ${ARTIFACTS}/bevformer_tiny_epoch_24_cp2_op13.${precision}.engine \
    --trt_plugins=${PLUGIN_PATH} \
    | tee ${ARTIFACTS}/evaluate_${precision}.log
done
```

The following historical reference results were collected by the earlier TensorRT 10.14 workflow on an NVIDIA RTX 6000 Ada Generation GPU, using 600 ordered calibration samples and all 6,019 nuScenes validation samples. They are retained as full-validation baselines; the DL4AGX-container integration was validated with three samples and did not rerun NDS or mAP.

| Precision | NDS | mAP |
| :-- | --: | --: |
| FP16 | 0.3546 | 0.2515 |
| INT8/FP16 | 0.3512 | 0.2505 |
| FP8/FP16 | 0.3526 | 0.2489 |

The accuracy run must complete all 6,019 validation samples. Serialized TensorRT engines are specific to the TensorRT version and GPU architecture used to build them.

## Performance

Following the PETR and FAR3D convention, performance is reported only as speedup normalized to FP16. These informational reference values use the archived full-validation engines and the median of five interleaved TensorRT 10.14 trials on the same NVIDIA RTX 6000 Ada Generation GPU. Each trial used the `trtexec` GPU Compute Time median with data transfers disabled, CUDA Graphs enabled, spin-wait, a 1-second warmup, a 10-second measurement window, and one inference stream.

| Precision | Speedup vs. FP16 |
| :-- | --: |
| FP16 | 1.00x |
| INT8/FP16 | 1.87x |
| FP8/FP16 | 1.18x |

TODO: Investigate why FP8 delivers less speedup than INT8 for BEVFormer-tiny.
