# FAR3D ONNX PTQ and Argoverse 2 evaluation

This example quantizes the FAR3D VoVNet image encoder to INT8 or FP8, keeps the decoder in its exported mixed FP16/FP32 precision, and evaluates TensorRT 11.1 engines on the Argoverse 2 validation set. It follows the [NVIDIA DL4AGX FAR3D workflow](https://github.com/NVIDIA/DL4AGX/tree/9f7b29104c253d5bc68334e7b83b3eecb72d4572/AV-Solutions/far3d-trt).

Build the shared `evaluator` and `modelopt` images as described in the [parent guide](../README.md#petr-and-far3d-containers). Use the evaluator for source setup, metadata, export, calibration, and accuracy evaluation. Use the ModelOpt image for AutoCast, quantization, and TensorRT engine builds.

## 1. Prepare FAR3D and Argoverse 2

Download the [Argoverse 2 sensor validation set](https://www.argoverse.org/av2.html) on the host, then start the evaluator with the workspace and dataset mounted:

```bash
docker run --rm -it --gpus=all --ipc=host \
  --user "$(id -u):$(id -g)" -e HOME=/tmp \
  -e USER="$(id -un)" -e LOGNAME="$(id -un)" \
  -v /path/to/workspace:/workspace \
  -v /path/to/av2_sensor:/data/av2:ro \
  modelopt-onnx-evaluator
```

Clone the pinned DL4AGX tree and apply its official FAR3D export patch. This is the only source patch in the workflow.

```bash
git clone https://github.com/NVIDIA/DL4AGX.git /workspace/DL4AGX
git -C /workspace/DL4AGX checkout 9f7b29104c253d5bc68334e7b83b3eecb72d4572
git -C /workspace/DL4AGX submodule update --init \
  AV-Solutions/far3d-trt/dependencies/Far3D \
  AV-Solutions/far3d-trt/dependencies/mmdetection3d
git -C /workspace/DL4AGX/AV-Solutions/far3d-trt/dependencies/Far3D \
  apply ../../patch/far3d.patch
```

Download the [FAR3D checkpoint](https://github.com/megvii-research/Far3D/releases/download/v1.0/iter_82548.pth). Keep the raw dataset read-only and store generated metadata in the workspace:

```text
/workspace/DL4AGX/AV-Solutions/far3d-trt/
├── data/av2/
│   └── val -> /data/av2/val
└── weights/iter_82548.pth
```

```bash
cd /workspace/DL4AGX/AV-Solutions/far3d-trt
# Replace DL4AGX's dataset-root symlink so generated metadata stays in the workspace.
unlink data/av2
mkdir -p data/av2 weights
ln -s /data/av2/val data/av2/val
```

## 2. Export and calibrate in the evaluator

```bash
cd /workspace/DL4AGX/AV-Solutions/far3d-trt
export PYTHONPATH=$PWD/dependencies/Far3D

python /opt/Model-Optimizer/examples/onnx_ptq/far3d/prepare_metadata.py data/av2
python tools/export_onnx.py \
  dependencies/Far3D/projects/configs/far3d.py \
  weights/iter_82548.pth

python /opt/Model-Optimizer/examples/onnx_ptq/far3d/prepare_calibration.py \
  dependencies/Far3D/projects/configs/far3d.py \
  far3d.encoder.onnx calibration/encoder
```

The calibration command writes 512 NPZ batches directly from the data loader. No temporary TensorRT engine or decoder calibration data is needed.

## 3. Optimize and build in the ModelOpt container

Restart the workspace with `modelopt-onnx-trt11`, then run:

```bash
cd /workspace/DL4AGX/AV-Solutions/far3d-trt

python -m modelopt.onnx.autocast \
  --onnx_path far3d.encoder.onnx \
  --output_path far3d.encoder.fp16.onnx \
  --calibration_data calibration/encoder/batch_0000.npz \
  --low_precision_type fp16 --keep_io_types --providers cuda:0 cpu

for precision in int8 fp8; do
  python /opt/Model-Optimizer/examples/onnx_ptq/quantize_vovnet.py \
    far3d.encoder.onnx calibration/encoder \
    --precision "$precision" --output "far3d.encoder.${precision}.onnx"
done

for precision in fp16 int8 fp8; do
  trtexec --onnx="far3d.encoder.${precision}.onnx" \
    --saveEngine="far3d.encoder.${precision}.engine" --skipInference
done
trtexec --onnx=far3d.decoder.onnx \
  --saveEngine=far3d.decoder.mixed.engine --skipInference
```

TensorRT 11.1 uses typed ONNX graphs; neither `--fp16` nor `--stronglyTyped` is needed. Serialized engines are not portable across TensorRT versions or GPU architectures.

## 4. Evaluate in the evaluator

Restart `modelopt-onnx-evaluator` with the same mounts:

```bash
cd /workspace/DL4AGX/AV-Solutions/far3d-trt
export PYTHONPATH=$PWD/dependencies/Far3D

for precision in fp16 int8 fp8; do
  python /opt/Model-Optimizer/examples/onnx_ptq/far3d/evaluate.py \
    dependencies/Far3D/projects/configs/far3d.py \
    "far3d.encoder.${precision}.engine" far3d.decoder.mixed.engine
done
```

Add `--max-samples 2` for a smoke test that also exercises recurrent decoder state. Full validation contains 23,522 frames.

## Reference accuracy and performance

Accuracy was measured with TensorRT 11.1.0.106 on an NVIDIA RTX 6000 Ada Generation GPU using 512 calibration batches.

| Encoder | Decoder | mAP |
| --- | --- | ---: |
| FP16 | Mixed FP16/FP32 | 0.241 |
| INT8 | Mixed FP16/FP32 | 0.235 |
| FP8 | Mixed FP16/FP32 | 0.239 |

Engine-only performance is normalized to the FP16 pipeline. Each comparison comprises one encoder pass plus the same exported mixed FP16/FP32 decoder; only the encoder precision changes.

Measurements use TensorRT 11.1.0.106 on an NVIDIA RTX 6000 Ada Generation GPU with five interleaved trials per engine component. Each component uses the median `trtexec`-reported GPU Compute Time with data transfers disabled and CUDA Graphs enabled. Component times are summed before normalization. Only speedups are reported.

| Pipeline | INT8 speedup vs. FP16 | FP8 speedup vs. FP16 |
| --- | ---: | ---: |
| FAR3D | 1.69x | 1.40x |
