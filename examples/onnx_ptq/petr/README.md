# PETR ONNX PTQ and nuScenes evaluation

This example quantizes the PETRv1 and PETRv2 image backbones and detection heads to INT8 or FP8 with Model Optimizer. It follows the [NVIDIA DL4AGX PETR TensorRT workflow](https://github.com/NVIDIA/DL4AGX/tree/master/AV-Solutions/petr-trt) and evaluates TensorRT engines on the nuScenes validation set.

The provided image uses `nvcr.io/nvidia/pytorch:25.06-py3`, CUDA 12.9, and TensorRT 10.11. PETR runs in an isolated Python 3.9 environment with PyTorch 2.8 `cu129` and MMCV compiled against the container's CUDA toolkit; Model Optimizer uses the base Python 3.12 environment.

## 1. Prepare PETR and nuScenes

Clone the reference repositories and apply the DL4AGX compatibility patch:

```bash
git clone https://github.com/NVIDIA/DL4AGX.git
git -C DL4AGX checkout 9f7b29104c253d5bc68334e7b83b3eecb72d4572
git clone https://github.com/megvii-research/PETR.git
git -C PETR checkout f7525f93467a33707ef401c587a52d5e7b34de74
git -C PETR apply ../DL4AGX/AV-Solutions/petr-trt/patch.diff
git clone https://github.com/open-mmlab/mmdetection3d.git PETR/mmdetection3d
git -C PETR/mmdetection3d checkout f1107977dfd26155fc1f83779ee6535d2468f449
mkdir -p PETR/ckpts PETR/data
ln -s /data/Dataset/nuScenes PETR/data/nuscenes
```

Download the `PETR-vov-p4-800x320_epoch24.pth` and `PETRv2-vov-p4-800x320_epoch24.pth` checkpoints linked from the DL4AGX README and rename them as shown below. Use nuScenes only under its [terms of use](https://www.nuscenes.org/terms-of-use).

```text
PETR/
├── ckpts/
│   ├── PETR-vov-p4-800x320_e24.pth
│   └── PETRv2-vov-p4-800x320_e24.pth
├── data/nuscenes/
│   ├── maps/
│   ├── samples/
│   ├── sweeps/
│   └── v1.0-trainval/
└── mmdetection3d/
```

Build and start the example image from the Model Optimizer checkout:

```bash
docker build \
  -f examples/onnx_ptq/petr/Dockerfile \
  -t petr-modelopt \
  .
```

The default MMCV build targets SM 8.9 and includes PTX for newer GPUs. Set `--build-arg TORCH_CUDA_ARCH_LIST=...` when building for an older GPU.

```bash
docker run --rm -it --gpus=all --shm-size=64G \
  --user "$(id -u):$(id -g)" \
  -e HOME=/tmp \
  -v /path/to/Model-Optimizer:/opt/Model-Optimizer \
  -v /path/to/PETR:/workspace/PETR \
  -v /path/to/DL4AGX:/workspace/DL4AGX \
  -v /path/to/nuscenes:/data/Dataset/nuScenes \
  petr-modelopt
```

The image builds MMCV 1.7.0 and MMDetection3D v1.0.0rc6 from verified upstream commits. The included patches add PyTorch 2.8 compatibility, remove the unused Lyft integration and the `plyfile`, TensorBoard, and KITTI-only scikit-image dependencies, and update the Python 3.9 runtime pins while preserving the nuScenes path used here.

Use the isolated Python 3.9 environment for PETR export, calibration preparation, and evaluation. Keep its site-packages first so the installed MMDetection3D 1.0.0rc6 package takes precedence over the v0.17.1 source tree used by the configs:

```bash
export PYTHONPATH=/opt/petr/lib/python3.9/site-packages:/workspace/PETR:/workspace/DL4AGX/AV-Solutions/petr-trt/export_eval
```

Inside the container, generate `nuscenes_infos_val.pkl` from the local nuScenes tables with the converter included in the pinned PETR checkout:

```bash
cd /workspace/PETR
/opt/petr/bin/python - <<'PY'
from tools.data_converter.nuscenes_converter import create_nuscenes_infos

create_nuscenes_infos(
    "/workspace/PETR/data/nuscenes",
    "nuscenes",
    version="v1.0-trainval",
    max_sweeps=10,
)
PY
```

PETRv2 also needs metadata for the previous camera sweeps. Run this step in the same container:

```bash
/opt/petr/bin/python \
  /opt/Model-Optimizer/examples/onnx_ptq/petr/prepare_sweep_metadata.py \
  /workspace/PETR/data/nuscenes \
  --split val
```

This creates `mmdet3d_nuscenes_30f_infos_val.pkl` without overwriting an existing file.

## 2. Export PETRv1 and PETRv2 to ONNX

Follow the DL4AGX exporter setup:

```bash
cd /workspace/DL4AGX/AV-Solutions/petr-trt/export_eval
ln -s /workspace/PETR/data data
mkdir -p onnx_files engines

/opt/petr/bin/python v1/v1_export_to_onnx.py \
  /workspace/PETR/projects/configs/petr/petr_vovnet_gridmask_p4_800x320.py \
  /workspace/PETR/ckpts/PETR-vov-p4-800x320_e24.pth \
  --eval bbox

/opt/petr/bin/python v2/v2_export_to_onnx.py \
  /workspace/PETR/projects/configs/petrv2/petrv2_vovnet_gridmask_p4_800x320.py \
  /workspace/PETR/ckpts/PETRv2-vov-p4-800x320_e24.pth \
  --eval bbox
```

The exporters create a backbone graph and a head graph. Simplify both graphs:

```bash
for version in v1 v2; do
  model="PETR${version}"
  /opt/petr/bin/python -m onnxsim \
    "onnx_files/${model}.extract_feat.onnx" \
    "onnx_files/sim_${model}.extract_feat.onnx"
  /opt/petr/bin/python -m onnxsim \
    "onnx_files/${model}.pts_bbox_head.forward.onnx" \
    "onnx_files/sim_${model}.pts_bbox_head.forward.onnx"
done
```

## 3. Prepare calibration batches

Build temporary FP32 engines for collecting the backbone and head inputs:

```bash
for version in v1 v2; do
  model="PETR${version}"
  trtexec \
    --onnx="onnx_files/sim_${model}.extract_feat.onnx" \
    --saveEngine="engines/${model}.backbone.calibration.engine" \
    --skipInference
  trtexec \
    --onnx="onnx_files/sim_${model}.pts_bbox_head.forward.onnx" \
    --saveEngine="engines/${model}.head.calibration.engine" \
    --skipInference
done
```

Collect 512 representative backbone and head input batches. The default interval samples across the 6,019-frame validation set:

```bash
/opt/petr/bin/python /opt/Model-Optimizer/examples/onnx_ptq/petr/prepare_calibration.py \
  v1 \
  /workspace/PETR/projects/configs/petr/petr_vovnet_gridmask_p4_800x320.py \
  /workspace/PETR/ckpts/PETR-vov-p4-800x320_e24.pth \
  onnx_files/sim_PETRv1.extract_feat.onnx \
  onnx_files/sim_PETRv1.pts_bbox_head.forward.onnx \
  engines/PETRv1.backbone.calibration.engine \
  engines/PETRv1.head.calibration.engine \
  calibration/PETRv1
```

Repeat with `v2`, the PETRv2 config, checkpoint, graphs, engines, and `calibration/PETRv2` output directory.

Use the base Python environment for Model Optimizer AutoCast, preserving FP32
model inputs and outputs, then build the reference engines. The example below
uses one representative batch for AutoCast node classification:

```bash
for version in v1 v2; do
  model="PETR${version}"
  env -u PYTHONPATH python -m modelopt.onnx.autocast \
    --onnx_path "onnx_files/sim_${model}.extract_feat.onnx" \
    --output_path "onnx_files/sim_${model}.extract_feat.fp16.onnx" \
    --calibration_data "calibration/${model}/backbone/batch_0000.npz" \
    --low_precision_type fp16 \
    --keep_io_types \
    --providers cuda:0 cpu
  env -u PYTHONPATH python -m modelopt.onnx.autocast \
    --onnx_path "onnx_files/sim_${model}.pts_bbox_head.forward.onnx" \
    --output_path "onnx_files/sim_${model}.pts_bbox_head.forward.fp16.onnx" \
    --calibration_data "calibration/${model}/head/batch_0000.npz" \
    --low_precision_type fp16 \
    --keep_io_types \
    --providers cuda:0 cpu
  trtexec \
    --onnx="onnx_files/sim_${model}.extract_feat.fp16.onnx" \
    --saveEngine="engines/${model}.backbone.fp16.engine" \
    --stronglyTyped \
    --skipInference
  trtexec \
    --onnx="onnx_files/sim_${model}.pts_bbox_head.forward.fp16.onnx" \
    --saveEngine="engines/${model}.head.fp16.engine" \
    --stronglyTyped \
    --skipInference
done
```

## 4. Quantize the ONNX models

Use the base Python environment for Model Optimizer. This command quantizes the backbone and keeps the head in FP16:

```bash
env -u PYTHONPATH python /opt/Model-Optimizer/examples/onnx_ptq/petr/quantize.py \
  --backbone-onnx onnx_files/sim_PETRv1.extract_feat.onnx \
  --head-onnx onnx_files/sim_PETRv1.pts_bbox_head.forward.onnx \
  --calibration-dir calibration/PETRv1 \
  --precision int8
```

Use `--precision fp8` for FP8. Add `--quantize-head` to quantize both the backbone and head. Repeat the commands for PETRv2. FP8 deployment requires an FP8-capable GPU.

The backbone quantizer preserves the accuracy-sensitive final VoVNet stage and FPN output layers in FP16, matching the exclusions used by the FAR3D example.

## 5. Build and evaluate TensorRT engines

Build the quantized backbone as a strongly typed engine:

```bash
precision=int8
model=PETRv1
trtexec \
  --onnx="onnx_files/sim_${model}.extract_feat.${precision}.onnx" \
  --saveEngine="engines/${model}.backbone.${precision}.engine" \
  --stronglyTyped \
  --skipInference
```

If `--quantize-head` was used, build the generated head graph with the same `trtexec` options. Otherwise, use the FP16 head engine from step 3.

Evaluate any backbone/head pairing. For example, INT8 backbone with FP16 head:

```bash
/opt/petr/bin/python /opt/Model-Optimizer/examples/onnx_ptq/petr/evaluate.py \
  v1 \
  /workspace/PETR/projects/configs/petr/petr_vovnet_gridmask_p4_800x320.py \
  /workspace/PETR/ckpts/PETR-vov-p4-800x320_e24.pth \
  engines/PETRv1.backbone.int8.engine \
  engines/PETRv1.head.fp16.engine
```

Use `--max-samples N` for an inference smoke test. Dataset metrics are skipped when only part of the validation set is processed.

Measure each engine with host/device transfers disabled and add the backbone and head median GPU compute times:

```bash
trtexec --loadEngine=engines/PETRv1.backbone.int8.engine \
  --noDataTransfers --useCudaGraph --warmUp=1000 --duration=10
trtexec --loadEngine=engines/PETRv1.head.fp16.engine \
  --noDataTransfers --useCudaGraph --warmUp=1000 --duration=10
```

## Results on the nuScenes validation set

The historical results below use TensorRT 11.1.0.106 and the previous PyTorch 1.13.1/CUDA 11.7 environment on an NVIDIA RTX 6000 Ada Generation GPU. Accuracy is measured over all 6,019 validation samples after calibration with 512 batches. GPU compute time is the sum of the backbone and head median times reported by `trtexec`; it excludes data transfers and PETRv2's reusable previous-frame feature extraction. Rerun the workflow to measure the current TensorRT 10.11/CUDA 12.9 environment.

| Model | Backbone precision | Head precision | Framework | GPU compute time (ms) | Accuracy (mAP) |
| --- | --- | --- | --- | ---: | ---: |
| PETRv1-vov-p4-800x320 | FP16 | FP16 | TensorRT 11.1 | 14.507 | 0.3781 |
| PETRv1-vov-p4-800x320 | INT8 | FP16 | TensorRT 11.1 | 9.992 | 0.3711 |
| PETRv1-vov-p4-800x320 | FP8 | FP16 | TensorRT 11.1 | 11.455 | 0.3757 |
| PETRv2-vov-p4-800x320 | FP16 | FP16 | TensorRT 11.1 | 19.349 | 0.4105 |
| PETRv2-vov-p4-800x320 | INT8 | FP16 | TensorRT 11.1 | 14.468 | 0.4017 |
| PETRv2-vov-p4-800x320 | FP8 | FP16 | TensorRT 11.1 | 16.242 | 0.4092 |

TensorRT engines are specific to the TensorRT version and GPU architecture used to build them. These x86 results are not directly comparable with the DRIVE Orin measurements in the DL4AGX reference.
