# PETR ONNX PTQ and nuScenes evaluation

This example quantizes the PETRv1 and PETRv2 image backbones and detection heads to INT8 or FP8 with Model Optimizer. It follows the [NVIDIA DL4AGX PETR TensorRT workflow](https://github.com/NVIDIA/DL4AGX/tree/master/AV-Solutions/petr-trt) and evaluates TensorRT engines on the nuScenes validation set.

PETR uses a legacy PyTorch/MMCV environment that is incompatible with current Model Optimizer dependencies. The provided image uses `nvcr.io/nvidia/pytorch:26.07-py3` with TensorRT 11.1 and isolates PETR in a Python 3.8 virtual environment.

## 1. Prepare PETR and nuScenes

Clone the reference repositories and apply the DL4AGX compatibility patch:

```bash
git clone https://github.com/NVIDIA/DL4AGX.git
git clone https://github.com/megvii-research/PETR.git
cd PETR
git apply ../DL4AGX/AV-Solutions/petr-trt/patch.diff
git clone https://github.com/open-mmlab/mmdetection3d.git -b v0.17.1
mkdir -p ckpts data
ln -s /path/to/nuscenes data/nuscenes
cd ..
```

Download the `PETR-vov-p4-800x320_epoch24.pth` and `PETRv2-vov-p4-800x320_epoch24.pth` checkpoints linked from the DL4AGX README, rename them as shown below, and prepare this layout:

```text
PETR/
├── ckpts/
│   ├── PETR-vov-p4-800x320_e24.pth
│   └── PETRv2-vov-p4-800x320_e24.pth
├── data/nuscenes/
│   ├── samples/
│   ├── sweeps/
│   ├── v1.0-trainval/
│   └── nuscenes_infos_val.pkl
└── mmdetection3d/
```

PETRv2 also needs metadata for the previous camera sweeps:

```bash
/opt/petr/bin/python \
  /opt/Model-Optimizer/examples/onnx_ptq/petr/prepare_sweep_metadata.py \
  /workspace/PETR/data/nuscenes \
  --split val
```

This creates `mmdet3d_nuscenes_30f_infos_val.pkl`.

Build and start the example image from the Model Optimizer checkout:

```bash
docker build \
  -f examples/onnx_ptq/petr/Dockerfile \
  -t petr-modelopt \
  .

docker run --rm -it --network=host --gpus=all --shm-size=64G \
  -v /path/to/Model-Optimizer:/opt/Model-Optimizer \
  -v /path/to/PETR:/workspace/PETR \
  -v /path/to/DL4AGX:/workspace/DL4AGX \
  petr-modelopt
```

Use the isolated Python 3.8 environment for PETR export, calibration preparation, and evaluation. Keep its site-packages first so the installed MMDetection3D 1.0.0rc6 package takes precedence over the v0.17.1 source tree used by the configs:

```bash
export PYTHONPATH=/opt/petr/lib/python3.8/site-packages:/workspace/PETR:/workspace/DL4AGX/AV-Solutions/petr-trt/export_eval
```

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
  python -m onnxsim \
    "onnx_files/${model}.extract_feat.onnx" \
    "onnx_files/sim_${model}.extract_feat.onnx"
  python -m onnxsim \
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

TensorRT 11 uses strongly typed networks and no longer accepts `--fp16`.
Create calibrated mixed-FP16 graphs with Model Optimizer AutoCast, preserving
FP32 model inputs and outputs, then build the reference engines. The example
below uses one representative batch for AutoCast node classification:

```bash
for version in v1 v2; do
  model="PETR${version}"
  python /opt/Model-Optimizer/examples/onnx_ptq/petr/convert_to_fp16.py \
    "onnx_files/sim_${model}.extract_feat.onnx" \
    "onnx_files/sim_${model}.extract_feat.fp16.onnx" \
    "calibration/${model}/backbone/batch_0000.npz"
  python /opt/Model-Optimizer/examples/onnx_ptq/petr/convert_to_fp16.py \
    "onnx_files/sim_${model}.pts_bbox_head.forward.onnx" \
    "onnx_files/sim_${model}.pts_bbox_head.forward.fp16.onnx" \
    "calibration/${model}/head/batch_0000.npz"
  trtexec \
    --onnx="onnx_files/sim_${model}.extract_feat.fp16.onnx" \
    --saveEngine="engines/${model}.backbone.fp16.engine" \
    --skipInference
  trtexec \
    --onnx="onnx_files/sim_${model}.pts_bbox_head.forward.fp16.onnx" \
    --saveEngine="engines/${model}.head.fp16.engine" \
    --skipInference
done
```

## 4. Quantize the ONNX models

Use the base Python environment for Model Optimizer. This command quantizes the backbone and keeps the head in FP16:

```bash
python /opt/Model-Optimizer/examples/onnx_ptq/petr/quantize.py \
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

Results below use TensorRT 11.1.0.106 on an NVIDIA RTX 6000 Ada Generation GPU. Accuracy is measured over all 6,019 validation samples after calibration with 512 batches. GPU compute time is the sum of the backbone and head median times reported by `trtexec`; it excludes data transfers and PETRv2's reusable previous-frame feature extraction.

| Model | Backbone precision | Head precision | Framework | GPU compute time (ms) | Accuracy (mAP) |
| --- | --- | --- | --- | ---: | ---: |
| PETRv1-vov-p4-800x320 | FP16 | FP16 | TensorRT 11.1 | 14.507 | 0.3781 |
| PETRv1-vov-p4-800x320 | INT8 | FP16 | TensorRT 11.1 | 9.992 | 0.3711 |
| PETRv1-vov-p4-800x320 | FP8 | FP16 | TensorRT 11.1 | 11.455 | 0.3757 |
| PETRv2-vov-p4-800x320 | FP16 | FP16 | TensorRT 11.1 | 19.349 | 0.4105 |
| PETRv2-vov-p4-800x320 | INT8 | FP16 | TensorRT 11.1 | 14.468 | 0.4017 |
| PETRv2-vov-p4-800x320 | FP8 | FP16 | TensorRT 11.1 | 16.242 | 0.4092 |

TensorRT engines are specific to the TensorRT version and GPU architecture used to build them. These x86 results are not directly comparable with the DRIVE Orin measurements in the DL4AGX reference.
