# PETR ONNX PTQ and nuScenes evaluation

This example quantizes the VoVNet image backbone in PETRv1 and PETRv2 to INT8 or FP8, keeps the detection head in mixed FP16/FP32, and evaluates TensorRT 11.1 engines on the nuScenes validation set. It follows the [NVIDIA DL4AGX PETR workflow](https://github.com/NVIDIA/DL4AGX/tree/9f7b29104c253d5bc68334e7b83b3eecb72d4572/AV-Solutions/petr-trt).

Build the shared `evaluator` and `modelopt` images as described in the [parent guide](../README.md#petr-and-far3d-containers). Use the evaluator for source setup, export, calibration, and accuracy evaluation. Use the ModelOpt image for AutoCast, quantization, and TensorRT engine builds.

## 1. Prepare PETR and nuScenes

Place the nuScenes data and pre-generated `nuscenes_infos_val.pkl` and `mmdet3d_nuscenes_30f_infos_val.pkl` metadata files in the host dataset directory. Then start the evaluator with the workspace and dataset mounted:

```bash
docker run --rm -it --gpus=all --ipc=host \
  --user "$(id -u):$(id -g)" -e HOME=/tmp \
  -e USER="$(id -un)" -e LOGNAME="$(id -un)" \
  -v /path/to/workspace:/workspace \
  -v /path/to/nuscenes:/data/nuscenes:ro \
  modelopt-onnx-evaluator
```

The final workspace layout is:

```text
/workspace/
├── DL4AGX/
└── PETR/
    ├── ckpts/
    │   ├── PETR-vov-p4-800x320_e24.pth
    │   └── PETRv2-vov-p4-800x320_e24.pth
```

Pin the source repositories. PETR does not need a patch; dataset paths are passed through its existing configuration overrides.
The mmdetection3d checkout supplies the base configuration files that PETR references by relative path.

```bash
git clone https://github.com/NVIDIA/DL4AGX.git /workspace/DL4AGX
git -C /workspace/DL4AGX checkout 9f7b29104c253d5bc68334e7b83b3eecb72d4572

git clone https://github.com/megvii-research/PETR.git /workspace/PETR
git -C /workspace/PETR checkout f7525f93467a33707ef401c587a52d5e7b34de74
git clone https://github.com/open-mmlab/mmdetection3d.git /workspace/PETR/mmdetection3d
git -C /workspace/PETR/mmdetection3d checkout f1107977dfd26155fc1f83779ee6535d2468f449
mkdir -p /workspace/PETR/ckpts
```

Download the checkpoints linked from the DL4AGX guide. Use nuScenes only under its [terms of use](https://www.nuscenes.org/terms-of-use).

## 2. Export and calibrate in the evaluator

```bash
cd /workspace/DL4AGX/AV-Solutions/petr-trt/export_eval
export PYTHONPATH=/workspace/PETR:$PWD
mkdir -p onnx_files engines calibration data
ln -s /data/nuscenes data/nuscenes

V1_CONFIG=/workspace/PETR/projects/configs/petr/petr_vovnet_gridmask_p4_800x320.py
V2_CONFIG=/workspace/PETR/projects/configs/petrv2/petrv2_vovnet_gridmask_p4_800x320.py
V1_CHECKPOINT=/workspace/PETR/ckpts/PETR-vov-p4-800x320_e24.pth
V2_CHECKPOINT=/workspace/PETR/ckpts/PETRv2-vov-p4-800x320_e24.pth
V1_INFO=/data/nuscenes/nuscenes_infos_val.pkl
V2_INFO=/data/nuscenes/mmdet3d_nuscenes_30f_infos_val.pkl
```

The dataset symlink resolves the `data/nuscenes/...` camera paths stored in the metadata files.

Export both models without modifying the PETR checkout:

```bash
python v1/v1_export_to_onnx.py "$V1_CONFIG" "$V1_CHECKPOINT" --eval bbox \
  --cfg-options \
  data.val.data_root=/data/nuscenes/ data.val.ann_file="$V1_INFO" \
  data.test.data_root=/data/nuscenes/ data.test.ann_file="$V1_INFO"

python v2/v2_export_to_onnx.py "$V2_CONFIG" "$V2_CHECKPOINT" --eval bbox \
  --cfg-options \
  data.val.data_root=/data/nuscenes/ data.val.ann_file="$V2_INFO" \
  data.test.data_root=/data/nuscenes/ data.test.ann_file="$V2_INFO"

for model in PETRv1 PETRv2; do
  python -m onnxsim "onnx_files/${model}.extract_feat.onnx" \
    "onnx_files/${model}.backbone.onnx"
  python -m onnxsim "onnx_files/${model}.pts_bbox_head.forward.onnx" \
    "onnx_files/${model}.head.onnx"
done
```

Collect 512 backbone batches and one representative head batch directly from PyTorch. No temporary TensorRT engines are needed.

```bash
python /opt/Model-Optimizer/examples/onnx_ptq/petr/prepare_calibration.py \
  v1 "$V1_CONFIG" "$V1_CHECKPOINT" \
  onnx_files/PETRv1.backbone.onnx onnx_files/PETRv1.head.onnx calibration/PETRv1 \
  --cfg-options \
  data.test.data_root=/data/nuscenes/ data.test.ann_file="$V1_INFO"

python /opt/Model-Optimizer/examples/onnx_ptq/petr/prepare_calibration.py \
  v2 "$V2_CONFIG" "$V2_CHECKPOINT" \
  onnx_files/PETRv2.backbone.onnx onnx_files/PETRv2.head.onnx calibration/PETRv2 \
  --cfg-options \
  data.test.data_root=/data/nuscenes/ data.test.ann_file="$V2_INFO"
```

## 3. Optimize and build in the ModelOpt container

Restart the workspace with `modelopt-onnx-trt11`, then return to the export directory:

```bash
cd /workspace/DL4AGX/AV-Solutions/petr-trt/export_eval

for model in PETRv1 PETRv2; do
  python -m modelopt.onnx.autocast \
    --onnx_path "onnx_files/${model}.backbone.onnx" \
    --output_path "onnx_files/${model}.backbone.fp16.onnx" \
    --calibration_data "calibration/${model}/backbone/batch_0000.npz" \
    --low_precision_type fp16 --keep_io_types --providers cuda:0 cpu
  python -m modelopt.onnx.autocast \
    --onnx_path "onnx_files/${model}.head.onnx" \
    --output_path "onnx_files/${model}.head.fp16.onnx" \
    --calibration_data "calibration/${model}/head/batch_0000.npz" \
    --low_precision_type fp16 --keep_io_types --providers cuda:0 cpu

  for precision in int8 fp8; do
    python /opt/Model-Optimizer/examples/onnx_ptq/quantize_vovnet.py \
      "onnx_files/${model}.backbone.onnx" "calibration/${model}/backbone" \
      --precision "$precision" \
      --output "onnx_files/${model}.backbone.${precision}.onnx"
  done

  trtexec --onnx="onnx_files/${model}.head.fp16.onnx" \
    --saveEngine="engines/${model}.head.fp16.engine" --skipInference
  for precision in fp16 int8 fp8; do
    trtexec --onnx="onnx_files/${model}.backbone.${precision}.onnx" \
      --saveEngine="engines/${model}.backbone.${precision}.engine" --skipInference
  done
done
```

TensorRT 11.1 uses typed ONNX graphs; the removed `--fp16` builder flag is not used.

## 4. Evaluate in the evaluator

Restart `modelopt-onnx-evaluator` with the same mounts, restore the variables from step 2, and run each backbone precision with the shared FP16 head:

```bash
cd /workspace/DL4AGX/AV-Solutions/petr-trt/export_eval
export PYTHONPATH=/workspace/PETR:$PWD

for precision in fp16 int8 fp8; do
  python /opt/Model-Optimizer/examples/onnx_ptq/petr/evaluate.py \
    v1 "$V1_CONFIG" "$V1_CHECKPOINT" \
    "engines/PETRv1.backbone.${precision}.engine" engines/PETRv1.head.fp16.engine \
    --cfg-options \
    data.test.data_root=/data/nuscenes/ data.test.ann_file="$V1_INFO"

  python /opt/Model-Optimizer/examples/onnx_ptq/petr/evaluate.py \
    v2 "$V2_CONFIG" "$V2_CHECKPOINT" \
    "engines/PETRv2.backbone.${precision}.engine" engines/PETRv2.head.fp16.engine \
    --cfg-options \
    data.test.data_root=/data/nuscenes/ data.test.ann_file="$V2_INFO"
done
```

Add `--max-samples 1` for an export-to-inference smoke test. Full validation contains 6,019 samples.

## Reference accuracy

These TensorRT 11.1.0.106 results were measured on an NVIDIA RTX 6000 Ada Generation GPU with 512 calibration batches.

| Model | Backbone | Head | mAP |
| --- | --- | --- | ---: |
| PETRv1 | FP16 | FP16 | 0.3778 |
| PETRv1 | INT8 | FP16 | 0.3707 |
| PETRv1 | FP8 | FP16 | 0.3756 |
| PETRv2 | FP16 | FP16 | 0.4102 |
| PETRv2 | INT8 | FP16 | 0.4024 |
| PETRv2 | FP8 | FP16 | 0.4093 |
