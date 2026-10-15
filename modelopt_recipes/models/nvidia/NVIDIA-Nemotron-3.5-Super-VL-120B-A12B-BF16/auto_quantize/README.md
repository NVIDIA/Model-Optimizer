# NVIDIA Nemotron 3.5 Super VL AutoQuantize

`nvfp4_mse_fp8_mtp_fixed_at_4p9bits` searches decoder modules and `lm_head` over
weight-MSE NVFP4, FP8, and BF16 using gradient-based sensitivity scores. It is
tuned for `nvidia/NVIDIA-Nemotron-3.5-Super-VL-120B-A12B-BF16`, not a default for
all Nemotron-H checkpoints. The search generates a mixed-precision candidate;
it does not reproduce the published QAD-refined NVFP4 weights.

MTP routed/shared experts stay fixed at MSE-calibrated NVFP4 W4A4. MTP KV uses
calibrated FP8 scales, while backbone KV uses fixed-range FP8 casting. Other MTP
modules and the vision/audio stack remain BF16. Fixed MTP and non-language
modules are excluded from the search cost, so the 4.9-bit budget is not the
whole-checkpoint bits per parameter.

```bash
python examples/hf_ptq/hf_ptq.py \
  --pyt_ckpt_path nvidia/NVIDIA-Nemotron-3.5-Super-VL-120B-A12B-BF16 \
  --recipe models/nvidia/NVIDIA-Nemotron-3.5-Super-VL-120B-A12B-BF16/auto_quantize/nvfp4_mse_fp8_mtp_fixed_at_4p9bits \
  --calib_size 128 \
  --use_seq_device_map \
  --export_path <quantized-checkpoint>
```

Review the checkpoint's modeling code before adding `--trust_remote_code` if
required. The example AutoQuantize CLI currently accepts text calibration only;
do not copy the sibling PTQ recipe's `--calib_with_images` invocation.
