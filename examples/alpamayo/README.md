# Quantizing Alpamayo 1

[Alpamayo 1](https://github.com/nvlabs/alpamayo) (formerly Alpamayo-R1) is a
~10B vision-language-action model trained by NVIDIA for autonomous vehicle
research. It takes multi-camera video and egomotion history as input and
produces a Chain-of-Causation reasoning trace plus a future driving trajectory.
See the paper, [*Alpamayo-R1: Bridging Reasoning and Action Prediction for
Generalizable Autonomous Driving in the Long
Tail*](https://arxiv.org/abs/2511.00088), and the
[nvlabs/alpamayo](https://github.com/nvlabs/alpamayo) repository for details.

This example produces FP8, NVFP4, and mixed-precision quantized checkpoints of
Alpamayo using ModelOpt. Quantization calibration runs on a small dataset of 16
AV clips (`0417_16rows_train_set_for_calibration_25.10.parquet`).

## Setup

Clone Alpamayo and install it into the current environment so `alpamayo_r1` is
importable:

```bash
git clone https://github.com/nvlabs/alpamayo  # tested @ 4cda35d
pip install ./alpamayo
```

Follow the Alpamayo README to request access to the gated model weights and the
Physical AI AV dataset, then authenticate with `hf auth login`.

## Usage

`quantize.py` loads an Alpamayo checkpoint, calibrates it on the 16 clips, and
exports an HF-style quantized checkpoint.

### FP8 / NVFP4

By default the script saves **fake-quantized** weights (fp16 weights plus
quantizer state) — useful for accuracy evaluation:

```bash
python quantize.py --ckpt nvidia/Alpamayo-R1-10B --output-dir ./alpamayo-fp8 --quantize fp8
```

Pass `--real-quant` to save **real-quantized** weights packed into the
low-precision storage format (NVFP4 = E2M1 nibbles + per-block FP8 scales),
which run on the hardware low-precision GEMM path:

```bash
python quantize.py --ckpt nvidia/Alpamayo-R1-10B --output-dir ./alpamayo-nvfp4 --quantize nvfp4 --real-quant
```

The vision tower is always kept in high precision, and small action-projection
heads whose dimensions are not multiples of 16 are left unquantized (they break
the real-quant GEMM backends).

### AutoQuantize (mixed precision)

`--quantize auto` runs ModelOpt's AutoQuantize, which searches per layer between
NVFP4 and FP8 under an effective-bits budget (`--auto_quantize_bits`, default
6.5):

```bash
python quantize.py --ckpt nvidia/Alpamayo-R1-10B --output-dir ./alpamayo-auto --quantize auto --auto_quantize_bits 6.5
```

AutoQuantize chooses a per-layer format using a **gradient-based sensitivity
score**: it backpropagates a loss through the model and estimates how much each
candidate format perturbs that loss, then picks the cheapest assignment that
stays within the bit budget. Here the loss is the flow-matching objective — an
MSE between the action expert's predicted velocity field `v_pred` and the
target `v_target = x_1 - x_0` from a teacher-forced forward pass on the
calibration clips. Layers the loss is sensitive to keep more bits (FP8); the
rest go to NVFP4.
