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
calibration clips. Layers the loss is sensitive to keep more bits
(FP8/unquantized); less sensitive layers go to NVFP4.

## Quantization-aware distillation

`qad.py` recovers accuracy lost to quantization by distilling the quantized
checkpoint that `quantize.py` produced. The student is that checkpoint's VLM;
the teacher is the FP16 VLM of the original checkpoint. Only the VLM is
trained — the action expert diffusion head stays frozen. The loss is KL
divergence between student and teacher logits: we leverage the existing
QADTrainer in ModelOpt.

The loss is prompt-only (ends at `<|cot_start|>`), covering system prompt, image,
and trajectory tokens. This avoids the cost of rollout sampling. Sequence augmentation
and sampling for generated tokens are **not currently implemented**—only prompt-only
targets are supported.

```bash
# 1. Quantize.
python quantize.py --ckpt nvidia/Alpamayo-R1-10B --output-dir ./alpamayo-auto --quantize auto

# 2. Distill the quantized VLM against the FP16 one, then export.
torchrun --standalone --nproc_per_node 8 qad.py \
    --student_ckpt ./alpamayo-auto \
    --output_dir ./alpamayo-auto-qad \
    --parquet ./train_clips.parquet --limit_train 2000 \
    --val_offset 2000 --limit_val 4 \
    --max_steps 500 --fsdp2 --grad_ckpt --export
```

Training will print progress updates, ending with final loss metrics:

```text
[qad] student (quantized) ./alpamayo-auto | teacher (FP) nvidia/Alpamayo-R1-10B
[qad] train clips=2000 val clips=4
[qad] final train loss: 0.123456 | eval loss: 0.125678
[qad] trained VLM saved to ./alpamayo-auto-qad
[qad] export: loaded 2048 trained tensors
[qad] export: wrote full model to ./alpamayo-auto-qad-full
```

`--fsdp2` shards the student and the teacher across GPUs; `--grad_ckpt` trades
recomputation for activation memory.

Training clips are yours to supply: `--parquet` and `--limit_train` are both
required, and only the `key` column, holding clip ids, is read. `--train_offset`,
`--limit_train`, `--val_offset` and `--limit_val` carve the train and validation
slices out of that list; pick them so the two slices do not overlap. Batch size is fixed at 1, so
use `--grad_accum` for a larger effective batch.

Set `--pai_revision` (or `PAI_REVISION`) to pin the PhysicalAI-AV dataset to a
commit. This is required to read a pre-warmed cache with `HF_HUB_OFFLINE=1`,
and makes clip loading reproducible.

### Exporting

Training saves the VLM alone, which is not loadable as an Alpamayo model:
`--export` reassembles one — it reloads the quantized checkpoint for its
quantizer structure, loads the trained tensors into its VLM, and saves the whole
model to `<output_dir>-full`, which `AlpamayoR1.from_pretrained` can read back.

To export an earlier run without retraining:

```bash
python qad.py --student_ckpt ./alpamayo-auto --output_dir ./unused \
    --parquet ./train_clips.parquet --limit_train 1 \
    --trained_vlm ./alpamayo-auto-qad --export_dir ./alpamayo-auto-qad-full
```
