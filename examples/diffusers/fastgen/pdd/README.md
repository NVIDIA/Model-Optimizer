# PDD for Qwen-Image

Parallel Decoding Distillation (PDD) trains one diffusion-transformer call to predict several
consecutive rectified-flow intervals. This example uses a 128-interval shifted-flow grid and a
Qwen-Image student with 128 output heads. A block schedule such as `[32, 32, 32, 32]` therefore
generates with four transformer calls; inference may choose any positive block sizes that sum to
128.

The frozen Qwen-Image teacher constructs the PDD target. The complete student transformer,
including the widened output projection, is finetuned at a constant learning rate of `5e-5`; this
is not a heads-only run. Training samples aligned target spans from 4 through 64 intervals, so the
same checkpoint supports multiple inference schedules.

## Ownership

ModelOpt owns only the PDD model transformation, Qwen execution adapter, loss, and fused sampler.
NeMo AutoModel owns the ordinary training lifecycle: dataloader iteration, backward, gradient
clipping, optimizer and learning-rate state, step scheduling, SIGTERM handling, checkpoint save,
`LATEST`, and resume. The example does not define a custom training loop or checkpoint manager and
does not modify AutoModel, Diffusers, or Qwen source.

## Prepare the student

Widen the Qwen output projection before AutoModel constructs FSDP and the optimizer:

```bash
python examples/diffusers/fastgen/pdd/prepare_qwen_image.py \
  --config examples/diffusers/fastgen/pdd/configs/qwen_image.yaml \
  --model-source Qwen/Qwen-Image \
  --output-dir models/qwen_image_pdd_student
```

The output is a full Diffusers pipeline overlay with a widened transformer. Point
`model.pretrained_model_name_or_path` at this directory.

## Train and resume

```bash
pip install -r examples/diffusers/fastgen/requirements.txt
export MODELOPT_FASTGEN_DATASET_CACHE_DIR=/absolute/path/to/qwen_image_cache

torchrun --standalone --nproc-per-node=8 \
  examples/diffusers/fastgen/pdd/finetune.py \
  --config examples/diffusers/fastgen/pdd/configs/qwen_image.yaml \
  --fsdp.dp_size=8
```

The cache must contain `metadata.json`, its declared tensor shards, and
`negative_prompt_embedding.pt`. `MODELOPT_FASTGEN_DATASET_CACHE_DIR` overrides the configured
cache root; paths declared by the dataset remain confined to that root.

The checked-in recipe targets 50,000 optimizer steps with global batch size 256, local batch size
4, and constant `5e-5` learning rate. `checkpoint.restore_from: LATEST` lets AutoModel resume the
latest native checkpoint. For wall-time-limited Slurm jobs, request an early signal such as
`#SBATCH --signal=TERM@1200`; AutoModel saves at the next completed step and exits. Keep
`step_scheduler.max_steps` at the overall training target rather than imposing a per-job step
limit.

## Generate an image

At the final step, `checkpoint.save_consolidated: final` writes a Diffusers-compatible transformer
under the native checkpoint's `model/consolidated` directory. A periodic or SIGTERM checkpoint
contains AutoModel's generated `model/consolidate.sh` helper for the same conversion.

```bash
python examples/diffusers/fastgen/pdd/inference_qwen_image.py \
  --config examples/diffusers/fastgen/pdd/configs/qwen_image.yaml \
  --model-dir models/qwen_image_pdd_student \
  --transformer-dir /path/to/final-checkpoint/model/consolidated \
  --blocks 32,32,32,32 \
  --prompt "a small red cube on a white table" \
  --seed 42 --height 1024 --width 1024 \
  --output pdd-qwen.png
```

For an effectiveness or speed comparison, use identical prompts, seeds, resolution, dtype, and
hardware for the original Qwen-Image baseline and PDD treatment. Warm up both paths before timing;
report image quality separately from transformer-call reduction and wall-clock latency.
