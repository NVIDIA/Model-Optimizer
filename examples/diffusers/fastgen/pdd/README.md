# PDD distillation for Qwen-Image

[Parallel Decoding Distillation (PDD)](https://arxiv.org/abs/2607.26004) trains one
diffusion-transformer call to predict several consecutive rectified-flow intervals. This example
uses a 128-interval shifted-flow grid and a Qwen-Image student with 128 output heads. A block
schedule such as `[32, 32, 32, 32]` therefore generates with four transformer calls.

The frozen Qwen-Image teacher constructs the PDD target using Qwen's native packed, per-token CFG
rescale. The checked-in configuration adapts the paper's data-free Midpoint algorithm: it trains the
full student from on-policy trajectories carried from fresh noise, using a constant `1e-5` learning
rate for 3,000 steps. It samples target spans up to 64 intervals and advances each carried
trajectory by 16 intervals, so student inputs during training start at multiples of 16.

For this recipe, we recommend positive inference block sizes that are multiples of 16, no larger
than 64, and sum to 128. Equal-block schedules can use 2, 4, or 8 transformer calls. Uneven schedules
such as `[16, 32, 32, 48]` are theoretically supported but have not been tested for generation
quality.

Data-free removes image supervision, not text conditioning. Training still consumes positive
prompt embeddings and masks plus a static negative-prompt embedding for teacher CFG. Supply these
tensors through a user-prepared FastGen cache. With `prompt_only: true`, cached image latents are
omitted from the training batch and never enter the PDD objective.

> [!NOTE]
> Qwen-Image is a third-party model with its own license terms. Review the
> [Qwen-Image model card](https://huggingface.co/Qwen/Qwen-Image) before downloading or
> redistributing weights or derivatives.

## Install

Run the commands below from the root of a Model-Optimizer source checkout; the `examples/` tree
is not included in the `nvidia-modelopt` pip package.

```bash
pip install -e ".[all]"
pip install -r examples/diffusers/requirements.txt
pip install -r examples/diffusers/fastgen/pdd/requirements.txt
```

The PDD requirements pin NeMo AutoModel to `0.5.0` and Diffusers to `0.39.0`, a narrower range
than the shared DMD2 requirements.

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

## Prepare conditioning data

Prepare a standard Qwen-Image cache from an image-caption dataset that you are permitted to use by
following the [DMD2 data preparation guide](../dmd2/README.md#requirements--self-contained-data-path).
The shared cache format contains image latents for use by other FastGen methods, but PDD's
prompt-only dataloader excludes them from training.

## Training

```bash
torchrun --standalone --nproc-per-node=8 \
  examples/diffusers/fastgen/pdd/finetune.py \
  --config examples/diffusers/fastgen/pdd/configs/qwen_image.yaml \
  --data.dataloader.cache_dir=/absolute/path/to/qwen_image_cache \
  --fsdp.dp_size=8
```

The cache must contain `metadata.json`, its declared prompt-embedding shards, and
`negative_prompt_embedding.pt`. Override `data.dataloader.cache_dir` on the command line when the
runtime cache location differs from the YAML. Sample payloads and the negative embedding resolve
from that root, and paths declared by the dataset remain confined to it.

The checked-in recipe targets 3,000 optimizer steps with global batch size 2,048, local batch size
4, and constant learning rate `1e-5`.

### Checkpoints & resuming

Use a new, empty checkpoint directory for the first job.
AutoModel auto-detects the latest checkpoint in that directory on later jobs. In-flight data-free
trajectories are transient, as in FastGen's carry callback, and restart from fresh noise after a
resume; AutoModel restores the model, optimizer, scheduler, RNG, and dataloader. For
wall-time-limited Slurm jobs, request an early signal such as
`#SBATCH --signal=TERM@1200`; AutoModel saves at the next completed step and exits. Keep
`step_scheduler.max_steps` at the overall training target rather than imposing a per-job step
limit.

## Inference

At the final step, `checkpoint.save_consolidated: final` writes a Diffusers-compatible transformer
under the native checkpoint's `model/consolidated` directory. A periodic or SIGTERM checkpoint
contains AutoModel's generated `model/consolidate.sh` helper for the same conversion.

```bash
python examples/diffusers/fastgen/pdd/inference_qwen_image.py \
  --config examples/diffusers/fastgen/pdd/configs/qwen_image.yaml \
  --model-dir models/qwen_image_pdd_student \
  --transformer-dir /path/to/final-checkpoint/model/consolidated \
  --pdd-steps 4 \
  --prompt "a small red cube on a white table" \
  --seed 42 --height 1024 --width 1024 \
  --output pdd-qwen.png
```

`--pdd-steps 4` divides the grid into four equal blocks of 32 intervals. The step count must be
positive and divide `grid_size` evenly. Alternatively, use `--blocks 16,32,32,48` for an uneven
schedule; `--blocks` and `--pdd-steps` cannot be used together.

For an effectiveness or speed comparison, use identical prompts, seeds, resolution, dtype, and
hardware for the original Qwen-Image baseline and PDD treatment. Warm up both paths before timing;
report image quality separately from transformer-call reduction and wall-clock latency.
