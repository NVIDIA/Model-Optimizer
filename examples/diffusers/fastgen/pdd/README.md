# PDD for Qwen-Image

Parallel Decoding Distillation (PDD) trains one diffusion-transformer call to predict several
consecutive rectified-flow intervals. This example uses a 128-interval shifted-flow grid and a
Qwen-Image student with 128 output heads. A block schedule such as `[32, 32, 32, 32]` therefore
generates with four transformer calls; inference may choose any positive block sizes that sum to
128.

The frozen Qwen-Image teacher constructs the PDD target using Qwen's native packed, per-token CFG
rescale. The checked-in configuration adapts the paper's data-free Midpoint algorithm and
hyperparameters to the current FastGen prompt cache: it trains the full student from on-policy
trajectories carried from fresh noise, using a constant `1e-5` learning rate for 3,000 steps. It
samples target spans up to 64 intervals and advances each carried trajectory by 16 intervals,
supporting 2-, 4-, and 8-NFE inference schedules.

Data-free removes image supervision, not text conditioning. Training still consumes positive
prompt embeddings and masks plus a static negative-prompt embedding for teacher CFG. The current
FastGen cache can provide those tensors; its cached image latents are omitted from the training
batch and never enter the PDD objective. That synthetic-DALL-E3 prompt corpus is an available
experiment input, not a claim to reproduce the paper's Pi-Flow prompt set or its OneIG,
DPG-Bench, and GenEval checkpoint-selection protocol. Canonical prompt and evaluation parity
remain experiment work rather than product-code requirements.

## Ownership

ModelOpt owns only the PDD model transformation, Qwen execution adapter, loss, and fused sampler.
NeMo AutoModel owns the ordinary training lifecycle: dataloader iteration, backward, gradient
clipping, optimizer and learning-rate state, step scheduling, SIGTERM handling, checkpoint save,
`LATEST`, and resume. The example does not define a custom training loop or checkpoint manager and
does not modify files in AutoModel, Diffusers, or Qwen.

The example pins AutoModel 0.5.0. That release does not expose setup hooks for preserving Qwen's
FP32 timestep input or freezing parameters before optimizer construction, so `pdd/compat.py`
temporarily adapts those two setup calls inside a serialized context and restores them immediately
after `TrainDiffusionRecipe.setup()`.

ModelOpt also binds an instance-local Qwen forward for PDD execution. Diffusers owns the joint
attention-mask behavior, but casts normalized timesteps to BF16; the PDD path preserves the FP32
grid value used by the original FastGen implementation. The override otherwise follows the pinned
Diffusers forward and rejects unsupported execution modes explicitly.

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

The cache must contain `metadata.json`, its declared prompt-embedding shards, and
`negative_prompt_embedding.pt`. `MODELOPT_FASTGEN_DATASET_CACHE_DIR` overrides the configured
cache root; paths declared by the dataset remain confined to that root.

The checked-in recipe targets 3,000 optimizer steps with global batch size 2,048, local batch size
4, and constant learning rate `1e-5`. Use a new, empty checkpoint directory for the first job.
AutoModel auto-detects the latest checkpoint in that directory on later jobs. In-flight data-free
trajectories are transient, as in FastGen's carry callback, and restart from fresh noise after a
resume; AutoModel restores the model, optimizer, scheduler, RNG, and dataloader. For
wall-time-limited Slurm jobs, request an early signal such as
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
