# FastGen Quantization-Aware Distillation

This example trains a quantized diffusion student against a frozen, BF16
Diffusers teacher with ModelOpt's distillation API. It is a standalone FastGen
recipe: it does not use DMD2, reduce the sampling schedule, create a fake-score
model, or add a GAN/EMA training phase.

The initial Qwen-Image recipe uses the official `Qwen/Qwen-Image` Diffusers
checkpoint as the teacher. Set `qad.teacher_model_name_or_path` to
`nvidia/Qwen-Image-Flash` when the four-step DMD2-trained Qwen-Image checkpoint
should be the teacher instead. Both follow the standard Diffusers checkpoint
interface. QAD intentionally does not interpret FastGen/DMD2 intermediate
checkpoint sidecars or standalone transformer safetensors as teacher inputs.

Every training micro-batch samples one noisy latent and one timestep, then sends
the same latent, timestep, prompt conditioning, and guidance inputs to the
teacher and student.

## Supported students

The `qad.student.mode` field selects one of two bundle validation contracts. In
both cases, `model.pretrained_model_name_or_path` is the only student artifact
path: it points to a complete, calibrated Diffusers pipeline written by
`quantize.py --output-bundle`. QAD restores the pipeline's weights and
component-local ModelOpt state together before FSDP; it does not accept a second
quantizer-state or transformer-checkpoint path.

### Regular NVFP4

Set `qad.student.mode=nvfp4` and point
`model.pretrained_model_name_or_path` at a regular NVFP4 training bundle. The
bundle includes the calibrated weights and ModelOpt quantizer topology/state.
This mode trains all student parameters, so its only valid `train_scope` is
`all`.

Use [`configs/qwen_image_nvfp4.yaml`](configs/qwen_image_nvfp4.yaml) as the
starting configuration.

### NVFP4 SVDQuant with Hugging Face PEFT

Set `qad.student.mode=nvfp4_svdquant` and point
`model.pretrained_model_name_or_path` at a user-prepared, ModelOpt-enabled Diffusers
training bundle. The bundle must contain the complete SVDQuant student:

- a DiffusionPipeline root with `model_index.json` (not only a standalone
  transformer `save_pretrained` directory);
- the ModelOpt topology and quantizer state;
- the residual weights produced by SVDQuant calibration; and
- the Hugging Face PEFT A/B factors for the SVDQuant low-rank branch.

For the standard Diffusers layout, the transformer files and ModelOpt sidecar
are under `transformer/`, including `transformer/modelopt_state.pth`. The path
given to QAD is the parent DiffusionPipeline directory.

A standalone weight-free NVFP4 quantizer-state file is not a QAD student bundle.
This is especially important for SVDQuant: calibration subtracts the low-rank
branch from the original weight, so both the resulting residual weight and the
PEFT factors are required. Deployment artifacts are not training bundles and
must not be used here.

The SVDQuant topology is restored before FSDP and before optimizer construction.
`qad.student.train_scope=all` is the default and trains both the residual/base
parameters and the PEFT factors. Set it to `lora_only` to freeze every student
parameter except the SVDQuant PEFT A/B factors. In both scopes,
`pre_quant_scale` remains ModelOpt buffer state and is never placed in the
optimizer.

Use [`configs/qwen_image_svdquant_nvfp4.yaml`](configs/qwen_image_svdquant_nvfp4.yaml)
as the starting configuration.

QAD is restore-only in both modes. It does not calibrate a student during
distributed training.

## Prepare a student bundle

Patch Diffusers ModelMixin support and save the complete pipeline through the
quantization entry point. `quantize.py` does this automatically before model
load and calls `pipe.save_pretrained(output_bundle)` after calibration. For
example:

```bash
# Regular NVFP4
python examples/diffusers/quantization/quantize.py \
  --model qwen-image \
  --override-model-path /path/to/Qwen-Image \
  --model-dtype BFloat16 \
  --format fp4 \
  --quant-algo max \
  --block-size 16 \
  --batch-size 1 \
  --calib-size 32 \
  --n-steps 50 \
  --extra-param true_cfg_scale=4.0 \
  --extra-param "negative_prompt= " \
  --output-bundle /path/to/Qwen-Image-NVFP4-Calib32

# NVFP4 SVDQuant, rank 32
python examples/diffusers/quantization/quantize.py \
  --model qwen-image \
  --override-model-path /path/to/Qwen-Image \
  --model-dtype BFloat16 \
  --format fp4 \
  --quant-algo svdquant \
  --lowrank 32 \
  --block-size 16 \
  --batch-size 1 \
  --calib-size 32 \
  --n-steps 50 \
  --extra-param true_cfg_scale=4.0 \
  --extra-param "negative_prompt= " \
  --output-bundle /path/to/Qwen-Image-NVFP4-SVDQuant-Calib32
```

The saved root must contain `model_index.json`; the converted transformer must
contain `transformer/modelopt_state.pth`. For Qwen-Image-Flash, use `--n-steps 4`
and `--extra-param true_cfg_scale=1.0`; omit `negative_prompt`. Standard output
includes ModelOpt's full quantizer summary; capture it with `tee` and retain that
log with the bundle.

## Distillation losses

Output distillation is always MSE. The canonical setting is:

```yaml
qad:
  output_loss:
    type: mse
    weight: 1.0
  task_loss:
    weight: 0.0
```

At `weight: 1.0`, the optimized objective is pure teacher-output MSE and the
ordinary flow-matching target has weight zero because `task_loss.weight` defaults
to `0.0`. All coefficients are independent and additive. For example, setting
both output and task weights to `0.5` produces an equal output-MSE/flow-matching
mixture; adding layerwise terms does not silently renormalize either coefficient.

Optional layerwise MSE can be added without changing the output loss:

```yaml
qad:
  layerwise:
    enabled: true
    pairs:
      - student_layer: transformer_blocks.29
        teacher_layer: transformer_blocks.29
        selector: hidden_states
        weight: 0.05
```

Each pair is an exact module name relative to the student or teacher
transformer. Its weight is additive to the output/task objective. Start with
output-only training: layer hooks retain activations and therefore increase
memory use, especially when activation checkpointing is enabled.

The recipe logs the flow-matching loss, output MSE, every configured layerwise
MSE, and the final combined loss separately.

## Configuration and launch

The entry point is `examples/diffusers/fastgen/qad/finetune.py`. It uses the
same YAML plus dotted-command-line override convention as the other FastGen
recipes:

```bash
torchrun --nproc-per-node=4 \
  examples/diffusers/fastgen/qad/finetune.py \
  --config examples/diffusers/fastgen/qad/configs/qwen_image_svdquant_nvfp4.yaml \
  --fsdp.dp_size=4 \
  --model.pretrained_model_name_or_path=/path/to/qwen-image-nvfp4-svdquant-training-bundle \
  --data.dataloader.cache_dir=/path/to/qwen_image_1024p \
  --checkpoint.checkpoint_dir=/path/to/qad/checkpoints
```

Cluster launchers can keep the established `CONFIG`, `RUN_ID`, and
`EXTRA_ARGS` interface. For example:

```bash
EXTRA_ARGS="--step_scheduler.max_steps=50000 \
--step_scheduler.ckpt_every_steps=1000 \
--step_scheduler.num_epochs=200 \
--step_scheduler.global_batch_size=64 \
--optim.learning_rate=2e-6 \
--lr_scheduler.min_lr=2e-6 \
--fsdp.dp_size=64 \
--qad.teacher_model_name_or_path=Qwen/Qwen-Image \
--qad.output_loss.weight=1.0 \
--qad.task_loss.weight=0.0 \
--qad.student.mode=nvfp4_svdquant \
--model.pretrained_model_name_or_path=/path/to/qwen-image-nvfp4-svdquant-training-bundle \
--qad.student.train_scope=all \
--data.dataloader.cache_dir=/path/to/qwen_image_1024p" \
CONFIG=examples/diffusers/fastgen/qad/configs/qwen_image_svdquant_nvfp4.yaml \
RUN_ID=qad_qwen_image_svdquant_nvfp4_16n \
NODES=16 \
GPUS_PER_NODE=4 \
TIME=05:00:00 \
PARTITION=batch \
bash /path/to/experiments/qad_qwen_image/launch.sh
```

The launcher must invoke `examples/diffusers/fastgen/qad/finetune.py`.
Pointing the existing DMD2 launcher at a QAD YAML is not sufficient when that
launcher still hard-codes `dmd2_finetune.py`.

The launch environment contains no Attention Grill settings. It also contains
no DMD2 timestep, fake-score, discriminator, negative-prompt, GAN, or EMA
settings.

## Restore and checkpoint invariants

On a fresh run the recipe restores the complete student first, constructs its
final ModelOpt/PEFT topology, applies FSDP, builds the optimizer from the selected
training scope, and only then creates the frozen teacher and distillation
controller. On resume, the same immutable student source reconstructs the
topology before the QAD checkpoint is loaded.

The teacher and the transient ModelOpt distillation controller are not training
checkpoint payloads. Checkpoints contain the student state required by the
selected training scope together with optimizer, scheduler, dataloader, RNG, and
global-step state. Resolved dotted CLI overrides are materialized into the saved
`config.yaml`. Resume validates the student bundle, quantization mode, train
scope, teacher, and loss configuration before loading optimizer shards; do not
change them while resuming an existing run.
