# DMD2 on Wan 2.2 5B — AutoModel integration (fastgen Phase 1)

> [!WARNING]
> **Third-Party License Notice — Wan 2.2**
>
> Wan 2.2 is a third-party model developed and provided by Wan-AI. It is **not**
> covered by the Apache 2.0 license that governs NVIDIA Model Optimizer. By downloading
> and using Wan 2.2 weights with Model Optimizer you must comply with Wan-AI's license.
> Any derivative models or fine-tuned weights produced through DMD2 distillation remain
> subject to Wan-AI's license and are **not** covered by Apache 2.0.

Distributed training example that exercises `modelopt.torch.fastgen.DMDPipeline`
end-to-end on `Wan-AI/Wan2.2-TI2V-5B-Diffusers` under FSDP2. Intended target:
**validate the DMD2 math in the real training environment** — once this loop runs
clean we layer CFG, the discriminator, and the real-data path on top (Phase 2).

This example subclasses
[`TrainDiffusionRecipe`](https://github.com/NVIDIA-NeMo/Automodel/blob/main/nemo_automodel/recipes/diffusion/train.py)
from NeMo AutoModel and swaps the flow-matching loss for
[`DMDPipeline`](../../../modelopt/torch/fastgen/methods/dmd.py).

## Scope — Phase 1 vs Phase 2

| | Phase 1 (this directory) | Phase 2 (roadmap) |
|---|---|---|
| Student update | VSD only | VSD + CFG + GAN generator term |
| Fake-score update | DSM | DSM (same) |
| Discriminator update | **not run** (no discriminator) | toy multiscale MLP + R1 |
| Data | mock (AutoModel `build_mock_dataloader`) | real preprocessed `.meta` cache |
| CFG | `guidance_scale: null` | negative-prompt precompute + CFG |
| Checkpointing | student DCP + fake_score DCP + EMA + DMD scalar state | + discriminator DCP |

## What the training loop does

Each step:

```
┌─────────────────────────────────────────────────────────────────────┐
│ if (global_step % student_update_freq == 0):   # student phase     │
│     loss = compute_student_loss(latents, noise, text_embeds)       │
│     loss["total"].backward()                                       │
│     student_optimizer.step()                                        │
│     dmd.update_ema()                                                │
│ else:                                           # fake-score phase │
│     loss = compute_fake_score_loss(latents, noise, text_embeds)    │
│     loss["total"].backward()                                       │
│     fake_score_optimizer.step()                                     │
└─────────────────────────────────────────────────────────────────────┘
```

The YAML's `dmd2.recipe_path: general/distillation/dmd2_wan22_5b` pulls the
canonical Wan 2.2 5B hyperparameters (`student_update_freq=5`,
`num_train_timesteps=1000`, `fake_score_pred_type=x0`,
`sample_t_cfg: shifted(5.0)`, `t_list=[0.999, 0.833, 0.0]`, etc.). Flat keys under
the `dmd2:` block apply targeted overrides on top.

## Install

From the repo root:

```bash
pip install -e ".[all]"                                  # ModelOpt + diffusers + torch
pip install -r examples/diffusers/fastgen/requirements.txt  # nemo_automodel
```

`nemo_automodel[diffusion]` pulls in `diffusers`, `accelerate`, the WAN
preprocessing helpers, and the
[`TrainDiffusionRecipe`](https://github.com/NVIDIA-NeMo/Automodel/blob/main/nemo_automodel/recipes/diffusion/train.py)
we subclass here.

## Quick start

### Target smoke — 8×H100, full Wan 2.2 5B latent shape, mock data

```bash
torchrun --nproc-per-node=8 \
    examples/diffusers/fastgen/dmd2_finetune.py \
    --config examples/diffusers/fastgen/configs/dmd2_wan22_5b.yaml
```

Expected behaviour:
- 100 optimiser steps, `student_update_freq=5` so 20 student phases + 80 fake-score
  phases (the first step fires on the student phase).
- `[STEP 0] phase=student ...`, `[STEP 1..4] phase=fake_score ...`,
  `[STEP 5] phase=student ...` in the logs.
- Memory per H100 in the ~50–65 GiB range (three 5B transformers sharded 8-way +
  activations + AdamW states for the trainable pair).
- Checkpoint lands at step 100 under `/tmp/dmd2-wan22-5b-phase1/epoch_0_step_100/`.

### Fast iteration — 2 GPUs, shrunken mock latents, ~2 min

Same recipe but scale the mock latent tensor down so each forward is cheap:

```bash
torchrun --nproc-per-node=2 \
    examples/diffusers/fastgen/dmd2_finetune.py \
    --config examples/diffusers/fastgen/configs/dmd2_wan22_5b.yaml \
    --step_scheduler.max_steps=20 \
    --fsdp.dp_size=2 \
    --data.mock.num_channels=8 \
    --data.mock.num_frame_latents=4 \
    --data.mock.spatial_h=16 \
    --data.mock.spatial_w=16 \
    --wandb.mode=offline
```

This is the "did my change compile and run" smoke loop. Runs the exact same DMD2
code path as the full-scale recipe, so logic bugs surface here.

### Resume from a checkpoint

Point `checkpoint.restore_from` at an absolute path or a dir name relative to
`checkpoint.checkpoint_dir`:

```bash
torchrun --nproc-per-node=8 \
    examples/diffusers/fastgen/dmd2_finetune.py \
    --config examples/diffusers/fastgen/configs/dmd2_wan22_5b.yaml \
    --checkpoint.restore_from=epoch_0_step_100 \
    --step_scheduler.max_steps=200
```

The recipe restores the student (via `TrainDiffusionRecipe`), plus the sidecar
files this example writes: `fake_score/` (DCP), `fake_score_optimizer/` (DCP),
`ema_shadow.pt`, and `dmd_state.pt` (carries the DMD iteration counter).

## Config reference

| Section | Key | Role |
|---|---|---|
| `model` | `pretrained_model_name_or_path` | HF path. Defaults to `Wan-AI/Wan2.2-TI2V-5B-Diffusers`. |
| `model` | `mode` | Must be `finetune` — loads the pretrained HF weights. |
| `step_scheduler` | `global_batch_size`, `local_batch_size`, `max_steps`, `ckpt_every_steps`, `log_every` | Standard AutoModel knobs. |
| `dmd2` | `recipe_path` | Built-in fastgen recipe to hydrate DMDConfig from. |
| `dmd2` | `gan_loss_weight_gen`, `guidance_scale`, `fake_score_pred_type`, etc. | Any `DMDConfig` field can be overridden here. |
| `dmd2` | `fake_score_lr` | Separate LR for the fake-score optimizer. Defaults to student LR. |
| `optim` | `learning_rate`, `optimizer.weight_decay`, `optimizer.betas` | Student AdamW knobs; re-used for the fake_score optimizer unless `dmd2.fake_score_lr` overrides. |
| `fsdp` | `dp_size`, `tp_size`, `cp_size`, `pp_size`, `activation_checkpointing` | Passed through to AutoModel's FSDP2Manager. |
| `data` | `use_mock`, `mock.*` | Toggle AutoModel's mock dataloader. All `mock.*` fields feed `build_mock_dataloader`. |
| `checkpoint` | `enabled`, `checkpoint_dir`, `model_save_format`, `restore_from` | Standard AutoModel Checkpointer. |

## Troubleshooting

**`CUDA out of memory` at fake_score load time.** Wan 2.2 5B is ~10 GiB bf16, and we
hold student + teacher + fake_score. On 80 GiB cards, FSDP2-sharded 8-way across the
three models fits comfortably; on 40 GiB cards you need more GPUs or a smaller dp_size.
For the fastest iteration drop to the 2-GPU shrunken-latent smoke above.

**`RuntimeError: teacher._fastgen_captured is missing`.** This means the GAN branch
of `compute_student_loss` fired without feature-capture hooks installed. In Phase 1
`gan_loss_weight_gen` is pinned to 0.0, so if you see this error you have overridden
`gan_loss_weight_gen` somewhere without attaching hooks — either revert the override or
call `mtf.plugins.wan22.attach_feature_capture(teacher, feature_indices=[15, 22, 29])`
in your fork of `setup()`.

**`ValueError: guidance_scale is set but negative_encoder_hidden_states was not provided.`**
Phase 1 deliberately leaves `negative_encoder_hidden_states=None`. If you override the
YAML's `dmd2.guidance_scale` away from `null`, you also need to precompute a negative
prompt embedding during `setup()` — wait for Phase 2 or do it yourself in your fork.

**Dataloader yields empty batches.** Check `data.mock.length >= step_scheduler.local_batch_size * fsdp.dp_size`; `build_mock_dataloader` drops incomplete batches when using the distributed sampler.

**Training loss is NaN on step 0 with mock data.** Mock latents are `torch.randn`,
which is a reasonable prior. NaN on step 0 almost certainly means the transformer is
receiving an out-of-range timestep — verify that `dmd2.num_train_timesteps` is `1000`
(diffusers convention for Wan 2.2) and that you haven't overridden `pred_type` away
from `flow`.

## Reference

- Fastgen library: [`modelopt/torch/fastgen/`](../../../modelopt/torch/fastgen/)
- Built-in recipe: [`modelopt_recipes/general/distillation/dmd2_wan22_5b.yaml`](../../../modelopt_recipes/general/distillation/dmd2_wan22_5b.yaml)
- FastGen reference math: `FastGen/fastgen/methods/distribution_matching/dmd2.py`
  (not shipped with Model-Optimizer)
- AutoModel recipe we subclass:
  [`nemo_automodel/recipes/diffusion/train.py`](https://github.com/NVIDIA-NeMo/Automodel/blob/main/nemo_automodel/recipes/diffusion/train.py)
