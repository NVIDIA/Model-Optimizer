# DMD2 distillation for Qwen-Image

Distill [`Qwen/Qwen-Image`](https://huggingface.co/Qwen/Qwen-Image) into a **few-step
generator** with DMD2 (Distribution Matching Distillation). The distilled student
produces images in as few as **1–4 sampling steps** while matching the base model's
output distribution. Built on `modelopt.torch.fastgen` and NeMo AutoModel's
[`TrainDiffusionRecipe`](https://github.com/NVIDIA-NeMo/Automodel/blob/main/nemo_automodel/recipes/diffusion/train.py).

> [!NOTE]
> Qwen-Image is a third-party model with its own license terms. Review the
> [Qwen-Image model card](https://huggingface.co/Qwen/Qwen-Image) before downloading or
> redistributing weights or derivatives.

## Requirements & self-contained data path

This example runs against **stock upstream `nemo_automodel`** (`>=0.4.0,<1.0`; see
`requirements.txt`) from a **source checkout** of Model-Optimizer — the `examples/` tree is not
shipped in the `nvidia-modelopt` pip package. Install the example dependencies with:

```bash
pip install -r examples/diffusers/fastgen/requirements.txt
```

> [!TIP]
> Prefer not to install `nemo_automodel` yourself? Use the **NeMo AutoModel container**, which
> bundles it (with the diffusion extras) — then you only need a source checkout of Model-Optimizer
> for the `examples/` tree and can skip the `pip install` above:
>
> ```bash
> docker run --gpus all -it --rm --shm-size=8g nvcr.io/nvidia/nemo-automodel:26.04
> ```

The DMD2 data loading (`fastgen_data/`) and raw-image preprocessing (`preprocess/`) are
**vendored into this example** (from NeMo-AutoModel, Apache-2.0) so that **no modifications to
`nemo_automodel` are required**. The entry points put this directory on `sys.path`, so the
configs reference the vendored builders as `_target_: fastgen_data.build_*`. The DMD2 math in
`modelopt/torch/fastgen/` is unchanged.

**Build the training cache from raw images** (Qwen-Image VAE latents + text embeddings):

```bash
python examples/diffusers/fastgen/preprocess_qwen_image.py image \
    --image_dir <raw images> --output_dir <cache dir> --processor qwen_image \
    --caption_format meta_json
```

The CFG negative-prompt embedding (the config's `negative_prompt_embedding_path`) is generated
once from the same Qwen text encoder:

```bash
python examples/diffusers/fastgen/make_negative_prompt_embedding.py \
    --output <cache dir>/negative_prompt_embedding.pt
```

Then point the config's `data.dataloader.cache_dir` at `<cache dir>` and its
`negative_prompt_embedding_path` at `<cache dir>/negative_prompt_embedding.pt`, and train (below).

## How DMD2 works

DMD2 trains three networks together:

| Model | Role |
|---|---|
| **Student** | the few-step generator you keep |
| **Fake-score** | a diffusion model that tracks the *student's* current output distribution |
| **Teacher** | the frozen base Qwen-Image model (the *target* distribution) |

The distribution-matching gradient pushes the student toward the teacher and away from
the fake-score. Training alternates between two phases, controlled by `student_update_freq`:

```text
each step:
  if step % student_update_freq == 0:   # student phase
      update the student (distribution-matching [+ optional GAN] loss)
      update the student EMA
  else:                                  # fake-score phase
      update the fake-score network to track the student
```

The canonical config additionally enables **CFG** (classifier-free guidance on the
teacher) and a lightweight **GAN** branch (a discriminator head on a teacher feature
block, plus an R1 gradient penalty) for sharper samples.

## Install

From the repo root:

```bash
pip install -e ".[all]"                                      # ModelOpt + torch + diffusers
pip install -r examples/diffusers/fastgen/requirements.txt   # nemo_automodel
```

`nemo_automodel[diffusion]` pulls in diffusers, accelerate, and the `TrainDiffusionRecipe`
this example subclasses.

## Real-data training

`configs/dmd2_qwen_image.yaml` is the canonical config: 4-step student, CFG, and the
GAN + R1 branch, trained on a preprocessed latent cache. Before launching, provide:

- **A preprocessed Qwen-Image latent cache** — set `data.dataloader.cache_dir`.
- **A precomputed negative-prompt embedding** (required for CFG) — set
  `data.dataloader.negative_prompt_embedding_path`.
- **An output directory** — set `checkpoint.checkpoint_dir`.

The model path defaults to `Qwen/Qwen-Image`; point it at a local snapshot to avoid
re-downloading on every job. Then:

```bash
torchrun --nproc-per-node=8 \
    examples/diffusers/fastgen/dmd2_finetune.py \
    --config examples/diffusers/fastgen/configs/dmd2_qwen_image.yaml \
    --step_scheduler.max_steps=5000
```

Any `DMDConfig` field can be overridden on the CLI (e.g. `--dmd2.guidance_scale=3.5`).

### Checkpoints & resuming

Checkpoints land under `checkpoint.checkpoint_dir`. Alongside the student, the recipe
saves the DMD2 sidecars needed to resume exactly: the fake-score model + optimizer, the
student EMA (`ema_shadow.pt`), and the DMD iteration counter (`dmd_state.pt`). With
`restore_from: LATEST` a re-launch auto-resumes from the newest checkpoint; pin a
specific one with `--checkpoint.restore_from=epoch_0_step_500`.

## Inference

After training, sample from the distilled student. The pipeline loads your consolidated
student transformer plus the base Qwen-Image VAE / text encoder / tokenizer:

```python
import torch
from inference_dmd2_qwen_image import QwenImageDMDInferencePipeline

pipe = QwenImageDMDInferencePipeline.from_pretrained(
    student_path="/path/to/checkpoint/epoch_0_step_500/model/consolidated",
    base_pipeline_path="Qwen/Qwen-Image",
    ema_path=None,                      # or ".../ema_shadow.pt" to sample the EMA weights
    torch_dtype=torch.bfloat16,
).to("cuda")

image = pipe(
    prompt="a small red cube on a white table",
    num_inference_steps=4,              # match the student_sample_steps you trained with
    height=1024, width=1024,
    generator=torch.Generator("cuda").manual_seed(42),
).images[0]
image.save("sample.png")
```

Or run the bundled CLI for a quick check:

```bash
python examples/diffusers/fastgen/inference_dmd2_qwen_image.py \
    --student_path /path/to/checkpoint/.../model/consolidated \
    --base_pipeline_path Qwen/Qwen-Image \
    --prompt "a small red cube on a white table" \
    --height 512 --width 512
```

Set `num_inference_steps` to the number of steps the student was trained for
(`dmd2.student_sample_steps` — e.g. 4 for the canonical config, or 1 for a single-step
student).

## Config reference

| Section | Key | Role |
|---|---|---|
| `model` | `pretrained_model_name_or_path` | Qwen-Image HF id or local snapshot. |
| `model` | `mode` | `finetune` — loads the pretrained weights. |
| `step_scheduler` | `global_batch_size`, `local_batch_size`, `max_steps`, `ckpt_every_steps`, `log_every` | Standard AutoModel scheduling knobs. |
| `dmd2` | `recipe_path` | Built-in fastgen recipe to hydrate `DMDConfig` from (`general/distillation/dmd2_qwen_image`). |
| `dmd2` | `pipeline_plugin` | `qwen_image` — selects `QwenImageDMDPipeline` (2×2 patch packing / img_shapes). |
| `dmd2` | `student_sample_steps` | Number of student sampling steps (e.g. 4). |
| `dmd2` | `guidance_scale` | CFG strength on the teacher (`null` disables CFG; requires a negative-prompt embedding when set). |
| `dmd2` | `gan_loss_weight_gen`, `gan_r1_reg_weight`, `gan_feature_indices`, … | GAN branch (set `gan_loss_weight_gen: 0` to disable). |
| `dmd2` | `fake_score_lr`, `discriminator_lr` | Separate LRs for the fake-score / discriminator optimizers. |
| `dmd2` | `sample_t_cfg`, `ema` | Timestep sampling + student EMA settings. |
| `optim` | `learning_rate`, `optimizer.*` | Student AdamW knobs. |
| `fsdp` | `dp_size`, `tp_size`, `activation_checkpointing`, … | FSDP2 parallelism (set `dp_size` to your GPU count). |
| `data` | `dataloader._target_`, `cache_dir`, `negative_prompt_embedding_path` | Latent cache dir + optional CFG negative-prompt embedding. |
| `checkpoint` | `checkpoint_dir`, `model_save_format`, `restore_from` | Output dir, save format, resume behavior. |

## Troubleshooting

**`CUDA out of memory`.** Training holds three Qwen-Image transformers (student + teacher
- fake-score) plus optimizer state. Shard across more GPUs (raise `--fsdp.dp_size`),
or enable `--fsdp.activation_checkpointing=true`.

**Loss is `NaN` on step 0.** Almost always an out-of-range timestep — confirm you haven't
overridden `dmd2.pred_type` away from `flow` (Qwen-Image is a rectified-flow model) or
changed the timestep schedule.

**`guidance_scale is set but negative_encoder_hidden_states was not provided`.** CFG needs
a precomputed negative-prompt embedding. Set `data.dataloader.negative_prompt_embedding_path`,
or set `dmd2.guidance_scale: null` to disable CFG.

**Dataloader yields empty batches.** Ensure your cache has at least
`local_batch_size * fsdp.dp_size` items; the distributed sampler drops incomplete batches.

## Reference

- Fastgen library: [`modelopt/torch/fastgen/`](../../../modelopt/torch/fastgen/)
- Built-in recipe: [`modelopt_recipes/general/distillation/dmd2_qwen_image.yaml`](../../../modelopt_recipes/general/distillation/dmd2_qwen_image.yaml)
- AutoModel recipe this example subclasses:
  [`nemo_automodel/recipes/diffusion/train.py`](https://github.com/NVIDIA-NeMo/Automodel/blob/main/nemo_automodel/recipes/diffusion/train.py)
