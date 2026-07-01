# DMD2 distillation for Qwen-Image and Qwen-Image-Edit

Distill [`Qwen/Qwen-Image`](https://huggingface.co/Qwen/Qwen-Image) into a **few-step
generator** with DMD2 (Distribution Matching Distillation). The distilled student
produces images in as few as **1–4 sampling steps** while matching the base model's
output distribution. Built on `modelopt.torch.fastgen` and NeMo AutoModel's
[`TrainDiffusionRecipe`](https://github.com/NVIDIA-NeMo/Automodel/blob/main/nemo_automodel/recipes/diffusion/train.py).

The paired image-edit path supports
[`Qwen/Qwen-Image-Edit-2511`](https://huggingface.co/Qwen/Qwen-Image-Edit-2511),
including its one-or-more reference-image input contract.

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

## Qwen-Image-Edit-2511 training

Image editing uses `configs/dmd2_qwen_image_edit_2511.yaml`. It is not a text-to-image
cache with an extra tensor: Edit-2511 conditions every transformer call with packed
reference-image latents, and its Qwen2.5-VL prompt embedding jointly encodes the edit
instruction and those same ordered references. Stable `diffusers>=0.37.0` is required so
the transformer's `zero_cond_t` path applies `t=0` modulation to the reference tokens.

Preprocess native SpatialEdit WebDataset shards directly, without extracting the image
corpus:

```bash
python examples/diffusers/fastgen/preprocess_qwen_image_edit.py \
    --input-dir /path/to/SpatialEdit-500K \
    --output-dir /path/to/qwen_image_edit_2511_cache \
    --model-name /path/to/Qwen-Image-Edit-2511 \
    --gpu-id 0
```

The preprocessor also accepts `--manifest pairs.jsonl`. A row supplies a target, an edit
instruction, and one or more ordered references; image values may be local paths or
`{"archive": "/path/shard.tar", "member": "sample.0.jpg"}` descriptors:

```json
{"id":"sample-1","target":"target.png","conditioning":["source.png"],"prompt":"Move the red cube left."}
```

Each cached sample contains the target latent, a list of deterministic reference latents,
and image-aware positive **and negative** embeddings. Consequently the edit dataloader
does not take `negative_prompt_embedding_path`; one global text-only negative embedding
would omit the reference-image visual tokens.

Keep the edit dataloader at `batch_size: 1` with the current sampler. It buckets target
resolution only; batching multiple samples also requires matching reference count and every
reference-slot shape. The collate rejects incompatible batches instead of padding image tokens.

```bash
torchrun --nproc-per-node=8 \
    examples/diffusers/fastgen/dmd2_finetune.py \
    --config examples/diffusers/fastgen/configs/dmd2_qwen_image_edit_2511.yaml \
    --model.pretrained_model_name_or_path=/path/to/Qwen-Image-Edit-2511 \
    --data.dataloader.cache_dir=/path/to/qwen_image_edit_2511_cache \
    --fsdp.dp_size=8 --step_scheduler.global_batch_size=8
```

The `qwen_image_edit` plugin packs the noisy target first, appends every clean reference,
builds `img_shapes=[target, reference_1, ...]`, and slices the transformer prediction back
to the target-token prefix. The same references are forwarded through student,
teacher, fake-score, CFG, backward-simulation, and GAN paths.

### Checkpoints & resuming

Checkpoints land under `checkpoint.checkpoint_dir`. Alongside the student, the recipe
saves the DMD2 sidecars needed to resume exactly: the fake-score model + optimizer, the
DMD iteration counter (`dmd_state.pt`), and, when EMA is enabled, the student EMA
(`ema_shadow.pt`). With
`restore_from: LATEST` a re-launch auto-resumes from the newest checkpoint; pin a
specific one with `--checkpoint.restore_from=epoch_0_step_500`.

## Quantization-aware training (QAT)

Continue a full-precision DMD2 run with the **student quantized**, so the few-step model
stays accurate at FP8/NVFP4. QAT here is **restore-only**: the trainer loads a ModelOpt
quantizer state (recipe + frozen `amax`) from disk and **never calibrates**. Only the
student is quantized; the frozen teacher and trainable fake-score stay full precision so
the distribution-matching gradient is exact, and `amax` stays frozen for the whole run.

QAT is driven by a `dmd2.quant` block — there's no dedicated config file. The cleanest
way to launch is to **reuse the exact config + overrides of the full-precision run you're
continuing** and add only the three `dmd2.quant.*` keys (plus a reduced LR), so the QAT
run is provably identical to the FP run except for quantization and learning rate. The
CLI parser creates the `dmd2.quant` subtree even when it's absent from the YAML.

| Key | Role |
| --- | --- |
| `dmd2.quant.enabled` | Turn QAT on (restore-only student quantization). |
| `dmd2.quant.quant_state_path` | The `transformer.pt` from step 1 below (recipe + frozen `amax`). |
| `dmd2.quant.init_weights_from` | FP DMD2 checkpoint to warm-start student / fake-score / optimizers from on the first launch (the run `amax` was calibrated against). |

1. **Calibrate once** with the quantization example to produce the quantizer state
   (`amax`, no weights) for a trained student checkpoint:

   ```bash
   python examples/diffusers/quantization/quantize.py \
       --model qwen-image-dmd2 --format fp8 \
       --extra-param student_path=<.../epoch_4_step_15999/model/consolidated> \
       --quantized-torch-ckpt-save-path <.../epoch_4_step_15999/quant>
   # -> writes <.../epoch_4_step_15999/quant/transformer.pt>
   ```

2. **Launch QAT** by re-running the FP run's command with a new output dir, a reduced
   student LR, and the three quant keys appended:

   ```bash
   torchrun --nproc-per-node=<gpus> \
       examples/diffusers/fastgen/dmd2_finetune.py \
       --config examples/diffusers/fastgen/configs/<the FP run's config>.yaml \
       --checkpoint.checkpoint_dir=<NEW output dir> \
       <... the FP run's other overrides, unchanged ...> \
       --optim.learning_rate=<FP lr / 10> --lr_scheduler.min_lr=<FP lr / 10> \
       --dmd2.quant.enabled=true \
       --dmd2.quant.quant_state_path=<.../epoch_4_step_15999/quant/transformer.pt> \
       --dmd2.quant.init_weights_from=<.../epoch_4_step_15999>
   ```

On the first launch (empty `checkpoint_dir`) the student / fake-score / discriminator /
optimizers warm-start from `init_weights_from`, then the student is quantized from
`quant_state_path`. `restore_from: LATEST` auto-resumes the new `checkpoint_dir`
thereafter. Because QAT is restore-only — amax never recalibrates — the recipe re-applies
`quant_state_path` on every resume rather than persisting a per-checkpoint copy, so keep
that file accessible for the whole run (it's the only quantization dependency). The saved
student weights are clean full precision (`model/consolidated` is a normal
`QwenImageTransformer2DModel`); re-apply `quant_state_path` to deploy or evaluate the
quantized QAT student via the quantization example.

> The `quant_state_path` `amax` must have been calibrated against the student in
> `init_weights_from`, with the same few-step schedule (`dmd2.sample_t_cfg.t_list`) the
> student trains/infers with. Pass `dmd2.quant.enabled=true` on every resume too (it is
> what tells the recipe to quantize). Reduce only the student LR by keeping
> `--dmd2.fake_score_lr` / `--dmd2.discriminator_lr` at the FP value.

For Qwen-Image-Edit, the restore-only QAT path itself is unchanged because the student is
still a `QwenImageTransformer2DModel`. The quantizer state must, however, be calibrated on
the exact edit student using target + reference tokens, multimodal image/instruction
embeddings, and the same `t_list`. The existing `--model qwen-image-dmd2` text-only
calibrator does **not** provide representative activation ranges for Edit-2511. Also
calibrate the non-EMA weights restored by `dmd2.quant.init_weights_from`; do not calibrate
an EMA overlay for that warm start.

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

For an edit student, use the companion pipeline and pass one or more ordered references:

```python
from inference_dmd2_qwen_image_edit import QwenImageEditDMDInferencePipeline
from diffusers.utils import load_image

pipe = QwenImageEditDMDInferencePipeline.from_pretrained(
    student_path="/path/to/checkpoint/model/consolidated",
    base_pipeline_path="Qwen/Qwen-Image-Edit-2511",
).to("cuda")
image = pipe(
    [load_image("source.png")],
    "Move the red cube left.",
    num_inference_steps=4,
).images[0]
```

## Config reference

| Section | Key | Role |
|---|---|---|
| `model` | `pretrained_model_name_or_path` | Qwen-Image HF id or local snapshot. |
| `model` | `mode` | `finetune` — loads the pretrained weights. |
| `step_scheduler` | `global_batch_size`, `local_batch_size`, `max_steps`, `ckpt_every_steps`, `log_every` | Standard AutoModel scheduling knobs. |
| `dmd2` | `recipe_path` | Built-in fastgen recipe to hydrate `DMDConfig` from (`general/distillation/dmd2_qwen_image`). |
| `dmd2` | `pipeline_plugin` | `qwen_image` for T2I or `qwen_image_edit` for target + reference token packing. |
| `dmd2` | `student_sample_steps` | Number of student sampling steps (e.g. 4). |
| `dmd2` | `guidance_scale` | CFG strength on the teacher (`null` disables CFG; requires a negative-prompt embedding when set). |
| `dmd2` | `gan_loss_weight_gen`, `gan_r1_reg_weight`, `gan_feature_indices`, … | GAN branch (set `gan_loss_weight_gen: 0` to disable). |
| `dmd2` | `fake_score_lr`, `discriminator_lr` | Separate LRs for the fake-score / discriminator optimizers. |
| `dmd2` | `sample_t_cfg`, `ema` | Timestep sampling + student EMA settings. |
| `optim` | `learning_rate`, `optimizer.*` | Student AdamW knobs. |
| `fsdp` | `dp_size`, `tp_size`, `activation_checkpointing`, … | FSDP2 parallelism (set `dp_size` to your GPU count). |
| `data` | `dataloader._target_`, `cache_dir`, `negative_prompt_embedding_path` | Latent cache. The static negative path is T2I-only; edit negatives are cached per sample. |
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
