# Knowledge Distillation with NeMo AutoModel

This guide shows how to run knowledge distillation on Puzzletron-compressed AnyModel (heterogeneous) checkpoints using **NeMo AutoModel**. AutoModel enables efficient training of any HuggingFace model with a unified API; here we extend it to load heterogeneous checkpoints and use TP-friendly KD loss.

## Overview

1. **AutoModel + AnyModel**: We monkey-patch NeMo AutoModel so `from_pretrained(..., anymodel_descriptor=..., block_configs_path=...)` can load heterogeneous checkpoints. The patch uses ModelOpt’s `ModelDescriptorFactory` and `deci_x_patcher` to apply per-layer configs during model init.
2. **Custom KD recipe**: For distillation we use a custom recipe (`recipe.py`) that adds pipeline-parallel (PP) support, better logging, and TP-friendly KD loss. Pretraining is unchanged and uses AutoModel’s built-in recipe. Once the AutoModel repo gains these features, the custom recipe can be dropped and the upstream KD recipe used instead.
3. **KD loss** (`loss.py`): We provide a TP-aware KD on precomputed logits only; CE is computed separately and mixed with `kd_ratio`.

**Supported parallelisms**  
FSDP is fully supported. Pipeline parallelism (PP) is supported for most models; exceptions are those whose layer naming does not follow the usual HuggingFace convention. Tensor parallelism (TP) and sequence parallelism (SP) are mostly supported—a known exception is GPT-OSS due to sink tokens (AutoModel has the same limitation; it is not specific to AnyModel). Context parallelism (CP) is supported for all models tested. Expert parallelism (EP) is not supported: AutoModel relies on custom (non–HuggingFace) model implementations for EP, which conflicts with the goal of supporting any HF model.

## Setup

**Requirements**

- NeMo AutoModel (install from source or use a container that provides it).
- ModelOpt installed (`pip install nvidia-modelopt` or install from the Model-Optimizer repo).
- For KD: this example’s `recipe.py`, `loss.py`, and `patch_automodel.py` (the run entrypoint always applies the patch before loading models).

**Environment**

Set `PYTHONPATH` so that the Model-Optimizer root is on the path (for ModelOpt and, if you run this example as a module, for `automodel_distillation`):

```bash
export PYTHONPATH="/path/to/Model-Optimizer:${PYTHONPATH}"
```

If you use a NeMo AutoModel container, ensure the AutoModel package is installed (e.g. clone AutoModel and `pip install -e .`). Upgrade HuggingFace Transformers if needed (e.g. for compatibility):

```bash
python -m pip install -e /path/to/AutoModel
python -m pip install -U omegaconf fire transformers
```

## Configuration

- **pretrain.yaml** – Pretrain/finetune on an AnyModel checkpoint. Set `model.pretrained_model_name_or_path` and `model.anymodel_descriptor` (e.g. `gpt_oss_20b`, `llama`, `qwen2`, `qwen3`). Optional: `model.block_configs_path`; if omitted, block configs are auto-detected from `<checkpoint_dir>/block_configs.json`.
- **kd.yaml** – Knowledge distillation. Set `model.pretrained_model_name_or_path` and `model.anymodel_descriptor` for the student, and `teacher_model.pretrained_model_name_or_path` and `teacher_model.anymodel_descriptor` for the teacher.

Paths and descriptors can be overridden from the command line (see below).

## Run

**Apply the patch and run KD**

Before loading models, the run entrypoint calls `apply_patch()` so that `from_pretrained` accepts `anymodel_descriptor` and `block_configs_path`. Then it loads the config and runs the chosen recipe.

Run from the **automodel_distillation** directory so that `run.py` can import `patch_automodel` and `recipe`:

```bash
cd /path/to/Model-Optimizer/examples/puzzletron/automodel_distillation
torchrun --nproc_per_node=2 \
  -m run \
  --mode kd \
  -c kd.yaml
```

Override config (e.g. paths and descriptor) on the command line:

```bash
torchrun --nproc_per_node=2 \
  -m run \
  --mode kd \
  -c kd.yaml \
  model.pretrained_model_name_or_path=/path/to/student \
  model.anymodel_descriptor=gpt_oss_20b \
  teacher_model.pretrained_model_name_or_path=/path/to/teacher \
  teacher_model.anymodel_descriptor=gpt_oss_20b
```

**Pretrain (uses AutoModel’s built-in recipe)**

```bash
torchrun --nproc_per_node=2 \
  -m run \
  --mode pretrain \
  -c pretrain.yaml \
  model.pretrained_model_name_or_path=/path/to/checkpoint \
  model.anymodel_descriptor=gpt_oss_20b
```

**Note:** If you run from a different layout (e.g. from the Model-Optimizer repo root or under another package name), set `PYTHONPATH` to include this directory so `run` can import `patch_automodel` and `recipe`, and ensure the config `kd_loss_fn._target_` (e.g. `loss.KDLoss`) resolves to the correct module.

## Example: Running on a cluster

Below is an example job setup: NeMo AutoModel container, clone AutoModel main, install it and upgrade Transformers, then run KD from a directory that contains your config and run script (e.g. a copy of this example or the RealAnyModel layout).

```bash
# Submit interactive job (example with your cluster’s submit_job)
submit_job --partition interactive --time 2.0 \
  --image nvcr.io/nvidia/nemo-automodel:25.11.00 \
  --mounts "/path/to/AutoModel/:/opt/Automodel/,/lustre:/lustre" \
  --interactive --gpu 2 --skip_image_check --email_mode=never \
  --command='bash'

# Inside the container
source /opt/venv/bin/activate
cd /opt/Automodel/
python -m pip install -e .
python -m pip install -U omegaconf fire transformers
python -m pip uninstall nvidia-modelopt
cd /path/to/Model-Optimizer
python -m pip install -e .

# Run KD (from your project dir that has run.py, kd.yaml, patch_automodel, loss, recipe)
cd ./examples/puzzletron/automodel_distillation/
torchrun --nproc_per_node 2 -m run --mode kd -c kd.yaml 2>&1 | tee logs
```

Use your own paths for mounts, checkpoint dirs, and config overrides as needed.

## Files in this example

| File | Purpose |
|------|--------|
| `patch_automodel.py` | Monkey-patch so `from_pretrained` accepts `anymodel_descriptor` and `block_configs_path`; uses ModelOpt’s `deci_x_patcher`. |
| `loss.py` | KDLoss: TP-aware KD on precomputed logits (CE is mixed via `kd_ratio` in the recipe). |
| `recipe.py` | Custom KD recipe (PP support, logging, TP-friendly KD). |
| `run.py` | Entrypoint: applies patch, then runs pretrain or KD using the config. |
| `pretrain.yaml` | Pretrain config (no hardcoded paths; override on CLI). |
| `kd.yaml` | KD config (no hardcoded paths; override on CLI). |
