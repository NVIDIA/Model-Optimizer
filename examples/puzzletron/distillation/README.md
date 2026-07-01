# Puzzletron Knowledge Distillation (Megatron Bridge)

Knowledge distillation (KD) for heterogeneous AnyModel / Puzzletron students and teachers,
driven by [Megatron Bridge](https://github.com/NVIDIA/Megatron-Bridge).

This example shows how to:

1. Distill an HF model into a (potentially smaller, potentially heterogeneous) student
   in Megatron-Core format.
2. Export the trained MCore checkpoint back to HuggingFace format for inference / eval.

The two scripts are:

| Script | Purpose |
|--------|---------|
| `distill.py` | Run distillation with student + teacher loaded from HF checkpoints. |
| `export_to_hf.py` | Convert a trained MCore checkpoint back to HuggingFace format. |

A shared `_common.py` provides the `MODEL_REGISTRY`, HF / Bridge loading helpers, and the
default `kd-container-default.yaml` discovery.

## Supported Models

`--student` / `--teacher` accept the following registry keys (defined in `_common.py`):

| Key      | HuggingFace model                                  | AnyModel converter |
|----------|----------------------------------------------------|--------------------|
| `gptoss` | `openai/gpt-oss-20b`                               | `gpt_oss`          |
| `nemo2`  | `nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16`       | `nemotron_h_v2`    |
| `llama`  | `meta-llama/Llama-3.2-3B-Instruct`                 | `llama`            |
| `qwen`   | `Qwen/Qwen3-8B`                                    | `qwen3`            |

If `--student-checkpoint` / `--teacher-checkpoint` is omitted the model is fetched from
the Hub. For local Puzzletron checkpoints the script reads `block_configs` from the HF
`config.json` so heterogeneous layer types are honored.

## Prerequisites

The recommended environment is the NeMo container (e.g. `nvcr.io/nvidia/nemo:26.02`) with
Megatron Bridge available under `/opt/Megatron-Bridge`. From the repo root:

```bash
python -m pip install -e ".[hf,puzzletron,dev-test]"
python -m pip install -r examples/puzzletron/requirements.txt

export PYTHONPATH="$(pwd):/opt/Megatron-Bridge/src:/opt/Megatron-Bridge/3rdparty/Megatron-LM:${PYTHONPATH:-}"
```

> **GPT-OSS only:** Megatron Bridge ships an upstream GPT-OSS bridge that does not yet
> handle Puzzletron's heterogeneous GPT-OSS layouts. ModelOpt now registers a patched
> `GPTOSSBridge` automatically (imported via the `puzzletron_mbridge` plugin from
> `modelopt/torch/puzzletron/plugins/mbridge/gpt_oss.py`), overriding the upstream bridge
> for `GptOssForCausalLM` — no manual file overlay is required.

## Run Distillation

### Minimal example

Distill a `llama` teacher into an `pruned llama` student on a single node:

```bash
torchrun --nproc-per-node=8  examples/puzzletron/distillation/distill.py \
--student llama \
--teacher llama \
--student-checkpoint /puzzletron/workspaces/Llama-3.1-8B-Instruct/mip/puzzle_solutions/target_memory_78000MiB-num_params_7G/solutions--checkpoints/solution_0/ \
--teacher-checkpoint /puzzletron/workspaces/Llama-3.1-8B-Instruct/ckpts/teacher/   \
--config-file examples/puzzletron/distillation/kd-container-llama.yaml \
--tensor-model-parallel-size 1 \
--pipeline-model-parallel-size 8 \
--expert-model-parallel-size 1 \
--expert-tensor-parallel-size 1 \
train.train_iters=1000 \
checkpoint.save=/puzzletron/workspaces/Llama-3.1-8B-Instruct/kd/puzzle_solutions/target_memory_78000MiB-num_params_7G-intermediate/ \
logger.wandb_exp_name=Llama-3.1-8B-Instruct-target_memory_78000MiB-num_params_7G-intermediate
```

Distill a `qwen3` teacher into an `pruned qwen` student on a single node (to fit into single gpu we limit the sequence length):

```bash
torchrun --nproc-per-node=8  examples/puzzletron/distillation/distill.py \
--student qwen \
--teacher qwen \
--student-checkpoint /puzzletron/workspaces/Qwen3-8B/mip/puzzle_solutions/target_memory_78000MiB-num_params_8G/solutions--checkpoints/solution_0/ \
--teacher-checkpoint /puzzletron/workspaces/Qwen3-8B/ckpts/teacher/ \
--config-file examples/puzzletron/distillation/kd-container-qwen.yaml \
--tensor-model-parallel-size 1 \
--pipeline-model-parallel-size 4 \
--expert-model-parallel-size 1 \
--expert-tensor-parallel-size 1 \
train.train_iters=1000 \
checkpoint.save=/puzzletron/workspaces/Qwen3-8B/kd/puzzle_solutions/target_memory_78000MiB-num_params_8G/ \
logger.wandb_exp_name=Qwen3-8B-intermediate \
model.seq_length=1024 \
dataset.seq_length=1024
```

Distill a `gpt-oss` teacher into an `pruned gpt-oss` student on a single node (to fit into single gpu we limit the sequence length):

```bash
torchrun --nproc-per-node=8  examples/puzzletron/distillation/distill.py \
--student gptoss \
--teacher gptoss \
--student-checkpoint /puzzletron/workspaces/any_model_gpt_oss_20b/mip/puzzle_solutions/stats_num_params_10914757184/solutions--checkpoints/solution_0/ \
--teacher-checkpoint /puzzletron/workspaces/any_model_gpt_oss_20b/ckpts/teacher \
--config-file examples/puzzletron/distillation/kd-container-qwen.yaml \
--tensor-model-parallel-size 1 \
--pipeline-model-parallel-size 4 \
--expert-model-parallel-size 1 \
--expert-tensor-parallel-size 1 \
train.train_iters=1000 \
checkpoint.save=/puzzletron/workspaces/any_model_gpt_oss_20b/kd/puzzle_solutions/stats_num_params_10914757184/solutions--checkpoints/solution_0/ \
logger.wandb_exp_name=GptOss-20b-intermediate \
model.seq_length=1024 \
dataset.seq_length=1024
```

### Heterogeneous student + teacher (e.g. GPT-OSS → Nemotron-H)

```bash
torchrun --nproc_per_node=8 examples/puzzletron/distillation/distill.py \
    --student nemo2  --student-checkpoint /path/to/nemotron-student \
    --teacher gptoss --teacher-checkpoint /path/to/gptoss-teacher \
    model.tensor_model_parallel_size=4 \
    model.expert_model_parallel_size=4
```

### Parallelism flags

The `--tensor/pipeline/expert-model-parallel-size` and `--expert-tensor-parallel-size`
flags are convenience shortcuts; the same fields can be set via YAML or Hydra dotlist.

```bash
torchrun --nproc_per_node=8 examples/puzzletron/distillation/distill.py \
    --student llama --teacher nemo2 \
    --tensor-model-parallel-size 4 \
    --pipeline-model-parallel-size 1 \
    --expert-model-parallel-size 1 \
    --expert-tensor-parallel-size 1
```

### Hydra-style CLI overrides

Anything not matching a known flag is treated as an OmegaConf dotlist override applied
to the full `ConfigContainer`:

```bash
torchrun --nproc_per_node=8 examples/puzzletron/distillation/distill.py \
    --student llama --teacher nemo2 \
    --teacher-checkpoint /path/to/teacher-hf \
    train.train_iters=50000 \
    optimizer.lr=1e-4 \
    checkpoint.save=./outputs/my-run \
    logger.wandb_project=my_project \
    logger.wandb_exp_name=llama-from-nemo2
```

### Configuration precedence

1. Defaults from Megatron Bridge `_pretrain_common()`.
2. YAML from `--config-file` (defaults to `kd-container-default.yaml` in this directory).
3. Hydra-style CLI overrides (highest priority).

The bundled YAMLs are:

| File | Purpose |
|------|---------|
| `kd-container-default.yaml` | Path-free defaults; safe starting point for any model. |
| `kd-container-llama.yaml` | Llama-specific KD recipe. |
| `kd-container-nemotron3.yaml` | Nemotron-H v3 recipe (intermediate-layer KD, real dataset). |
| `kd-container-qwen.yaml` | Qwen3 recipe. |

The default YAML leaves dataset paths as `null` — set `dataset.per_split_data_args_path`
or `dataset.blend_per_split` (and adjust `dataset.path_to_cache`) before running real
training. See the inline comment in `kd-container-default.yaml` for the
`blend_per_split` schema, and the `data_prep_*.ipynb` notebooks for tokenization.

### Debugging the layer / provider patchers

Per-iteration MoE diagnostics and patcher debug logging are gated behind an env var:

```bash
MBRIDGE_PATCHER_DEBUG=1 torchrun ... examples/puzzletron/distillation/distill.py ...
```

## Export to HuggingFace

After training, convert the saved MCore checkpoint back to HF format. The student HF
checkpoint is needed only to obtain its `config.json` / `block_configs` and tokenizer
files — no weights are loaded from it for the export itself.

```bash
torchrun --nproc_per_node=1 examples/puzzletron/distillation/export_to_hf.py \
    --student gptoss \
    --student-hf-checkpoint /path/to/student-hf \
    --student-mcore-checkpoint /path/to/kd-checkpoints/iter_0000400 \
    --output-hf-checkpoint /path/to/exported-hf
```

The exported directory will contain the converted weights plus the tokenizer / config
files copied verbatim from `--student-hf-checkpoint`
(`config.json`, `tokenizer.json`, `tokenizer_config.json`, `special_tokens_map.json`,
`chat_template.jinja`).

## Layout

```text
examples/puzzletron/distillation/
├── README.md                       # this file
├── distill.py                      # KD entrypoint (HF -> Bridge -> distill loop)
├── export_to_hf.py                 # MCore -> HF checkpoint export
├── _common.py                      # MODEL_REGISTRY + shared HF/Bridge helpers
├── hooks.py                        # PuzzletronHooks: thin DistillHooks glue
├── kd-container-default.yaml       # Default ConfigContainer overrides
└── kd-container-{llama,nemotron3,qwen}.yaml  # Per-model recipes
```

The reusable "patcher technique" logic now lives in the ModelOpt library under
`modelopt/torch/puzzletron/plugins/mbridge/` and is imported by the scripts above:

```text
modelopt/torch/puzzletron/plugins/mbridge/
├── __init__.py                     # registers bridges + re-exports patcher API
├── base.py                         # HeterogeneousBridgeMixin / GenericHeterogeneousProvider
├── llama.py, qwen3.py, gpt_oss.py  # block_configs-aware Megatron-Bridge adapters
├── block_config_utils.py           # Per-layer block_configs loader & translation
├── layer_patchers.py               # mbridge_patcher: per-layer MCore overrides
├── provider_patch.py               # ModelProviderMixin / DistillationProvider patches
└── distill_patches.py              # Reusable distillation helpers (MoE aux-loss fix, sync)
```
