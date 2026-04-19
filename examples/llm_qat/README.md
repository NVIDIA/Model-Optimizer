# Quantization Aware Training (QAT) and Distillation (QAD)

Quantization Aware Training (QAT) improves model accuracy beyond post-training quantization (PTQ) at low precisions (e.g., INT4, FP4 on [NVIDIA Blackwell](https://www.nvidia.com/en-us/data-center/technologies/blackwell-architecture/)). Quantization Aware Distillation (QAD) further improves accuracy by using the original full-precision model as a teacher.

For background on how QAT enables low-precision accuracy recovery, see the [QAT/QAD blog post](https://developer.nvidia.com/blog/how-quantization-aware-training-enables-low-precision-accuracy-recovery/).

<div align="center">

| **Section** | **Description** | **Link** | **Docs** |
| :---: | :---: | :---: | :---: |
| Quick Start | Prerequisites and setup | \[[Link](#quick-start)\] | |
| End-to-End Example | Run QAT/QAD in 3 steps: quantize, train, export | \[[Link](#run-end-to-end-qatqad-example)\] | |
| Background | How QAT/QAD work and when to use each | \[[Link](#background)\] | \[[docs](https://nvidia.github.io/Model-Optimizer/guides/1_quantization.html)\] |
| Support Matrix | Supported models, quantization formats, and backends | \[[Link](#support-matrix)\] | |
| QLoRA | Model training with reduced GPU memory | \[[Link](#qlora-real-quantization)\] | |
| Advanced Topics | FSDP2 config, YAML options | \[[Link](#advanced-topics)\] | |
| Results | Accuracy benchmarks | \[[Link](#results)\] | |
| Resources | Extra links and references | \[[Link](#resources)\] | |

</div>

## Quick Start

### Prerequisites

Please refer to [llm_ptq/README.md](../llm_ptq/README.md#pre-requisites) for prerequisites.

The Qwen3-8B example below requires a minimum of **2 x 80GB GPUs**.

## Run End-to-End QAT/QAD Example

All arguments can be specified via YAML config, CLI flags, or both (CLI overrides YAML). See [Advanced Topics](#advanced-topics).

### QAT

Quantize, fine-tune on labeled data, and export:

```sh
# 1. Quantize
python quantize.py \
  --model_name_or_path Qwen/Qwen3-8B \
  --dataset_config configs/dataset/blend.yaml \
  --recipe general/ptq/nvfp4_default-fp8_kv \
  --quantize_output_dir qwen3-8b-quantized

# 2. Train
accelerate launch --config-file configs/accelerate/fsdp2.yaml train.py \
  --config configs/train/qat_nvfp4.yaml \
  --model_name_or_path qwen3-8b-quantized \
  --output_dir qwen3-8b-qat-nvfp4

# 3. Export
python export.py --pyt_ckpt_path qwen3-8b-qat-nvfp4 --export_path qwen3-8b-qat-deploy
```

### QAD

Quantize, recover accuracy using the original model as teacher, and export:

```sh
# 1. Quantize
python quantize.py \
  --model_name_or_path Qwen/Qwen3-8B \
  --dataset_config configs/dataset/blend.yaml \
  --recipe general/ptq/nvfp4_default-fp8_kv \
  --quantize_output_dir qwen3-8b-quantized

# 2. Train with distillation
accelerate launch --config-file configs/accelerate/fsdp2.yaml train.py \
  --config configs/train/qad_nvfp4.yaml \
  --model_name_or_path qwen3-8b-quantized \
  --teacher_model Qwen/Qwen3-8B \
  --output_dir qwen3-8b-qad-nvfp4

# 3. Export
python export.py --pyt_ckpt_path qwen3-8b-qad-nvfp4 --export_path qwen3-8b-qad-deploy
```

Exported checkpoints can be deployed on [TensorRT-LLM](https://github.com/NVIDIA/TensorRT-LLM), [vLLM](https://github.com/vllm-project/vllm), or [SGLang](https://github.com/sgl-project/sglang). See [llm_ptq/README.md](../llm_ptq/README.md#deployment) for deployment instructions. For quick accuracy evaluation without exporting, see [Native Fake-Quantized Evaluation](#native-fake-quantized-evaluation).

> **Note:** To see the full QAT flow in a single script (quantize + train + save), see [simple_qat_train.py](simple_qat_train.py):
>
> ```sh
> python simple_qat_train.py --model-path meta-llama/Llama-3.2-3B --recipe general/ptq/nvfp4_default-fp8_kv
> ```

## Background

### What is QAT?

**Quantization Aware Training (QAT)** inserts simulated quantization operations into the model graph and then fine-tunes the model so its weights learn to compensate for quantization error. During training, quantization scales are frozen while weights are updated. QAT is a general technique — it learns from labeled data on a quantized model.

```python
import modelopt.torch.quantization as mtq
from modelopt.recipe import load_recipe

# 1. Load a quantization recipe
recipe = load_recipe("general/ptq/nvfp4_default-fp8_kv")

# 2. Quantize the model in-place
model = mtq.quantize(model, recipe.quantize, forward_loop)

# 3. Fine-tune the quantized model
trainer.train()
trainer.save_model()
```

> ModelOpt provides accelerated quantization kernels using Triton for NVFP4 QAT. See the [installation guide](https://nvidia.github.io/Model-Optimizer/getting_started/_installation_for_Linux.html#accelerated-quantization-with-triton-kernels).

### What is QAD?

**Quantization Aware Distillation (QAD)** is a special case of QAT that uses a teacher model (typically the original unquantized model) to guide the quantized student via a distillation loss. QAD is a **pure accuracy recovery technique** — its goal is to recover accuracy lost from quantization, not to teach the model a new task.

To learn more, read the [QAT/QAD blog post](https://developer.nvidia.com/blog/how-quantization-aware-training-enables-low-precision-accuracy-recovery/).

### When to Use QAT vs QAD

| | **QAT** (without distillation) | **QAD** (with distillation) |
|-|---------|----------------------|
| **What it does** | Fine-tunes a quantized model on labeled data | Recovers quantization accuracy using the original model as teacher |
| **When to use** | The model is already quantized and you want to fine-tune it for a **new task** (e.g., fine-tuning a [GPT-OSS](../gpt-oss/) quantized checkpoint) | You want the **best possible accuracy recovery** after quantization |
| **Recommended workflow** | Start from a quantized checkpoint, fine-tune with task-specific data | Full-precision fine-tuning first, then QAD to recover quantization loss |

**QAD is Model Optimizer's recommended strategy for accuracy recovery after quantization.** In our experiments, full-precision fine-tuning followed by QAD delivers the best accuracy, especially at aggressive quantization levels (e.g., NVFP4). The optimal balance between QAT and QAD for a given model and task is an active area of research.

### Using `QATTrainer` and `QADTrainer`

`QATTrainer` is a drop-in replacement for HuggingFace's `Trainer` that handles quantization-aware training seamlessly with various distributed backends (FSDP2, DeepSpeed, DDP):

```python
from modelopt.torch.quantization.plugins.transformers_trainer import QATTrainer

trainer = QATTrainer(
    model=model,            # pre-quantized model
    processing_class=tokenizer,
    args=training_args,
    **data_module,
)
trainer.train()
trainer.save_model()
```

`QADTrainer` extends `QATTrainer` with distillation. Pass the teacher model and a `DistillArguments` instance:

```python
from modelopt.torch.distill.plugins.huggingface import DistillArguments
from modelopt.torch.quantization.plugins.transformers_trainer import QADTrainer

distill_args = DistillArguments(
    distill=True,
    teacher_model="Qwen/Qwen3-8B",
    criterion="logits_loss",
)

trainer = QADTrainer(
    model=model,            # pre-quantized model
    processing_class=tokenizer,
    args=training_args,
    distill_args=distill_args,
    **data_module,
)
trainer.train()
trainer.save_model()
```

### Quantization Recipes

Recipes are declarative YAML files that specify the quantization configuration. Built-in recipes are available in [`modelopt_recipes/`](../../modelopt_recipes/):

```sh
# List available built-in recipes
ls modelopt_recipes/general/ptq/
```

See [custom calibration](https://nvidia.github.io/Model-Optimizer/guides/_pytorch_quantization.html#advanced-configuration-creation) for creating your own recipe.

## Support Matrix

### Supported Models

| Model | Chat Template | Support |
|-------|---------------|---------|
| Qwen2, 2.5, 3, 3.5 dense models | ChatML | Yes (chat + pretrain) |
| Llama 2, 3, 3.1 | Llama | Yes (pretrain only) |

> **Note:** Chat-template-based assistant masking (`apply_chat_template`) is supported for ChatML models (Qwen, Nemotron). For other model families, use `apply_chat_template: false` in your dataset config to use pretrain-style tokenization where all tokens are trainable.

### Supported Quantization Formats

| Format | Precision | Recipe | Use Case |
|--------|-----------|--------|----------|
| **NVFP4** | W4A4 + FP8 KV | `general/ptq/nvfp4_default-fp8_kv` | Maximum compression for Blackwell GPUs |
| **FP8** | W8A8 + FP8 KV | `general/ptq/fp8_default-fp8_kv` | Balanced speed and accuracy |
| **INT4** weight-only | W4A16 | `general/ptq/int4_blockwise_weight_only` | Deployable on all Ampere or later GPUs |

> **NVFP4** uses 4-bit FP weights and activations (E2M1 with FP8 dynamic scales) plus FP8 KV cache. Partial variants are available for quantizing only specific layers (e.g., MLP-only, MoE experts-only) — see [`modelopt_recipes/general/ptq/`](../../modelopt_recipes/general/ptq/) for all options.

### Supported Backends

| Backend | Config File | Notes |
|---------|------------|-------|
| FSDP2 | `configs/accelerate/fsdp2.yaml` | **Recommended** |
| DDP | `configs/accelerate/ddp.yaml` | Add `--gradient_checkpointing True` |
| DeepSpeed | `configs/accelerate/deepspeed.yaml` | Add `--gradient_checkpointing True` |

Replace `--config-file configs/accelerate/fsdp2.yaml` with the desired backend config in any of the commands above.

## QLoRA (Real Quantization)

[QLoRA](https://arxiv.org/pdf/2305.14314) reduces training memory by quantizing LoRA backbone weights with real quantization via `mtq.compress()`.

```sh
# 1. Quantize with compression
python quantize.py \
  --model_name_or_path Qwen/Qwen3-8B \
  --dataset_config configs/dataset/blend.yaml \
  --recipe general/ptq/nvfp4_default-fp8_kv \
  --compress True \
  --quantize_output_dir qwen3-8b-quantized

# 2. Train with QLoRA
accelerate launch --config-file configs/accelerate/ddp.yaml train.py \
  --config configs/train/qlora_nvfp4.yaml \
  --model_name_or_path qwen3-8b-quantized \
  --output_dir qwen3-8b-fp4-qlora

# 3. Export
python export.py \
  --pyt_ckpt_path qwen3-8b-fp4-qlora \
  --export_path qwen3-8b-fp4-qlora-hf

# 4. Serve with vLLM
vllm serve qwen3-8b-fp4-qlora-hf/base_model --enable-lora \
  --lora-modules adapter=qwen3-8b-fp4-qlora-hf --port 8000 \
  --tokenizer qwen3-8b-fp4-qlora-hf
```

> QLoRA export is not currently supported with FSDP2.

## Advanced Topics

<details>
<summary><b>FSDP2 and Model-Specific Layer Wrapping</b></summary>

The default `fsdp2.yaml` uses `TRANSFORMER_BASED_WRAP` with `fsdp_transformer_layer_cls_to_wrap: Qwen3DecoderLayer`. This setting is **model-specific** — if you are training a different model architecture, you must update it to match your model's decoder layer class.

You can either:

1. **Override via CLI** (recommended for one-off runs):

   ```sh
   accelerate launch --config-file configs/accelerate/fsdp2.yaml \
     --fsdp_transformer_layer_cls_to_wrap LlamaDecoderLayer \
     train.py --config configs/train/qat_nvfp4.yaml ...
   ```

2. **Create a custom config** (recommended for repeated use):

   ```sh
   cp configs/accelerate/fsdp2.yaml configs/accelerate/fsdp2_llama.yaml
   # Edit fsdp2_llama.yaml: change Qwen3DecoderLayer -> LlamaDecoderLayer
   ```

Common layer class names:

| Model Family | `fsdp_transformer_layer_cls_to_wrap` |
|---|---|
| Qwen2, Qwen2.5, Qwen3 | `Qwen3DecoderLayer` (or `Qwen2DecoderLayer`) |
| Llama 2, 3, 3.1 | `LlamaDecoderLayer` |

</details>

<details>
<summary><b>Configuration</b></summary>

There are two types of configs:

- **Dataset configs** (`configs/dataset/`): Define the dataset blend — sources, `blend_size` (total samples), and `splits` (train/eval/test ratios). These are self-contained and determine what gets cached.
- **Training configs** (`configs/train/`): Define training hyperparameters plus runtime caps (`train_samples`, `eval_samples`) that subset the pre-built dataset without retriggering caching.

`quantize.py` only needs `--dataset_config` and `--recipe`. `train.py` uses a full training config via `--config`. All arguments can be specified via YAML, CLI flags, or both (CLI overrides YAML). See [ARGUMENTS.md](ARGUMENTS.md) for the full reference.

```sh
# YAML + CLI override
accelerate launch --config-file configs/accelerate/fsdp2.yaml train.py \
  --config configs/train/qat_nvfp4.yaml --learning_rate 5e-5
```

See [Dataset Configuration](configs/dataset/README.md) for custom dataset blends and adding new datasets.

</details>

<details>
<summary><b>Pre-Building the Dataset</b></summary>

You can pre-tokenize and cache the dataset before training using `dataset_utils.py`. This is useful for large blends or multi-node setups where you want to build the cache once and reuse it across experiments.

```sh
python dataset_utils.py \
  --dataset_config configs/dataset/blend.yaml \
  --model_name_or_path Qwen/Qwen3-8B
```

The cached dataset is stored under `.dataset_cache/tokenized/` by default (configurable via `--dataset_cache_dir`). The cache key depends on the dataset config (`blend_size`, `splits`, sources) and tokenizer — changing `train_samples` or `eval_samples` in the training config does **not** invalidate the cache.

</details>

## Results

\[Coming Soon\]

## Native Fake-Quantized Evaluation

ModelOpt quantized models can be saved and restored without exporting to a deployment platform. This is useful for fast evaluation with fake quantization using standard LLM benchmarks (MMLU, WikiText, etc.). See [HuggingFace checkpointing](https://nvidia.github.io/Model-Optimizer/guides/2_save_load.html#modelopt-save-restore-using-huggingface-checkpointing-apis) for details.

```sh
cd ../llm_eval

python lm_eval_hf.py --model hf \
    --tasks mmlu,wikitext \
    --model_args pretrained=../llm_qat/qwen3-8b-qat-nvfp4 \
    --batch_size 4
```

See [llm_eval/README.md](../llm_eval/README.md) for supported tasks.

## Pre-Quantized Checkpoints

- Ready-to-deploy checkpoints: [Hugging Face - NVIDIA Model Optimizer Collection](https://huggingface.co/collections/nvidia/inference-optimized-checkpoints-with-model-optimizer)
- Deployable on [TensorRT-LLM](https://github.com/NVIDIA/TensorRT-LLM), [vLLM](https://github.com/vllm-project/vllm) and [SGLang](https://github.com/sgl-project/sglang)

## Resources

- [Roadmap](https://github.com/NVIDIA/Model-Optimizer/issues/146)
- [Documentation](https://nvidia.github.io/Model-Optimizer)
- [Benchmarks](../benchmark.md)
- [Release Notes](https://nvidia.github.io/Model-Optimizer/reference/0_changelog.html)
- [File a bug](https://github.com/NVIDIA/Model-Optimizer/issues/new?template=1_bug_report.md)
- [Feature Request](https://github.com/NVIDIA/Model-Optimizer/issues/new?template=2_feature_request.md)
