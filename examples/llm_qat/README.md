# Quantization Aware Training (QAT) and Distillation (QAD)

This tutorial shows how to run QAT and QAD with Hugging Face Transformers: set up the environment, quantize a model, train it, evaluate the checkpoint, and export it for deployment.

For background on QAT and QAD and help choosing between Hugging Face, Megatron Bridge, and Megatron-LM, start with the [QAT/QAD guide](https://nvidia.github.io/Model-Optimizer/guides/quantization_aware_training_and_distillation.html).

<div align="center">

| **Section** | **Description** | **Link** |
| :---: | :---: | :---: |
| Quick Start | Prerequisites and setup | \[[Link](#quick-start)\] |
| End-to-End Example | Run QAT/QAD in 3 steps: quantize, train, export | \[[Link](#run-end-to-end-qatqad-example)\] |
| Arguments | Full CLI/YAML argument reference | \[[Link](ARGUMENTS.md)\] |
| Support Matrix | Supported models, quantization formats, and backends | \[[Link](#support-matrix)\] |
| QLoRA | Model training with reduced GPU memory | \[[Link](#qlora-real-quantization)\] |
| Advanced Topics | Trainer APIs, FSDP2 config, YAML options | \[[Link](#advanced-topics)\] |
| Results | Accuracy benchmarks | \[[Link](#results)\] |
| Resources | Extra links and references | \[[Link](#resources)\] |

</div>

## Quick Start

### Prerequisites

Please refer to [hf_ptq/README.md](../hf_ptq/README.md#pre-requisites) for container
recommendations and base ModelOpt installation guidance. For this QAT/QAD example,
install the Hugging Face dependencies and the example-specific requirements:

<!-- modelopt-doc-test:begin
id = "llm-qat-install"
profile = "cpu"
timeout_seconds = 7200
manual = true
min_gpus = 0
-->
<!-- modelopt-doc-test:setup python
from _test_utils.doc_tests.fixtures.installation import prepare_installation, verify_installation
prepare_installation(ctx)
-->
<!-- modelopt-doc-test:run -->
```bash
pip install -U nvidia-modelopt[hf]
pip install --no-build-isolation -r examples/llm_qat/requirements.txt
```
<!-- modelopt-doc-test:verify python
verify_installation(ctx)
-->
<!-- modelopt-doc-test:end -->

`--no-build-isolation` lets FlashAttention build against the PyTorch installed by
the first command.

The Qwen3-8B example below requires a minimum of **2 x 80GB GPUs**.

> ModelOpt provides accelerated quantization kernels using Triton for NVFP4 QAT. See the [installation guide](https://nvidia.github.io/Model-Optimizer/getting_started/_installation_for_Linux.html#accelerated-quantization-with-triton-kernels).

## Run End-to-End QAT/QAD Example

All arguments can be set via YAML, CLI, or both (CLI overrides YAML). See
[ARGUMENTS.md](ARGUMENTS.md), `--help`, and [Configuration](#advanced-configuration).

### Quantization Recipes

Recipes are declarative YAML files that specify the quantization configuration. Built-in recipes are available in [`modelopt_recipes/`](../../modelopt_recipes/):

<!-- modelopt-doc-test:begin
id = "llm-qat-recipes"
profile = "cpu"
timeout_seconds = 30
-->
<!-- modelopt-doc-test:run -->

```sh
# From the Model-Optimizer repository root, list available built-in recipes
ls modelopt_recipes/general/ptq/
```

<!-- modelopt-doc-test:end -->

See [custom calibration](https://nvidia.github.io/Model-Optimizer/guides/_pytorch_quantization.html#advanced-configuration-creation) for creating your own recipe.

### QAT

Quantize, fine-tune on labeled data, and export. Start from the repository root:

<!-- modelopt-doc-test:begin
id = "llm-qat-quickstart"
profile = "gpu"
timeout_seconds = 900
-->
<!-- modelopt-doc-test:setup python
from _test_utils.doc_tests.fixtures.llm_qat import prepare_llm_qat_workspace, verify_llm_qat_workspace
workspace = prepare_llm_qat_workspace(ctx.repo, ctx.tmp)
ctx.cwd = workspace
ctx.env.update(HF_HUB_OFFLINE="1", HF_DATASETS_OFFLINE="1", TOKENIZERS_PARALLELISM="false")
-->
<!-- modelopt-doc-test:run -->

```sh
cd examples/llm_qat

# 1. Quantize
python quantize.py \
  --model_name_or_path Qwen/Qwen3-8B \
  --dataset_config configs/dataset/blend.yaml \
  --recipe general/ptq/nvfp4_default-kv_fp8 \
  --output_dir qwen3-8b-quantized

# 2. Train
accelerate launch --config-file configs/accelerate/fsdp2.yaml train.py \
  --config configs/train/qat_nvfp4.yaml \
  --model_name_or_path qwen3-8b-quantized \
  --output_dir qwen3-8b-qat-nvfp4

# 3. Export
python export.py --pyt_ckpt_path qwen3-8b-qat-nvfp4 --export_path qwen3-8b-qat-deploy
```

<!-- modelopt-doc-test:verify python
verify_llm_qat_workspace(workspace)
-->
<!-- modelopt-doc-test:end -->

### QAD

Quantize, recover accuracy using the original model as teacher, and export:

<!-- modelopt-doc-test:begin
id = "llm-qad-quickstart"
profile = "gpu"
timeout_seconds = 900
-->
<!-- modelopt-doc-test:setup python
from _test_utils.doc_tests.fixtures.llm_qat import prepare_llm_qat_workspace, verify_llm_qat_workspace
workspace = prepare_llm_qat_workspace(ctx.repo, ctx.tmp)
ctx.cwd = workspace / "examples/llm_qat"
ctx.env.update(HF_HUB_OFFLINE="1", HF_DATASETS_OFFLINE="1", TOKENIZERS_PARALLELISM="false")
-->
<!-- modelopt-doc-test:run -->
```sh
# 1. Quantize
python quantize.py \
  --model_name_or_path Qwen/Qwen3-8B \
  --dataset_config configs/dataset/blend.yaml \
  --recipe general/ptq/nvfp4_default-kv_fp8 \
  --output_dir qwen3-8b-quantized

# 2. Train with distillation
accelerate launch --config-file configs/accelerate/fsdp2.yaml train.py \
  --config configs/train/qad_nvfp4.yaml \
  --model_name_or_path qwen3-8b-quantized \
  --teacher_model Qwen/Qwen3-8B \
  --output_dir qwen3-8b-qad-nvfp4

# 3. Export
python export.py --pyt_ckpt_path qwen3-8b-qad-nvfp4 --export_path qwen3-8b-qad-deploy
```

<!-- modelopt-doc-test:verify python
verify_llm_qat_workspace(workspace, variant="qad")
-->
<!-- modelopt-doc-test:end -->

Exported checkpoints can be deployed on [TensorRT-LLM](https://github.com/NVIDIA/TensorRT-LLM), [vLLM](https://github.com/vllm-project/vllm), or [SGLang](https://github.com/sgl-project/sglang). See [hf_ptq/README.md](../hf_ptq/README.md#deployment) for deployment instructions. For quick accuracy evaluation without exporting, see [Native Fake-Quantized Evaluation](#native-fake-quantized-evaluation).

> [!NOTE]
> For a minimal end-to-end demo (quantize + train + save in one script), see [simple_qat_train.py](simple_qat_train.py). It runs on a **single GPU** only and is intended as a quick introduction to the QAT flow (without transformer trainer)—not for distributed training.
>
> <!-- modelopt-doc-test:begin
> id = "llm-qat-simple"
> profile = "gpu"
> timeout_seconds = 900
> -->
> <!-- modelopt-doc-test:setup python
> from _test_utils.doc_tests.fixtures.llm_qat import prepare_llm_qat_workspace, verify_llm_qat_workspace
> workspace = prepare_llm_qat_workspace(ctx.repo, ctx.tmp, llama=True)
> ctx.cwd = workspace / "examples/llm_qat"
> ctx.env.update(HF_HUB_OFFLINE="1", HF_DATASETS_OFFLINE="1", TOKENIZERS_PARALLELISM="false")
> -->
> <!-- modelopt-doc-test:run -->
> ```sh
> python simple_qat_train.py --model-path meta-llama/Llama-3.2-3B --recipe general/ptq/nvfp4_default-kv_fp8
> ```
>
> <!-- modelopt-doc-test:verify python
> assert (ctx.cwd / "qat_model/modelopt_state.pth").is_file()
> assert (ctx.cwd / "qat_model/model.safetensors").stat().st_size > 0
> -->
> <!-- modelopt-doc-test:end -->
>
> For multi-GPU training (FSDP2, DDP, DeepSpeed), use [train.py](train.py) with `accelerate launch` as shown in the [commands](#qat) above.

> [!TIP]
> For more performant QAD, please refer to [examples/megatron_bridge/README.md](../megatron_bridge/README.md) for example scripts for PTQ / QAD with Megatron-Bridge which is generally more performant than the Hugging Face scripts.

## Support Matrix

### Supported Models

| Model | Chat Template | Support |
|-------|---------------|---------|
| Qwen2, 2.5, 3, 3.5 dense models; Nemotron ChatML models | ChatML | Yes (chat + assistant-only labels + pretrain) |
| Models with `{% generation %}` chat templates | Model-specific | Yes (chat + assistant-only labels + pretrain) |
| Other models with HuggingFace chat templates, including Llama 2, 3, 3.1 | Model-specific | Yes (chat full-label + pretrain) |

> **Note:** `apply_chat_template` controls chat formatting. `train_only_assistant_tokens` controls label masking: `auto` uses assistant-only labels when native `{% generation %}` masks or the tested Qwen/Nemotron ChatML heuristic is available, then falls back to all non-padding chat-template tokens; set `train_only_assistant_tokens: true` to require native or ChatML assistant-only labels, or `false` to always train on all chat-template tokens.

### Supported Quantization Formats

Built-in recipes support full-model, partial-layer, and mixed-precision quantization. Common entry points:

| Format | Precision | Example Recipe | Use Case |
|--------|-----------|----------------|----------|
| **NVFP4** | W4A4 + FP8 KV | `general/ptq/nvfp4_default-kv_fp8` | FP4 compute and compression on Blackwell GPUs |
| **FP8** | W8A8 + FP8 KV | `general/ptq/fp8_default-kv_fp8` | Near-BF16 accuracy on Hopper or later GPUs |
| **INT4** weight-only | W4A16 | `general/ptq/int4_blockwise_weight_only` | Deployable on all Ampere or later GPUs |
| **Partial / mixed** | Pattern-specific | `general/ptq/nvfp4_mlp_only-kv_fp8` | Quantize selected layers or combine precisions |

> Recipes can target different layers or GEMMs with different precisions, such as NVFP4
> for MLP/MoE GEMMs and FP8 for attention GEMMs or KV cache. See
> [`modelopt_recipes/general/ptq/`](../../modelopt_recipes/general/ptq/) and
> [`modelopt_recipes/configs/ptq/`](../../modelopt_recipes/configs/ptq/) for built-in
> options and reusable recipe units.

### Supported Backends

| Backend | Config File | Notes |
|---------|------------|-------|
| FSDP2 | `configs/accelerate/fsdp2.yaml` | **Recommended** |
| DDP | `configs/accelerate/ddp.yaml` | Add `--gradient_checkpointing True` |
| DeepSpeed | `configs/accelerate/deepspeed.yaml` | Add `--gradient_checkpointing True` |

Replace `--config-file configs/accelerate/fsdp2.yaml` with the desired backend config in any of the commands above.

## QLoRA (Real Quantization)

[QLoRA](https://arxiv.org/pdf/2305.14314) reduces training memory by quantizing LoRA backbone weights with real quantization via `mtq.compress()`.

<!-- modelopt-doc-test:begin
id = "llm-qlora-quickstart"
profile = "gpu"
timeout_seconds = 900
-->
<!-- modelopt-doc-test:setup python
from _test_utils.doc_tests.fixtures.llm_qat import prepare_llm_qat_workspace, verify_llm_qat_workspace
workspace = prepare_llm_qat_workspace(ctx.repo, ctx.tmp)
ctx.cwd = workspace / "examples/llm_qat"
ctx.env.update(HF_HUB_OFFLINE="1", HF_DATASETS_OFFLINE="1", TOKENIZERS_PARALLELISM="false")
-->
<!-- modelopt-doc-test:run -->
```sh
# 1. Quantize with compression
python quantize.py \
  --model_name_or_path Qwen/Qwen3-8B \
  --dataset_config configs/dataset/blend.yaml \
  --recipe general/ptq/nvfp4_default-kv_fp8 \
  --compress True \
  --output_dir qwen3-8b-quantized

# 2. Train with QLoRA
accelerate launch --config-file configs/accelerate/ddp.yaml train.py \
  --config configs/train/qlora_nvfp4.yaml \
  --model_name_or_path qwen3-8b-quantized \
  --output_dir qwen3-8b-fp4-qlora

# 3. Export
python export.py \
  --pyt_ckpt_path qwen3-8b-fp4-qlora \
  --export_path qwen3-8b-fp4-qlora-hf

```

<!-- modelopt-doc-test:verify python
import json
trained = ctx.cwd / "qwen3-8b-fp4-qlora"
exported = ctx.cwd / "qwen3-8b-fp4-qlora-hf"
import math
state = json.loads((trained / "trainer_state.json").read_text())
assert state["global_step"] == 2
losses = [entry["train_loss"] for entry in state["log_history"] if "train_loss" in entry]
assert losses and all(math.isfinite(value) for value in losses)
assert (exported / "adapter_config.json").is_file()
assert (exported / "adapter_model.safetensors").stat().st_size > 0
assert (exported / "base_model/config.json").is_file()
-->
<!-- modelopt-doc-test:end -->

Serve the exported adapter with vLLM:

<!-- modelopt-doc-test:begin
id = "llm-qlora-serve"
profile = "gpu"
timeout_seconds = 900
manual = true
requires = ["vllm"]
-->
<!-- modelopt-doc-test:setup python
from _test_utils.doc_tests.fixtures.serving import prepare_adapter_server, verify_adapter_server
prepare_adapter_server(ctx)
-->
<!-- modelopt-doc-test:run background -->
```sh
# 4. Serve with vLLM
vllm serve qwen3-8b-fp4-qlora-hf/base_model --enable-lora \
  --lora-modules adapter=qwen3-8b-fp4-qlora-hf --port "${PORT:-8000}" \
  --tokenizer qwen3-8b-fp4-qlora-hf
```

<!-- modelopt-doc-test:verify python
verify_adapter_server(ctx)
-->
<!-- modelopt-doc-test:end -->

> QLoRA export is not currently supported with FSDP2.

## Advanced Topics

### Quantize and Fine-Tune with Python

<!-- modelopt-doc-test:begin
id = "llm-qat-python"
profile = "gpu"
timeout_seconds = 900
-->
<!-- modelopt-doc-test:setup python
from _test_utils.doc_tests.fixtures.llm_qat import prepare_python_api, verify_python_api
globals().update(prepare_python_api(ctx, quantized=False))
-->
<!-- modelopt-doc-test:run -->
```python
import modelopt.torch.quantization as mtq
from modelopt.recipe import load_recipe

# 1. Load a quantization recipe
recipe = load_recipe("general/ptq/nvfp4_default-kv_fp8")

# 2. Quantize the model in-place
model = mtq.quantize(model, recipe.quantize, forward_loop)

# 3. Fine-tune the quantized model
trainer.train()
trainer.save_model()
```

<!-- modelopt-doc-test:verify python
verify_python_api(trainer, initial_weights)
-->
<!-- modelopt-doc-test:end -->

### Using `QATTrainer` and `QADTrainer`

`QATTrainer` is a drop-in replacement for HuggingFace's `Trainer` that handles quantization-aware training seamlessly with various distributed backends (FSDP2, DeepSpeed, DDP):

<!-- modelopt-doc-test:begin
id = "llm-qat-trainer"
profile = "gpu"
timeout_seconds = 900
-->
<!-- modelopt-doc-test:setup python
from _test_utils.doc_tests.fixtures.llm_qat import prepare_python_api, verify_python_api
globals().update(prepare_python_api(ctx, quantized=True))
-->
<!-- modelopt-doc-test:run -->
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

<!-- modelopt-doc-test:verify python
verify_python_api(trainer, initial_weights)
-->
<!-- modelopt-doc-test:end -->

`QADTrainer` extends `QATTrainer` with distillation. Load the teacher first and pass a `DistillArgsWithTeacherModel` instance:

<!-- modelopt-doc-test:begin
id = "llm-qad-trainer"
profile = "gpu"
timeout_seconds = 900
-->
<!-- modelopt-doc-test:setup python
from _test_utils.doc_tests.fixtures.llm_qat import prepare_python_api, verify_python_api
globals().update(prepare_python_api(ctx, quantized=True))
-->
<!-- modelopt-doc-test:run -->
```python
from transformers import AutoModelForCausalLM

from modelopt.torch.distill.plugins.huggingface import DistillArgsWithTeacherModel
from modelopt.torch.quantization.plugins.transformers_trainer import QADTrainer

teacher = AutoModelForCausalLM.from_pretrained("Qwen/Qwen3-8B", dtype=model.dtype)
distill_args = DistillArgsWithTeacherModel(
    distill=True,
    teacher_model=teacher,
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

<!-- modelopt-doc-test:verify python
verify_python_api(trainer, initial_weights)
-->
<!-- modelopt-doc-test:end -->

<details>
<summary><b>FSDP2 and Model-Specific Layer Wrapping</b></summary>

The default `fsdp2.yaml` uses `TRANSFORMER_BASED_WRAP` with `fsdp_transformer_layer_cls_to_wrap: Qwen3DecoderLayer`. This setting is **model-specific** — if you are training a different model architecture, you must update it to match your model's decoder layer class.

You can either:

1. **Override via CLI** (recommended for one-off runs):

   <!-- modelopt-doc-test:begin
   id = "llm-qat-llama-wrap"
   profile = "gpu"
   timeout_seconds = 900
   -->
   <!-- modelopt-doc-test:setup python
   from _test_utils.doc_tests.fixtures.llm_qat import prepare_quantized_input
   prepare_quantized_input(ctx, llama=True)
   (ctx.cwd / "llama-quantized").symlink_to(ctx.cwd / "qwen3-8b-quantized", target_is_directory=True)
   -->
   <!-- modelopt-doc-test:run -->
   ```sh
   accelerate launch --config-file configs/accelerate/fsdp2.yaml \
     --fsdp_transformer_layer_cls_to_wrap LlamaDecoderLayer \
     train.py --config configs/train/qat_nvfp4.yaml \
     --model_name_or_path llama-quantized --output_dir llama-qat
   ```

   <!-- modelopt-doc-test:verify python
   import json
   assert json.loads((ctx.cwd / "llama-qat/trainer_state.json").read_text())["global_step"] == 2
   -->
   <!-- modelopt-doc-test:end -->

2. **Create a custom config** (recommended for repeated use):

   <!-- modelopt-doc-test:begin
   id = "llm-qat-copy-config"
   profile = "cpu"
   timeout_seconds = 900
   -->
   <!-- modelopt-doc-test:setup python
   import shutil
   ctx.cwd = ctx.tmp
   shutil.copytree(ctx.repo / "examples/llm_qat/configs", ctx.tmp / "configs")
   -->
   <!-- modelopt-doc-test:run -->
   ```sh
   cp configs/accelerate/fsdp2.yaml configs/accelerate/fsdp2_llama.yaml
   # Edit fsdp2_llama.yaml: change Qwen3DecoderLayer -> LlamaDecoderLayer
   ```

   <!-- modelopt-doc-test:verify python
   assert (ctx.cwd / "configs/accelerate/fsdp2_llama.yaml").read_bytes() == (ctx.cwd / "configs/accelerate/fsdp2.yaml").read_bytes()
   -->
   <!-- modelopt-doc-test:end -->

Common layer class names:

| Model Family | `fsdp_transformer_layer_cls_to_wrap` |
|---|---|
| Qwen2, Qwen2.5, Qwen3 | `Qwen3DecoderLayer` (or `Qwen2DecoderLayer`) |
| Llama 2, 3, 3.1 | `LlamaDecoderLayer` |

</details>

<details id="advanced-configuration">
<summary><b>Configuration</b></summary>

There are two types of configs:

- **Dataset configs** (`configs/dataset/`): Define the dataset blend — sources, `blend_size` (total samples), and `splits` (train/eval/test ratios). These are self-contained and determine what gets cached.
- **Training configs** (`configs/train/`): Define training hyperparameters plus runtime caps (`train_samples`, `eval_samples`) that subset the pre-built dataset without retriggering caching.

`quantize.py` only needs `--dataset_config` and `--recipe`. `train.py` uses a full training config via `--config`. All arguments can be specified via YAML, CLI flags, or both (CLI overrides YAML). See [ARGUMENTS.md](ARGUMENTS.md) for the full reference, regenerated with `python_pwd examples/llm_qat/arguments.py --generate_docs examples/llm_qat/ARGUMENTS.md`.

<!-- modelopt-doc-test:begin
id = "llm-qat-cli-override"
profile = "gpu"
timeout_seconds = 900
-->
<!-- modelopt-doc-test:setup python
from _test_utils.doc_tests.fixtures.llm_qat import prepare_quantized_input
prepare_quantized_input(ctx)
-->
<!-- modelopt-doc-test:run -->
```sh
# YAML + CLI override
accelerate launch --config-file configs/accelerate/fsdp2.yaml train.py \
  --config configs/train/qat_nvfp4.yaml --learning_rate 5e-5 \
  --model_name_or_path qwen3-8b-quantized --output_dir qwen3-8b-qat-override
```

<!-- modelopt-doc-test:verify python
import json
state = json.loads((ctx.cwd / "qwen3-8b-qat-override/trainer_state.json").read_text())
assert state["global_step"] == 2
assert any(entry.get("learning_rate") == 5e-5 for entry in state["log_history"])
-->
<!-- modelopt-doc-test:end -->

See [Dataset Configuration](configs/dataset/README.md) for custom dataset blends and adding new datasets.

</details>

<details>
<summary><b>Pre-Building the Dataset</b></summary>

You can pre-tokenize and cache the dataset before training using `dataset_utils.py`. This is useful for large blends or multi-node setups where you want to build the cache once and reuse it across experiments.

<!-- modelopt-doc-test:begin
id = "llm-qat-dataset"
profile = "cpu"
timeout_seconds = 900
-->
<!-- modelopt-doc-test:setup python
from _test_utils.doc_tests.fixtures.llm_qat import prepare_llm_qat_workspace, verify_llm_qat_workspace
workspace = prepare_llm_qat_workspace(ctx.repo, ctx.tmp)
ctx.cwd = workspace / "examples/llm_qat"
ctx.env.update(HF_HUB_OFFLINE="1", HF_DATASETS_OFFLINE="1", TOKENIZERS_PARALLELISM="false")
-->
<!-- modelopt-doc-test:run -->
```sh
python dataset_utils.py \
  --dataset_config configs/dataset/blend.yaml \
  --model_name_or_path Qwen/Qwen3-8B
```

<!-- modelopt-doc-test:verify python
from datasets import load_from_disk
caches = list((ctx.cwd / ".dataset_cache/tokenized").glob("*/dataset_dict.json"))
assert len(caches) == 1
cached = load_from_disk(str(caches[0].parent))
assert len(cached["train"]) == 16 and len(cached["eval"]) == 4
assert {"input_ids", "attention_mask", "labels"} <= set(cached["train"].column_names)
-->
<!-- modelopt-doc-test:end -->

The cached dataset is stored under `.dataset_cache/tokenized/` by default (configurable via `--dataset_cache_dir`). The cache key depends on the dataset config (`blend_size`, `splits`, sources) and tokenizer — changing `train_samples` or `eval_samples` in the training config does **not** invalidate the cache.

</details>

## Results

\[Coming Soon\]

## Native Fake-Quantized Evaluation

ModelOpt quantized models can be saved and restored without exporting to a deployment platform. This is useful for fast evaluation with fake quantization using standard LLM benchmarks (MMLU, WikiText, etc.). See [HuggingFace checkpointing](https://nvidia.github.io/Model-Optimizer/guides/2_save_load.html#modelopt-save-restore-using-huggingface-checkpointing-apis) for details.

<!-- modelopt-doc-test:begin
id = "llm-qat-evaluation"
profile = "gpu"
timeout_seconds = 7200
manual = true
requires = ["lm_eval"]
-->
<!-- modelopt-doc-test:setup python
import os
from pathlib import Path
checkpoint = Path(os.environ["MODELOPT_DOC_EVAL_CHECKPOINT"]).resolve()
assert (checkpoint / "modelopt_state.pth").is_file()
ctx.cwd = ctx.tmp / "examples/llm_qat"
ctx.cwd.mkdir(parents=True)
(ctx.cwd / "qwen3-8b-qat-nvfp4").symlink_to(checkpoint, target_is_directory=True)
import shutil
shutil.copytree(ctx.repo / "examples/llm_eval", ctx.cwd.parent / "llm_eval")
ctx.env["EVAL_OUTPUT"] = str(ctx.tmp / "eval-results")
-->
<!-- modelopt-doc-test:run -->
```sh
cd ../llm_eval

python lm_eval_hf.py --model hf \
    --tasks mmlu,wikitext \
    --model_args pretrained=../llm_qat/qwen3-8b-qat-nvfp4 \
    --batch_size 4 --output_path "${EVAL_OUTPUT:-eval-results}"
```

<!-- modelopt-doc-test:verify python
import json
files = list((ctx.tmp / "eval-results").rglob("results*.json"))
assert files and json.loads(files[0].read_text())["results"]
-->
<!-- modelopt-doc-test:end -->

See [llm_eval/README.md](../llm_eval/README.md) for supported tasks.

## Pre-Quantized Checkpoints

- Ready-to-deploy checkpoints: [Hugging Face - NVIDIA Model Optimizer Collection](https://huggingface.co/collections/nvidia/inference-optimized-checkpoints-with-model-optimizer)
- Deployable on [TensorRT-LLM](https://github.com/NVIDIA/TensorRT-LLM), [vLLM](https://github.com/vllm-project/vllm) and [SGLang](https://github.com/sgl-project/sglang)

## Resources

- [Roadmap](https://github.com/NVIDIA/Model-Optimizer/issues/1699)
- [Documentation](https://nvidia.github.io/Model-Optimizer)
- [Benchmarks](../benchmark.md)
- [Release Notes](https://nvidia.github.io/Model-Optimizer/reference/0_changelog.html)
- [File a bug](https://github.com/NVIDIA/Model-Optimizer/issues/new?template=1_bug_report.md)
- [Feature Request](https://github.com/NVIDIA/Model-Optimizer/issues/new?template=2_feature_request.md)
