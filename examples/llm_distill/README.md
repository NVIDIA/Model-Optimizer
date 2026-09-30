# Knowledge Distillation

Knowledge Distillation is a machine learning technique where a compact "student" model learns to replicate the behavior of a larger, more complex "teacher" model to achieve comparable performance with improved efficiency.

Model Optimizer's Distillation is a set of wrappers and utilities to easily perform Knowledge Distillation among teacher and student models. Given a pretrained teacher model, Distillation has the potential to train a smaller student model faster and/or with higher accuracy than the student model could achieve on its own.

This section focuses on demonstrating how to apply Model Optimizer to perform knowledge distillation with ease.

<div align="center">

| **Section** | **Description** | **Link** | **Docs** |
| :------------: | :------------: | :------------: | :------------: |
| Pre-Requisites | Required & optional packages to use this technique | \[[Link](#pre-requisites)\] | |
| Getting Started | Learn how to optimize your models using distillation to produce more intellegant smaller models | \[[Link](#getting-started)\] | \[[docs](https://nvidia.github.io/Model-Optimizer/guides/4_distillation.html)\] |
| Support Matrix | View the support matrix to see compatibility and feature availability across different models | \[[Link](#support-matrix)\] | |
| Distillation with Megatron-Bridge | Learn how to distill your models with Megatron-Bridge Framework | \[[Link](#knowledge-distillation-kd-in-nvidia-megatron-bridge-framework)\] | \[[docs](https://nvidia.github.io/Model-Optimizer/guides/4_distillation.html)\] |
| Distillation with Megatron-LM | Learn how to distill your models with Megatron-LM Framework | \[[Link](#knowledge-distillation-kd-in-nvidia-megatron-lm-framework)\] | |
| Distillation with Huggingface | Learn how to distill your models with Hugging Face | \[[Link](#knowledge-distillation-kd-for-huggingface-models)\] | \[[docs](https://nvidia.github.io/Model-Optimizer/guides/4_distillation.html)\] |
| Resources | Extra links to relevant resources | \[[Link](#resources)\] | |

</div>

## Pre-Requisites

### Docker

For Hugging Face models, please use the PyTorch docker image (e.g., `nvcr.io/nvidia/pytorch:26.01-py3`).
Visit our [installation docs](https://nvidia.github.io/Model-Optimizer/getting_started/2_installation.html) for more information.

Also follow the installation steps below to upgrade to the latest version of Model Optimizer and install example-specific dependencies.

### Local Installation

For Hugging Face models, install Model Optimizer with `hf` dependencies using `pip` from [PyPI](https://pypi.org/project/nvidia-modelopt/) and install the requirements for the example:

```bash
pip install -U nvidia-modelopt[hf]
pip install -r requirements.txt
```

## Getting Started

### Set up your base models

First obtain both a pretrained model to act as the teacher and a (usually smaller) model to serve as the student.

```python
from transformers import AutoModelForCausalLM

# Define student & teacher
student_model = AutoModelForCausalLM.from_pretrained("student-model-id-or-path")
teacher_model = AutoModelForCausalLM.from_pretrained("teacher-model-id-or-path")
```

### Set up the KDTrainer

For HuggingFace models, ModelOpt provides `KDTrainer`, a drop-in replacement for HuggingFace's `Trainer` that
handles the teacher forward pass and KD loss computation internally. Unlike the general-purpose Distillation API,
`KDTrainer` does **not** call `mtd.convert()` and does not wrap the student in a `DistillationModel` — the student
stays a plain HuggingFace model, and the teacher is kept on the trainer and forwarded explicitly during loss
computation.

```python
from modelopt.torch.distill.plugins.huggingface import KDTrainer

trainer = KDTrainer(
    student_model,
    training_args,
    distill_args={"teacher_model": teacher_model},  # criterion defaults to "logits_loss"
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
)
```

`KDTrainer` can be mixed in with other HuggingFace trainers (e.g. `SFTTrainer`) via normal Python multiple
inheritance, as done in [`main.py`](main.py):

```python
class KDSFTTrainer(KDTrainer, SFTTrainer):
    pass
```

> [!NOTE]
> `KDTrainer` currently only supports logit-level (output) distillation. Hidden-state / intermediate-layer
> distillation is not yet supported by `KDTrainer`. Until that support lands, use `mtd.convert()` and
> `DistillationModel` directly (see [Distillation](https://nvidia.github.io/Model-Optimizer/guides/4_distillation.html))
> for hidden-state KD.

### Distill during training

Since `KDTrainer` overrides `compute_loss()` to run the teacher forward pass and compute the KD loss, training is
just the normal HuggingFace `Trainer` loop — no manual loss computation is required:

```python
trainer.train()
```

> [!NOTE]
> `compute_loss()` returns the KD loss on its own; it does not combine it with the original student
> cross-entropy loss. Weighted combination of CE and KD losses is not yet supported by `KDTrainer`, though it
> is a planned feature. During evaluation, the CE loss is still computed and reported separately as the
> `eval_ce_loss` metric.

> [!NOTE]
> `KDTrainer` requires FSDP2 when FSDP is enabled; FSDP1 is not supported. Note that HuggingFace Trainer uses
> DataParallel by default, which may break distributed teacher/student forwarding — use FSDP2, DeepSpeed, or DDP
> instead (see [`accelerate_config/fsdp2.yaml`](accelerate_config/fsdp2.yaml)).

### Export trained model

Because the student is never wrapped in a `DistillationModel`, no `mtd.export()` step is needed — `trainer.save_model()`
saves the student directly in its original HuggingFace format.

```python
trainer.save_model(training_args.output_dir)
```

## Support Matrix

### Current out of the box components

Loss criterion:

- `mtd.LogitsDistillationLoss()` - Standard KL-Divergence on output logits
- `mtd.MGDLoss()` - Masked Generative Distillation loss for 2D convolutional outputs
- `mtd.MFTLoss()` - KL-divergence loss with Minifinetuning threshold modification

Loss balancers:

- `mtd.StaticLossBalancer()` - Combines original student loss and KD loss into a single weighted sum (without changing over time)

### Supported Models

> [!NOTE]
> The following are models that were confirmed to run with ModelOpt distillation, but it is absolutely not limited to these

| Model | type | confirmed compatible |
| :---: | :---: | :---: |
| Nemotron | mamba hybrid | ✅ |
| Llama 3 | llama | ✅ |
| Llama 4 | llama | ✅ |
| Gemma 2 | gemma | ✅ |
| Gemma 3 | gemma | ✅ |
| Phi 3 | phi | ✅ |
| Qwen 2 | qwen2 | ✅ |
| Qwen 3 | qwen3 | ✅ |
| Mamba | mamba | ✅ |

## Knowledge Distillation (KD) in NVIDIA Megatron-Bridge Framework

Checkout the stand-alone distillation script in the [examples/megatron_bridge/](../megatron_bridge/README.md) for example scripts for KD with Megatron-Bridge which is generally more performant than the Hugging Face scripts.

## Knowledge Distillation (KD) in NVIDIA Megatron-LM Framework

Checkout the Knowledge Distillation example in the [Megatron-LM repository](https://github.com/NVIDIA/Megatron-LM/tree/main/examples/post_training/modelopt).

## Knowledge Distillation (KD) for HuggingFace Models

In this e2e example we finetune Llama-3.2 models on the [smol-smoltalk-Interaction-SFT](https://huggingface.co/datasets/ReactiveAI/smol-smoltalk-Interaction-SFT)
dataset as a minimal example to demonstrate a simple way of integrating Model Optimizer's KD feature.

We replace normal supervised finetuning (SFT) of a Llama-3.2-1B base model by distilling information from Llama-3.2-3B-Instruct which has already been instruction-finetuned.

> [!NOTE]
> We can fit the following in memory using [FSDP](https://huggingface.co/docs/accelerate/en/usage_guides/fsdp) enabled on 8x RTX 6000 (total ~400GB VRAM)

```bash
accelerate launch --config-file ./accelerate_config/fsdp2.yaml \
    main.py \
    --teacher_name_or_path 'meta-llama/Llama-3.2-3B-Instruct' \
    --student_name_or_path 'meta-llama/Llama-3.2-1B' \
    --output_dir ./llama3.2-distill \
    --max_length 2048 \
    --per_device_train_batch_size 4 \
    --per_device_eval_batch_size 8 \
    --max_steps 200 \
    --logging_steps 5
```

## Resources

- 📅 [Roadmap](https://github.com/NVIDIA/Model-Optimizer/issues/1699)
- 📖 [Documentation](https://nvidia.github.io/Model-Optimizer)
- 🎯 [Benchmarks](../benchmark.md)
- 💡 [Release Notes](https://nvidia.github.io/Model-Optimizer/reference/0_changelog.html)
- 🐛 [File a bug](https://github.com/NVIDIA/Model-Optimizer/issues/new?template=1_bug_report.md)
- ✨ [File a Feature Request](https://github.com/NVIDIA/Model-Optimizer/issues/new?template=2_feature_request.md)
