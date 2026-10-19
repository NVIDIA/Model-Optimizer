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
For Megatron-Bridge or Megatron-LM models, use the NeMo container (e.g., `nvcr.io/nvidia/nemo:26.02`) which has all the dependencies installed.
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

### Set up the meta model

As Knowledge Distillation involves (at least) two models, ModelOpt simplifies the integration process by wrapping both student and teacher into one meta model.

Please see an example Distillation setup below. This example assumes the outputs of `teacher_model` and `student_model` are logits.

```python
import modelopt.torch.distill as mtd

distillation_config = {
    "teacher_model": teacher_model,
    "criterion": mtd.LogitsDistillationLoss(),  # callable receiving student and teacher outputs, in order
    "loss_balancer": mtd.StaticLossBalancer(),  # combines multiple losses; omit if only one distillation loss used
}

distillation_model = mtd.convert(student_model, mode=[("kd_loss", distillation_config)])
```

The `teacher_model` can be either a `nn.Module`, a callable which returns an `nn.Module`, or a tuple of `(model_cls, args, kwargs)`. The `criterion` is the distillation loss used between student and teacher tensors. The `loss_balancer` determines how the original and distillation losses are combined (if needed).

See [Distillation](https://nvidia.github.io/Model-Optimizer/guides/4_distillation.html) for more info.

### Distill during training

To Distill from teacher to student, simply use the meta model in the usual training loop, while also using the meta model’s `.compute_kd_loss()` method to compute the distillation loss, in addition to the original user loss.

An example of Distillation training is given below:

```python
# Setup the data loaders. As example:
train_loader = get_train_loader()

# Define user loss function. As example:
loss_fn = get_user_loss_fn()

for input, labels in train_dataloader:
    distillation_model.zero_grad()
    # Forward through the wrapped models
    out = distillation_model(input)
    # Same loss as originally present
    loss = loss_fn(out, labels)
    # Combine distillation and user losses
    loss_total = distillation_model.compute_kd_loss(student_loss=loss)
    loss_total.backward()
```

> [!NOTE]
> DataParallel may break ModelOpt’s Distillation feature. Note that HuggingFace Trainer uses DataParallel by default.

### Export trained model

The model can easily be reverted to its original class for further use (i.e deployment) without any ModelOpt modifications attached.

```python
model = mtd.export(distillation_model)
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

Checkout the stand-alone distillation script in the [examples/megatron_bridge/](../megatron_bridge/README.md).

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

- 📅 [Roadmap](https://github.com/NVIDIA/Model-Optimizer/issues/146)
- 📖 [Documentation](https://nvidia.github.io/Model-Optimizer)
- 🎯 [Benchmarks](../benchmark.md)
- 💡 [Release Notes](https://nvidia.github.io/Model-Optimizer/reference/0_changelog.html)
- 🐛 [File a bug](https://github.com/NVIDIA/Model-Optimizer/issues/new?template=1_bug_report.md)
- ✨ [File a Feature Request](https://github.com/NVIDIA/Model-Optimizer/issues/new?template=2_feature_request.md)
