.. _quantization-aware-training:

===============================================
Quantization-Aware Training and Distillation
===============================================

Quantization-aware training (QAT) and quantization-aware distillation (QAD) recover
quality lost when a model is quantized. Both train with simulated quantization
enabled, so the resulting checkpoint retains its ModelOpt quantization state for
deployment.

**Quantization Aware Training (QAT)** inserts simulated quantization operations
into the model graph and then fine-tunes the model so its weights learn to
compensate for quantization error. During training, quantization scales are frozen
while weights are updated. QAT is a general technique — it learns from labeled
data on a quantized model using the usual cross-entropy (CE) loss.

**Quantization Aware Distillation (QAD)** is a special case of QAT that uses a
frozen BF16 teacher (typically the original unquantized model) to guide the
quantized student via a logit-level KL-divergence loss. QAD is a **pure accuracy
recovery technique** — its goal is to recover accuracy lost from quantization, not
to teach the model a new task. It requires the teacher during training and
therefore uses more memory and compute than QAT.

In both cases, start from a PTQ checkpoint and retain its quantization configuration
during training.

When to Use QAT vs QAD
======================

.. list-table::
   :header-rows: 1
   :stub-columns: 1
   :widths: 20 40 40

   * -
     - QAT (without distillation)
     - QAD (with distillation)
   * - What it does
     - Fine-tunes a quantized model on labeled data
     - Recovers quantization accuracy using the original model as teacher
   * - When to use
     - The model is already quantized and you want to fine-tune it for a
       **new task** (e.g., fine-tuning a `GPT-OSS
       <https://github.com/NVIDIA/Model-Optimizer/tree/main/examples/gpt-oss>`_
       quantized checkpoint)
     - You want the **best possible accuracy recovery** after quantization
   * - Recommended workflow
     - Start from a quantized checkpoint, fine-tune with task-specific data
     - Full-precision fine-tuning first, then QAD to recover quantization loss

QAD rationale
=============

**QAD is Model Optimizer's recommended strategy for accuracy recovery after
quantization.**

In our experiments, full-precision fine-tuning followed by QAD
delivers the best accuracy, especially at aggressive quantization levels (e.g.,
NVFP4).

`Quantization-Aware Distillation for NVFP4 Inference Accuracy Recovery
<https://arxiv.org/abs/2601.20088>`_ recommends QAD for recovery after aggressive
quantization, especially for models that have passed through multi-stage
post-training such as SFT, RL, or model merging. The teacher signal makes recovery
more robust when training-data quality or coverage is limited.

The optimal balance between QAT and QAD for a given model and task is an
active area of research.

To learn more, read the `QAT/QAD blog post
<https://developer.nvidia.com/blog/how-quantization-aware-training-enables-low-precision-accuracy-recovery/>`_.

The
`Nemotron 3.5 Lightning QAD blog
<https://developer.nvidia.com/blog/developing-nemotron-3-5-lightning-nvfp4-with-qad-using-nvidia-model-optimizer/>`_
discusses the PTQ-to-QAD-to-export workflow and its scale-handling considerations.

Choose a framework
==================

ModelOpt supports QAT with Hugging Face, Megatron-Bridge, and Megatron-LM.

.. list-table::
   :header-rows: 1
   :widths: 20 40 40

   * - Framework
     - Advantages
     - Trade-offs
   * - Hugging Face
     - Starts directly from Hugging Face checkpoints and is the simplest path for
       small to medium models. It supports FSDP2, DDP, and DeepSpeed through
       Accelerate, with no conversion to Megatron-Core.
     - Its parallelism is less efficient for large-scale training, so it is better
       suited to smaller models than the Megatron-based options.
   * - Megatron-Bridge
     - Automatically converts Hugging Face models to Megatron-Core and uses
       Megatron-LM's distributed training stack. It is a convenient scalable
       workflow without a separate conversion step.
     - The high-level workflow exposes fewer customization points than working
       directly in Megatron-LM.
   * - Megatron-LM
     - Provides the most control over model configuration, data, parallelism, and
       training behavior, making it the most customizable option for large-model
       training.
     - Requires manually converting the model to Megatron-Core and managing that
       checkpoint workflow.

Implementation guides
=====================

Use the framework README as the executable source of truth. Each guide owns its
prerequisites, commands, data preparation, distributed topology, and export options.

* `Hugging Face QAT/QAD Quick Start
  <https://github.com/NVIDIA/Model-Optimizer/tree/main/examples/llm_qat#quick-start>`_
* `Megatron-Bridge README
  <https://github.com/NVIDIA/Model-Optimizer/tree/main/examples/megatron_bridge>`_
* `Megatron-LM ModelOpt post-training documentation
  <https://github.com/NVIDIA/Megatron-LM/tree/main/examples/post_training/modelopt>`_

QAD launcher examples
===================================

Model Optimizer includes a `launcher
<https://github.com/NVIDIA/Model-Optimizer/tree/main/tools/launcher>`_ for
running supported QAD pipelines as one-click commands. Follow the launcher's
`Quick Start <https://github.com/NVIDIA/Model-Optimizer/tree/main/tools/launcher#quick-start>`_
to set it up.

The `NVIDIA Nemotron 3.5 Lightning QAD launcher examples
<https://github.com/NVIDIA/Model-Optimizer/tree/main/tools/launcher/examples/nvidia/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-BF16>`_
provide Slurm pipelines for Megatron-Bridge and Megatron-LM. Both create an NVFP4
student with PTQ, distill it from the BF16 teacher, and export a deployable Hugging
Face checkpoint.

Review the example directory's README and the selected YAML before running. Adapt
the model and data locations, output paths, and Slurm topology for your environment.

Run the Megatron-Bridge pipeline:

.. code-block:: bash

    cd tools/launcher
    source .env-slurm
    uv run launch.py \
        --yaml examples/nvidia/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-BF16/mbridge_qad.yaml \
        --yes

Run the Megatron-LM pipeline:

.. code-block:: bash

    cd tools/launcher
    source .env-slurm
    uv run launch.py \
        --yaml examples/nvidia/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-BF16/megatron_lm_qad.yaml \
        --yes
