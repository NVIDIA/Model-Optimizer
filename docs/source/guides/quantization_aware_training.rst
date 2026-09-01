.. _quantization-aware-training:

===============================================
Quantization-Aware Training and Distillation
===============================================

Quantization-aware training (QAT) and quantization-aware distillation (QAD) recover
quality lost when a model is quantized. Both train with simulated quantization
enabled, so the resulting checkpoint retains its ModelOpt quantization state for
deployment.

QAT versus QAD
==============

QAT is standard supervised fine-tuning of a quantized model. It uses the usual
cross-entropy (CE) loss against labeled data while the quantized forward pass lets
the weights adapt to quantization error. Use QAT to adapt a quantized model to a
task or dataset.

QAD uses knowledge distillation: a frozen BF16 teacher guides the quantized student
with a logit-level KL-divergence loss. Use QAD after PTQ to recover accuracy lost
specifically to quantization. It requires the teacher during training and therefore
uses more memory and compute than QAT.

In both cases, start from a PTQ checkpoint and retain its quantization configuration
during training. Choose QAT for task adaptation; choose QAD for quantization-accuracy
recovery.

QAD rationale
=============

`Quantization-Aware Distillation for NVFP4 Inference Accuracy Recovery
<https://arxiv.org/abs/2601.20088>`_ recommends QAD for recovery after aggressive
quantization, especially for models that have passed through multi-stage
post-training such as SFT, RL, or model merging. The teacher signal makes recovery
more robust when training-data quality or coverage is limited.

For broader background, see `How Quantization-Aware Training Enables Low-Precision
Accuracy Recovery
<https://developer.nvidia.com/blog/how-quantization-aware-training-enables-low-precision-accuracy-recovery/>`_. The
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
  <https://github.com/NVIDIA/Model-Optimizer/tree/main/examples/llm_qat#quick-start>`_:
  run ``examples/llm_qat/quantize.py``, then ``train.py`` with a QAT or QAD
  configuration, and finally ``export.py``. QAD training also supplies the BF16
  teacher through ``--teacher_model``.
* `Megatron-Bridge README
  <https://github.com/NVIDIA/Model-Optimizer/tree/main/examples/megatron_bridge>`_:
  run ``examples/megatron_bridge/quantize.py``, ``distill.py``, and
  ``export_quantized_megatron_to_hf.py``. ``distill.py`` is the QAD entry point;
  use the Megatron-Bridge SFT application for CE-only QAT.
* `Megatron-LM ModelOpt post-training documentation
  <https://github.com/NVIDIA/Megatron-LM/tree/main/examples/post_training/modelopt>`_:
  run ``quantize.sh``, then ``finetune.sh`` for QAT or QAD, and ``export.sh``.

Megatron container setup
------------------------

Megatron-Bridge and Megatron-LM need Megatron-Core, and their framework
dependencies. Run them in an NVIDIA GPU Docker environment such as a current
`NeMo container <https://catalog.ngc.nvidia.com/orgs/nvidia/-/containers/nemo/-/tags>`_,
and bind-mount a workspace for checkpoints and the Hugging Face cache so artifacts
survive the container.

Megatron-Bridge entry points live in the Model Optimizer checkout, so mount that
checkout into the container. See the
`Megatron-Bridge prerequisites
<https://github.com/NVIDIA/Model-Optimizer/tree/main/examples/megatron_bridge#pre-requisites>`_.

For Megatron-LM, use the bundled ModelOpt post-training scripts at
``/opt/Megatron-Bridge/3rdparty/Megatron-LM/examples/post_training/modelopt``,
or install Megatron-LM separately. Retain the container's ModelOpt installation
unless you explicitly need a newer version.

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
