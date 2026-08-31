.. _quantization-aware-training:

===============================================
Quantization-Aware Training and Distillation
===============================================

Quantization-aware training (QAT) and quantization-aware distillation (QAD) recover
quality that is lost when a model is quantized. Both train a model with simulated
quantization enabled, so the resulting checkpoint retains its ModelOpt quantization
state and can be exported for deployment.

QAT versus QAD
==============

QAT is standard supervised fine-tuning of a quantized model. The training objective
is the usual cross-entropy (CE) loss against labeled data, while the quantized
forward pass lets the weights adapt to quantization error. Use QAT when the goal is
to adapt a quantized model to a task or dataset.

QAD uses knowledge distillation instead: a frozen BF16 teacher guides the quantized
student with a logit-level KL-divergence loss. It is usually used after PTQ, with
the original BF16 model as the teacher, to recover quality lost specifically to
quantization. QAD requires the teacher model during training and therefore uses more
memory and compute than QAT.

In both cases, begin with a PTQ checkpoint and keep its quantization configuration
unchanged during training. After training, export the quantized checkpoint using the
export workflow for the framework that produced it.

QAD workflow and rationale
==========================
The paper `Quantization-Aware Distillation for NVFP4 Inference Accuracy Recovery <https://arxiv.org/abs/2601.20088>`_ recommends QAD for accuracy
recovery after aggressive quantization, particularly for models that have gone
through multi-stage post-training such as SFT, RL, or model merging. It reports that
QAD is more stable and less complex to engineer than conventional QAT in these
settings, and that the teacher signal makes recovery more robust when training data
quality or coverage is limited. QAT remains the direct choice when task-specific CE
fine-tuning is the objective.

QAD is a two-stage workflow:

#. Start from the BF16 checkpoint and use PTQ to create a quantized student
   checkpoint. Because QAD is intended to recover the remaining accuracy gap, this
   stage can use a more aggressive recipe than a PTQ-only deployment would accept.
#. Train that PTQ checkpoint against the frozen BF16 teacher. Each student forward
   pass uses simulated quantization and the distillation loss aligns the student
   logits with the teacher. Export the resulting QAD checkpoint for deployment.

The PTQ recipe determines how quantization scales are handled during QAD. Dynamic
scales from max-calibrated PTQ checkpoints can be recomputed during training;
scales from MSE-based static PTQ checkpoints should remain frozen, because repeating
the scale search each step is prohibitively expensive.

For a broad overview of how QAT and QAD can recover accuracy lost to quantization, see `How Quantization-Aware
Training Enables Low-Precision Accuracy Recovery <https://developer.nvidia.com/blog/how-quantization-aware-training-enables-low-precision-accuracy-recovery/>`_. For information on applying QAD to Nemotron models, see our blog on `Developing Nemotron 3.5 Lightning NVFP4 with QAD <https://developer.nvidia.com/blog/developing-nemotron-3-5-lightning-nvfp4-with-qad-using-nvidia-model-optimizer/>`_, which
walks through this PTQ-to-QAD-to-export process, including selection of the student
recipe, training data and sequence length, and scale handling.



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

Run QAT or QAD
==============

Hugging Face
------------

The Hugging Face workflow is manual; there is no launcher example. The
`Hugging Face QAT/QAD examples <https://github.com/NVIDIA/Model-Optimizer/tree/main/examples/llm_qat>`_
provide end-to-end QAT and QAD commands. In summary:

#. Quantize the Hugging Face model with ``examples/llm_qat/quantize.py``.
#. For QAT, run ``examples/llm_qat/train.py`` with the quantized checkpoint and a
   QAT configuration. This trains with CE loss on the labeled dataset.
#. For QAD, run the same training script with a QAD configuration and
   ``--teacher_model`` pointing at the BF16 model.
#. Export the trained checkpoint with ``examples/llm_qat/export.py``.

For example, the QAT and QAD training configurations in the examples are
``configs/train/qat_nvfp4.yaml`` and ``configs/train/qad_nvfp4.yaml`` respectively.


Run the Megatron workflows in a NeMo container
----------------------------------------------

Megatron-Bridge and Megatron-LM require CUDA, Megatron-Core, and their framework
dependencies. The `NeMo container catalog <https://catalog.ngc.nvidia.com/orgs/nvidia/-/containers/nemo/-/tags>`_
lists the latest available image; the commands below use ``nvcr.io/nvidia/nemo:26.08``.
Run them on a host with NVIDIA GPUs and Docker configured for GPU access.

Prepare a directory for checkpoints and Hugging Face caches, then start an
interactive container. Replace ``$HF_TOKEN`` with a token that can read the selected
model and dataset. The default command uses the Model Optimizer package bundled in
the container and is sufficient for the Megatron-LM workflow.

.. code-block:: bash

    export QAT_WORK_DIR="$PWD/qat-qad-work"
    mkdir -p "$QAT_WORK_DIR/hf-cache"

    docker run --rm -it --gpus all --shm-size=16g --net=host --ulimit memlock=-1 \
        -e HF_TOKEN="$HF_TOKEN" \
        -v "$QAT_WORK_DIR":/workspace \
        -v "$QAT_WORK_DIR/hf-cache":/root/.cache/huggingface \
        nvcr.io/nvidia/nemo:26.08 bash

Inside the container, authenticate with ``hf auth login --token "$HF_TOKEN"`` if
the token was not provided through the environment. The mounted ``/workspace``
directory preserves checkpoints after the container exits. The examples below use a
small Qwen model and one GPU to demonstrate the command shape; increase the GPU
count and choose TP, PP, CP, and EP to fit the target model and sequence length.

The Megatron-Bridge examples are source entry points and are not bundled in the
NeMo image. Before running the Megatron-Bridge commands below, mount the
Model-Optimizer checkout and its Python packages by adding these mounts to the
``docker run`` command (run the command from the repository root):

.. code-block:: bash

    -v "$PWD":/opt/Model-Optimizer
    -v "$PWD/modelopt":/opt/venv/lib/python3.12/site-packages/modelopt
    -v "$PWD/modelopt_recipes":/opt/venv/lib/python3.12/site-packages/modelopt_recipes

These mounts keep the Megatron-Bridge scripts and ModelOpt package at the same
revision. For Megatron-LM alone, omit them and use the bundled scripts at
``/opt/Megatron-Bridge/3rdparty/Megatron-LM``.

Megatron-Bridge
---------------

Megatron-Bridge converts the Hugging Face model to Megatron-Core automatically. The `Megatron-Bridge README <https://github.com/NVIDIA/Model-Optimizer/tree/main/examples/megatron_bridge>`_
includes the complete PTQ, QAD, data-preparation, and export options.

First create a quantized Megatron checkpoint; it is the input to either QAT or QAD:

.. code-block:: bash

    cd /opt/Model-Optimizer/examples/megatron_bridge
    export MODEL=Qwen/Qwen3-0.6B
    export PTQ_CKPT=/workspace/qwen3-0.6b-nvfp4-megatron

    torchrun --nproc_per_node 1 quantize.py \
        --hf_model_name_or_path "$MODEL" \
        --quant_cfg nvfp4 \
        --calib_batch_size 1 \
        --calib_num_samples 128 \
        --seq_length 4096 \
        --export_megatron_path "$PTQ_CKPT"

For QAT, load ``$PTQ_CKPT`` into your Megatron-Bridge SFT configuration, restore
the ModelOpt state, and train with the normal CE objective. This keeps the student
fake-quantized during SFT. The repository's Megatron-Bridge entry point is focused
on distillation; use the framework's SFT application for this CE-only step.

For QAD, run the included ``distill.py`` entry point. ``--student_megatron_path``
restores the quantized student and its ModelOpt state, while the BF16 Hugging Face
model is loaded as the frozen teacher. The following mock-data command verifies the
end-to-end path; replace ``--use_mock_data`` with ``--data_paths`` (or ``--sft`` and
``--sft_dataset_root``) for a real run.

.. code-block:: bash

    torchrun --nproc_per_node 1 distill.py \
        --teacher_hf_path "$MODEL" \
        --student_hf_path "$MODEL" \
        --student_megatron_path "$PTQ_CKPT" \
        --use_mock_data \
        --seq_length 512 \
        --mbs 1 \
        --gbs 8 \
        --train_iters 100 \
        --eval_interval 10 \
        --eval_iters 4 \
        --output_dir /workspace/qwen3-0.6b-nvfp4-qad

Export the final QAD checkpoint to a deployable unified Hugging Face checkpoint:

.. code-block:: bash

    torchrun --nproc_per_node 1 export_quantized_megatron_to_hf.py \
        --hf_model_name_or_path "$MODEL" \
        --megatron_path /workspace/qwen3-0.6b-nvfp4-qad/checkpoints \
        --export_unified_hf_path /workspace/qwen3-0.6b-nvfp4-qad-hf

There is also a complete Slurm launcher example for QAD. It tokenizes the data,
performs NVFP4 PTQ, distills the quantized student, and exports a unified Hugging
Face checkpoint:

.. code-block:: bash

    cd tools/launcher
    source .env-slurm
    uv run launch.py \
        --yaml examples/nvidia/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-BF16/mbridge_qad.yaml \
        --yes

See the `Megatron-Bridge launcher configuration <https://github.com/NVIDIA/Model-Optimizer/blob/main/tools/launcher/examples/nvidia/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-BF16/mbridge_qad.yaml>`_
before running it, and adjust the model, data paths, output paths, and Slurm topology
for your environment.

Megatron-LM
-----------

Megatron-LM can be used when more control over training parameters is needed. See the `Megatron-LM post-training README <https://github.com/NVIDIA/Megatron-LM/tree/main/examples/post_training/modelopt>`_ for more details. 

For manual QAT, first apply PTQ, then fine-tune the quantized checkpoint with the
ModelOpt-enabled Megatron-LM post-training scripts. QAT uses the standard SFT/CE
objective. From the NeMo container started above, run:

.. code-block:: bash

    cd /opt/Megatron-Bridge/3rdparty/Megatron-LM/examples/post_training/modelopt
    export MODEL=meta-llama/Llama-3.2-1B-Instruct
    export HF_MODEL_CKPT="$MODEL"
    export PTQ_CKPT=/workspace/llama-3.2-1b-nvfp4-ptq

    TP=1 PP=1 EP=1 ETP=1 \
        MLM_MODEL_SAVE="$PTQ_CKPT" \
        MLM_EXTRA_ARGS="--calib-size 128 --calib-max-sequence-length 4096" \
        bash ./quantize.sh "$MODEL" NVFP4_DEFAULT_CFG

.. note::

   Megatron-LM defaults to the `NVIDIA Nemotron Post-Training Dataset v2
   <https://huggingface.co/datasets/nvidia/Nemotron-Post-Training-Dataset-v2>`_
   for PTQ calibration. It is gated on Hugging Face, but is the preferred
   calibration dataset for this workflow. Request access and provide ``HF_TOKEN``
   as shown above. To use another dataset, set
   ``--calib-dataset-path-or-name`` in ``MLM_EXTRA_ARGS``; choose data that is
   representative of the target workload.

    TP=1 PP=1 EP=1 ETP=1 \
        MLM_MODEL_CKPT="$PTQ_CKPT" \
        MLM_MODEL_SAVE=/workspace/llama-3.2-1b-nvfp4-qat \
        DATASET=Magpie-Align/Magpie-Llama-3.1-Pro-MT-300K-Filtered \
        MLM_EXTRA_ARGS="--train-samples 1000 --lr-decay-samples 1000" \
        bash ./finetune.sh "$MODEL"

Replace the example ``DATASET`` value with a dataset appropriate for the task. The
second command detects the ModelOpt state in the quantized checkpoint, retains
simulated quantization, and uses the normal supervised objective. Export it after
training:

.. code-block:: bash

    TP=1 PP=1 EP=1 ETP=1 \
        MLM_MODEL_CKPT=/workspace/llama-3.2-1b-nvfp4-qat \
        EXPORT_DIR=/workspace/llama-3.2-1b-nvfp4-qat-hf \
        bash ./export.sh "$MODEL"

For QAD, import the BF16 Hugging Face model as a Megatron-Core teacher with
Megatron-Bridge, then use the PTQ checkpoint as the student. This follows the
`Megatron-LM checkpoint prerequisite <https://github.com/NVIDIA/Megatron-LM/blob/main/examples/post_training/modelopt/README.md#megatron-core-checkpoint-prerequisite>`_.
The imported checkpoint includes the Megatron-Core configuration required by the
Megatron-LM QAD workflow.

.. code-block:: bash

    export TEACHER_CKPT=/workspace/llama-3.2-1b-bf16-mcore

    bash /opt/Megatron-Bridge/scripts/conversion/convert.sh import \
        --executor local \
        --device gpu \
        --gpus-per-node 1 \
        --hf-model "$MODEL" \
        --megatron-path "$TEACHER_CKPT"

    TP=1 PP=1 EP=1 ETP=1 \
        MLM_MODEL_CKPT="$PTQ_CKPT" \
        MLM_MODEL_SAVE=/workspace/llama-3.2-1b-nvfp4-qad \
        DATASET=Magpie-Align/Magpie-Llama-3.1-Pro-MT-300K-Filtered \
        MLM_EXTRA_ARGS="--export-kd-teacher-load $TEACHER_CKPT --train-samples 1000 --lr-decay-samples 1000" \
        bash ./finetune.sh "$MODEL"

The second command adds the frozen teacher and the default logit-level distillation
loss to the quantized student training. Use ``bash ./export.sh "$MODEL"`` as in the QAT
flow, replacing ``MLM_MODEL_CKPT`` and ``EXPORT_DIR`` with the QAD output paths.

The following launcher example runs the complete Megatron-LM QAD flow: import the
BF16 model, create an NVFP4 student with PTQ, distill the student, and export it to
Hugging Face format.

.. code-block:: bash

    cd tools/launcher
    source .env-slurm
    uv run launch.py \
        --yaml examples/nvidia/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-BF16/megatron_lm_qad.yaml \
        --yes

Review the `Megatron-LM launcher configuration <https://github.com/NVIDIA/Model-Optimizer/blob/main/tools/launcher/examples/nvidia/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-BF16/megatron_lm_qad.yaml>`_
before launching. In particular, adapt its data, checkpoint paths, and distributed
topology to the target model and cluster.

Additional resources
====================

* `Hugging Face QAT/QAD examples <https://github.com/NVIDIA/Model-Optimizer/tree/main/examples/llm_qat>`_
* `Megatron-Bridge examples <https://github.com/NVIDIA/Model-Optimizer/tree/main/examples/megatron_bridge>`_
* `Megatron-LM ModelOpt post-training examples <https://github.com/NVIDIA/Megatron-LM/tree/main/examples/post_training/modelopt>`_
* `Quantization-Aware Distillation for NVFP4 Inference Accuracy Recovery <https://arxiv.org/abs/2601.20088>`_
* `How Quantization-Aware Training Enables Low-Precision Accuracy Recovery <https://developer.nvidia.com/blog/how-quantization-aware-training-enables-low-precision-accuracy-recovery/>`_
* `Developing Nemotron 3.5 Lightning NVFP4 with QAD Using NVIDIA Model Optimizer <https://developer.nvidia.com/blog/developing-nemotron-3-5-lightning-nvfp4-with-qad-using-nvidia-model-optimizer/>`_
