# Knowledge Distillation with Megatron-Bridge

This guide shows how to perform knowledge distillation on Puzzletron-compressed AnyModel checkpoints using Megatron-Bridge.

## Overview

1. Set up the environment with Megatron-Bridge
2. Convert AnyModel checkpoints (student and teacher) to Megatron-Bridge format
3. Run knowledge distillation training

## Setup

> **Temporary Setup:** The NeMo docker container includes Megatron-Bridge (main branch), but Puzzletron requires a specific version/branch of Megatron-Bridge that is not included by default. This manual setup is required to use the Puzzletron-compatible version. Once the container includes the required version, this setup step will no longer be necessary.

**Note:** Set `$WORKSPACE` to your project root directory before running these commands:

```bash
export WORKSPACE=/path/to/your/project
```

1. **Clone Megatron-Bridge:**

   Clone [Megatron-Bridge](https://github.com/NVIDIA-NeMo/Megatron-Bridge) and checkout the specific commit required for Puzzletron:

   ```bash
   cd $WORKSPACE
   git clone https://github.com/NVIDIA-NeMo/Megatron-Bridge.git
   cd Megatron-Bridge
   git checkout 960a718cb8989676b258e107d538642717e22e39
   ```

2. **Initialize Megatron-Bridge submodules:**

   ```bash
   cd $WORKSPACE/Megatron-Bridge
   git submodule init
   git submodule update
   ```

3. **Start Docker container with mounts:**

   Use the [NeMo 25.11 container](https://catalog.ngc.nvidia.com/orgs/nvidia/containers/nemo?version=25.11):

   ```bash
   docker run --gpus all -it --rm \
     -v $WORKSPACE:/workspace \
     -v $WORKSPACE/Megatron-Bridge/3rdparty/Megatron-LM:/opt/megatron-lm \
     nvcr.io/nvidia/nemo:25.11 \
     /bin/bash
   ```

   **Note:** The mount `/opt/megatron-lm` is required because Megatron-Bridge depends on the Megatron-LM submodule.

4. **Set up the environment inside the container:**

   ```bash
   export PYTHONPATH="/workspace/Megatron-Bridge/src:/workspace/Model-Optimizer:${PYTHONPATH}"
   ```

## Step 1: Convert Checkpoints to Megatron-Bridge Format

Convert both student and teacher checkpoints:

```bash
# Convert student checkpoint
torchrun --nproc_per_node=1 examples/puzzletron/mbridge_distillation/import_anymodel_to_mbridge.py \
    --input-ckpt-path /path/to/student/anymodel/checkpoint \
    --output-ckpt-path /path/to/student/mbridge/checkpoint

# Convert teacher checkpoint
torchrun --nproc_per_node=1 examples/puzzletron/mbridge_distillation/import_anymodel_to_mbridge.py \
    --input-ckpt-path /path/to/teacher/anymodel/checkpoint \
    --output-ckpt-path /path/to/teacher/mbridge/checkpoint
```

## Step 2: Run Knowledge Distillation

Run distillation with tokenized dataset:

```bash
torchrun --nproc_per_node=8 examples/puzzletron/mbridge_distillation/distill_anymodel.py \
    --student-mbridge-ckpt /path/to/student/mbridge/checkpoint/iter_0000000 \
    --teacher-mbridge-ckpt /path/to/teacher/mbridge/checkpoint/iter_0000000 \
    --data-path /path/to/tokenized/dataset \
    --output-dir ./distilled_output \
    dataset.sequence_length=8192 \
    model.tensor_model_parallel_size=8 \
    model.teacher.tensor_model_parallel_size=8 \
    train.global_batch_size=4 \
    train.micro_batch_size=1 \
    train.train_iters=5000 \
    logger.log_interval=1
```

The distilled checkpoint will be saved to `--output-dir`.
