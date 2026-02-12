# MBridge Distillation Examples

This directory contains examples for knowledge distillation using Megatron-Bridge.

Based on: <https://docs.nvidia.com/nemo/megatron-bridge/0.2.0/training/distillation.html>

## Setup Steps

**Note:** Set `$WORKSPACE` to your project root directory before running these commands:

```bash
export WORKSPACE=...
```

1. **Initialize Megatron-Bridge submodules:**

   ```bash
   cd $WORKSPACE/Megatron-Bridge
   git submodule init
   git submodule update
   ```

2. **Start docker container with mounts:**

   ```bash
   submit_job --partition interactive --time 4 \
     --image $WORKSPACE/docker/modelopt_puzzletron_nemo_25_11.sqsh \
     --mounts $WORKSPACE:/workspace,$WORKSPACE/Megatron-Bridge/3rdparty/Megatron-LM:/opt/megatron-lm \
     --interactive --gpu 1
   ```

**Note:** The mount `/opt/megatron-lm` is required because Megatron-Bridge depends on the Megatron-LM submodule.

## Examples

- **`hf_model_distillation/`**: Standard HuggingFace model distillation
  - See [hf_model_distillation/README.md](hf_model_distillation/README.md) for usage

- **`decilm_model_distillation/`**: Puzzletron DeciLM model distillation
  - See [decilm_model_distillation/README.md](decilm_model_distillation/README.md) for details
