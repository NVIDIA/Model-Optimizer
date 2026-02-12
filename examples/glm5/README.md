# GLM-5 NVFP4 Quantization

This guide describes how to quantize the GLM-5 model (bf16) to NVFP4 using the
DeepSeek V3.2 PTQ pipeline. GLM-5 shares the same MoE + MLA architecture as
DeepSeek V3 (256 routed experts, MLA attention with DSA indexer), so we reuse
the DeepSeek inference code with a GLM-5-specific config.

## Prerequisites

- SLURM cluster with 8 GPUs (H100 80GB recommended)
- Container image with Model-Optimizer, PyTorch, and `fast_hadamard_transform`
  pre-installed (e.g. `modelopt-v2.sqsh`)
- HuggingFace bf16 checkpoint of GLM-5
- (Optional) FP8 checkpoint of GLM-5 for the MTP head

## Overview

The pipeline has four steps:

1. **Convert** the HF bf16 checkpoint to DeepSeek's sharded format (8-way TP)
2. **PTQ calibration** to compute per-layer amax statistics
3. **Quantize** bf16 weights to NVFP4 using the amax values
4. **(Optional) Extract MTP head** from FP8 checkpoint and add to the output

## Step 1: Convert HF checkpoint to DeepSeek format

The DeepSeek V3.2 model code uses a different weight naming convention and
shards weights across tensor-parallel ranks. This step converts and shards the
HF checkpoint using parallel subprocesses (one per rank) with `safetensors`
`get_slice` for memory-efficient reading.

```bash
srun --overlap --jobid=${JOBID} --nodes=1 --ntasks=1 \
    --container-image="${CONTAINER_IMAGE}" \
    --container-mounts="${CONTAINER_MOUNTS}" \
    bash -c '
pip install --no-deps -e /path/to/Model-Optimizer
cd /path/to/Model-Optimizer/examples/deepseek

python DeepSeek-V3.2-Exp/inference/convert.py \
    --hf-ckpt-path /path/to/glm-5-bf16 \
    --save-path /path/to/glm-5-ds \
    --n-experts 256 \
    --model-parallel 8
'
```

The conversion:
- Auto-detects the MTP layer (layer 78) via `config.json` and skips it
- Patches `tokenizer_config.json` to remove incompatible fields
  (`tokenizer_class`, `is_local`, `extra_special_tokens`)
- Produces 8 shards of ~177GB each

## Step 2: PTQ calibration

Run PTQ calibration across 8 GPUs using `torchrun`. This inserts quantizers
into the model and calibrates amax values on sample data.

```bash
srun --overlap --jobid=${JOBID} --nodes=1 --ntasks=1 \
    --container-image="${CONTAINER_IMAGE}" \
    --container-mounts="${CONTAINER_MOUNTS}" \
    --export="ALL,HF_TOKEN=${HF_TOKEN}" \
    bash -c '
pip install --no-deps -e /path/to/Model-Optimizer
cd /path/to/Model-Optimizer/examples/deepseek

torchrun --nproc-per-node 8 --master_port=12346 ptq.py \
    --model_path /path/to/glm-5-ds \
    --config DeepSeek-V3.2-Exp/inference/config_glm5.json \
    --quant_cfg NVFP4_DEFAULT_CFG \
    --output_path /path/to/glm-5-nvfp4-amax \
    --trust_remote_code \
    --batch_size 8 \
    --calib_size 512
'
```

Notes:
- An `HF_TOKEN` is required because the calibration dataset
  (`nvidia/Nemotron-Post-Training-Dataset-v2`) is gated
- The `kernel.py` stub in `examples/deepseek/` provides pure PyTorch
  implementations of `act_quant`, `fp8_gemm`, and `fp8_index` to replace the
  tilelang-based kernels (which require CUDA 12's `libnvrtc`)
- Output: 8 amax files + `hf_quant_config.json`

## Step 3: Quantize to NVFP4

Apply the calibrated amax values to quantize weights from the original HF
checkpoint to NVFP4 format.

```bash
# First copy config/tokenizer files to the output directory
mkdir -p /path/to/glm-5-nvfp4
cp /path/to/glm-5-bf16/*.json /path/to/glm-5-nvfp4/
cp /path/to/glm-5-bf16/*token* /path/to/glm-5-nvfp4/

# Run quantization (requires 1 GPU)
python examples/deepseek/quantize_to_nvfp4.py \
    --amax_path /path/to/glm-5-nvfp4-amax \
    --hf_path /path/to/glm-5-bf16 \
    --fp4_path /path/to/glm-5-nvfp4 \
    --world_size 8
```

This iterates through all 282 safetensors shards, quantizing weights that have
amax entries to NVFP4 and passing through non-quantized weights (norms, embed,
gate) as bf16. Takes ~35 minutes on a single GPU.

## Step 4 (Optional): Add MTP head

The MTP (Multi-Token Prediction) head at layer 78 is excluded from
quantization. If you have an FP8 checkpoint with the MTP head, extract it:

```bash
python examples/glm5/extract_mtp_head.py \
    --fp8_index /path/to/glm-5-fp8/model.safetensors.index.json \
    --fp8_dir /path/to/glm-5-fp8 \
    --nvfp4_dir /path/to/glm-5-nvfp4 \
    --mtp_layer 78
```

This extracts all layer-78 tensors into `mtp-fp8.safetensors` and updates
`model.safetensors.index.json` to include them.

## Convenience script

`examples/deepseek/run_glm5_ptq.sh` combines steps 1 and 2 into a single SLURM
job. Usage:

```bash
# Allocate a SLURM job (3-4 hours recommended)
salloc --nodes=1 --time=3:59:00 --account=<account> --partition=batch --no-shell

# Run conversion + PTQ
bash examples/deepseek/run_glm5_ptq.sh <job_id>

# Skip conversion if already done
bash examples/deepseek/run_glm5_ptq.sh <job_id> --skip-convert
```

## Key modifications to DeepSeek V3.2 code

The following changes were made to support GLM-5:

- **`convert.py`**: Parallel subprocess conversion with `get_slice`, MTP layer
  auto-detection, tokenizer config patching
- **`model.py`**: Added `gate_bias` config flag to `ModelArgs` (GLM-5 has gate
  bias, DSv3 gate bias was hardcoded to `dim == 7168`)
- **`config_glm5.json`**: GLM-5 model config in DeepSeek V3.2 format
- **`kernel.py`**: Pure PyTorch stubs for `act_quant`, `fp8_gemm`, `fp8_index`
  (replaces tilelang kernels that need CUDA 12)
- **`quantize_to_nvfp4.py`**: Conditional `ds_kernel` import, component-level
  `_remap_key` to avoid corrupting indexer keys, bf16 source support

## GLM-5 model config

| Parameter | Value |
|---|---|
| Hidden dim | 6144 |
| Layers | 78 + 1 MTP |
| Attention heads | 64 |
| Routed experts | 256 (8 activated) |
| Shared experts | 1 |
| Dense layers | 3 |
| Q LoRA rank | 2048 |
| KV LoRA rank | 512 |
| Vocab size | 154880 |
