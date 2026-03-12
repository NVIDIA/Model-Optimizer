# Environment Setup for ModelOpt PTQ

## Container Selection

The recommended container is the TensorRT-LLM release image from NVIDIA NGC, which includes ModelOpt, PyTorch, and CUDA pre-installed:

```text
nvcr.io/nvidia/tensorrt-llm/release:<version>
```

Check the [ModelOpt installation guide](https://nvidia.github.io/TensorRT-Model-Optimizer/getting_started/2_installation.html) for the compatible container version.

To use a custom ModelOpt branch inside the container, install it editable:

```bash
pip install -e /path/to/Model-Optimizer[hf] --no-deps
```

## Local Execution (No Container)

```bash
pip install nvidia-modelopt[hf]
# or from source
pip install -e ".[hf]"

python examples/llm_ptq/hf_ptq.py --model_dir <model> --qformat nvfp4 --output_dir <output>
```

## Docker

```bash
docker run --gpus all \
    -v /path/to/models:/models \
    -v /path/to/output:/output \
    nvcr.io/nvidia/tensorrt-llm/release:<version> \
    python /workspace/ptq_script.py \
        --model_path /models/my-model \
        --export_path /output/quantized
```

## SLURM with Enroot/Pyxis

### Import container image

Convert Docker image to enroot squashfs for reuse. Set writable cache paths first — the default `/raid/containers` may not be writable:

```bash
export ENROOT_CACHE_PATH=/path/to/writable/enroot-cache
export ENROOT_DATA_PATH=/path/to/writable/enroot-data
export TMPDIR=/path/to/writable/tmp
mkdir -p "$ENROOT_CACHE_PATH" "$ENROOT_DATA_PATH" "$TMPDIR"

enroot import --output /path/to/container.sqsh \
    docker://nvcr.io#nvidia/tensorrt-llm/release:<version>
```

### SLURM launch script

Container flags (`--container-image`, `--container-mounts`) only work with `srun`, NOT as `#SBATCH` directives.

```bash
#!/bin/bash
#SBATCH --job-name=ptq
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=8
#SBATCH --time=04:00:00
#SBATCH --output=logs/ptq-%j.out
#SBATCH --error=logs/ptq-%j.err

srun \
    --container-image="/path/to/container.sqsh" \
    --container-mounts="/data:/data" \
    --container-workdir="/workspace" \
    --no-container-mount-home \
    bash -c "
        pip install --upgrade huggingface_hub 2>/dev/null || true
        pip install -e /data/Model-Optimizer[hf] --no-deps 2>/dev/null || true
        python /data/ptq_script.py \
            --model_path /data/model \
            --export_path /data/output
    "
```

### Multi-node PTQ (FSDP2)

For models too large for a single node, use `examples/llm_ptq/multinode_ptq.py` with accelerate and FSDP2. Edit `examples/llm_ptq/fsdp2.yaml`:

- Set `num_machines` and `num_processes` to match SLURM allocation
- Set `fsdp_transformer_layer_cls_to_wrap` to the model's decoder layer class name

```bash
accelerate launch --config_file fsdp2.yaml multinode_ptq.py \
    --model_dir <model> --qformat nvfp4 --output_dir <output>
```

## GPU Memory Estimation

The model must fit in BF16 across all GPUs during PTQ. Rule of thumb:

| Model Size | BF16 Memory | Recommended Setup |
|------------|-------------|-------------------|
| 7-13B | 14-26 GB | 1x H100-80GB |
| 34-70B | 68-140 GB | 2-4x H100-80GB |
| 100-200B | 200-400 GB | 4-8x H100-80GB |
| 200B+ | 400+ GB | Multi-node FSDP2 |

Use `device_map="auto"` for automatic multi-GPU sharding.

## Common Environment Errors

### `mkdir: cannot create directory '/raid/containers/cache/...': Permission denied`

Default enroot cache directory not writable. Set `ENROOT_CACHE_PATH`, `ENROOT_DATA_PATH`, and `TMPDIR` to writable paths before `enroot import`.

### `sbatch: unrecognized option '--container-image'`

Pyxis container flags only work with `srun`. Move container flags from `#SBATCH` directives into the `srun` command.

### `ImportError: cannot import name '...' from 'huggingface_hub'`

Container's `huggingface_hub` is too old. Add `pip install --upgrade huggingface_hub` before other pip installs.

### `DatasetNotFoundError: Dataset '...' is a gated dataset`

Use an ungated dataset for calibration (e.g., `cnn_dailymail`), or authenticate with `huggingface-cli login`.

### OOM during model loading

Use `device_map="auto"` and request more GPUs. Reduce `--batch_size` to 1. For very large models, consider multi-node FSDP2.
