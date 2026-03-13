# Deployment Environment Setup

## Framework Installation

### vLLM

```bash
pip install vllm
```

Minimum version: 0.10.1

### SGLang

```bash
pip install "sglang[all]"
```

Minimum version: 0.4.10

### TRT-LLM

TRT-LLM is best installed via NVIDIA container:

```bash
docker pull nvcr.io/nvidia/tensorrt-llm/release:<version>
```

Or via pip (requires CUDA toolkit):

```bash
pip install tensorrt-llm
```

Minimum version: 0.17.0

## SLURM Deployment

For SLURM clusters, deploy inside a container. Container flags MUST be on the `srun` line:

```bash
#!/bin/bash
#SBATCH --job-name=deploy
#SBATCH --account=<account>
#SBATCH --partition=<partition>
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=<num_gpus>
#SBATCH --time=04:00:00
#SBATCH --output=deploy_%j.log

srun \
    --container-image="<path/to/container.sqsh>" \
    --container-mounts="<data_root>:<data_root>" \
    --container-workdir="<workdir>" \
    --no-container-mount-home \
    bash -c "python -m vllm.entrypoints.openai.api_server \
        --model <checkpoint_path> \
        --quantization modelopt \
        --tensor-parallel-size <num_gpus> \
        --host 0.0.0.0 --port 8000"
```

To access the server from outside the SLURM node, note the allocated hostname:

```bash
squeue -u $USER -o "%j %N %S"  # Get the node name
# Then SSH tunnel or use the node's hostname directly
```

## Docker Deployment

### vLLM with ModelOpt

A Dockerfile is available at `examples/vllm_serve/Dockerfile`:

```bash
docker build -f examples/vllm_serve/Dockerfile -t vllm-modelopt .

docker run --gpus all -p 8000:8000 vllm-modelopt \
    python -m vllm.entrypoints.openai.api_server \
        --model <checkpoint_path> \
        --quantization modelopt \
        --host 0.0.0.0 --port 8000
```
