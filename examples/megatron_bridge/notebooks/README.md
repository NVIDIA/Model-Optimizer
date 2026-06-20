# Running the Cosmos Reason 2 Notebook

## Overview

This notebook (`05_cosmos_reason_end_to_end.ipynb`) walks through the full compression pipeline for [Cosmos Reason 2](https://github.com/nvidia-cosmos/cosmos-reason2) — a reasoning-specialised Vision-Language Model (VLM) built for Physical AI applications such as robotic manipulation and autonomous navigation.

The pipeline covers:

| Stage | Tool | What it does |
|---|---|---|
| **Prune** | `prune_minitron.py` | Depth-prune the LM tower from 28 → 24 layers (1.7 B → ~1.4 B) using the Minitron approach for the LLM component |
| **Distill** | `distill.py` (Megatron-Bridge) | Knowledge distillation from the unpruned teacher on a Nemotron CC + Math + Code corpus |
| **Quantize** | `hf_ptq.py` | FP8 PTQ with image-aware calibration (~2× throughput vs BF16) |
| **Serve** | vLLM | OpenAI-compatible endpoint |
| **Evaluate** | lmms-eval + aiperf | 4-way accuracy comparison (MathVista / PAI-Bench / RealWorldQA) + throughput benchmarks |

The compressed checkpoint (pruned + distilled + FP8, ~1.7 B total) targets near-full recovery of the 2 B BF16 baseline while halving memory footprint and roughly doubling serving throughput on a single H100.

**Prerequisites:** HuggingFace access to [`nvidia/Cosmos-Reason2-2B`](https://huggingface.co/nvidia/Cosmos-Reason2-2B) (gated — request access before running). A single H100 is sufficient; end-to-end runtime is approximately 1–1.5 hours.

## 1. Clone Model-Optimizer and checkout the relevant branch
```bash
git clone git@github.com:NVIDIA/Model-Optimizer.git
cd <BASE_PATH>/Model-Optimizer
git fetch origin lmikaelyan/compress-vlms
git checkout lmikaelyan/compress-vlms
```

## 2. Build the Docker image
```bash
cd examples/megatron_bridge/notebooks
docker build -t modelopt-notebooks .
```

## 3. Request HuggingFace access.
Go to https://huggingface.co/nvidia/Cosmos-Reason2-2B and request access. You will need to use your HF token.

## 4. Launch JupyterLab
```bash
docker run --gpus all \
  --shm-size=16GB \
  -p 8888:8888 \
  -p 6006:6006 \
  --rm -it \
  -e HF_TOKEN=<YOUR_HF_TOKEN> \
  -v <BASE_PATH>/Model-Optimizer:/dli/Model-Optimizer \
  modelopt-notebooks \
  jupyter lab --ip=0.0.0.0 --port=8888 --no-browser --allow-root \
  --notebook-dir=/dli/Model-Optimizer/examples/megatron_bridge/notebooks \
  --IdentityProvider.token='' \
  --ServerApp.password=''
```

## 5. SSH tunnel (from your local machine)
```bash
ssh -N -L 8888:g242-p33-0094:8888 nvdl-code-01
```

## 6. Open in browser
http://127.0.0.1:8888/lab

## Dependencies installed by the Dockerfile

| Package | Version | Purpose |
|---|---|---|
| `jupyterlab` | 4.4.10 | Notebook UI |
| `ipywidgets` | 8.1.7 | Progress bars in notebook |
| `nvidia-modelopt` | from source (`lmikaelyan/compress-vlms`) | Pruning, distillation, PTQ |
| `lmms-eval` | v0.7.1 | VLM accuracy benchmarks (§1.6: MathVista / BLINK / RealWorldQA) |
| `python-Levenshtein` | latest | Required by lmms-eval string metrics |
