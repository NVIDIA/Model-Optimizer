# Running the Cosmos Reason 2 Notebook

## 1. Checkout the branch
```bash
cd /home/scratch.thannan_wwfo/robotic_dli/Model-Optimizer
git fetch origin lmikaelyan/compress-vlms
git checkout lmikaelyan/compress-vlms
```

## 2. Build the Docker image
```bash
cd examples/megatron_bridge/notebooks
docker build -t modelopt-notebooks .
```

## 3. Request HuggingFace access
Go to https://huggingface.co/nvidia/Cosmos-Reason2-2B and request access. (alreay have access)

## 4. Launch JupyterLab
```bash
docker run --gpus all \
  --shm-size=16GB \
  -p 8888:8888 \
  --rm -it \
  -e HF_TOKEN=<YOUR_HF_TOKEN> \
  -v /home/scratch.thannan_wwfo/robotic_dli/Model-Optimizer:/dli/Model-Optimizer \
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
