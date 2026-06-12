# Qwen-Image Quantization (FP8 / NVFP4 / NVFP4-SVDQuant)

A reproducible harness for quantizing [`Qwen/Qwen-Image`](https://huggingface.co/Qwen/Qwen-Image)
with the diffusers quantization example and exporting HuggingFace checkpoints.

## What it does

- Registers Qwen-Image in the diffusers quantization example (`--model qwen-image`).
- **Recipe**: quantizes only the linears under `transformer_blocks`, keeping the
  **first 2 and last 2** of the 60 blocks (and everything outside
  `transformer_blocks`: text encoder, VAE, embedders, norms, `proj_out`, …) in
  original precision. The exclusion is applied **before calibration** so that for
  SVDQuant the excluded blocks' weights stay bit-identical to the original.
- Produces three checkpoints: **FP8**, **NVFP4** (max), and **NVFP4 + SVDQuant**.
- Exports a HuggingFace unified checkpoint per component (safetensors + `config.json`).

### SVDQuant checkpoint format (AWQ-aligned)

For the SVDQuant export, the quantizer-owned tensors are promoted to clean,
module-level safetensors keys (mirroring how AWQ exports `pre_quant_scale`):

| Tensor | Safetensors key |
|--------|-----------------|
| AWQ smoothing scale (`input_quantizer._pre_quant_scale`) | `<module>.pre_quant_scale` |
| Low-rank factor A (`weight_quantizer.svdquant_lora_a`) | `<module>.svdquant_lora_a` |
| Low-rank factor B (`weight_quantizer.svdquant_lora_b`) | `<module>.svdquant_lora_b` |

They are embedded in the component's main safetensors (no sidecar). The
`config.json`'s `quantization_config` follows the `nvfp4_awq` shape with
`"pre_quant_scale": true` plus the SVDQuant `lora_rank`, so a consumer can
reconstruct `y = NVFP4_GEMM(x) + (x @ lora_a^T) @ lora_b^T`. (No in-repo runtime
applies this residual yet; the checkpoint is a documented on-disk artifact.)

## Layout (kernel-dev defaults)

| Env var | Default | Purpose |
|---------|---------|---------|
| `KERNEL_DEV_ROOT` | `/lustre/fsw/coreai_dlalgo_modelopt/users/jingyux/kernel-dev` | Root for container/models/output |
| `MODEL_DIR` | `${KERNEL_DEV_ROOT}/models/Qwen-Image` | Local model cache |
| `OUTPUT_DIR` | `${KERNEL_DEV_ROOT}/qwen_image_ckpts` | Exported checkpoints |
| `HF_TOKEN_FILE` | `${KERNEL_DEV_ROOT}/HF_TOKEN.txt` | Hugging Face token file |
| `FORMATS` | `fp8 nvfp4 svdquant` | Formats to run |
| `CALIB_SIZE` / `BATCH_SIZE` / `N_STEPS` / `LOWRANK` | `64 / 2 / 20 / 32` | Calibration knobs |

## 1. Build the container (once)

The diffusers example needs a recent `diffusers` (with `QwenImagePipeline`) and
modelopt installed from source. From a base NGC PyTorch image:

```bash
CONTAINER_DIR=/lustre/fsw/coreai_dlalgo_modelopt/users/jingyux/kernel-dev/container
mkdir -p "${CONTAINER_DIR}"

# Import a base image to an enroot squashfs (adjust the tag as needed).
enroot import -o "${CONTAINER_DIR}/modelopt-diffusers.sqsh" \
    docker://nvcr.io#nvidia/pytorch:25.04-py3

# Install modelopt (from source) + example deps into the container, then re-save.
srun --container-image="${CONTAINER_DIR}/modelopt-diffusers.sqsh" \
     --container-mounts=/lustre:/lustre --container-save="${CONTAINER_DIR}/modelopt-diffusers.sqsh" \
     bash -lc '
       cd /lustre/fsw/coreai_dlalgo_modelopt/users/jingyux/kernel-dev/source/Model-Optimizer &&
       pip install -e ".[dev]" &&
       pip install -U "diffusers>=0.35" "transformers>=4.52" accelerate datasets &&
       python -c "from diffusers import QwenImagePipeline; print(\"QwenImagePipeline OK\")"
     '
```

## 2. Run quantization

Inside the container (or via `srun`), run the harness:

```bash
srun --gpus=1 \
     --container-image=/lustre/fsw/coreai_dlalgo_modelopt/users/jingyux/kernel-dev/container/modelopt-diffusers.sqsh \
     --container-mounts=/lustre:/lustre \
     bash examples/diffusers/quantization/qwen_image_svdquant/run_qwen_image_quantization.sh
```

This downloads `Qwen/Qwen-Image` to `MODEL_DIR` (idempotent), then for each
format writes `${OUTPUT_DIR}/qwen-image-<fmt>/` (HF checkpoint + `sanity.png`).

Run a single format, or preview the commands without executing:

```bash
FORMATS=svdquant LOWRANK=32 bash .../run_qwen_image_quantization.sh
DRY_RUN=1 bash .../run_qwen_image_quantization.sh        # print planned commands only
```

The equivalent direct `quantize.py` invocation for SVDQuant:

```bash
python examples/diffusers/quantization/quantize.py \
    --model qwen-image --override-model-path "${MODEL_DIR}" --model-dtype BFloat16 \
    --format fp4 --quant-algo svdquant --lowrank 32 \
    --calib-size 64 --batch-size 2 --n-steps 20 \
    --hf-ckpt-dir "${OUTPUT_DIR}/qwen-image-svdquant" \
    --sanity-image-path "${OUTPUT_DIR}/qwen-image-svdquant/sanity.png"
```

## Notes

- `Qwen/Qwen-Image` loads without `trust_remote_code`.
- The transformer is ~20B params; calibration needs a GPU with enough memory
  (use `--cpu-offloading` if VRAM-limited).
- The `--sanity-image-path` image is generated from the **in-memory** quantized
  pipeline before the weights are packed for export (a functional check of
  quantized inference; it does not reload the exported checkpoint).
