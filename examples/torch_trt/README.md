# ModelOpt + Torch-TensorRT Deployment

End-to-end examples that quantize a PyTorch model with NVIDIA ModelOpt and
then compile the quantized graph with
[Torch-TensorRT](https://docs.pytorch.org/TensorRT/) for deployment.

The flow follows the
[Torch-TensorRT quantization guide](https://docs.pytorch.org/TensorRT/user_guide/shapes_precision/quantization.html):
ModelOpt inserts Q/DQ nodes into the eager PyTorch graph, then
`torch_tensorrt.compile(ir="dynamo")` converts those Q/DQ nodes into native
TensorRT precision layers.

## Setup

```bash
# From the NVIDIA TensorRT docker image (recommended):
docker run --gpus all -it --rm -v $(pwd):/workspace -w /workspace nvcr.io/nvidia/tensorrt:26.02-py3 bash

pip install -U "nvidia-modelopt[torch]"
pip install -r examples/torch_trt/requirements.txt
```

Torch-TensorRT itself follows the
[official install instructions](https://docs.pytorch.org/TensorRT/getting_started/installation.html) —
the version pulled by `pip` must match your installed PyTorch.

## Usage

```bash
# FP8 / NVFP4 default model is google/vit-large-patch16-224
python examples/torch_trt/quantize_and_compile_vit.py \
    --precision fp8/nvfp4 \
    --calib_samples 128 \
    --batch_size 1

# Quantize but don't TRT-compile (handy on a non-TRT host)
python examples/torch_trt/quantize_and_compile_vit.py \
    --precision fp8/nvfp4 \
    --skip_trt

# Custom model + custom recipe
python examples/torch_trt/quantize_and_compile_vit.py \
    --model_id <huggingface/model-id> \
    --recipe <recipe-path-relative-to-modelopt_recipes-or-absolute-yaml>
```

## What the example does

1. Loads a HuggingFace model (default: `google/vit-large-patch16-224`).
2. Builds a tiny calibration loader from `zh-plus/tiny-imagenet` (avoids the
   gated `ILSVRC/imagenet-1k` repo so the example runs unauthenticated).
3. Runs `mtq.quantize` with one of the recipes shipped under
   [`modelopt_recipes/`](../../modelopt_recipes/). The default recipes
   target ViT; pass `--recipe <path>` to use a different one for a
   different model.
4. Compiles the quantized model with `torch_tensorrt.compile` and prints a
   median-latency benchmark against the BF16 eager baseline.

## ViT-specific recipes shipped with the example

These are the recipes the CLI selects by default when `--model_id` points
at a HF ViT classifier. They are **not** thin wrappers around the modelopt
defaults — they're tuned for the HF ViT module layout.

| Flag | Recipe path | Key differences from the default |
|------|-------------|----------------------------------|
| `--precision fp8` | `huggingface/vit/ptq/fp8` | W8A8 FP8 **plus** MHA-aware FP8 on every per-block `nn.LayerNorm` output (shared Q/DQ feeds Q/K/V + MLP), FP8 attention Q/K/V BMM + softmax slots, patch-embedding `nn.Conv2d` left FP16, `classifier` head left in FP16, final `vit.layernorm` left FP16. |
| `--precision nvfp4` | `huggingface/vit/ptq/nvfp4` | Same skip list as the FP8 recipe; encoder Linear weights/inputs run NVFP4 W4A4 (E2M1, block 16, FP8 scales). Attention BMMs, softmax, and per-block LayerNorm outputs stay at FP8 — NVFP4 is too aggressive there. Uses `awq_lite` calibration. |

Each recipe is self-contained (no `$import` of shared snippets) and uses
the "specific-enable" style: narrow `parent_class` + path scoping on the
enable rules means no `enable: false` carve-outs are needed.

## Hardware requirements

| Recipe | Minimum GPU |
|--------|-------------|
| `fp8`   | Hopper (H100) / Ada (RTX 4090 / 6000 Ada) — compute capability 8.9+ |
| `nvfp4` | Blackwell (B100/B200) — TRT ≥ 10.8 |

Older GPUs will still let `mtq.quantize` succeed (it emits fake-quant
nodes in PyTorch), but `torch_tensorrt.compile` will not find a real
low-precision kernel and the speedup column will be ~1×.

### Resuming from a saved checkpoint

Pass `--save_dir <path>` to persist the modelopt-quantized model
(`vit_modelopt_state.pt`). To reload without recalibrating, restore it
before the TRT compile step with:

```python
import modelopt.torch.opt as mto
mto.restore(model, "vit_modelopt_state.pt")
```

## Custom recipes

Use `--recipe <path>` to plug in a different recipe — either a path
relative to `modelopt_recipes/` (resolved against the built-in library) or
an absolute filesystem path to a YAML file. The recipe must declare
`metadata.recipe_type: ptq` and a `quantize:` section; see existing
`modelopt_recipes/huggingface/vit/ptq/*.yaml` for the patterns used here.
