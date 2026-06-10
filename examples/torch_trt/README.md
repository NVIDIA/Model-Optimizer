# ModelOpt + Torch-TensorRT Deployment

End-to-end examples that quantize a PyTorch model with NVIDIA ModelOpt and
then compile the quantized graph with
[Torch-TensorRT](https://docs.pytorch.org/TensorRT/) for deployment.

The flow follows the
[Torch-TensorRT quantization guide](https://docs.pytorch.org/TensorRT/user_guide/shapes_precision/quantization.html):
ModelOpt inserts Q/DQ nodes into the eager PyTorch graph, then
`torch_tensorrt.compile(ir="dynamo")` converts those Q/DQ nodes into native
TensorRT precision layers.

## How this differs from the ONNX examples

All three of these examples reach the same destination — a low-precision
TensorRT engine — but quantize at a different point in the pipeline and emit a
different artifact, so they suit different deployment stacks:

| | Torch-TensorRT (this example) | [`torch_onnx`](../torch_onnx/) | [`onnx_ptq`](../onnx_ptq/) |
|---|---|---|---|
| Starting point | a PyTorch / HF model | a PyTorch / timm model | an already-exported ONNX model |
| Quantize on | the eager PyTorch graph (`mtq.quantize`) | the eager PyTorch graph (`mtq.quantize`) | the ONNX graph directly (ModelOpt ONNX PTQ) |
| Export step | none — the FX/Dynamo graph stays in-process | `torch.onnx.export` of the Q/DQ graph, postprocessed for TRT | none — Q/DQ inserted straight into the ONNX graph |
| Intermediate artifact | none | a Q/DQ ONNX file | a Q/DQ ONNX file |
| Compiler + runtime | `torch_tensorrt.compile(ir="dynamo")` → a `torch.nn.Module` you call from PyTorch | TensorRT builds a standalone engine from the ONNX | TensorRT builds a standalone engine from the ONNX |
| Best when | PyTorch-native serving; you want a drop-in compiled module | you quantize in PyTorch but deploy via a portable ONNX → TRT engine | you only have an ONNX model and never touch PyTorch |

This example and [`torch_onnx`](../torch_onnx/) share the same PyTorch front end
(`mtq.quantize`), so the numerics are identical — they differ only in the back
end: this one keeps the graph in-process and hands it to Torch-TensorRT, while
`torch_onnx` exports a portable ONNX artifact for the standalone TensorRT
runtime. [`onnx_ptq`](../onnx_ptq/) instead quantizes the ONNX graph directly,
for when you start from an ONNX model rather than PyTorch. Pick this example
when your serving stack is PyTorch-native and you'd rather avoid an ONNX export
step.

## Setup

```bash
# From the NVIDIA TensorRT docker image (recommended):
docker run --gpus all -it --rm -v $(pwd):/workspace -w /workspace nvcr.io/nvidia/tensorrt:26.02-py3 bash

pip install -U "nvidia-modelopt"
pip install -r examples/torch_trt/requirements.txt
```

Torch-TensorRT itself follows the
[official install instructions](https://docs.pytorch.org/TensorRT/getting_started/installation.html) —
the version pulled by `pip` must match your installed PyTorch.

## Usage

```bash
# FP8 / NVFP4 default model is google/vit-large-patch16-224
python examples/torch_trt/torch_tensorrt_ptq.py \
    --precision fp8/nvfp4 \
    --calib_samples 512 \
    --batch_size 1

# Quantize but don't TRT-compile (handy on a non-TRT host)
python examples/torch_trt/torch_tensorrt_ptq.py \
    --precision fp8/nvfp4 \
    --skip_trt

# Custom model + custom recipe
python examples/torch_trt/torch_tensorrt_ptq.py \
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
4. Compiles the quantized model with `torch_tensorrt.compile` and verifies
   that the compiled-model argmax matches the fake-quant argmax on a sample
   input.

## Measuring ImageNet accuracy

`torch_tensorrt_accuracy.py` reuses the pipeline above and reports ImageNet-1k
top-1 / top-5 accuracy via the `onnx_ptq` example's `evaluate()` harness
([`examples/onnx_ptq/evaluation.py`](../onnx_ptq/evaluation.py)):

```bash
python examples/torch_trt/torch_tensorrt_accuracy.py \
    --precision fp8 \
    --baseline \
    --eval_data_size 5000
```

- `--baseline` also scores the unquantized model. It is Torch-TensorRT-compiled
  the same way as the quantized model, so every reported number comes from the
  same TRT runtime (pass `--skip_trt` to score the eager / fake-quant models).
- The Torch-TRT engine is compiled for one static batch shape, so the eval path
  requires `--batch_size 1`.
- Validation uses the gated `ILSVRC/imagenet-1k` split (accept its license / set
  `HF_TOKEN`), or point `--imagenet_path` at a local copy. `evaluate()` shuffles
  the split, so a partial `--eval_data_size` draws a different random subset each
  run — omit it (full set) for a stable, comparable score.
- `--results_path results.csv` writes the metrics table to CSV.

## ViT-specific recipes shipped with the example

These are the recipes the CLI selects by default when `--model_id` points at a
HF ViT classifier. They are tuned for the HF ViT module layout and are composed
from the shared `$import` building blocks under
[`modelopt_recipes/configs/`](../../modelopt_recipes/configs/)
(`numerics/{fp8,nvfp4}`, `ptq/units/{w8a8_fp8_fp8,attention_qkv_fp8}`) rather
than spelling out each `quant_cfg` entry.

| Flag | Recipe path | What it quantizes |
|------|-------------|-------------------|
| `--precision fp8` | `huggingface/vit/ptq/fp8` | W8A8 FP8 (E4M3) on every weight + input quantizer — encoder Linears, the patch-embed `nn.Conv2d`, the `classifier` head, and per-block `nn.LayerNorm` inputs — plus FP8 on the attention Q/K/V BMMs and softmax. Output quantizers disabled. |
| `--precision nvfp4` | `huggingface/vit/ptq/nvfp4` | NVFP4 W4A4 (E2M1, block 16, FP8 scales) on the encoder `nn.Linear` weights/inputs, with the patch-embed `nn.Conv2d`, the `classifier` head, and the attention Q/K/V BMMs + softmax held at FP8. Uses `awq_lite` calibration. |

## Hardware requirements

| Recipe | Minimum GPU |
|--------|-------------|
| `fp8`   | Hopper (H100) / Ada (RTX 4090 / 6000 Ada) — compute capability 8.9+ |
| `nvfp4` | Blackwell (B100/B200) — TRT ≥ 10.8 |

Older GPUs will still let `mtq.quantize` succeed (it emits fake-quant
nodes in PyTorch), but `torch_tensorrt.compile` will not find a real
low-precision kernel.

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
