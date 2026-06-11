# Torch-TensorRT Quantization

[Torch-TensorRT](https://docs.pytorch.org/TensorRT/) compiles a PyTorch model into an optimized TensorRT engine with no separate export or runtime. This example quantizes a PyTorch / HuggingFace model with NVIDIA Model Optimizer and then compiles the quantized graph in-framework with Torch-TensorRT for deployment.

Quantization is an effective model optimization technique that compresses your models. Model Optimizer inserts Q/DQ nodes into the eager PyTorch graph; `torch_tensorrt.compile(ir="dynamo")` then converts those Q/DQ nodes into native TensorRT precision layers — including FP8 and NVFP4 — following the [Torch-TensorRT quantization guide](https://docs.pytorch.org/TensorRT/user_guide/shapes_precision/quantization.html).

This section focuses on the in-framework Torch-TensorRT path: a PyTorch front end (`mtq.quantize`) feeding a Dynamo-compiled TensorRT engine, demonstrated end-to-end on a HuggingFace ViT image classifier. If you instead want a portable ONNX → TensorRT artifact, or you start from an ONNX model, see the sibling [`torch_onnx`](../torch_onnx/) and [`onnx_ptq`](../onnx_ptq/) examples (compared in the [Support Matrix](#support-matrix)).

<div align="center">

| **Section** | **Description** | **Link** | **Docs** |
| :------------: | :------------: | :------------: | :------------: |
| Pre-Requisites | Required packages and installation | \[[Link](#pre-requisites)\] | |
| Getting Started | Quantize and compile a ViT in a few lines | \[[Link](#getting-started)\] | \[[docs](https://docs.pytorch.org/TensorRT/user_guide/shapes_precision/quantization.html)\] |
| Support Matrix | How this path compares to the ONNX examples | \[[Link](#support-matrix)\] | |
| ViT Recipes | The FP8 / NVFP4 recipes shipped with the example | \[[Link](#vit-recipes)\] | |
| Usage | CLI flags for the quantize and accuracy scripts | \[[Link](#usage)\] | |
| Evaluate Accuracy | Measure ImageNet top-1 / top-5 accuracy | \[[Link](#evaluate-accuracy)\] | |
| Custom Recipes | Plug in your own recipe / model | \[[Link](#custom-recipes)\] | |
| Resources | Roadmap, docs, benchmarks, and support | \[[Link](#resources)\] | |

</div>

## Pre-Requisites

### Docker

Please use the TensorRT docker image (e.g., `nvcr.io/nvidia/tensorrt:26.02-py3`) or visit our [installation docs](https://nvidia.github.io/Model-Optimizer/getting_started/2_installation.html) for more information.

```bash
docker run --gpus all -it --rm -v $(pwd):/workspace -w /workspace nvcr.io/nvidia/tensorrt:26.02-py3 bash
```

Also follow the installation steps below to upgrade to the latest version of Model Optimizer and install example-specific dependencies.

### Local Installation

```bash
pip install -U "nvidia-modelopt[hf]"
pip install -r requirements.txt
```

### Hardware Requirements

The low-precision kernels Torch-TensorRT emits need a GPU that supports the target format:

<div align="center">

| Recipe | Minimum GPU |
| :---: | :---: |
| `fp8` | Ada / Hopper — compute capability 8.9+ |
| `nvfp4` | Blackwell — compute capability 10.0+ |

</div>

> [!NOTE]
> Older GPUs still let `mtq.quantize` succeed — it emits fake-quant nodes in PyTorch — but `torch_tensorrt.compile` will not find a real low-precision kernel for an unsupported format.

## Getting Started

Quantize a HuggingFace ViT, then compile the Q/DQ graph with Torch-TensorRT into a single `torch.nn.Module` you call from PyTorch:

```python
import torch
import torch_tensorrt

import modelopt.torch.quantization as mtq
from modelopt.recipe import load_recipe
from modelopt.torch.quantization.utils import export_torch_mode

# 1. Quantize the eager PyTorch model with a Model Optimizer PTQ recipe.
recipe = load_recipe("huggingface/vit/ptq/fp8")  # or huggingface/vit/ptq/nvfp4
mtq.quantize(model, recipe.quantize.model_dump(), forward_loop=calibrate)

# 2. Compile the quantized (Q/DQ) graph with Torch-TensorRT.
#    export_torch_mode() makes Model Optimizer emit Q/DQ in the TRT-friendly form,
#    and min_block_size=1 lets single-node Q/DQ + matmul subgraphs become TRT
#    precision layers (per the Torch-TensorRT quantization guide).
with export_torch_mode():
    trt_model = torch_tensorrt.compile(
        model,
        ir="dynamo",
        min_block_size=1,
        truncate_double=True,
        inputs=[torch_tensorrt.Input(
            min_shape=(1, 3, 224, 224),
            opt_shape=(128, 3, 224, 224),
            max_shape=(1024, 3, 224, 224),
            dtype=torch.float16,
        )],
    )

logits = trt_model(pixel_values)  # call it like any nn.Module
```

The runnable script [`torch_tensorrt_ptq.py`](./torch_tensorrt_ptq.py) wraps this flow end-to-end. It:

1. Loads a HuggingFace ViT classifier (default `google/vit-large-patch16-224`).
1. Builds a tiny calibration loader from `zh-plus/tiny-imagenet` (avoids the gated `ILSVRC/imagenet-1k` repo, so the example runs unauthenticated).
1. Runs `mtq.quantize` with one of the recipes under [`modelopt_recipes/`](../../modelopt_recipes/) (see [ViT Recipes](#vit-recipes)).
1. Saves the quantized Model Optimizer state (FP16 weights + Q/DQ metadata) to `<save_dir>/vit_modelopt_state.pt` for reuse without recalibration (see [Custom Recipes](#custom-recipes)).
1. Compiles the quantized model with `torch_tensorrt.compile` and verifies that the compiled-model argmax matches the fake-quant argmax on a sample input.

```bash
# Default model is google/vit-large-patch16-224, default recipe is the ViT FP8 recipe.
python torch_tensorrt_ptq.py --calib_samples 1024 --batch_size 128

# NVFP4 instead of FP8.
python torch_tensorrt_ptq.py --recipe huggingface/vit/ptq/nvfp4

# Quantize but don't TRT-compile (handy on a non-TRT host).
python torch_tensorrt_ptq.py --skip_trt
```

> [!NOTE]
> Both `torch_tensorrt_ptq.py` and the accuracy script ([`torch_tensorrt_accuracy.py`](./torch_tensorrt_accuracy.py)) run the model in `float16`.

## Support Matrix

All three of these examples reach the same destination — a low-precision TensorRT engine — but quantize at a different point in the pipeline and emit a different artifact, so they suit different deployment stacks:

<div align="center">

| | Torch-TensorRT (this example) | [`torch_onnx`](../torch_onnx/) | [`onnx_ptq`](../onnx_ptq/) |
| :---: | :---: | :---: | :---: |
| Starting point | a PyTorch / HF model | a PyTorch / timm model | an already-exported ONNX model |
| Quantize on | the eager PyTorch graph (`mtq.quantize`) | the eager PyTorch graph (`mtq.quantize`) | the ONNX graph directly (ONNX PTQ) |
| Export step | none — the FX/Dynamo graph stays in-process | `torch.onnx.export` of the Q/DQ graph, postprocessed for TRT | none — Q/DQ inserted straight into the ONNX graph |
| Intermediate artifact | none | a Q/DQ ONNX file | a Q/DQ ONNX file |
| Compiler + runtime | `torch_tensorrt.compile(ir="dynamo")` → a `torch.nn.Module` you call from PyTorch | TensorRT builds a standalone engine from the ONNX | TensorRT builds a standalone engine from the ONNX |
| Best when | PyTorch-native serving; you want a drop-in compiled module | you quantize in PyTorch but deploy via a portable ONNX → TRT engine | you only have an ONNX model and never touch PyTorch |

</div>

This example and [`torch_onnx`](../torch_onnx/) share the same PyTorch front end (`mtq.quantize`), so the numerics are identical — they differ only in the back end: this one keeps the graph in-process and hands it to Torch-TensorRT, while `torch_onnx` exports a portable ONNX artifact for the standalone TensorRT runtime. [`onnx_ptq`](../onnx_ptq/) instead quantizes the ONNX graph directly, for when you start from an ONNX model rather than PyTorch. Pick this example when your serving stack is PyTorch-native and you'd rather avoid an ONNX export step.

## ViT Recipes

These are the recipes the CLI selects by default when `--model_id` points at a HF ViT classifier. They are tuned for the HF ViT module layout and are composed from the shared `$import` building blocks under [`modelopt_recipes/configs/`](../../modelopt_recipes/configs/) (`numerics/{fp8,nvfp4}`, `ptq/units/{w8a8_fp8_fp8,attention_qkv_fp8,w4a4_nvfp4_nvfp4}`) rather than spelling out each `quant_cfg` entry.

<div align="center">

| `--recipe` value | Calibration | What it quantizes |
| :---: | :---: | :--- |
| `huggingface/vit/ptq/fp8` (default) | `max` | Per-tensor FP8 (E4M3) on every weight + input quantizer matched by the `*weight_quantizer` / `*input_quantizer` globs — encoder Linears, the patch-embed `nn.Conv2d` projection, and the `classifier` head — plus FP8 on the attention Q/K/V BMMs and softmax. All output quantizers disabled. |
| `huggingface/vit/ptq/nvfp4` | `awq_lite` | Dynamic NVFP4 W4A4 (E2M1, block size 16, FP8 E4M3 scales) on all weight/input quantizers, with the patch-embed `nn.Conv2d` projection, the `classifier` head, and the attention Q/K/V BMMs + softmax held at per-tensor FP8. All output quantizers disabled. |

</div>

## Usage

### `torch_tensorrt_ptq.py`

[Script](./torch_tensorrt_ptq.py) — quantize and (optionally) Torch-TensorRT-compile a ViT.

<div align="center">

| Flag | Default | Description |
| :---: | :---: | :--- |
| `--model_id` | `google/vit-large-patch16-224` | HuggingFace model id of the ViT classifier to quantize. |
| `--recipe` | `huggingface/vit/ptq/fp8` | Recipe path (relative to `modelopt_recipes/` or an absolute YAML). Pass `huggingface/vit/ptq/nvfp4` for NVFP4. |
| `--calib_samples` | `1024` | Number of tiny-imagenet samples to use for calibration. |
| `--batch_size` | `128` | Batch size for calibration / TRT compile. |
| `--save_dir` | `./modelopt_quantized` | Directory the quantized Model Optimizer state-dict (FP16 weights + Q/DQ metadata) is always saved to, as `vit_modelopt_state.pt` — re-usable across runs without recalibration. |
| `--skip_trt` | off | Quantize + run the fake-quant model only; skip `torch_tensorrt.compile`. Useful for environments without Torch-TensorRT installed. |
| `--layer_info_path` | unset | If set, write the compiled TRT engine's per-layer info (`get_layer_info()`) to this file. |

</div>

```bash
# Custom model + custom recipe, saving the quantized state elsewhere.
python torch_tensorrt_ptq.py \
    --model_id <huggingface/model-id> \
    --recipe <recipe-path-relative-to-modelopt_recipes-or-absolute-yaml> \
    --save_dir ./my_quantized

# Dump the compiled engine's per-layer info to inspect FP8/NVFP4 fusion.
python torch_tensorrt_ptq.py --layer_info_path ./vit_fp8_layers.txt
```

### `torch_tensorrt_accuracy.py`

[Script](./torch_tensorrt_accuracy.py) — quantize, compile, and score on ImageNet (see [Evaluate Accuracy](#evaluate-accuracy)).

<div align="center">

| Flag | Default | Description |
| :---: | :---: | :--- |
| `--model_id` | `google/vit-large-patch16-224` | HuggingFace model id of the ViT classifier to quantize and score. |
| `--recipe` | `huggingface/vit/ptq/fp8` | Recipe path (relative to `modelopt_recipes/` or an absolute YAML). Pass `huggingface/vit/ptq/nvfp4` for NVFP4. |
| `--calib_samples` | `1024` | Number of tiny-imagenet samples to use for calibration. |
| `--batch_size` | `128` | Calibration / compile / eval batch size. The Torch-TRT engine is dynamic (`min=1`, `opt=max(--batch_size, 2)`, `max=1024`) and handles any batch including the trailing partial batch. |
| `--eval_data_size` | full 50k | Number of ImageNet validation images to score. |
| `--imagenet_path` | `ILSVRC/imagenet-1k` | HF dataset card or local path to the ImageNet validation set (gated). |
| `--baseline` | off | Also score the unquantized model as a reference. It is Torch-TensorRT-compiled like the quantized model (or run eager under `--skip_trt`) so the comparison is apples-to-apples. |
| `--skip_trt` | off | Score the fake-quant (Model Optimizer) model; skip `torch_tensorrt.compile`. Useful for environments without Torch-TensorRT installed. |
| `--results_path` | unset | If set, write the accuracy results to this CSV path. |

</div>

## Evaluate Accuracy

[`torch_tensorrt_accuracy.py`](./torch_tensorrt_accuracy.py) reuses the quantize → compile pipeline above and reports ImageNet-1k top-1 / top-5 accuracy via the `onnx_ptq` example's `evaluate()` harness ([`examples/onnx_ptq/evaluation.py`](../onnx_ptq/evaluation.py)):

```bash
python torch_tensorrt_accuracy.py \
    --recipe huggingface/vit/ptq/fp8 \
    --batch_size 128 \
    --baseline \
    --eval_data_size 5000 \
    --results_path results.csv
```

- `--baseline` also scores the unquantized model. It is Torch-TensorRT-compiled the same way as the quantized model, so every reported number comes from the same TRT runtime (pass `--skip_trt` to score the eager / fake-quant models instead).
- The eval uses a **dynamic** engine (default `--batch_size 128`) for both precisions, so it serves the trailing partial batch at any batch size.
- `--results_path results.csv` writes the metrics table (`Metric`, `Top1 (%)`, `Top5 (%)`) to CSV.

> [!NOTE]
> Validation uses the gated `ILSVRC/imagenet-1k` split: accept its license / set `HF_TOKEN`, or point `--imagenet_path` at a local copy. `evaluate()` shuffles the split, so a partial `--eval_data_size` draws a different random subset each run — omit it (full 50k set) for a stable, comparable score.

## Custom Recipes

Use `--recipe <path>` to plug in a different recipe — either a path relative to `modelopt_recipes/` (resolved against the built-in recipe library) or an absolute filesystem path to a YAML file. The recipe is loaded via `modelopt.recipe.load_recipe`, must declare `metadata.recipe_type: ptq` and a `quantize:` section, and its `quantize` config is passed straight to `mtq.quantize`. See the existing [`modelopt_recipes/huggingface/vit/ptq/*.yaml`](../../modelopt_recipes/huggingface/vit/ptq/) for the patterns used here.

### Resuming From a Saved Checkpoint

`torch_tensorrt_ptq.py` always saves the quantized Model Optimizer state to `<save_dir>/vit_modelopt_state.pt` (default `--save_dir ./modelopt_quantized`) via `mto.save`. To reload it without recalibrating, restore it onto a freshly-loaded model before the TRT compile step:

```python
import modelopt.torch.opt as mto

mto.restore(model, "./modelopt_quantized/vit_modelopt_state.pt")
```

> [!NOTE]
> See the [save / restore guide](https://nvidia.github.io/Model-Optimizer/guides/2_save_load.html) for the full `mto.save` / `mto.restore` workflow.

## Resources

- 📅 [Roadmap](https://github.com/NVIDIA/Model-Optimizer/issues/146)
- 📖 [Documentation](https://nvidia.github.io/Model-Optimizer)
- 🎯 [Benchmarks](../benchmark.md)
- 💡 [Release Notes](https://nvidia.github.io/Model-Optimizer/reference/0_changelog.html)
- 🐛 [File a bug](https://github.com/NVIDIA/Model-Optimizer/issues/new?template=1_bug_report.md)
- ✨ [File a Feature Request](https://github.com/NVIDIA/Model-Optimizer/issues/new?template=2_feature_request.md)
