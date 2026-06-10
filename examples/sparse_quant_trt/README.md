<!--
SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Sparse + INT8 → ONNX → TensorRT example

End-to-end example that takes a Hugging Face LLM (default
[`Qwen/Qwen2.5-1.5B-Instruct`](https://huggingface.co/Qwen/Qwen2.5-1.5B-Instruct))
all the way to a running TensorRT engine:

```text
[2:4 weight sparsity]  ->  INT8 W8A8 SmoothQuant PTQ  ->  [QAT]
  ->  finalize  ->  torch -> ONNX export (opset 20)
  ->  TensorRT engine build (trtexec, --stronglyTyped)
  ->  validate structured-sparse INT8 kernels  ->  real text generation
```

A single script, [`pipeline.py`](pipeline.py), drives the whole flow.

Both **2:4 structured sparsity** (`--sparsity`) and **QAT** (`--qat`) are **optional and OFF by
default** — the default run is plain INT8 W8A8 SmoothQuant, which preserves accuracy and produces
coherent generations.

## Tested environment

Developed and tested inside the `nvcr.io/nvidia/pytorch:26.01-py3` Docker container:

| Component | Version / commit |
| --- | --- |
| Docker container | `nvcr.io/nvidia/pytorch:26.01-py3` |
| PyTorch | `2.10.0a0+a36e1d39eb` (git `a36e1d39eb`) |
| ONNX | `1.18.0` |
| TensorRT | `10.14.1.48` (trtexec `v101401`) |
| CUDA | `13.1` |
| NVIDIA ModelOpt | `0.45.0rc0` (this repository, installed editable) |
| transformers | `5.9.0` (supported range `>=4.56,<5.10`) |
| accelerate | `1.13.0` |
| GPU | NVIDIA RTX 6000 Ada Generation (sm_89) |

## Setup

Run inside the container above. ModelOpt is installed editable; `transformers`/`accelerate` and the
ONNX-export helper dependencies are added without disturbing the container's `torch 2.10` / `onnx 1.18`:

```bash
# editable ModelOpt (the container ships a pip constraint pinning an older version, so clear it)
PIP_CONSTRAINT= pip install -e . --no-deps
pip install --upgrade-strategy only-if-needed "transformers>=4.56,<5.10" "accelerate>=1.0.0" \
    omegaconf "pulp<4.0" "pydantic>=2.0" rich safetensors regex scipy
# ONNX-export helper deps used by modelopt.torch._deploy (do NOT upgrade onnx off 1.18)
pip install --upgrade-strategy only-if-needed onnxruntime onnx-graphsurgeon \
    "onnxconverter-common~=1.16.0" "onnxslim>=0.1.76"
```

## Usage

```bash
# Default: INT8 W8A8 SmoothQuant -> ONNX -> strongly-typed TensorRT -> text generation
python pipeline.py

# 2:4 sparsity + INT8 (TensorRT selects structured-sparse INT8 kernels).
# Add --qat to (start to) recover the accuracy that sparsity costs.
python pipeline.py --sparsity [--qat]

# Iterate on the TensorRT build/inference only, reusing an already-exported ONNX
python pipeline.py --reuse-onnx

# Also build an FP16 (unquantized, dense) baseline engine and compare performance
python pipeline.py --compare-baseline
```

### Key options

| Flag | Default | Description |
| --- | --- | --- |
| `--model-dir` | `/models/Qwen2.5-1.5B-Instruct` | HF model directory |
| `--sparsity` | off | Apply 2:4 structured sparsity before PTQ (auto-enables INT8 output quantizers) |
| `--quant-attention` / `--no-quant-attention` | on | INT8-quantize the attention math (q/k/v_bmm + softmax) in addition to the linear projections |
| `--qat` | off | Run the **example** QAT fine-tune after PTQ |
| `--weights-dtype` | `fp16` | ONNX/engine weight dtype (`fp16` or `fp32`); both build strongly-typed |
| `--calib-dataset` | `cnn_dailymail` | Calibration dataset (`cnn_dailymail` or a `nemotron-*` dataset) |
| `--calib-samples` / `--calib-seq` | `1024` / `512` | Calibration size / sequence length (mirrors `examples/llm_ptq/hf_ptq.py`) |
| `--seq-len` | `128` | Representative sequence length for ONNX export + trtexec optimization profile |
| `--prompt` / `--max-new-tokens` | *"What is the capital of France? ..."* / `32` | Final real-inference prompt and length |
| `--reuse-onnx` | off | Skip the torch stages and build TensorRT from an existing ONNX |
| `--compare-baseline` | off | Also build an FP16 (unquantized, dense) engine and profile both with trtexec |
| `--profiling-runs` | `1` | trtexec profiling runs for `--compare-baseline` (each run = 500 inferences) |

Run `python pipeline.py --help` for the full list.

## Performance comparison (`--compare-baseline`)

With `--compare-baseline`, the script additionally builds an **FP16 (unquantized, dense)** engine
from the same model and profiles both engines with `trtexec`, using the same profiling parameters
as ModelOpt's `modelopt/torch/_deploy/_runtime/tensorrt/engine_builder.py`:

```text
trtexec --loadEngine=<engine> --shapes=input_ids:1x<seq_len> \
        --warmUp=500 --avgRuns=500 --iterations=500*<profiling-runs> \
        --noDataTransfers --useCudaGraph --useSpinWait
```

It parses `Throughput` (qps) and the median `GPU Compute Time` / `Latency` from the trtexec output
and prints a side-by-side table plus the optimized engine's throughput/latency speedup over the
FP16 baseline. Both engines are profiled at the same fixed shape for an apples-to-apples comparison.

## What each stage does

1. **Load** the model in fp32 with eager attention (fp32 keeps SmoothQuant calibration of a sparse
   model numerically stable; eager attention exports as plain matmul+softmax).
2. **Calibration loop** — `get_dataset_dataloader(...) -> create_forward_loop(...)`, mirroring
   `examples/llm_ptq/hf_ptq.py`.
3. **2:4 sparsity** *(optional)* — `mts.sparsify(model, "sparse_magnitude")`.
4. **INT8 W8A8 SmoothQuant PTQ** — `mtq.quantize(model, mtq.INT8_SMOOTHQUANT_CFG, forward_loop)`.
   By default the **attention math** (`q/k/v_bmm` + `softmax`) is INT8-quantized too, not just the
   linear projections (`--no-quant-attention` for linears-only). With `--sparsity`, INT8 **output**
   quantizers are also enabled (see below).
5. **QAT** *(optional)* — a minimal **example** loop; replace it with your own dataset/training
   pipeline for real recovery.
6. **Finalize + ONNX export** — opset 20, `dynamo=False`; for `fp16` the model is cast and exported
   natively so the graph is self-consistently typed.
7. **TensorRT build** — `trtexec --stronglyTyped --sparsity={enable|disable}`.
8. **Validate + generate** — parse the trtexec sparsity report and run greedy text generation
   through the engine.

## When are structured-sparse INT8 kernels actually used?

TensorRT's structured-sparse path validates the 2:4 pattern and reports, in the verbose build log:

```text
(Sparsity) Found N layer(s) eligible to use sparse tactics: ...
(Sparsity) Chose M layer(s) using sparse tactics: ...
```

- **`Found`** = the weights pass the 2:4 pattern check (every 4 elements along the reduction axis
  have ≥2 zeros).
- **`Chose`** = TensorRT actually selected a sparse INT8 kernel (its tactic timer found it fastest).

A sparse INT8 GEMM is chosen only when its epilogue keeps data in **INT8** — which is why
`--sparsity` auto-enables INT8 output quantizers (otherwise each GEMM dequantizes to fp16/fp32 and
the dense kernel wins). With this, on the default model TensorRT reports roughly
`Found 196 / Chose 140` (sparse kernels selected for the q/o/gate/up/down projections across all
layers; the small k/v projections stay dense). The script prints a `PASS` when `Chose > 0`.

## ⚠️ Accuracy note

One-shot 2:4 **magnitude** sparsity zeros half the weights and causes **severe** accuracy
degradation on its own — the model produces gibberish until recovered with **QAT/SAT
fine-tuning**. The built-in `--qat` loop is a smoke-level **example**; recovering 50% sparsity
requires a real training run on a representative dataset. Use `--sparsity` to demonstrate the
sparse-kernel path; use the default (INT8-only) run for accuracy-preserving generation.

## Outputs

Written to `--out-dir` (default `/workspace/out`):

- `<model-name>.onnx` (+ `.onnx_data` external weights) — INT8 QDQ graph
- `<model-name>.engine` — the TensorRT engine
- `trtexec_build.log`, `layer_info.json` — build log and per-layer info used for sparse-kernel
  validation
