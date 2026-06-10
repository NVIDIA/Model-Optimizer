# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Measure ImageNet top-1/top-5 accuracy of a Torch-TensorRT ViT.

Pipeline:

1. Quantize a HuggingFace ViT with a ModelOpt recipe and compile it with
   ``torch_tensorrt.compile(ir="dynamo")`` — reusing the sibling example
   ``torch_tensorrt_ptq.py``.
2. Score the compiled model on the ImageNet-1k validation split using the
   ``onnx_ptq`` example's ``evaluate`` API (``examples/onnx_ptq/evaluation.py``).

The compiled Torch-TRT module is a ``torch.nn.Module``, so ``evaluate`` runs it
exactly like an eager model. A thin :class:`_EvalAdapter` bridges the two
contracts: it casts the dataloader's float32 image batches to the model's
compute dtype and unwraps HF ``ImageClassifierOutput`` to a plain logits tensor.

Example::

    python torch_tensorrt_accuracy.py --batch_size 128 --eval_data_size 5000 --baseline

``--imagenet_path`` defaults to the gated ``ILSVRC/imagenet-1k`` HF dataset
(accept its license / set ``HF_TOKEN``), or point it at a local copy. Note the
``evaluate`` API shuffles the validation set, so a partial ``--eval_data_size``
samples a different random subset each run; use the full set for a stable score.
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import torch

# Reuse the quantize -> torch_tensorrt.compile pipeline from the sibling example.
_THIS_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(_THIS_DIR))
import torch_tensorrt_ptq as ttptq  # noqa: E402

# Reuse the ImageNet accuracy harness from the onnx_ptq example (sibling dir).
_ONNX_PTQ_DIR = _THIS_DIR.parent / "onnx_ptq"
sys.path.insert(0, str(_ONNX_PTQ_DIR))
from evaluation import evaluate  # noqa: E402


class _EvalAdapter(torch.nn.Module):
    """Adapt a compiled/eager ViT to the ``onnx_ptq`` ``evaluate`` contract.

    ``evaluate_accuracy`` feeds float32 image batches, calls ``model(inputs)``,
    and reads ``outputs.data``. This adapter casts inputs to the model's compute
    dtype (the dataloader yields FP32) and unwraps an HF ``ImageClassifierOutput``
    to the bare logits tensor.
    """

    def __init__(self, model: torch.nn.Module, dtype: torch.dtype):
        super().__init__()
        self.model = model
        self._dtype = dtype

    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        out = self.model(pixel_values.to(self._dtype))
        return out.logits if hasattr(out, "logits") else out


def build_processor_transform(processor):
    """Return a ``PIL.Image -> (C, H, W) float tensor`` transform from the HF processor.

    Using the model's own image processor keeps eval preprocessing (resize,
    normalization mean/std) consistent with how the ViT was trained, which is
    more faithful for a HuggingFace checkpoint than a generic timm transform.
    The model and ``ILSVRC/imagenet-1k`` share the standard 1000-class ordering,
    so predicted indices line up with the dataset labels.
    """

    def _transform(image):
        return processor(images=image, return_tensors="pt")["pixel_values"][0]

    return _transform


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--model_id",
        default="google/vit-large-patch16-224",
        help="HuggingFace model id of the ViT classifier to quantize and score.",
    )
    parser.add_argument(
        "--recipe",
        default=ttptq.DEFAULT_RECIPE,
        help="Recipe path (relative to modelopt_recipes/ or an absolute YAML). "
        "Defaults to the ViT FP8 recipe; pass huggingface/vit/ptq/nvfp4 for NVFP4.",
    )
    parser.add_argument(
        "--calib_samples",
        type=int,
        default=1024,
        help="Number of tiny-imagenet samples to use for calibration.",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=128,
        help="Calibration / compile / eval batch size. The Torch-TRT engine is "
        "dynamic (min=1, opt=--batch_size, max=1024) and handles any batch incl. "
        "the trailing partial batch, so any --batch_size (e.g. 128) works.",
    )
    parser.add_argument(
        "--eval_data_size",
        type=int,
        default=None,
        help="Number of ImageNet validation images to score (default: full 50k).",
    )
    parser.add_argument(
        "--imagenet_path",
        default="ILSVRC/imagenet-1k",
        help="HF dataset card or local path to the ImageNet validation set (gated).",
    )
    parser.add_argument(
        "--baseline",
        action="store_true",
        help="Also score the unquantized model as a reference. It is "
        "Torch-TensorRT-compiled like the quantized model (or run eager under "
        "--skip_trt) so the comparison is apples-to-apples.",
    )
    parser.add_argument(
        "--skip_trt",
        action="store_true",
        help="Score the fake-quant (modelopt) model; skip torch_tensorrt.compile. "
        "Useful for environments without torch_tensorrt installed.",
    )
    parser.add_argument(
        "--results_path",
        default=None,
        help="If set, write the accuracy results to this CSV path.",
    )
    args = parser.parse_args()

    dynamic = True

    if not torch.cuda.is_available():
        raise SystemExit("This example requires a CUDA-capable GPU.")
    device = torch.device("cuda")
    dtype = torch.float16

    model, processor = ttptq.load_model_and_processor(args.model_id, device, dtype)
    transform = build_processor_transform(processor)

    def run_eval(m: torch.nn.Module) -> tuple[float, float]:
        top1, top5 = evaluate(
            _EvalAdapter(m, dtype),
            transform,
            batch_size=args.batch_size,
            num_examples=args.eval_data_size,
            device="cuda",
            dataset_path=args.imagenet_path,
        )
        return top1, top5

    image_size = model.config.image_size
    num_channels = model.config.num_channels
    example_input = torch.randn(
        args.batch_size, num_channels, image_size, image_size, device=device, dtype=dtype
    )
    runtime = "fake-quant" if args.skip_trt else "torch-trt"

    def to_eval_model(m: torch.nn.Module, what: str) -> torch.nn.Module:
        """Logits-wrap and (unless --skip_trt) Torch-TensorRT-compile ``m`` for eval.

        The baseline is compiled the same way as the quantized model so all
        reported numbers come from the same Torch-TRT runtime.
        """
        wrapped = ttptq.ViTLogitsWrapper(m).to(device).eval()
        if args.skip_trt:
            return wrapped
        print(f"\nCompiling {what} with Torch-TensorRT (dynamic={dynamic}) ...")
        return ttptq.compile_with_torch_tensorrt(wrapped, example_input, dynamic=dynamic)

    results: list[list[str | float]] = [["Metric", "Top1 (%)", "Top5 (%)"]]

    # Baseline must be built + scored before in-place quantization mutates `model`.
    if args.baseline:
        prec = str(dtype).rsplit(".", 1)[-1]  # e.g. "float16"
        base_tag = f"baseline-{prec} ({runtime})"
        base_eval = to_eval_model(model, "unquantized baseline")
        print(f"\n=== {base_tag} ===")
        top1, top5 = run_eval(base_eval)
        print(f"{base_tag}   top1={top1:.2f}%  top5={top5:.2f}%")
        results.append([base_tag, top1, top5])
        del base_eval
        torch.cuda.empty_cache()

    calib_batches = ttptq.build_calibration_loader(
        processor, args.calib_samples, args.batch_size, device, dtype
    )
    ttptq.quantize_with_recipe(model, args.recipe, calib_batches)

    label = Path(args.recipe).stem  # e.g. "fp8" / "nvfp4"
    tag = f"{label} ({runtime})"
    eval_model = to_eval_model(model, f"{label} model")
    print(f"\n=== {tag} ===")
    top1, top5 = run_eval(eval_model)
    print(f"{tag}   top1={top1:.2f}%  top5={top5:.2f}%")
    results.append([tag, top1, top5])

    if args.results_path:
        with open(args.results_path, "w", newline="") as f:
            csv.writer(f).writerows(results)
        print(f"\nWrote results to {args.results_path}")


if __name__ == "__main__":
    main()
