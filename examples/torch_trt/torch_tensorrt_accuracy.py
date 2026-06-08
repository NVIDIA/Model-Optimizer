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

    python torch_tensorrt_accuracy.py --precision fp8 --eval_data_size 5000 --baseline

``--imagenet_path`` defaults to the gated ``ILSVRC/imagenet-1k`` HF dataset
(accept its license / set ``HF_TOKEN``), or point it at a local copy. Note the
``evaluate`` API shuffles the validation set, so a partial ``--eval_data_size``
samples a different random subset each run; use the full set for a stable score.
"""

from __future__ import annotations

import argparse
import csv
import json
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
        "--precision",
        choices=sorted(ttptq.PRECISION_TO_RECIPE),
        default="fp8",
        help="Which ViT recipe variant to apply.",
    )
    parser.add_argument(
        "--recipe",
        default=None,
        help="Override the recipe path (relative to modelopt_recipes/ or absolute). "
        "If unset, the recipe is picked by --precision.",
    )
    parser.add_argument(
        "--calib_samples",
        type=int,
        default=128,
        help="Number of tiny-imagenet samples to use for calibration.",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=1,
        help="Calibration / compile / eval batch size. The Torch-TRT engine is "
        "compiled for this single static batch shape and the onnx_ptq evaluate() "
        "dataloader keeps the trailing partial batch, so the Torch-TRT path "
        "requires --batch_size 1; larger batches are only allowed with --skip_trt.",
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
        help="Also score the un-quantized BF16 model for a reference point.",
    )
    parser.add_argument(
        "--skip_trt",
        action="store_true",
        help="Score the fake-quant (modelopt) model; skip torch_tensorrt.compile. "
        "Useful for environments without torch_tensorrt installed.",
    )
    parser.add_argument(
        "--no_pretrained",
        action="store_true",
        help="Build the model from config with random weights (fast smoke test).",
    )
    parser.add_argument(
        "--model_kwargs",
        default=None,
        help="JSON string of ViTConfig overrides applied when --no_pretrained is set.",
    )
    parser.add_argument(
        "--results_path",
        default=None,
        help="If set, write the accuracy results to this CSV path.",
    )
    args = parser.parse_args()

    # The Torch-TRT engine is compiled for one static batch shape, and the reused
    # onnx_ptq evaluate() dataloader does not drop the trailing partial batch, so a
    # batch size that doesn't divide the validation set would crash the static
    # engine mid-run. Fail fast. The fake-quant (--skip_trt) path is a plain eager
    # module and tolerates any batch size.
    if not args.skip_trt and args.batch_size != 1:
        raise SystemExit(
            "The Torch-TensorRT path requires --batch_size 1 (the engine is compiled "
            "for a single static batch shape). Use --skip_trt to score the fake-quant "
            "model at a larger batch size."
        )

    if not torch.cuda.is_available():
        raise SystemExit("This example requires a CUDA-capable GPU.")
    device = torch.device("cuda")
    dtype = torch.float16

    config_overrides = json.loads(args.model_kwargs) if args.model_kwargs else None
    model, processor = ttptq.load_model_and_processor(
        args.model_id,
        device,
        dtype,
        pretrained=not args.no_pretrained,
        config_overrides=config_overrides,
    )
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

    results: list[list[str | float]] = [["Metric", "Top1 (%)", "Top5 (%)"]]

    # Baseline must run before in-place quantization mutates `model`.
    if args.baseline:
        prec = str(dtype).rsplit(".", 1)[-1]  # e.g. "float16"
        print(f"\n=== Baseline ({prec}) ===")
        top1, top5 = run_eval(model)
        print(f"baseline ({prec})   top1={top1:.2f}%  top5={top5:.2f}%")
        results.append([f"baseline_{prec}", top1, top5])

    calib_batches = ttptq.build_calibration_loader(
        processor, args.calib_samples, args.batch_size, device, dtype
    )
    recipe_path = args.recipe or ttptq.PRECISION_TO_RECIPE[args.precision]
    ttptq.quantize_with_recipe(model, recipe_path, calib_batches)

    wrapped = ttptq.ViTLogitsWrapper(model).to(device).eval()
    if args.skip_trt:
        print("\n--skip_trt set; scoring the fake-quant model (no Torch-TensorRT compile).")
        eval_model: torch.nn.Module = wrapped
        tag = f"{args.precision} (fake-quant)"
    else:
        image_size = model.config.image_size
        num_channels = model.config.num_channels
        example_input = torch.randn(
            args.batch_size, num_channels, image_size, image_size, device=device, dtype=dtype
        )
        eval_model = ttptq.compile_with_torch_tensorrt(wrapped, example_input)
        tag = f"{args.precision} (torch-trt)"

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
