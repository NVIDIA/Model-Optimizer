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

"""Quantize a HuggingFace ViT model with ModelOpt and deploy it with Torch-TensorRT.

Pipeline:

1. Load ``google/vit-large-patch16-224`` (`ViTForImageClassification`) from HF.
2. Build a calibration loader from `zh-plus/tiny-imagenet` (same pattern as the
   `torch_onnx` example) so the recipe runs end-to-end without ImageNet access.
3. Run ``mtq.quantize`` with one of the ViT-specific recipes
   (`modelopt_recipes/huggingface/vit/ptq/`). Two non-default variants are
   shipped:

   * ``fp8`` -> ``fp8_mha-classifier_skip``: W8A8 FP8 with an MHA-aware
     LayerNorm output quantizer, FP8 attention BMM/softmax slots, and the
     `classifier` head left in FP16.
   * ``nvfp4`` -> ``nvfp4_linear-fp8_conv-classifier_skip``: NVFP4 W4A4 on
     encoder Linear layers, FP8 override on the patch-embedding Conv2d (TRT
     has no NVFP4 kernel for 4D Conv inputs), AWQ-lite calibration, and the
     `classifier` head left in FP16.

4. Compile the quantized model with ``torch_tensorrt.compile`` (Dynamo IR,
   ``min_block_size=1``) and run an end-to-end sanity check + small benchmark
   against the eager BF16 baseline.

This script is intentionally CLI-driven and side-effect-free outside of the
optional ``--save_dir`` checkpoint. The quantized graph keeps Q/DQ nodes; the
TRT compile step is what turns them into TRT precision layers.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
from datasets import load_dataset
from transformers import AutoImageProcessor, ViTForImageClassification

import modelopt.torch.opt as mto
import modelopt.torch.quantization as mtq
from modelopt.recipe import ModelOptPTQRecipe, load_recipe
from modelopt.torch.quantization.utils import export_torch_mode

# Maps the user-facing precision flag to the ViT-specific recipe under
# `modelopt_recipes/huggingface/vit/ptq/`. The recipe loader resolves this
# relative path against the built-in recipe library.
PRECISION_TO_RECIPE: dict[str, str] = {
    "fp8": "huggingface/vit/ptq/fp8",
    "nvfp4": "huggingface/vit/ptq/nvfp4",
}


def load_model_and_processor(model_id: str, device: torch.device, dtype: torch.dtype):
    """Pull the HF ViT classifier and its preprocessor."""
    print(f"Loading {model_id} (dtype={dtype})...")
    processor = AutoImageProcessor.from_pretrained(model_id)
    model = ViTForImageClassification.from_pretrained(model_id, torch_dtype=dtype)
    model.eval().to(device)
    return model, processor


def build_calibration_loader(
    processor,
    num_samples: int,
    batch_size: int,
    device: torch.device,
    dtype: torch.dtype,
):
    """Build a calibration tensor stream from tiny-imagenet.

    tiny-imagenet avoids the gated `ILSVRC/imagenet-1k` repo so this example
    runs unauthenticated. Images go through the HF processor (resize + center
    crop + ImageNet normalization), which is exactly the eval-time transform
    used by the released `vit-large-patch16-224` checkpoint.
    """
    print(f"Loading calibration data ({num_samples} samples)...")
    dataset = load_dataset("zh-plus/tiny-imagenet", split="train")
    dataset = dataset.shuffle(seed=42).select(range(num_samples))

    tensors: list[torch.Tensor] = []
    for sample in dataset:
        image = sample["image"]
        if image.mode != "RGB":
            image = image.convert("RGB")
        # HF image processors emit `pixel_values` of shape (1, 3, H, W).
        pixel_values = processor(images=image, return_tensors="pt")["pixel_values"]
        tensors.append(pixel_values.squeeze(0))

    batched = torch.stack(tensors).to(device=device, dtype=dtype)
    return torch.split(batched, batch_size)


def quantize_with_recipe(model, recipe_path: str, calib_batches):
    """Resolve the YAML recipe and run `mtq.quantize`.

    Returns the quantized model. The graph still uses high-precision math at
    this point — Q/DQ nodes have been inserted around weights and activations
    and amax values populated, but no kernel substitution has happened yet.
    """
    print(f"Loading recipe: {recipe_path}")
    recipe = load_recipe(recipe_path)
    if not isinstance(recipe, ModelOptPTQRecipe):
        raise TypeError(f"Expected PTQ recipe, got {type(recipe).__name__}")
    quant_cfg = recipe.quantize.model_dump()

    def forward_loop(model_):
        with torch.no_grad():
            for batch in calib_batches:
                model_(pixel_values=batch)

    print("Running mtq.quantize ...")
    mtq.quantize(model, quant_cfg, forward_loop=forward_loop)
    mtq.print_quant_summary(model)
    return model


class ViTLogitsWrapper(torch.nn.Module):
    """Returns raw logits as a single tensor.

    HF's `ViTForImageClassification.forward` returns an `ImageClassifierOutput`
    dataclass. `torch_tensorrt.compile` (and `torch.export`) need a tensor-tree
    return, so we unwrap it here. The wrapper holds the quantized model as a
    submodule; Q/DQ nodes flow through unchanged.
    """

    def __init__(self, vit_model: torch.nn.Module):
        super().__init__()
        self.vit = vit_model

    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        return self.vit(pixel_values=pixel_values).logits


def compile_with_torch_tensorrt(model: torch.nn.Module, example_input: torch.Tensor):
    """Compile the quantized model with Torch-TensorRT (Dynamo IR).

    `min_block_size=1` follows the Torch-TRT quantization guide — it makes the
    partitioner accept single-node TRT subgraphs, which is what we want so the
    Q/DQ + matmul pairs become TRT precision layers instead of falling back to
    eager. The compile step expects fake-quant operators in the graph; we run
    it under `export_torch_mode` so modelopt's Q/DQ are exported in the
    TRT-friendly form.
    """
    import torch_tensorrt

    print("Compiling with torch_tensorrt.compile (Dynamo IR)...")
    with export_torch_mode():
        trt_model = torch_tensorrt.compile(
            model,
            ir="dynamo",
            arg_inputs=[example_input],
            min_block_size=1,
            # The recipes export weights in BF16; TRT picks the FP8/NVFP4
            # kernel from the Q/DQ pattern, not from this list.
            enabled_precisions={torch.bfloat16, torch.float16, torch.float32},
            truncate_double=True,
        )
    return trt_model


def benchmark(model: torch.nn.Module, example_input: torch.Tensor, n_warmup: int, n_iters: int):
    """Median-of-`n_iters` latency over `example_input`. CUDA-event timed."""
    torch.cuda.synchronize()
    with torch.no_grad():
        for _ in range(n_warmup):
            model(example_input)
    torch.cuda.synchronize()

    starts = [torch.cuda.Event(enable_timing=True) for _ in range(n_iters)]
    ends = [torch.cuda.Event(enable_timing=True) for _ in range(n_iters)]
    with torch.no_grad():
        for i in range(n_iters):
            starts[i].record()
            model(example_input)
            ends[i].record()
    torch.cuda.synchronize()
    times = sorted(s.elapsed_time(e) for s, e in zip(starts, ends))
    return times[len(times) // 2]


def _argmax_logits(out) -> torch.Tensor:
    """Handle either an HF `ImageClassifierOutput` or a raw tensor."""
    logits = out.logits if hasattr(out, "logits") else out
    return logits.argmax(dim=-1)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model_id",
        default="google/vit-large-patch16-224",
        help="HuggingFace model id of the ViT classifier to quantize.",
    )
    parser.add_argument(
        "--precision",
        choices=sorted(PRECISION_TO_RECIPE),
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
        help="Batch size for calibration / TRT compile / benchmarking.",
    )
    parser.add_argument(
        "--benchmark_iters",
        type=int,
        default=50,
        help="Number of timed iterations (after warmup) per benchmark phase.",
    )
    parser.add_argument(
        "--save_dir",
        type=str,
        default=None,
        help="If set, save the quantized modelopt state-dict here (BF16 weights "
        "+ Q/DQ metadata) — re-usable across runs without recalibration.",
    )
    parser.add_argument(
        "--skip_trt",
        action="store_true",
        help="Quantize + run the BF16-fake-quant model only; skip torch_tensorrt.compile. "
        "Useful for environments without torch_tensorrt installed.",
    )
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise SystemExit("This example requires a CUDA-capable GPU.")
    device = torch.device("cuda")
    # ViT-Large is a transformer in BF16 on the released checkpoint; the Q/DQ
    # nodes operate on top of BF16 master weights either way.
    dtype = torch.bfloat16

    model, processor = load_model_and_processor(args.model_id, device, dtype)
    image_size = model.config.image_size
    num_channels = model.config.num_channels
    example_input = torch.randn(
        args.batch_size, num_channels, image_size, image_size, device=device, dtype=dtype
    )

    # Baseline forward + benchmark for a comparison number that survives
    # quantization. argmax preserves the predicted-class check below.
    print("\n=== Baseline (BF16) ===")
    with torch.no_grad():
        baseline_pred = _argmax_logits(model(example_input))
    baseline_latency = benchmark(
        lambda x: model(x), example_input, n_warmup=5, n_iters=args.benchmark_iters
    )
    print(f"Baseline argmax class: {baseline_pred.tolist()}")
    print(f"Baseline latency: {baseline_latency:.3f} ms (median over {args.benchmark_iters} iters)")

    calib_batches = build_calibration_loader(
        processor, args.calib_samples, args.batch_size, device, dtype
    )

    recipe_path = args.recipe or PRECISION_TO_RECIPE[args.precision]
    quantize_with_recipe(model, recipe_path, calib_batches)

    if args.save_dir:
        save_path = Path(args.save_dir)
        save_path.mkdir(parents=True, exist_ok=True)
        ckpt = save_path / "vit_modelopt_state.pt"
        mto.save(model, ckpt)
        print(f"Saved quantized modelopt state to {ckpt}")

    print("\n=== Fake-quant (modelopt, BF16 math) ===")
    with torch.no_grad():
        fq_pred = _argmax_logits(model(example_input))
    fq_match = (fq_pred == baseline_pred).all().item()
    print(f"Quantized argmax class: {fq_pred.tolist()} (matches baseline: {fq_match})")

    if args.skip_trt:
        print("\n--skip_trt set; not compiling with Torch-TensorRT.")
        return

    wrapped = ViTLogitsWrapper(model).to(device).eval()
    trt_model = compile_with_torch_tensorrt(wrapped, example_input)

    print("\n=== Torch-TensorRT compiled ===")
    with torch.no_grad():
        trt_pred = trt_model(example_input).argmax(dim=-1)
    trt_match = (trt_pred == baseline_pred).all().item()
    trt_latency = benchmark(trt_model, example_input, n_warmup=5, n_iters=args.benchmark_iters)
    print(f"TRT argmax class: {trt_pred.tolist()} (matches baseline: {trt_match})")
    print(f"TRT latency: {trt_latency:.3f} ms (median over {args.benchmark_iters} iters)")
    speedup = baseline_latency / trt_latency if trt_latency > 0 else float("inf")
    print(f"\nSpeedup vs. BF16 baseline: {speedup:.2f}x")


if __name__ == "__main__":
    main()
