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

"""Quantize a HuggingFace ViT model with ModelOpt and compile with Torch-TensorRT.

Pipeline:

1. Load ``google/vit-large-patch16-224`` (`ViTForImageClassification`) from HF.
2. Build a calibration loader from `zh-plus/tiny-imagenet` so the recipe runs
   end-to-end without ImageNet access.
3. Run ``mtq.quantize`` with one of the ViT-specific recipes under
   `modelopt_recipes/huggingface/vit/ptq/` (FP8 or NVFP4).
4. Compile the quantized model with ``torch_tensorrt.compile(ir="dynamo",
   min_block_size=1)`` and verify the compiled-model argmax matches the
   fake-quant argmax on a sample input.

The quantized graph keeps Q/DQ nodes; the TRT compile step is what turns
them into TRT precision layers.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from datasets import load_dataset
from transformers import AutoImageProcessor, ViTConfig, ViTForImageClassification

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


def load_model_and_processor(
    model_id: str,
    device: torch.device,
    dtype: torch.dtype,
    pretrained: bool = True,
    config_overrides: dict | None = None,
):
    """Pull the HF ViT classifier and its preprocessor.

    With ``pretrained=False`` the model is built from a config with random
    weights (test path); ``config_overrides`` lets the caller shrink it
    (e.g. ``{"num_hidden_layers": 1, "hidden_size": 64, ...}``). The
    preprocessor is always loaded from ``model_id`` since it only carries
    a small JSON config.
    """
    print(f"Loading {model_id} (dtype={dtype}, pretrained={pretrained})...")
    processor = AutoImageProcessor.from_pretrained(model_id)
    if pretrained:
        model = ViTForImageClassification.from_pretrained(model_id, torch_dtype=dtype)
    else:
        config = ViTConfig.from_pretrained(model_id)
        for k, v in (config_overrides or {}).items():
            setattr(config, k, v)
        model = ViTForImageClassification(config).to(dtype)
    model.eval().to(device)
    return model, processor


def build_calibration_loader(
    processor,
    num_samples: int,
    batch_size: int,
    device: torch.device,
    dtype: torch.dtype,
):
    """Build a calibration tensor stream from tiny-imagenet."""
    print(f"Loading calibration data ({num_samples} samples)...")
    dataset = load_dataset("zh-plus/tiny-imagenet", split="train")
    dataset = dataset.shuffle(seed=42).select(range(num_samples))

    tensors: list[torch.Tensor] = []
    for sample in dataset:
        image = sample["image"]
        if image.mode != "RGB":
            image = image.convert("RGB")
        pixel_values = processor(images=image, return_tensors="pt")["pixel_values"]
        tensors.append(pixel_values.squeeze(0))

    batched = torch.stack(tensors).to(device=device, dtype=dtype)
    return torch.split(batched, batch_size)


def quantize_with_recipe(model, recipe_path: str, calib_batches):
    """Resolve the YAML recipe and run `mtq.quantize`."""
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
    return, so we unwrap it here.
    """

    def __init__(self, vit_model: torch.nn.Module):
        super().__init__()
        self.vit = vit_model

    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        return self.vit(pixel_values=pixel_values).logits


def compile_with_torch_tensorrt(model: torch.nn.Module, example_input: torch.Tensor):
    """Compile the quantized model with Torch-TensorRT (Dynamo IR, strongly-typed).

    `min_block_size=1` follows the Torch-TRT quantization guide so single-node
    Q/DQ + matmul subgraphs become TRT precision layers. `export_torch_mode`
    makes modelopt emit Q/DQ in the TRT-friendly form during `torch.export`.
    """
    import torch_tensorrt

    print("Compiling with torch_tensorrt.compile (Dynamo IR)...")
    # `aten.cat.default` is force-executed in PyTorch because torch_tensorrt
    # 2.10's cat converter chokes on the cls-token + patch-embedding concat
    # in HF ViT (BFloat16 path: `TypeError: Got unsupported ScalarType
    # BFloat16`; FP16 path: rank-(-1) TRT tensor that trips the downstream
    # `embeddings + position_embeddings` add). The cat is a tiny [1,1,H]
    # + [1,N,H] concat that runs once per forward, so falling back to
    # PyTorch costs essentially nothing.
    with export_torch_mode():
        trt_model = torch_tensorrt.compile(
            model,
            ir="dynamo",
            arg_inputs=[example_input],
            min_block_size=1,
            truncate_double=True,
            torch_executed_ops={torch.ops.aten.cat.default},
        )
    return trt_model


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
        help="Batch size for calibration / TRT compile.",
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
        help="Quantize + run the fake-quant model only; skip torch_tensorrt.compile. "
        "Useful for environments without torch_tensorrt installed.",
    )
    parser.add_argument(
        "--no_pretrained",
        action="store_true",
        help="Build the model from config with random weights instead of "
        "downloading pretrained weights. Useful for fast e2e tests.",
    )
    parser.add_argument(
        "--model_kwargs",
        type=str,
        default=None,
        help="JSON string of ViTConfig overrides applied when --no_pretrained "
        'is set (e.g. \'{"num_hidden_layers": 1, "hidden_size": 64, '
        '"intermediate_size": 128, "num_attention_heads": 2}\').',
    )
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise SystemExit("This example requires a CUDA-capable GPU.")
    device = torch.device("cuda")
    dtype = torch.bfloat16

    config_overrides = json.loads(args.model_kwargs) if args.model_kwargs else None
    model, processor = load_model_and_processor(
        args.model_id,
        device,
        dtype,
        pretrained=not args.no_pretrained,
        config_overrides=config_overrides,
    )
    image_size = model.config.image_size
    num_channels = model.config.num_channels
    example_input = torch.randn(
        args.batch_size, num_channels, image_size, image_size, device=device, dtype=dtype
    )

    print("\n=== Baseline (BF16) ===")
    with torch.no_grad():
        baseline_pred = _argmax_logits(model(example_input))
    print(f"Baseline argmax class: {baseline_pred.tolist()}")

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

    print("\n=== Fake-quant (modelopt) ===")
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
    print(f"TRT argmax class: {trt_pred.tolist()} (matches baseline: {trt_match})")


if __name__ == "__main__":
    main()
