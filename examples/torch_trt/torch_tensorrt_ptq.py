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
3. Run ``mtq.quantize`` with the ViT-specific FP8 recipe under
   `modelopt_recipes/huggingface/vit/ptq/`.
4. Compile the quantized model with ``torch_tensorrt.compile(ir="dynamo",
   min_block_size=1)`` and verify the compiled-model argmax matches the
   fake-quant argmax on a sample input.

The quantized graph keeps Q/DQ nodes; the TRT compile step is what turns
them into TRT precision layers.
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

# Default ViT PTQ recipe under `modelopt_recipes/huggingface/vit/ptq/`. The
# recipe loader resolves this relative path against the built-in recipe library;
# pass `--recipe` for a different one.
DEFAULT_RECIPE = "huggingface/vit/ptq/fp8"


def load_model_and_processor(model_id: str, device: torch.device, dtype: torch.dtype):
    """Pull the HF ViT classifier and its preprocessor."""
    print(f"Loading {model_id} (dtype={dtype})...")
    processor = AutoImageProcessor.from_pretrained(model_id)
    # `gelu_fast` selects the tanh-approximation GELU rather than the erf-based
    # default. Eager attention runs softmax through `F.softmax` instead of the
    # fused SDPA kernel, so the recipe's attention softmax-P quantizer
    # (`p_bmm_quantizer` on HF attention) is exercised during calibration and
    # emits Q/DQ around the softmax output on export.
    model = ViTForImageClassification.from_pretrained(
        model_id,
        torch_dtype=dtype,
        hidden_act="gelu_fast",
        attn_implementation="eager",
    )
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
    """Compile the quantized model with Torch-TensorRT (Dynamo IR, strongly-typed)."""
    # Imported here (not at module scope) so the quantize-only `--skip_trt` path
    # still runs on hosts without torch_tensorrt installed.
    import torch_tensorrt

    print("Compiling with torch_tensorrt.compile (Dynamo IR, dynamic batch)...")
    n, c, h, w = example_input.shape
    # torch.export specializes a size-1 dynamic dim to a constant, so trace at
    # opt batch >= 2; min=1 still serves batch 1 at runtime.
    opt_n = max(int(n), 2)
    with export_torch_mode(), torch_tensorrt.dynamo.Debugger(log_level="error"):
        trt_model = torch_tensorrt.compile(
            model,
            ir="dynamo",
            min_block_size=1,
            truncate_double=True,
            inputs=[
                torch_tensorrt.Input(
                    min_shape=(1, c, h, w),
                    opt_shape=(opt_n, c, h, w),
                    max_shape=(1024, c, h, w),
                    dtype=example_input.dtype,
                )
            ],
        )
    return trt_model


def dump_trt_layer_info(trt_model: torch.nn.Module, path: Path) -> None:
    """Write the per-layer engine info of every TRT submodule to ``path``.

    A Dynamo-compiled module can hold several ``TorchTensorRTModule`` subgraphs
    (the parts that fell back to PyTorch sit between them), so we concatenate the
    ``get_layer_info()`` JSON of each.
    """
    import torch_tensorrt

    infos = [
        mod.get_layer_info()
        for _, mod in trt_model.named_modules()
        if isinstance(mod, torch_tensorrt.dynamo.runtime.TorchTensorRTModule)
    ]
    if not infos:
        print("No TorchTensorRTModule found; nothing to dump (whole graph fell back to PyTorch?).")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(infos))
    print(f"Wrote TRT layer info ({len(infos)} engine(s)) to {path}")


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
        "--recipe",
        default=DEFAULT_RECIPE,
        help="Recipe path (relative to modelopt_recipes/ or an absolute YAML). "
        "Defaults to the ViT FP8 recipe.",
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
        help="Batch size for calibration / TRT compile.",
    )
    parser.add_argument(
        "--save_dir",
        type=str,
        default="./modelopt_quantized",
        help="Directory to save the quantized modelopt state-dict (FP16 weights "
        "+ Q/DQ metadata) — re-usable across runs without recalibration.",
    )
    parser.add_argument(
        "--skip_trt",
        action="store_true",
        help="Quantize + run the fake-quant model only; skip torch_tensorrt.compile. "
        "Useful for environments without torch_tensorrt installed.",
    )
    parser.add_argument(
        "--layer_info_path",
        default=None,
        help="If set, write the compiled TRT engine's per-layer info "
        "(get_layer_info()) to this file.",
    )
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise SystemExit("This example requires a CUDA-capable GPU.")
    device = torch.device("cuda")
    dtype = torch.float16

    model, processor = load_model_and_processor(args.model_id, device, dtype)
    image_size = model.config.image_size
    num_channels = model.config.num_channels
    example_input = torch.randn(
        args.batch_size, num_channels, image_size, image_size, device=device, dtype=dtype
    )

    print("\n=== Baseline (FP16) ===")
    with torch.no_grad():
        baseline_pred = _argmax_logits(model(example_input))
    print(f"Baseline argmax class: {baseline_pred.tolist()}")

    calib_batches = build_calibration_loader(
        processor, args.calib_samples, args.batch_size, device, dtype
    )

    quantize_with_recipe(model, args.recipe, calib_batches)

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

    if args.layer_info_path:
        dump_trt_layer_info(trt_model, Path(args.layer_info_path))

    print("\n=== Torch-TensorRT compiled ===")
    with torch.no_grad():
        trt_pred = trt_model(example_input).argmax(dim=-1)
    trt_match = (trt_pred == baseline_pred).all().item()
    print(f"TRT argmax class: {trt_pred.tolist()} (matches baseline: {trt_match})")


if __name__ == "__main__":
    main()
