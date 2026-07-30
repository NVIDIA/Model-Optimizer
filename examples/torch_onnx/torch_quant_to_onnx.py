# SPDX-FileCopyrightText: Copyright (c) 2023-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

import argparse
import json
import re
import subprocess
import sys
import warnings
from pathlib import Path
from typing import Any

# Add onnx_ptq to path for shared modules
sys.path.insert(0, str(Path(__file__).parent.parent / "onnx_ptq"))

import timm
import torch
import torch.multiprocessing as mp
import torch.nn.functional as F
from datasets import load_dataset
from download_example_onnx import export_to_onnx
from evaluation import evaluate

import modelopt.torch.quantization as mtq
from modelopt.recipe import load_config, load_recipe
from modelopt.recipe.presets import MODEL_QUANT_PRESET_DIR
from modelopt.torch.quantization.config import QuantizeConfig
from modelopt.torch.quantization.nn import TensorQuantizer
from modelopt.torch.quantization.plugins.custom import CUSTOM_POST_CONVERSION_PLUGINS

"""
Quantize a timm vision model and export to ONNX for TensorRT deployment.

Supports FP8, INT8, MXFP8, NVFP4, and AUTO (mixed-precision) quantization modes end-to-end
(quantize + ONNX export + TRT build). INT4_AWQ is quantize/export-only; it is not compatible
with ``--trt_build``.

The script will:
1. Load a pretrained timm model (e.g., ViT, Swin, ResNet).
2. Quantize the model using the specified mode. For models with Conv2d layers,
   Conv2d quantization is automatically overridden for TensorRT compatibility
   (FP8 for MXFP8/NVFP4, INT8 for INT4_AWQ).
3. Export the quantized model to ONNX with FP16 weights.
4. Optionally evaluate accuracy on ImageNet-1k before and after quantization.
"""


mp.set_start_method("spawn", force=True)  # Needed for data loader with multiple workers


def load_quant_config(recipe: str) -> dict:
    """Load a quantization config from a recipe YAML.

    ``recipe`` is either a preset basename under
    ``modelopt_recipes/configs/ptq/presets/model/`` (e.g. ``nvfp4``) or a path to a
    ``QuantizeConfig`` YAML file (filesystem or built-in recipe library).
    """
    if "/" not in recipe and not recipe.endswith((".yml", ".yaml")):
        recipe = f"{MODEL_QUANT_PRESET_DIR}/{recipe}"
    return load_config(recipe, schema_type=QuantizeConfig).model_dump()


_RESNET_RECIPES = {
    "fp8": "timm/resnet/ptq/fp8",
    "int8": "timm/resnet/ptq/int8",
    "auto": "timm/resnet/auto_quantize/fp8_int8_at_8p0bits",
}


_FP8_CONV_OVERRIDE: list = [
    {
        "parent_class": "nn.Conv2d",
        "quantizer_name": "*weight_quantizer",
        "cfg": {"num_bits": (4, 3), "axis": None},
    },
    {
        "parent_class": "nn.Conv2d",
        "quantizer_name": "*input_quantizer",
        "cfg": {"num_bits": (4, 3), "axis": None},
    },
]

_INT8_CONV_OVERRIDE: list = [
    {
        "parent_class": "nn.Conv2d",
        "quantizer_name": "*weight_quantizer",
        "cfg": {"num_bits": 8, "axis": 0},
    },
    {
        "parent_class": "nn.Conv2d",
        "quantizer_name": "*input_quantizer",
        "cfg": {"num_bits": 8, "axis": None},
    },
]

# FP8 MHA-aware config entries: quantize LayerNorm output so TRT can fuse the shared
# Q/DQ across all downstream Q/K/V/FC consumers. Softmax-output Q/DQ is handled by the
# FP8 ONNX exporter's post-processing pass (fixed 1/448 scale, data-independent).
_FP8_MHA_OVERRIDE: list = [
    {
        "parent_class": "nn.LayerNorm",
        "quantizer_name": "*output_quantizer",
        "cfg": {"num_bits": (4, 3), "axis": None},
    },
    {
        "parent_class": "nn.LayerNorm",
        "quantizer_name": "*input_quantizer",
        "enable": False,
    },
]

# Auto-quantize format presets that use block quantization and need Conv2d overrides for TRT.
# TRT DynamicQuantize requires 2D/3D input, but Conv2d operates on 4D tensors.
_NEEDS_FP8_CONV_OVERRIDE: set[str] = {
    "nvfp4_awq_lite",
    "nvfp4",
    "mxfp8",
}
_NEEDS_INT8_CONV_OVERRIDE: set[str] = {"int4_awq"}


def get_quant_config(quantize_mode, recipe=None):
    """Get quantization config, overriding Conv2d for TRT compatibility.

    The config is loaded from ``recipe`` when given, else from the preset YAML
    matching ``quantize_mode``. TensorRT only supports FP8 and INT8 for Conv layers.
    - For FP8: add MHA-aware LayerNorm output quantizer so TRT fuses shared Q/DQ into
      downstream attention matmuls. Softmax-output Q/DQ is inserted by the FP8 ONNX
      exporter's post-processing (fixed 1/448 scale, no calibration needed).
    - For MXFP8, NVFP4: override Conv2d to FP8
    - For INT4_AWQ: override Conv2d to INT8
    """
    config: dict = load_quant_config(recipe or quantize_mode)
    if quantize_mode == "fp8":
        config["quant_cfg"].extend(_FP8_MHA_OVERRIDE)
    elif quantize_mode in ("mxfp8", "nvfp4"):
        warnings.warn(
            f"TensorRT only supports FP8/INT8 for Conv layers. "
            f"Overriding Conv2d quantization to FP8 for '{quantize_mode}' mode."
        )
        config["quant_cfg"].extend(_FP8_CONV_OVERRIDE)
    elif quantize_mode == "int4_awq":
        warnings.warn(
            "TensorRT only supports FP8/INT8 for Conv layers. "
            "Overriding Conv2d quantization to INT8 for 'int4_awq' mode."
        )
        config["quant_cfg"].extend(_INT8_CONV_OVERRIDE)
    return config


def _get_resnet_recipe(model, quantize_mode):
    if not isinstance(model, timm.models.resnet.ResNet):
        return None
    recipe_path = _RESNET_RECIPES.get(quantize_mode)
    return load_recipe(recipe_path) if recipe_path is not None else None


def _add_resnet_residual_quantizers(model):
    block_types = (timm.models.resnet.BasicBlock, timm.models.resnet.Bottleneck)
    for block in (module for module in model.modules() if isinstance(module, block_types)):
        if block.downsample is None:
            block.downsample = torch.nn.Sequential()
        elif not isinstance(block.downsample, torch.nn.Sequential):
            block.downsample = torch.nn.Sequential(block.downsample)
        if "residual_quantizer" in block.downsample._modules:
            continue
        residual_quantizer = TensorQuantizer()
        residual_quantizer.disable()
        block.downsample.add_module("residual_quantizer", residual_quantizer)


def filter_func(name):
    """Filter function to exclude certain layers from quantization.

    ``downsample.reduction`` (Swin/SwinV2) is excluded because it operates on 4D tensors
    and TRT's DynamicQuantize layer (used for MXFP8/NVFP4) requires 2D/3D input.
    Other 4D-input layers (e.g. Swin's ``norm1``, ``downsample.norm``, top-level ``norm``)
    are handled dynamically by ``_disable_high_rank_input_quantizers`` via a forward-pass
    rank probe — that avoids false positives on ViT, whose same-named ``norm`` sees 3D input.
    """
    pattern = re.compile(
        r".*(time_emb_proj|time_embedding|conv_in|conv_out|conv_shortcut|add_embedding|"
        r"pos_embed|time_text_embed|context_embedder|norm_out|x_embedder|patch_embed|cpb_mlp|"
        r"maxpool|global_pool|downsample\.reduction).*"
    )
    return pattern.match(name) is not None


def _disable_high_rank_input_quantizers(model, input_shape, device):
    """Disable quantizers on Linear/LayerNorm modules that receive 4D+ input.

    TRT's MXFP8/NVFP4 ``DynamicQuantize`` op only supports 2D/3D input, so Swin's
    per-block ``norm1``, ``downsample.norm``, and top-level ``norm`` (all 4D in Swin
    but 3D in ViT) must be skipped. A forward pass with hooks identifies them at
    runtime, so this works across architectures without hardcoded paths.
    """
    high_rank: set[str] = set()
    handles = []
    for name, mod in model.named_modules():
        if isinstance(mod, (torch.nn.Linear, torch.nn.LayerNorm)):

            def hook(m, inp, out, _n=name):
                if inp and hasattr(inp[0], "ndim") and inp[0].ndim > 3:
                    high_rank.add(_n)

            handles.append(mod.register_forward_hook(hook))

    was_training = model.training
    model.eval()
    try:
        with torch.no_grad():
            model(torch.randn(input_shape, device=device))
    finally:
        for h in handles:
            h.remove()
        model.train(was_training)

    if not high_rank:
        return
    prefixes = tuple(n + "." for n in high_rank)
    mtq.disable_quantizer(model, lambda n: n.startswith(prefixes))


def _disable_low_channel_conv_input_quantizers(model):
    """Disable ``input_quantizer`` on Conv2d modules whose ``in_channels <= 3``.

    The first Conv2d of an image backbone (e.g. ResNet50's ``conv1``) consumes raw
    RGB input, so ``in_channels == 3``. On Blackwell (compute capability 12.0) TRT
    fails to find an FP8/MXFP8/NVFP4 tactic for this first-layer Q→Conv fusion:

        Error Code 10: Could not find any implementation for node
        /conv1/input_quantizer/TRT_FP8QuantizeLinear ... [ElementWise]

    Ada (8.9) happens to have a tactic, which is why local runs pass. Disabling the
    input quantizer on the raw-RGB conv is also standard quantization practice —
    first/last layers are typically left in higher precision. Weight quantization
    still applies. Swin/ViT's ``patch_embed.proj`` is already excluded via
    ``filter_func``'s ``patch_embed`` pattern, so this helper is effectively the
    ResNet-shaped analogue.
    """
    for _, mod in model.named_modules():
        if isinstance(mod, torch.nn.Conv2d) and mod.in_channels <= 3:
            q = getattr(mod, "input_quantizer", None)
            if q is not None and q.is_enabled:
                q.disable()


def load_calibration_data(model, data_size, batch_size, device, with_labels=False):
    """Load and prepare calibration data.

    Args:
        model: The timm model being quantized; used to derive the calibration transforms so the
               data pipeline matches the exact model config (respects --no_pretrained and
               --model_kwargs).
        data_size: Number of samples to load
        batch_size: Batch size for data loader
        device: Device to load data to
        with_labels: If True, return dict with 'image' and 'label' keys (for auto_quantize)
                    If False, return just the images (for standard quantize)
    """
    dataset = load_dataset("zh-plus/tiny-imagenet")
    data_config = timm.data.resolve_model_data_config(model)
    transforms = timm.data.create_transform(**data_config, is_training=False)

    images = dataset["train"][:data_size]["image"]
    calib_tensor = [transforms(img) for img in images]
    calib_tensor = [t.to(device) for t in calib_tensor]

    if with_labels:
        labels = dataset["train"][:data_size]["label"]
        labels = torch.tensor(labels, device=device)
        calib_dataset = [{"image": img, "label": lbl} for img, lbl in zip(calib_tensor, labels)]
        return torch.utils.data.DataLoader(
            calib_dataset, batch_size=batch_size, shuffle=True, num_workers=4
        )
    else:
        return torch.utils.data.DataLoader(
            calib_tensor, batch_size=batch_size, shuffle=True, num_workers=4
        )


def _disable_dead_quantizers(model):
    """Disable quantizers whose calibrated ``amax`` is non-positive or NaN.

    ``export_fp8`` computes ``scale = 448 / amax`` and blows up on ``amax == 0``.
    This shows up on SwinV2 with ``--no_pretrained``: timm's ``res-post-norm`` scheme
    zero-inits each block's ``norm1``/``norm2`` weight and bias, so those LayerNorm
    outputs are exactly zero at init and the MHA override's output_quantizer
    calibrates to ``amax == 0``. Disable such dead quantizers — they have nothing
    meaningful to quantize and would otherwise break ONNX export.
    """
    for _, mod in model.named_modules():
        for attr in ("input_quantizer", "output_quantizer", "weight_quantizer"):
            q = getattr(mod, attr, None)
            if q is None or not q.is_enabled:
                continue
            amax = q.amax
            if amax is None or not torch.is_tensor(amax):
                continue
            if torch.any(torch.isnan(amax)) or torch.all(amax <= 0):
                q.disable()


def _calibrate_uncalibrated_quantizers(model, data_loader):
    """Calibrate FP8 quantizers that weren't calibrated by mtq.quantize().

    When MXFP8/NVFP4 modes override Conv2d to FP8, the FP8 quantizers may not
    be calibrated because the MXFP8/NVFP4 quantization pipeline skips standard
    calibration. This function explicitly calibrates those uncalibrated quantizers.
    """
    uncalibrated = []
    for _, module in model.named_modules():
        for attr_name in ("input_quantizer", "weight_quantizer"):
            if not hasattr(module, attr_name):
                continue
            quantizer = getattr(module, attr_name)
            if quantizer.is_enabled and not quantizer.block_sizes and quantizer.amax is None:
                quantizer.enable_calib()
                uncalibrated.append(quantizer)

    if not uncalibrated:
        return

    model.eval()
    with torch.no_grad():
        for batch in data_loader:
            model(batch)

    for quantizer in uncalibrated:
        quantizer.disable_calib()
        quantizer.load_calib_amax(strict=False)


def quantize_model(model, config, data_loader=None):
    """Quantize the model using the given config and calibration data."""
    if data_loader is not None:

        def forward_loop(model):
            for batch in data_loader:
                model(batch)

        quantized_model = mtq.quantize(model, config, forward_loop=forward_loop)
    else:
        quantized_model = mtq.quantize(model, config)

    # Disable filtered quantizers BEFORE calibrating override quantizers so we don't
    # waste time calibrating quantizers that are about to be turned off.
    mtq.disable_quantizer(quantized_model, filter_func)

    # Calibrate any FP8 override quantizers that weren't calibrated by mtq.quantize().
    if data_loader is not None:
        _calibrate_uncalibrated_quantizers(quantized_model, data_loader)

    # Drop quantizers whose calibration saw only zeros (e.g. SwinV2 zero-init norm1/norm2)
    # so ``export_fp8`` doesn't divide by zero.
    _disable_dead_quantizers(quantized_model)

    return quantized_model


def forward_step(model, batch):
    """Forward step function for auto_quantize scoring."""
    return model(batch["image"])


def loss_func(output, batch):
    """Loss function for auto_quantize gradient computation."""
    return F.cross_entropy(output, batch["label"])


def _disable_inplace_relu(model):
    """Replace inplace ReLU with non-inplace ReLU throughout the model.

    This is needed for auto_quantize which uses backward hooks for gradient-based
    sensitivity scoring. Inplace ReLU on views created by custom Functions causes
    PyTorch autograd errors.
    """
    for module in model.modules():
        if isinstance(module, torch.nn.ReLU) and module.inplace:
            module.inplace = False


def auto_quantize_model(
    model,
    data_loader,
    quantization_formats,
    effective_bits=None,
    num_calib_steps=512,
    num_score_steps=128,
    recipe=None,
):
    """Auto-quantize the model using optimal per-layer quantization search.

    Args:
        model: PyTorch model to quantize
        data_loader: DataLoader with image-label dict batches
        quantization_formats: List of quantization recipe names (preset basenames,
            e.g. ``nvfp4_awq_lite``) or config dicts
        effective_bits: Target effective bits constraint
        num_calib_steps: Number of calibration steps
        num_score_steps: Number of scoring steps for sensitivity analysis

    Returns:
        Tuple of (quantized_model, search_state_dict)
    """
    _disable_inplace_relu(model)
    if recipe is None:
        constraints = {"effective_bits": 4.8 if effective_bits is None else effective_bits}
        format_configs: list[dict[str, Any] | str] = []
        for fmt in quantization_formats:
            if isinstance(fmt, str):
                config = load_quant_config(fmt)
                if fmt in _NEEDS_FP8_CONV_OVERRIDE:
                    config["quant_cfg"].extend(_FP8_CONV_OVERRIDE)
                elif fmt in _NEEDS_INT8_CONV_OVERRIDE:
                    config["quant_cfg"].extend(_INT8_CONV_OVERRIDE)
                format_configs.append(config)
            else:
                format_configs.append(fmt)
        fixed_quantization_config = None
        module_search_spaces = None
        disabled_layers = None
        method = "gradient"
    else:
        auto_config = recipe.auto_quantize
        constraints = auto_config.constraints.model_dump(exclude_none=True)
        if effective_bits is not None:
            constraints["effective_bits"] = effective_bits
        format_configs = []
        fixed_quantization_config = recipe.quantize.model_dump()
        module_search_spaces = [
            {
                "module_name_patterns": search_space.module_name_patterns,
                "quantization_formats": [
                    candidate.model_dump() for candidate in search_space.candidate_formats
                ],
                "allow_no_quant": search_space.allow_no_quant,
            }
            for search_space in auto_config.module_search_spaces
        ]
        disabled_layers = auto_config.disabled_layers
        method = auto_config.auto_quantize_method

    format_count = len(format_configs) or sum(
        len(search_space["quantization_formats"]) for search_space in module_search_spaces or []
    )
    print(f"Starting auto-quantization search with {format_count} formats...")
    print(f"Effective bits constraint: {constraints['effective_bits']}")
    print(f"Calibration steps: {num_calib_steps}, Scoring steps: {num_score_steps}")

    quantized_model, search_state = mtq.auto_quantize(
        model,
        constraints=constraints,
        quantization_formats=format_configs,
        data_loader=data_loader,
        forward_step=forward_step,
        loss_func=loss_func,
        num_calib_steps=num_calib_steps,
        num_score_steps=num_score_steps,
        verbose=True,
        fixed_quantization_config=fixed_quantization_config,
        module_search_spaces=module_search_spaces,
        disabled_layers=disabled_layers,
        method=method,
    )

    # Disable quantization for specified layers
    mtq.disable_quantizer(quantized_model, filter_func)

    _disable_dead_quantizers(quantized_model)

    return quantized_model, search_state


def get_model_input_shape(model):
    """Get the input shape from timm model configuration."""
    data_config = timm.data.resolve_model_data_config(model)
    input_size = data_config["input_size"]
    return tuple(input_size)


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Quantize timm models to FP8, MXFP8, INT8, NVFP4, or use AUTO quantization. "
            "INT4_AWQ is supported for quantize/export only and is not compatible with --trt_build."
        )
    )

    # Model hyperparameters
    parser.add_argument(
        "--timm_model_name",
        default="vit_base_patch16_224",
        help="The timm model name to quantize.",
        type=str,
    )
    parser.add_argument(
        "--quantize_mode",
        choices=["fp8", "mxfp8", "int8", "nvfp4", "int4_awq", "auto"],
        default="mxfp8",
        help="Type of quantization to apply. Default is MXFP8.",
    )
    parser.add_argument(
        "--recipe",
        type=str,
        default=None,
        help=(
            "Quantization config recipe: a preset basename under "
            "modelopt_recipes/configs/ptq/presets/model/ or a path to a QuantizeConfig "
            "YAML. Defaults to the preset matching --quantize_mode. Not supported with "
            "--quantize_mode=auto."
        ),
    )
    parser.add_argument(
        "--onnx_save_path",
        required=True,
        help="The save path to save the ONNX model.",
        type=str,
    )
    parser.add_argument(
        "--calibration_data_size",
        type=int,
        default=512,
        help="Number of images to use in calibration [1-512]",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=1,
        help="Batch size for calibration and ONNX model export.",
    )
    parser.add_argument(
        "--evaluate",
        action="store_true",
        help="Evaluate the base and quantized models on ImageNet validation set.",
    )
    parser.add_argument(
        "--eval_data_size",
        type=int,
        default=None,
        help="Number of samples to use for evaluation. If None, use entire validation set.",
    )

    # Auto quantization specific arguments
    parser.add_argument(
        "--auto_quantization_formats",
        nargs="+",
        choices=[
            "nvfp4_awq_lite",
            "fp8",
            "mxfp8",
            "int8",
            "int4_awq",
        ],
        default=["nvfp4_awq_lite", "fp8"],
        help="Quantization preset recipes to search from for auto mode (e.g., nvfp4_awq_lite fp8)",
    )
    parser.add_argument(
        "--effective_bits",
        type=float,
        default=None,
        help=(
            "Target effective bits for auto quantization. Defaults to 4.8 without a recipe "
            "and overrides the ResNet recipe when provided."
        ),
    )
    parser.add_argument(
        "--num_score_steps",
        type=int,
        default=128,
        help="Number of scoring steps for auto quantization. Default is 128.",
    )
    parser.add_argument(
        "--trt_build",
        action="store_true",
        help="Build a TensorRT engine from the exported ONNX model using trtexec.",
    )
    parser.add_argument(
        "--no_pretrained",
        action="store_true",
        help="Don't load pretrained weights (useful for testing with random weights).",
    )
    parser.add_argument(
        "--model_kwargs",
        type=str,
        default=None,
        help="JSON string of extra model kwargs (e.g., '{\"depth\": 1}').",
    )

    args = parser.parse_args()

    if args.recipe and args.quantize_mode == "auto":
        parser.error(
            "--recipe is not supported with --quantize_mode=auto; "
            "use --auto_quantization_formats instead."
        )

    # Create model and move to appropriate device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_kwargs = json.loads(args.model_kwargs) if args.model_kwargs else {}
    model = timm.create_model(
        args.timm_model_name,
        pretrained=not args.no_pretrained,
        num_classes=1000,
        **model_kwargs,
    ).to(device)

    # Get input shape from model config
    input_size = get_model_input_shape(model)
    input_shape = (args.batch_size, *input_size)

    # Evaluate base model if requested
    if args.evaluate:
        print("\n=== Evaluating Base Model ===")
        data_config = timm.data.resolve_model_data_config(model)
        transforms = timm.data.create_transform(**data_config, is_training=False)
        top1, top5 = evaluate(
            model, transforms, batch_size=args.batch_size, num_examples=args.eval_data_size
        )
        print(f"Base Model - Top-1 Accuracy: {top1:.2f}%, Top-5 Accuracy: {top5:.2f}%")

    resnet_recipe = None if args.recipe else _get_resnet_recipe(model, args.quantize_mode)
    if resnet_recipe is not None:
        CUSTOM_POST_CONVERSION_PLUGINS.add(_add_resnet_residual_quantizers)

    # Quantize model based on mode
    if args.quantize_mode == "auto":
        # Auto quantization requires labels for loss computation
        data_loader = load_calibration_data(
            model,
            args.calibration_data_size,
            args.batch_size,
            device,
            with_labels=True,
        )

        quantized_model, _ = auto_quantize_model(
            model,
            data_loader,
            args.auto_quantization_formats,
            args.effective_bits,
            args.calibration_data_size,
            args.num_score_steps,
            recipe=resnet_recipe,
        )
    else:
        # Standard quantization - load calibration data
        # Note: MXFP8 is dynamic and does not need calibration itself, but when
        # Conv2d layers are overridden to FP8 (for TRT compatibility), those FP8
        # quantizers require calibration data.
        config = (
            resnet_recipe.quantize.model_dump()
            if resnet_recipe is not None
            else get_quant_config(args.quantize_mode, args.recipe)
        )

        data_loader = load_calibration_data(
            model,
            args.calibration_data_size,
            args.batch_size,
            device,
            with_labels=False,
        )

        quantized_model = quantize_model(model, config, data_loader)

    # MXFP8/NVFP4 lower their input quantizers to TRT DynamicQuantize (2D/3D only).
    # Disable quantizers on 4D-input layers (Swin's norm1 / downsample.norm / top-level norm).
    # Auto mode also needs this when an MXFP8/NVFP4 candidate format is in the search set.
    uses_dynamic_quantize = args.quantize_mode in ("mxfp8", "nvfp4") or (
        args.quantize_mode == "auto"
        and resnet_recipe is None
        and any(fmt in _NEEDS_FP8_CONV_OVERRIDE for fmt in args.auto_quantization_formats)
    )
    if uses_dynamic_quantize:
        _disable_high_rank_input_quantizers(quantized_model, input_shape, device)

    # FP8-family modes emit TRT_FP8QuantizeLinear on the first-layer conv; Blackwell has
    # no tactic for that 3-channel Q→Conv fusion. Skip for pure INT8 (unaffected).
    uses_fp8_conv_input = args.quantize_mode in ("fp8", "mxfp8", "nvfp4") or (
        args.quantize_mode == "auto"
        and (
            resnet_recipe is not None
            or any(fmt not in {"int8", "int4_awq"} for fmt in args.auto_quantization_formats)
        )
    )
    if uses_fp8_conv_input:
        _disable_low_channel_conv_input_quantizers(quantized_model)

    # Print quantization summary
    print("\nQuantization Summary:")
    mtq.print_quant_summary(quantized_model)

    # Evaluate quantized model if requested
    if args.evaluate:
        print("\n=== Evaluating Quantized Model ===")
        data_config = timm.data.resolve_model_data_config(quantized_model)
        transforms = timm.data.create_transform(**data_config, is_training=False)
        top1, top5 = evaluate(
            quantized_model,
            transforms,
            batch_size=args.batch_size,
            num_examples=args.eval_data_size,
        )
        print(f"Quantized Model - Top-1 Accuracy: {top1:.2f}%, Top-5 Accuracy: {top5:.2f}%")

    # Export to ONNX
    export_to_onnx(
        quantized_model,
        input_shape,
        args.onnx_save_path,
        device,
        weights_dtype="fp16",
    )

    print(f"Quantized ONNX model is saved to {args.onnx_save_path}")

    if args.trt_build:
        build_trt_engine(args.onnx_save_path)


def build_trt_engine(onnx_path):
    """Build a TensorRT engine from the exported ONNX model using trtexec."""
    cmd = [
        "trtexec",
        f"--onnx={onnx_path}",
        "--stronglyTyped",
        "--builderOptimizationLevel=4",
    ]
    print(f"\nBuilding TensorRT engine: {' '.join(cmd)}")
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
    except FileNotFoundError as e:
        raise RuntimeError(
            "trtexec not found on PATH; install TensorRT or drop --trt_build."
        ) from e
    except subprocess.TimeoutExpired as e:
        raise RuntimeError(f"trtexec timed out building {onnx_path} after 600s.") from e
    if result.returncode != 0:
        raise RuntimeError(
            f"TensorRT engine build failed for {onnx_path}:\n{result.stdout}\n{result.stderr}"
        )
    print("TensorRT engine build succeeded.")


if __name__ == "__main__":
    main()
