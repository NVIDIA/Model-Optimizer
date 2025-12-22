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
import re
import sys
from pathlib import Path

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

"""
This script is used to quantize a timm model using dynamic quantization like MXFP8 or NVFP4,
or using auto quantization for optimal per-layer quantization.

The script will:
1. Given the model name, create a timm torch model.
2. Quantize the torch model in MXFP8, NVFP4, INT4_AWQ, or AUTO mode.
3. Export the quantized torch model to ONNX format.
"""


mp.set_start_method("spawn", force=True)  # Needed for data loader with multiple workers

QUANT_CONFIG_DICT = {
    "fp8": mtq.FP8_DEFAULT_CFG,
    "int8": mtq.INT8_DEFAULT_CFG,
    "mxfp8": mtq.MXFP8_DEFAULT_CFG,
    "nvfp4": mtq.NVFP4_DEFAULT_CFG,
    "int4_awq": mtq.INT4_AWQ_CFG,
}


def filter_func(name):
    """Filter function to exclude certain layers from quantization."""
    pattern = re.compile(
        r".*(time_emb_proj|time_embedding|conv_in|conv_out|conv_shortcut|add_embedding|"
        r"pos_embed|time_text_embed|context_embedder|norm_out|x_embedder|patch_embed).*"
    )
    return pattern.match(name) is not None


def load_calibration_data(model_name, data_size, batch_size, device, with_labels=False):
    """Load and prepare calibration data.

    Args:
        model_name: Name of the timm model
        data_size: Number of samples to load
        batch_size: Batch size for data loader
        device: Device to load data to
        with_labels: If True, return dict with 'image' and 'label' keys (for auto_quantize)
                    If False, return just the images (for standard quantize)
    """
    dataset = load_dataset("zh-plus/tiny-imagenet")
    model = timm.create_model(model_name, pretrained=True, num_classes=1000)
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


def quantize_model(model, config, data_loader=None):
    """Quantize the model using the given config and calibration data."""
    if data_loader is not None:

        def forward_loop(model):
            for batch in data_loader:
                model(batch)

        quantized_model = mtq.quantize(model, config, forward_loop=forward_loop)
    else:
        quantized_model = mtq.quantize(model, config)

    mtq.disable_quantizer(quantized_model, filter_func)
    return quantized_model


def forward_step(model, batch):
    """Forward step function for auto_quantize scoring."""
    return model(batch["image"])


def loss_func(output, batch):
    """Loss function for auto_quantize gradient computation."""
    return F.cross_entropy(output, batch["label"])


def auto_quantize_model(
    model,
    data_loader,
    quantization_formats,
    effective_bits=4.8,
    num_calib_steps=512,
    num_score_steps=128,
):
    """Auto-quantize the model using optimal per-layer quantization search.

    Args:
        model: PyTorch model to quantize
        data_loader: DataLoader with image-label dict batches
        quantization_formats: List of quantization format config names or dicts
        effective_bits: Target effective bits constraint
        num_calib_steps: Number of calibration steps
        num_score_steps: Number of scoring steps for sensitivity analysis

    Returns:
        Tuple of (quantized_model, search_state_dict)
    """
    constraints = {"effective_bits": effective_bits}

    # Convert string format names to actual config objects
    format_configs = []
    for fmt in quantization_formats:
        if isinstance(fmt, str):
            format_configs.append(getattr(mtq, fmt))
        else:
            format_configs.append(fmt)

    print(f"Starting auto-quantization search with {len(format_configs)} formats...")
    print(f"Effective bits constraint: {effective_bits}")
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
    )

    # Disable quantization for specified layers
    mtq.disable_quantizer(quantized_model, filter_func)

    return quantized_model, search_state


def get_model_input_shape(model):
    """Get the input shape from timm model configuration."""
    data_config = timm.data.resolve_model_data_config(model)
    input_size = data_config["input_size"]
    return tuple(input_size)


def main():
    parser = argparse.ArgumentParser(
        description="Quantize timm models to FP8, MXFP8, INT8, NVFP4, INT4_AWQ, or use AUTO quantization"
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
            "NVFP4_AWQ_LITE_CFG",
            "FP8_DEFAULT_CFG",
            "MXFP8_DEFAULT_CFG",
            "INT8_DEFAULT_CFG",
            "INT4_AWQ_CFG",
        ],
        default=["NVFP4_AWQ_LITE_CFG", "FP8_DEFAULT_CFG"],
        help="Quantization formats to search from for auto mode (e.g., NVFP4_AWQ_LITE_CFG FP8_DEFAULT_CFG)",
    )
    parser.add_argument(
        "--effective_bits",
        type=float,
        default=4.8,
        help="Target effective bits for auto quantization constraint. Default is 4.8.",
    )
    parser.add_argument(
        "--num_score_steps",
        type=int,
        default=128,
        help="Number of scoring steps for auto quantization. Default is 128.",
    )

    args = parser.parse_args()

    # Create model and move to appropriate device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = timm.create_model(args.timm_model_name, pretrained=True, num_classes=1000).to(device)

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

    # Quantize model based on mode
    if args.quantize_mode == "auto":
        # Auto quantization requires labels for loss computation
        data_loader = load_calibration_data(
            args.timm_model_name,
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
        )
    else:
        # Standard quantization - only load calibration data if needed
        config = QUANT_CONFIG_DICT[args.quantize_mode]
        if args.quantize_mode == "mxfp8":
            data_loader = None
        else:
            data_loader = load_calibration_data(
                args.timm_model_name,
                args.calibration_data_size,
                args.batch_size,
                device,
                with_labels=False,
            )

        quantized_model = quantize_model(model, config, data_loader)

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


if __name__ == "__main__":
    main()
