#!/usr/bin/env python3
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

"""
Script to set a fixed batch size for ONNX models.

This script modifies an ONNX model with dynamic batch size to use a fixed batch size,
which is often beneficial for TensorRT performance benchmarking.

Usage:
    python set_batch_size.py resnet50_Opset17.onnx --batch-size 128 --output resnet50.bs128.onnx
"""

import argparse
import sys

import onnx

from modelopt.onnx.utils import check_model, infer_shapes, save_onnx


def _validate_onnx_model_path(path: str) -> None:
    """Ensure the model path has a .onnx extension for consistent output path generation."""
    if not path.lower().endswith(".onnx"):
        print(f"Error: Model path must end with '.onnx', got: {path}", file=sys.stderr)
        sys.exit(1)


def _validate_batch_size(batch_size: int) -> None:
    """Ensure batch size is a positive integer to prevent invalid model configurations."""
    if batch_size < 1:
        print(f"Error: Batch size must be a positive integer, got: {batch_size}", file=sys.stderr)
        sys.exit(1)


def set_batch_size(model_path: str, batch_size: int, output_path: str) -> None:
    """
    Set a fixed batch size for an ONNX model.

    Args:
        model_path: Path to input ONNX model
        batch_size: Desired batch size
        output_path: Path to save modified model
    """
    # Load the model
    print(f"Loading model from {model_path}...")
    model = onnx.load(model_path)

    # Get the input tensor
    graph = model.graph
    input_tensor = graph.input[0]

    print(
        f"Original input shape: {[d.dim_param or d.dim_value for d in input_tensor.type.tensor_type.shape.dim]}"
    )

    # Modify the batch dimension (first dimension)
    if len(input_tensor.type.tensor_type.shape.dim) > 0:
        input_tensor.type.tensor_type.shape.dim[0].dim_value = batch_size
        # Clear any symbolic dimension parameter
        input_tensor.type.tensor_type.shape.dim[0].ClearField("dim_param")

    # Also update output shapes if needed
    for output_tensor in graph.output:
        if len(output_tensor.type.tensor_type.shape.dim) > 0:
            output_tensor.type.tensor_type.shape.dim[0].dim_value = batch_size
            output_tensor.type.tensor_type.shape.dim[0].ClearField("dim_param")

    print(
        f"Modified input shape: {[d.dim_param or d.dim_value for d in input_tensor.type.tensor_type.shape.dim]}"
    )

    # Run shape inference to propagate the batch size through the model
    # Use modelopt's infer_shapes to support models with external data and large models
    print("Running shape inference...")
    try:
        model = infer_shapes(model)
    except Exception as e:
        print(f"Warning: Shape inference failed: {e}")
        print("Continuing without shape inference...")

    # Save the modified model (handles external data and IR > max ORT supported)
    print(f"Saving modified model to {output_path}...")
    save_onnx(model, output_path)

    # Verify the saved model (handles external data and large models)
    print("Verifying model...")
    check_model(model)
    print("✓ Model saved and verified successfully!")


def main():
    parser = argparse.ArgumentParser(
        description="Set a fixed batch size for an ONNX model",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Set batch size to 128 for ResNet50
  python set_batch_size.py resnet50_Opset17.onnx --batch-size 128 --output resnet50.bs128.onnx

  # Set batch size to 1 for single-image inference
  python set_batch_size.py resnet50_Opset17.onnx --batch-size 1 --output resnet50.bs1.onnx
        """,
    )

    parser.add_argument("model", help="Path to input ONNX model")
    parser.add_argument(
        "--batch-size", "-b", type=int, default=128, help="Batch size to set (default: 128)"
    )
    parser.add_argument(
        "--output", "-o", help="Path to save modified model (default: <model>_bs<batch_size>.onnx)"
    )

    args = parser.parse_args()

    _validate_onnx_model_path(args.model)
    _validate_batch_size(args.batch_size)

    # Generate output path if not provided (requires .onnx extension, validated above)
    if args.output is None:
        parts = args.model.rsplit(".", 1)
        base_name = parts[0] if len(parts) == 2 else args.model
        args.output = f"{base_name}.bs{args.batch_size}.onnx"

    set_batch_size(args.model, args.batch_size, args.output)


if __name__ == "__main__":
    main()
