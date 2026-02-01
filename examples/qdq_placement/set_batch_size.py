#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

import onnx
from onnx import shape_inference


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
    print("Running shape inference...")
    try:
        model = shape_inference.infer_shapes(model)
    except Exception as e:
        print(f"Warning: Shape inference failed: {e}")
        print("Continuing without shape inference...")

    # Save the modified model
    print(f"Saving modified model to {output_path}...")
    onnx.save(model, output_path)

    # Verify the saved model
    print("Verifying model...")
    onnx.checker.check_model(output_path)
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

    # Generate output path if not provided
    if args.output is None:
        base_name = args.model.rsplit(".", 1)[0]
        args.output = f"{base_name}.bs{args.batch_size}.onnx"

    set_batch_size(args.model, args.batch_size, args.output)


if __name__ == "__main__":
    main()
