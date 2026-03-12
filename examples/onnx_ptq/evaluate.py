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

import argparse
import csv

import timm
from evaluation import evaluate

from modelopt.torch._deploy._runtime import RuntimeRegistry
from modelopt.torch._deploy.device_model import DeviceModel
from modelopt.torch._deploy.utils import OnnxBytes


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--onnx_path",
        type=str,
        required=True,
        help="""Path to the image classification ONNX model with input shape of
        [batch_size,3,224,224] and output shape of [1,1000]""",
    )
    parser.add_argument(
        "--model_name",
        type=str,
        required=True,
        help="use for timm.create_model to load data config",
    )
    parser.add_argument(
        "--engine_path",
        type=str,
        default=None,
        help="Path to the TensorRT engine",
    )
    parser.add_argument(
        "--timing_cache_path",
        type=str,
        default=None,
        help="Path to the TensorRT timing cache",
    )
    parser.add_argument(
        "--imagenet_path",
        type=str,
        default="ILSVRC/imagenet-1k",
        help="HF dataset card or local path to the ImageNet dataset",
    )
    parser.add_argument("--batch_size", type=int, default=1, help="Batch size for evaluation")
    parser.add_argument(
        "--eval_data_size", type=int, default=None, help="Number of examples to evaluate"
    )
    parser.add_argument(
        "--engine_precision",
        type=str,
        default="stronglyTyped",
        choices=["best", "fp16", "stronglyTyped"],
        help="Precision mode for the TensorRT engine. \
            stronglyTyped is recommended, all other modes have been deprecated in TensorRT",
    )
    parser.add_argument(
        "--results_path", type=str, default=None, help="Save the results to the specified path"
    )

    args = parser.parse_args()
    deployment = {
        "runtime": "TRT",
        "precision": args.engine_precision,
    }

    # Create an ONNX bytes object with the specified path
    onnx_bytes = OnnxBytes(args.onnx_path).to_bytes()

    # Get the runtime client
    client = RuntimeRegistry.get(deployment)

    # Compile the ONNX model to TRT engine and create the device model
    compilation_args = {
        "engine_path": args.engine_path,
        "timing_cache_path": args.timing_cache_path,
    }
    compiled_model = client.ir_to_compiled(onnx_bytes, compilation_args)
    device_model = DeviceModel(client, compiled_model, metadata={})

    top1_accuracy, top5_accuracy = 0.0, 0.0
    model = timm.create_model(args.model_name, pretrained=False, num_classes=1000)
    data_config = timm.data.resolve_model_data_config(model)
    transforms = timm.data.create_transform(**data_config, is_training=False)

    top1_accuracy, top5_accuracy = evaluate(
        device_model,
        transforms,
        batch_size=args.batch_size,
        num_examples=args.eval_data_size,
        dataset_path=args.imagenet_path,
    )
    print(f"The top1 accuracy of the model is {top1_accuracy}%")
    print(f"The top5 accuracy of the model is {top5_accuracy}%")

    latency = device_model.get_latency()
    print(f"Inference latency of the model is {latency} ms")

    if args.results_path:
        results: list[list[str | float]] = [
            ["Metric", "Value"],
            ["Top 1", top1_accuracy],
            ["Top 5", top5_accuracy],
            ["Latency", latency],
        ]
        with open(args.results_path, "w", newline="") as csvfile:
            writer = csv.writer(csvfile)
            writer.writerows(results)


if __name__ == "__main__":
    main()
