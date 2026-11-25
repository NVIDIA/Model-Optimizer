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
from contextlib import nullcontext

import numpy as np
import torch

# This is a workaround for making the onnx export of models that use the torch RMSNorm work. We will
# need to move on to use dynamo based onnx export to properly fix the problem. The issue has been hit
# by both external users https://github.com/NVIDIA/Model-Optimizer/issues/262, and our
# internal users from MLPerf Inference.
#
if __name__ == "__main__":
    from diffusers.models.normalization import RMSNorm as DiffuserRMSNorm

    torch.nn.RMSNorm = DiffuserRMSNorm
    torch.nn.modules.normalization.RMSNorm = DiffuserRMSNorm

from onnx_utils.export import (
    _create_trt_dynamic_shapes,
    generate_dummy_kwargs_and_dynamic_axes_and_shapes,
    get_io_shapes,
    remove_nesting,
    update_dynamic_axes,
)
from quantize import ModelType, PipelineManager
from tqdm import tqdm

import modelopt.torch.opt as mto
from modelopt.torch._deploy._runtime import RuntimeRegistry
from modelopt.torch._deploy._runtime.tensorrt.constants import SHA_256_HASH_LENGTH
from modelopt.torch._deploy._runtime.tensorrt.tensorrt_utils import prepend_hash_to_bytes
from modelopt.torch._deploy.device_model import DeviceModel
from modelopt.torch._deploy.utils import get_onnx_bytes_and_metadata

MODEL_ID = {
    "sdxl-1.0": ModelType.SDXL_BASE,
    "sdxl-turbo": ModelType.SDXL_TURBO,
    "sd3-medium": ModelType.SD3_MEDIUM,
    "flux-dev": ModelType.FLUX_DEV,
    "flux-schnell": ModelType.FLUX_SCHNELL,
}

DTYPE_MAP = {
    "sdxl-1.0": torch.float16,
    "sdxl-turbo": torch.float16,
    "sd3-medium": torch.float16,
    "flux-dev": torch.bfloat16,
    "flux-schnell": torch.bfloat16,
}


@torch.inference_mode()
def generate_image(pipe, prompt, image_name, torch_autocast=False):
    context = torch.autocast("cuda") if torch_autocast else nullcontext()
    seed = 42
    with context:
        image = pipe(
            prompt,
            output_type="pil",
            num_inference_steps=30,
            generator=torch.Generator("cuda").manual_seed(seed),
        ).images[0]
    image.save(image_name)
    print(f"Image generated saved as {image_name}")


@torch.inference_mode()
def benchmark_backbone_standalone(
    pipe,
    num_warmup=10,
    num_benchmark=100,
    model_name="flux-dev",
    torch_autocast=False,
):
    """Benchmark the backbone model directly without running the full pipeline."""
    context = torch.autocast("cuda") if torch_autocast else nullcontext()
    backbone = pipe.transformer if hasattr(pipe, "transformer") else pipe.unet

    # Generate dummy inputs for the backbone
    dummy_kwargs, _, _ = generate_dummy_kwargs_and_dynamic_axes_and_shapes(model_name, backbone)

    # Extract the dict from the tuple and move to cuda
    dummy_kwargs_cuda = {
        k: v.cuda() if isinstance(v, torch.Tensor) else v for k, v in dummy_kwargs.items()
    }

    # Warmup
    print(f"Warming up: {num_warmup} iterations")
    for _ in tqdm(range(num_warmup), desc="Warmup"):
        with context:
            _ = backbone(**dummy_kwargs_cuda)

    # Benchmark
    torch.cuda.synchronize()
    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)

    print(f"Benchmarking: {num_benchmark} iterations")
    times = []
    for _ in tqdm(range(num_benchmark), desc="Benchmark"):
        with context:
            torch.cuda.profiler.cudart().cudaProfilerStart()
            start_event.record()
            _ = backbone(**dummy_kwargs_cuda)
            end_event.record()
            torch.cuda.synchronize()
            torch.cuda.profiler.cudart().cudaProfilerStop()
            times.append(start_event.elapsed_time(end_event))

    avg_latency = sum(times) / len(times)
    p50 = np.percentile(times, 50)
    p95 = np.percentile(times, 95)
    p99 = np.percentile(times, 99)

    print("\nBackbone-only inference latency:")
    print(f"  Average: {avg_latency:.2f} ms")
    print(f"  P50: {p50:.2f} ms")
    print(f"  P95: {p95:.2f} ms")
    print(f"  P99: {p99:.2f} ms")

    return avg_latency


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model",
        type=str,
        default="flux-dev",
        choices=["sdxl-1.0", "sdxl-turbo", "sd3-medium", "flux-dev", "flux-schnell"],
    )
    parser.add_argument(
        "--override-model-path",
        type=str,
        default=None,
        help="Path to the model if not using default paths in MODEL_ID mapping.",
    )
    parser.add_argument(
        "--restore-from", type=str, default=None, help="Path to the modelopt quantized checkpoint"
    )
    parser.add_argument(
        "--prompt",
        type=str,
        default="a photo of an astronaut riding a horse on mars",
        help="Input text prompt for the model",
    )
    parser.add_argument(
        "--onnx-load-path", type=str, default="", help="Path to load the ONNX model"
    )
    parser.add_argument(
        "--trt-engine-load-path", type=str, default=None, help="Path to load the TensorRT engine"
    )
    parser.add_argument(
        "--dq-only", action="store_true", help="Converts the ONNX model to a dq_only model"
    )
    parser.add_argument(
        "--torch",
        action="store_true",
        help="Use the torch pipeline for image generation or benchmarking",
    )
    parser.add_argument("--save-image-as", type=str, default=None, help="Name of the image to save")
    parser.add_argument(
        "--benchmark", action="store_true", help="Benchmark the model backbone inference time"
    )
    parser.add_argument(
        "--torch-compile", action="store_true", help="Use torch.compile() on the backbone model"
    )
    parser.add_argument(
        "--torch-autocast",
        action="store_true",
        help="Use torch.autocast() during inference or benchmarking",
    )
    parser.add_argument("--skip-image", action="store_true", help="Skip image generation")
    args = parser.parse_args()

    image_name = args.save_image_as if args.save_image_as else f"{args.model}.png"
    model_dtype = DTYPE_MAP[args.model]

    pipe = PipelineManager.create_pipeline_from(
        MODEL_ID[args.model],
        torch_dtype=model_dtype,
        override_model_path=args.override_model_path,
    )

    if args.torch_compile:
        assert args.torch, "Torch mode must be enabled when torch_compile is used"
    # Save the backbone (and other attributes) of the pipeline and move it to the GPU
    add_embedding = None
    cache_context = None
    if hasattr(pipe, "transformer"):
        backbone = pipe.transformer
        if hasattr(backbone, "cache_context"):
            cache_context = backbone.cache_context
    elif hasattr(pipe, "unet"):
        backbone = pipe.unet
        add_embedding = backbone.add_embedding
    else:
        raise ValueError("Pipeline does not have a transformer or unet backbone")

    if args.restore_from:
        mto.restore(backbone, args.restore_from)

    if args.torch:
        if args.torch_compile:
            print("Compiling backbone with torch.compile()...")
            backbone = torch.compile(backbone, mode="max-autotune")
        if hasattr(pipe, "transformer"):
            pipe.transformer = backbone
        elif hasattr(pipe, "unet"):
            pipe.unet = backbone
        pipe.to("cuda")

        if args.benchmark:
            benchmark_backbone_standalone(
                pipe,
                num_warmup=10,
                num_benchmark=100,
                model_name=args.model,
                torch_autocast=args.torch_autocast,
            )

        if not args.skip_image:
            generate_image(pipe, args.prompt, image_name, args.torch_autocast)
        return

    backbone.to("cuda")

    # Generate dummy inputs for the backbone
    dummy_inputs, dynamic_axes, dynamic_shapes = generate_dummy_kwargs_and_dynamic_axes_and_shapes(
        args.model, backbone
    )

    # Postprocess the dynamic axes to match the input and output names with DeviceModel
    if args.onnx_load_path == "":
        update_dynamic_axes(args.model, dynamic_axes)

    trt_dynamic_shapes = _create_trt_dynamic_shapes(dynamic_shapes)

    # We only need to remove the nesting for SDXL models as they contain the nested input added_cond_kwargs
    # which are renamed by the DeviceModel
    ignore_nesting = False
    if args.onnx_load_path != "" and args.model in ["sdxl-1.0", "sdxl-turbo"]:
        remove_nesting(trt_dynamic_shapes)
        ignore_nesting = True

    # Define deployment configuration
    deployment = {
        "runtime": "TRT",
        "precision": "stronglyTyped",
        "onnx_opset": "17",
        "verbose": "false",
    }

    client = RuntimeRegistry.get(deployment)

    # Export onnx model and get some required names from it
    onnx_bytes, metadata = get_onnx_bytes_and_metadata(
        model=backbone,
        dummy_input=dummy_inputs,
        onnx_load_path=args.onnx_load_path,
        dynamic_axes=dynamic_axes,
        onnx_opset=int(deployment["onnx_opset"]),
        remove_exported_model=False,
        dq_only=args.dq_only,
    )

    # Delete the original backbone and empty the cache
    del backbone
    torch.cuda.empty_cache()

    compilation_args = {"dynamic_shapes": trt_dynamic_shapes}
    if not args.trt_engine_load_path:
        # Compile the TRT engine from the exported ONNX model
        compiled_model = client.ir_to_compiled(onnx_bytes, compilation_args)
        # Clear onnx_bytes to free memory
        del onnx_bytes
        # Save TRT engine for future use
        with open(f"{args.model}.plan", "wb") as f:
            # Remove the SHA-256 hash from the compiled model, used to maintain state in the trt_client
            f.write(compiled_model[SHA_256_HASH_LENGTH:])
    else:
        with open(args.trt_engine_load_path, "rb") as f:
            compiled_model = f.read()
            # Prepend the SHA-256 hash from the compiled model, used to maintain state in the trt_client
            compiled_model = prepend_hash_to_bytes(compiled_model)

    # The output shapes will need to be specified for models with dynamic output dimensions
    device_model = DeviceModel(
        client,
        compiled_model,
        metadata,
        compilation_args,
        get_io_shapes(args.model, args.onnx_load_path, trt_dynamic_shapes),
        ignore_nesting,
    )

    # Set the backbone and other attributes to the device model
    if hasattr(pipe, "transformer"):
        pipe.transformer = device_model
        if cache_context:
            device_model.cache_context = cache_context
    elif hasattr(pipe, "unet"):
        pipe.unet = device_model
        device_model.add_embedding = add_embedding
    else:
        raise ValueError("Pipeline does not have a transformer or unet backbone")
    pipe.to("cuda")

    if not args.skip_image:
        generate_image(pipe, args.prompt, image_name, args.torch_autocast)
        print(f"Image generated using {args.model} model saved as {image_name}")

    if args.benchmark:
        print(
            f"Inference latency of the TensorRT optimized backbone: {device_model.get_latency()} ms"
        )


if __name__ == "__main__":
    main()
