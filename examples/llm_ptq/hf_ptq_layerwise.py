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

"""Layer-wise MXFP8 weight-only quantization for very large models.

This script converts model weights from bf16/fp16 to MXFP8 format one parameter at a time,
avoiding the need for full model forward passes. This is essential for models that are too
large to run forward passes on available GPUs (e.g., 500B+ parameter models).

MXFP8 is a dynamic quantization format -- scales are computed per-block from the weight values
alone, so no calibration data or forward passes are needed for weight conversion.

Usage:
    python examples/llm_ptq/hf_ptq_layerwise.py \
        --pyt_ckpt_path <model_path> \
        --export_path <output_path> \
        --qformat mxfp8 \
        --trust_remote_code

This script accepts the same arguments as hf_ptq.py for compatibility, but only the following
are used: --pyt_ckpt_path, --export_path, --qformat, --trust_remote_code,
--gpu_max_mem_percentage, --attn_implementation, --verbose.
"""

import argparse
import gc
import json
import os
import time

import torch
from example_utils import copy_custom_model_files, get_tokenizer
from tqdm import tqdm
from transformers import AutoModelForCausalLM

try:
    from modelopt import __version__ as modelopt_version
except Exception:
    modelopt_version = "unknown"
from modelopt.torch.quantization.qtensor import MXFP8QTensor

SUPPORTED_QFORMATS = ["mxfp8"]
MXFP8_BLOCK_SIZE = MXFP8QTensor.BLOCK_SIZE


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--pyt_ckpt_path",
        help="Specify where the PyTorch checkpoint path is",
        required=True,
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--qformat",
        help="Quantization format.",
        default="mxfp8",
        choices=SUPPORTED_QFORMATS,
    )
    # Accept but ignore calibration-related args for CLI compatibility with hf_ptq.py
    parser.add_argument("--batch_size", type=int, default=0)
    parser.add_argument("--calib_size", type=str, default="512")
    parser.add_argument("--calib_seq", type=int, default=512)
    parser.add_argument("--dataset", type=str, default=None)
    parser.add_argument("--export_path", default="exported_model")
    parser.add_argument("--export_fmt", default="hf", choices=["hf"])
    parser.add_argument(
        "--trust_remote_code",
        default=False,
        action="store_true",
    )
    parser.add_argument("--gpu_max_mem_percentage", type=float, default=0.8)
    parser.add_argument("--use_seq_device_map", default=False, action="store_true")
    parser.add_argument(
        "--verbose",
        default=True,
        action=argparse.BooleanOptionalAction,
    )
    parser.add_argument("--attn_implementation", default=None, type=str)
    parser.add_argument("--inference_tensor_parallel", type=int, default=1)
    parser.add_argument("--inference_pipeline_parallel", type=int, default=1)
    parser.add_argument("--kv_cache_qformat", default="none", type=str)
    parser.add_argument("--awq_block_size", default=0, type=int)
    parser.add_argument("--sparsity_fmt", default="dense", type=str)
    parser.add_argument("--auto_quantize_bits", default=None, type=float)
    parser.add_argument("--low_memory_mode", default=False, action="store_true")
    parser.add_argument("--calib_with_images", action="store_true")
    parser.add_argument("--auto_quantize_method", default="gradient", type=str)
    parser.add_argument("--auto_quantize_score_size", type=int, default=128)
    parser.add_argument("--auto_quantize_checkpoint", type=str, default=None)
    parser.add_argument("--moe_calib_experts_ratio", type=float, default=1.0)

    return parser.parse_args()


def _is_quantizable_weight(name: str, param: torch.Tensor) -> bool:
    """Check if a parameter should be quantized to MXFP8."""
    if param.ndim < 2:
        return False
    if "bias" in name:
        return False
    if param.shape[-1] % MXFP8_BLOCK_SIZE != 0:
        return False
    # Skip embedding and lm_head (they typically aren't quantized)
    if any(skip in name for skip in ["embed_tokens", "lm_head", "wte", "wpe"]):
        return False
    return True


def quantize_state_dict_mxfp8(model: torch.nn.Module) -> dict[str, torch.Tensor]:
    """Quantize model weights to MXFP8 format one parameter at a time.

    For each quantizable weight:
      - Computes E8M0 per-block scales
      - Quantizes weight to float8_e4m3fn
      - Stores both in the output state dict
      - Frees GPU memory after each parameter

    Non-quantizable parameters (biases, embeddings, etc.) are kept as-is.

    Returns:
        State dict with quantized weights and their scales.
    """
    new_state_dict = {}
    quantized_layers = []
    skipped_layers = []

    param_items = list(model.state_dict().items())
    for name, param in tqdm(param_items, desc="Quantizing weights"):
        if _is_quantizable_weight(name, param):
            param_gpu = param.to("cuda:0", non_blocking=True)

            scale = MXFP8QTensor.get_weights_scaling_factor(param_gpu)
            quantized_weight = MXFP8QTensor.quantize_with_scale(param_gpu, scale)

            scale_name = name.replace(".weight", ".weight_scale")
            if scale_name == name:
                scale_name = name + "_scale"

            new_state_dict[name] = quantized_weight.cpu()
            new_state_dict[scale_name] = scale.cpu()
            quantized_layers.append(name)

            del param_gpu, scale, quantized_weight
            torch.cuda.empty_cache()
            gc.collect()
        else:
            new_state_dict[name] = param.cpu()
            skipped_layers.append(name)

    print(f"\nQuantized {len(quantized_layers)} weight tensors to MXFP8")
    print(f"Skipped {len(skipped_layers)} parameters (biases, embeddings, etc.)")

    return new_state_dict


def build_mxfp8_quant_config(model: torch.nn.Module, quantized_state_dict: dict) -> dict:
    """Build the HF quantization_config for the exported config.json."""
    exclude_modules = []
    for name, module in model.named_modules():
        if hasattr(module, "weight"):
            full_weight_name = name + ".weight"
            if full_weight_name not in quantized_state_dict:
                continue
            weight = quantized_state_dict[full_weight_name]
            if weight.dtype != torch.float8_e4m3fn:
                exclude_modules.append(name.split(".")[-1])

    exclude_modules = sorted(set(exclude_modules))

    return {
        "config_groups": {
            "group_0": {
                "input_activations": {
                    "dynamic": True,
                    "num_bits": 8,
                    "type": "float",
                    "group_size": MXFP8_BLOCK_SIZE,
                },
                "weights": {
                    "dynamic": True,
                    "num_bits": 8,
                    "type": "float",
                    "group_size": MXFP8_BLOCK_SIZE,
                },
                "targets": ["Linear"],
            }
        },
        "ignore": exclude_modules,
        "quant_algo": "MXFP8",
        "kv_cache_scheme": None,
        "producer": {"name": "modelopt", "version": modelopt_version},
        "quant_method": "modelopt",
    }


def main(args: argparse.Namespace):
    if not torch.cuda.is_available():
        raise OSError("GPU is required for MXFP8 quantization.")

    if args.qformat not in SUPPORTED_QFORMATS:
        raise ValueError(
            f"This script only supports layerwise quantization for: {SUPPORTED_QFORMATS}. "
            f"Got: {args.qformat}. Use hf_ptq.py for other formats."
        )

    print(f"{'=' * 60}")
    print(f"Layer-wise MXFP8 weight quantization")
    print(f"Model: {args.pyt_ckpt_path}")
    print(f"Output: {args.export_path}")
    print(f"Format: {args.qformat}")
    print(f"{'=' * 60}")

    # Force eager execution
    torch.compiler.set_stance("force_eager")

    # Load model distributed across available GPUs + CPU via accelerate
    print("\nLoading model...")
    start_time = time.time()

    model_kwargs = {
        "torch_dtype": "auto",
        "device_map": "auto",
        "trust_remote_code": args.trust_remote_code,
    }
    if args.attn_implementation is not None:
        model_kwargs["attn_implementation"] = args.attn_implementation

    model = AutoModelForCausalLM.from_pretrained(args.pyt_ckpt_path, **model_kwargs)
    model.eval()

    load_time = time.time() - start_time
    print(f"Model loaded in {load_time:.1f}s")

    # Quantize weights layer by layer
    print("\nQuantizing weights to MXFP8...")
    start_time = time.time()

    quantized_state_dict = quantize_state_dict_mxfp8(model)

    quant_time = time.time() - start_time
    print(f"Quantization completed in {quant_time:.1f}s")

    # Build quantization config
    quant_config = build_mxfp8_quant_config(model, quantized_state_dict)

    # Save the model
    print(f"\nSaving quantized model to {args.export_path}...")
    start_time = time.time()

    os.makedirs(args.export_path, exist_ok=True)

    # Remove hf_quantizer if present so save_pretrained doesn't interfere
    if getattr(model, "hf_quantizer", None) is not None:
        model.hf_quantizer = None

    model.save_pretrained(
        args.export_path,
        state_dict=quantized_state_dict,
    )

    # Update config.json with quantization config
    config_path = os.path.join(args.export_path, "config.json")
    with open(config_path) as f:
        config_data = json.load(f)
    config_data["quantization_config"] = quant_config
    with open(config_path, "w") as f:
        json.dump(config_data, f, indent=4)

    # Save hf_quant_config.json for backward compatibility
    hf_quant_config = {
        "producer": quant_config["producer"],
        "quantization": {
            "quant_algo": "MXFP8",
            "kv_cache_quant_algo": None,
            "exclude_modules": quant_config["ignore"],
        },
    }
    with open(os.path.join(args.export_path, "hf_quant_config.json"), "w") as f:
        json.dump(hf_quant_config, f, indent=4)

    # Copy custom model files if trust_remote_code
    copy_custom_model_files(args.pyt_ckpt_path, args.export_path, args.trust_remote_code)

    # Save tokenizer
    tokenizer = get_tokenizer(args.pyt_ckpt_path, trust_remote_code=args.trust_remote_code)
    if tokenizer is not None:
        tokenizer.save_pretrained(args.export_path)

    save_time = time.time() - start_time
    print(f"Model saved in {save_time:.1f}s")

    if args.verbose:
        total_params = sum(p.numel() for p in model.parameters())
        quantized_params = sum(
            quantized_state_dict[k].numel()
            for k in quantized_state_dict
            if quantized_state_dict[k].dtype == torch.float8_e4m3fn
        )
        print(f"\n{'=' * 60}")
        print(f"Summary:")
        print(f"  Total parameters: {total_params:,}")
        print(f"  Quantized parameters: {quantized_params:,} ({100*quantized_params/total_params:.1f}%)")
        print(f"  Block size: {MXFP8_BLOCK_SIZE}")
        print(f"  Weight format: float8_e4m3fn")
        print(f"  Scale format: E8M0 (uint8)")
        print(f"  Output: {args.export_path}")
        print(f"{'=' * 60}")


if __name__ == "__main__":
    args = parse_args()
    main(args)
