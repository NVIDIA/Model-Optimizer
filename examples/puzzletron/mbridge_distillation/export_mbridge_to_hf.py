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
Megatron-HuggingFace Checkpoint Conversion Example

This script demonstrates how to convert models between HuggingFace and Megatron formats
using the AutoBridge import_ckpt and export_ckpt methods.

Features:
- Import HuggingFace models to Megatron checkpoint format
- Export Megatron checkpoints to HuggingFace format
- Support for various model architectures (GPT, Llama, etc.)
- Configurable model and conversion settings

Usage examples:
  # Import a HuggingFace model to Megatron format
  uv run python examples/conversion/convert_checkpoints.py import \
    --hf-model meta-llama/Llama-3.2-1B \
    --megatron-path ./checkpoints/llama3_2_1b

  # Export a Megatron checkpoint to HuggingFace format
  uv run python examples/conversion/convert_checkpoints.py export \
    --hf-model meta-llama/Llama-3.2-1B \
    --megatron-path ./checkpoints/llama3_2_1b \
    --hf-path ./exports/llama3_2_1b_hf

  # Import with custom settings
  uv run python examples/conversion/convert_checkpoints.py import \
    --hf-model ./local_model \
    --megatron-path ./checkpoints/custom_model \
    --torch-dtype bfloat16 \
    --device-map auto

  # Export without progress bar (useful for scripting)
  uv run python examples/conversion/convert_checkpoints.py export \
    --hf-model ./local_model \
    --megatron-path ./checkpoints/custom_model \
    --hf-path ./exports/custom_model_hf \
    --no-progress
"""

import argparse
import sys
from pathlib import Path

import torch
from megatron.bridge import AutoBridge


def validate_path(path: str, must_exist: bool = False) -> Path:
    """Validate and convert string path to Path object."""
    path_obj = Path(path)
    if must_exist and not path_obj.exists():
        raise ValueError(f"Path does not exist: {path}")
    return path_obj


def get_torch_dtype(dtype_str: str) -> torch.dtype:
    """Convert string to torch dtype."""
    dtype_map = {
        "float32": torch.float32,
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
    }
    if dtype_str not in dtype_map:
        raise ValueError(f"Unsupported dtype: {dtype_str}. Supported: {list(dtype_map.keys())}")
    return dtype_map[dtype_str]


def import_hf_to_megatron(
    hf_model: str,
    megatron_path: str,
    torch_dtype: str | None = None,
    device_map: str | None = None,
    trust_remote_code: bool = False,
) -> None:
    """
    Import a HuggingFace model and save it as a Megatron checkpoint.

    Args:
        hf_model: HuggingFace model ID or path to model directory
        megatron_path: Directory path where the Megatron checkpoint will be saved
        torch_dtype: Model precision ("float32", "float16", "bfloat16")
        device_map: Device placement strategy ("auto", "cuda:0", etc.)
        trust_remote_code: Allow custom model code execution
    """
    print(f"🔄 Starting import: {hf_model} -> {megatron_path}")

    # Prepare kwargs
    kwargs = {}
    if torch_dtype:
        kwargs["torch_dtype"] = get_torch_dtype(torch_dtype)
        print(f"   Using torch_dtype: {torch_dtype}")

    if device_map:
        kwargs["device_map"] = device_map
        print(f"   Using device_map: {device_map}")

    if trust_remote_code:
        kwargs["trust_remote_code"] = trust_remote_code
        print(f"   Trust remote code: {trust_remote_code}")

    # Import using the convenience method
    print(f"📥 Loading HuggingFace model: {hf_model}")
    AutoBridge.import_ckpt(
        hf_model_id=hf_model,
        megatron_path=megatron_path,
        **kwargs,
    )

    print(f"✅ Successfully imported model to: {megatron_path}")

    # Verify the checkpoint was created
    checkpoint_path = Path(megatron_path)
    if checkpoint_path.exists():
        print("📁 Checkpoint structure:")
        for item in checkpoint_path.iterdir():
            if item.is_dir():
                print(f"   📂 {item.name}/")
            else:
                print(f"   📄 {item.name}")


def export_megatron_to_hf(
    hf_model: str,
    megatron_path: str,
    hf_path: str,
    show_progress: bool = True,
    strict: bool = True,
) -> None:
    """
    Export a Megatron checkpoint to HuggingFace format.

    Args:
        megatron_path: Directory path where the Megatron checkpoint is stored
        hf_path: Directory path where the HuggingFace model will be saved
        show_progress: Display progress bar during weight export
    """
    print(f"🔄 Starting export: {megatron_path} -> {hf_path}")

    # Validate megatron checkpoint exists
    checkpoint_path = validate_path(megatron_path, must_exist=True)
    print(f"📂 Found Megatron checkpoint: {checkpoint_path}")

    # Look for configuration files to determine the model type
    config_files = list(checkpoint_path.glob("**/run_config.yaml"))
    if not config_files:
        # Look in iter_ subdirectories
        iter_dirs = [
            d for d in checkpoint_path.iterdir() if d.is_dir() and d.name.startswith("iter_")
        ]
        if iter_dirs:
            # Use the latest iteration
            latest_iter = max(iter_dirs, key=lambda d: int(d.name.replace("iter_", "")))
            config_files = list(latest_iter.glob("run_config.yaml"))

    if not config_files:
        raise FileNotFoundError(
            f"Could not find run_config.yaml in {checkpoint_path}. Please ensure this is a valid Megatron checkpoint."
        )

    print(f"📋 Found configuration: {config_files[0]}")

    # For demonstration, we'll create a bridge from a known config
    # This would typically be extracted from the checkpoint metadata
    bridge = AutoBridge.from_hf_pretrained(hf_model, trust_remote_code=True)

    # Export using the convenience method
    print("📤 Exporting to HuggingFace format...")
    bridge.export_ckpt(
        megatron_path=megatron_path,
        hf_path=hf_path,
        show_progress=show_progress,
        strict=strict,
    )

    print(f"✅ Successfully exported model to: {hf_path}")

    # Verify the export was created
    export_path = Path(hf_path)
    if export_path.exists():
        print("📁 Export structure:")
        for item in export_path.iterdir():
            if item.is_dir():
                print(f"   📂 {item.name}/")
            else:
                print(f"   📄 {item.name}")

    print("🔍 You can now load this model with:")
    print("   from transformers import AutoModelForCausalLM")
    print(f"   model = AutoModelForCausalLM.from_pretrained('{hf_path}')")


def main():
    """Main function to handle command line arguments and execute conversions."""
    parser = argparse.ArgumentParser(
        description="Convert models between HuggingFace and Megatron formats",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    subparsers = parser.add_subparsers(dest="command", help="Conversion direction")

    # Import subcommand (HF -> Megatron)
    import_parser = subparsers.add_parser(
        "import", help="Import HuggingFace model to Megatron checkpoint format"
    )
    import_parser.add_argument(
        "--hf-model", required=True, help="HuggingFace model ID or path to model directory"
    )
    import_parser.add_argument(
        "--megatron-path",
        required=True,
        help="Directory path where the Megatron checkpoint will be saved",
    )
    import_parser.add_argument(
        "--torch-dtype", choices=["float32", "float16", "bfloat16"], help="Model precision"
    )
    import_parser.add_argument(
        "--device-map", help='Device placement strategy (e.g., "auto", "cuda:0")'
    )
    import_parser.add_argument(
        "--trust-remote-code", action="store_true", help="Allow custom model code execution"
    )

    # Export subcommand (Megatron -> HF)
    export_parser = subparsers.add_parser(
        "export", help="Export Megatron checkpoint to HuggingFace format"
    )
    export_parser.add_argument(
        "--hf-model", required=True, help="HuggingFace model ID or path to model directory"
    )
    export_parser.add_argument(
        "--megatron-path",
        required=True,
        help="Directory path where the Megatron checkpoint is stored",
    )
    export_parser.add_argument(
        "--hf-path", required=True, help="Directory path where the HuggingFace model will be saved"
    )
    export_parser.add_argument(
        "--no-progress", action="store_true", help="Disable progress bar during export"
    )
    export_parser.add_argument(
        "--not-strict",
        action="store_true",
        help="Allow source and target checkpoint to have different keys",
    )

    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        return 1

    if args.command == "import":
        import_hf_to_megatron(
            hf_model=args.hf_model,
            megatron_path=args.megatron_path,
            torch_dtype=args.torch_dtype,
            device_map=args.device_map,
            trust_remote_code=args.trust_remote_code,
        )

    elif args.command == "export":
        export_megatron_to_hf(
            hf_model=args.hf_model,
            megatron_path=args.megatron_path,
            hf_path=args.hf_path,
            show_progress=not args.no_progress,
            strict=not args.not_strict,
        )
    else:
        raise RuntimeError(f"Unknown command: {args.command}")

    if torch.distributed.is_initialized():
        torch.distributed.barrier()
        torch.distributed.destroy_process_group()


if __name__ == "__main__":
    sys.exit(main())
