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
Export Megatron-Bridge checkpoint to HuggingFace format.

This script exports Megatron checkpoints to HuggingFace format using the AutoBridge
export_ckpt method.

Features:
- Export Megatron checkpoints to HuggingFace format
- Support for various model architectures (GPT, Llama, etc.)
- Configurable export settings

Usage examples:
  # Export a Megatron checkpoint to HuggingFace format
  python examples/puzzletron/mbridge_distillation/export_mbridge_to_hf.py \
    --hf-model meta-llama/Llama-3.1-8B-Instruct \
    --megatron-path ./checkpoints/distilled_model \
    --hf-path ./exports/distilled_model_hf

  # Export without progress bar (useful for scripting)
  python examples/puzzletron/mbridge_distillation/export_mbridge_to_hf.py \
    --hf-model meta-llama/Llama-3.1-8B-Instruct \
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
        hf_model: HuggingFace model ID or path (used for tokenizer and config template)
        megatron_path: Directory path where the Megatron checkpoint is stored
        hf_path: Directory path where the HuggingFace model will be saved
        show_progress: Display progress bar during weight export
        strict: Whether to perform strict validation during weight export
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
    """Main function to handle command line arguments and execute export."""
    parser = argparse.ArgumentParser(
        description="Export Megatron checkpoint to HuggingFace format",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    parser.add_argument(
        "--hf-model", required=True, help="HuggingFace model ID or path to model directory"
    )
    parser.add_argument(
        "--megatron-path",
        required=True,
        help="Directory path where the Megatron checkpoint is stored",
    )
    parser.add_argument(
        "--hf-path", required=True, help="Directory path where the HuggingFace model will be saved"
    )
    parser.add_argument(
        "--no-progress", action="store_true", help="Disable progress bar during export"
    )
    parser.add_argument(
        "--not-strict",
        action="store_true",
        help="Allow source and target checkpoint to have different keys",
    )

    args = parser.parse_args()

    export_megatron_to_hf(
        hf_model=args.hf_model,
        megatron_path=args.megatron_path,
        hf_path=args.hf_path,
        show_progress=not args.no_progress,
        strict=not args.not_strict,
    )

    if torch.distributed.is_initialized():
        torch.distributed.barrier()
        torch.distributed.destroy_process_group()


if __name__ == "__main__":
    sys.exit(main())
