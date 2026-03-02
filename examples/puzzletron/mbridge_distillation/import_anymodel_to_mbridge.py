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
Import AnyModel checkpoint to Megatron-Bridge format.

This script converts a HuggingFace AnyModel checkpoint to Megatron-Bridge format.

Usage:
    cd /workspace/Model-Optimizer

    export PYTHONPATH="/workspace/Model-Optimizer:${PYTHONPATH}"

    torchrun --nproc_per_node=1 examples/puzzletron/mbridge_distillation/import_anymodel_to_mbridge.py \
        --input-ckpt-path /path/to/anymodel/checkpoint \
        --output-ckpt-path /path/to/save/mbridge/checkpoint
"""

import argparse
from pathlib import Path

from megatron.bridge import AutoBridge

# Import all heterogeneous bridges to register them
# This will override homogeneous bridges (e.g., LlamaBridge, Qwen3Bridge) with
# heterogeneous versions (PuzzletronLlamaAnyModelBridge, PuzzletronQwen3AnyModelBridge)
# that support block_configs for AnyModel checkpoints.
import modelopt.torch.puzzletron.export.mbridge  # noqa: F401


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Convert AnyModel checkpoint to Megatron-Bridge format",
        formatter_class=argparse.RawTextHelpFormatter,
    )
    parser.add_argument(
        "--input-ckpt-path",
        type=str,
        required=True,
        help="Path to input AnyModel checkpoint (HuggingFace format)",
    )
    parser.add_argument(
        "--output-ckpt-path",
        type=str,
        required=True,
        help="Path to save output Megatron-Bridge checkpoint",
    )
    return parser.parse_args()


def main() -> None:
    """Main function to import HF AnyModel and save as Megatron checkpoint."""
    args = parse_args()

    input_path = Path(args.input_ckpt_path)
    output_path = Path(args.output_ckpt_path)

    print(f"Importing AnyModel checkpoint from: {input_path}")
    print(f"Saving Megatron-Bridge checkpoint to: {output_path}")
    print()

    # Create output directory if it doesn't exist
    output_path.mkdir(parents=True, exist_ok=True)

    # Import and save as Megatron checkpoint
    AutoBridge.import_ckpt(
        hf_model_id=str(input_path),
        megatron_path=str(output_path),
        trust_remote_code=True,
    )

    print(f"\n✓ Successfully saved Megatron-Bridge checkpoint to: {output_path}")


if __name__ == "__main__":
    main()
