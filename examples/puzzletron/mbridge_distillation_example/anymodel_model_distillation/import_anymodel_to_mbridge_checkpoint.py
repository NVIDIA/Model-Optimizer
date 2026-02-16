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
Import a HuggingFace AnyModel checkpoint and save it as a Megatron-Bridge checkpoint.

This script uses AutoBridge.import_ckpt() to:
1. Load a HuggingFace AnyModel model from a local directory
2. Convert it to Megatron format using PuzzletronLlamaAnyModelBridge
3. Save it as a native Megatron checkpoint that can be loaded with load_megatron_model()

IMPORTANT: This script should be run with torchrun to properly handle distributed setup:
    torchrun --nproc_per_node=1 import_anymodel_to_mbridge_checkpoint.py
"""

from pathlib import Path

# Import bridge to register it
import llama_anymodel_bridge  # noqa: F401
from megatron.bridge import AutoBridge


def main():
    """Main function to import HF AnyModel and save as Megatron checkpoint."""
    # HuggingFace AnyModel path (local directory)
    hf_model_path = "/workspace/puzzle_dir_anymodel/ckpts/teacher"
    # hf_model_path = "/workspace/hf_models/meta-llama/Llama-3.2-3B-Instruct"
    # Output directory for Megatron checkpoint
    megatron_checkpoint_path = "/workspace/mbridge_models/anymodel_teacher"

    print(f"Importing HuggingFace AnyModel from: {hf_model_path}")
    print(f"Saving Megatron checkpoint to: {megatron_checkpoint_path}")
    print()

    # Create output directory if it doesn't exist
    Path(megatron_checkpoint_path).mkdir(parents=True, exist_ok=True)

    # Import and save as Megatron checkpoint
    # Note: AutoBridge.import_ckpt internally initializes distributed training,
    # so it's best run with torchrun --nproc_per_node=1
    AutoBridge.import_ckpt(
        hf_model_path,  # Local HF AnyModel directory
        megatron_checkpoint_path,  # Target Megatron checkpoint directory
        dtype="bfloat16",  # Use bfloat16 for efficiency (dtype instead of torch_dtype)
        device_map="auto",  # Automatically place model on available devices
        trust_remote_code=True,  # Required for AnyModel models with custom code
    )

    print(f"\n✓ Successfully saved Megatron checkpoint to: {megatron_checkpoint_path}")
    print("\nYou can now load this checkpoint with:")
    print("  from megatron.bridge.training.model_load_save import load_megatron_model")
    print(f"  model = load_megatron_model('{megatron_checkpoint_path}')")


if __name__ == "__main__":
    main()
