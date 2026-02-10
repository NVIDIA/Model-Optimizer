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
Import a HuggingFace model and save it as a Megatron-Bridge checkpoint.

This script uses AutoBridge.import_ckpt() to:
1. Load a HuggingFace model from a local directory
2. Convert it to Megatron format
3. Save it as a native Megatron checkpoint that can be loaded with load_megatron_model()

IMPORTANT: This script should be run with torchrun to properly handle distributed setup:
    torchrun --nproc_per_node=1 import_hf_checkpoint.py

Or use the wrapper script: bash run_import_hf_checkpoint.sh
"""

from pathlib import Path

from megatron.bridge import AutoBridge


def main():
    """Main function to import HF model and save as Megatron checkpoint."""
    # HuggingFace model path (local directory)
    hf_model_path = ".../hf_models/meta-llama/Llama-3.2-3B-Instruct"

    # Output directory for Megatron checkpoint
    megatron_checkpoint_path = ".../mbridge_models/meta-llama/Llama-3.2-3B-Instruct"

    print(f"Importing HuggingFace model from: {hf_model_path}")
    print(f"Saving Megatron checkpoint to: {megatron_checkpoint_path}")
    print()

    # Create output directory if it doesn't exist
    Path(megatron_checkpoint_path).mkdir(parents=True, exist_ok=True)

    # Import and save as Megatron checkpoint
    # Note: AutoBridge.import_ckpt internally initializes distributed training,
    # so it's best run with torchrun --nproc_per_node=1
    AutoBridge.import_ckpt(
        hf_model_path,  # Local HF model directory
        megatron_checkpoint_path,  # Target Megatron checkpoint directory
        dtype="bfloat16",  # Use bfloat16 for efficiency (dtype instead of torch_dtype)
        device_map="auto",  # Automatically place model on available devices
    )

    print(f"\n✓ Successfully saved Megatron checkpoint to: {megatron_checkpoint_path}")
    print("\nYou can now load this checkpoint with:")
    print("  from megatron.bridge.training.model_load_save import load_megatron_model")
    print(f"  model = load_megatron_model('{megatron_checkpoint_path}')")


if __name__ == "__main__":
    main()
