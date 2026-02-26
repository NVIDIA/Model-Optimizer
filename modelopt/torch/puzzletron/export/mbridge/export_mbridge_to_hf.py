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
"""Export utilities for Megatron-Bridge checkpoints."""

import shutil
from pathlib import Path

from megatron.bridge import AutoBridge

from modelopt.torch.utils import print_rank_0


def export_to_hf_and_copy_config(
    student_hf_path: str,
    checkpoint_dir: str,
    train_iters: int,
    hf_export_path: str,
    hf_model: str,
) -> None:
    """
    Export Megatron checkpoint to HuggingFace format and copy config.json from student model.

    Args:
        student_hf_path: Path to the original student HuggingFace model (source of config.json)
        checkpoint_dir: Base directory where Megatron checkpoints are stored
        train_iters: Number of training iterations (used to construct final checkpoint path)
        hf_export_path: Directory path where the HuggingFace model will be saved
        hf_model: HuggingFace model ID to use as template for export (e.g., meta-llama/Llama-3.1-8B-Instruct)
    """
    print_rank_0(f"\n{'=' * 80}")
    print_rank_0("Exporting to HuggingFace format...")
    print_rank_0(f"{'=' * 80}\n")

    # Construct path to final checkpoint iteration (format: iter_0000100 for 100 iterations)
    final_iter_dir = Path(checkpoint_dir) / f"iter_{train_iters:07d}"
    print_rank_0(f"📂 Using final checkpoint: {final_iter_dir}")

    # Use the final iteration directory for export (export_ckpt will validate it exists)
    megatron_path = str(final_iter_dir)

    # Create bridge using standard model ID (not AnyModel checkpoint) to avoid sharding structure issues
    print_rank_0("🌉 Creating bridge...")
    print_rank_0(f"   Using model ID: {hf_model}")
    bridge = AutoBridge.from_hf_pretrained(hf_model, trust_remote_code=True)

    print_rank_0("📤 Exporting to HuggingFace format...")
    bridge.export_ckpt(
        megatron_path=megatron_path,
        hf_path=hf_export_path,
        show_progress=True,
        strict=True,
    )

    print_rank_0(f"✅ Successfully exported model to: {hf_export_path}")

    # Copy config.json from student model to exported model (preserves block_configs)
    student_config_path = Path(student_hf_path) / "config.json"
    exported_config_path = Path(hf_export_path) / "config.json"

    print_rank_0(f"📋 Copying config.json from student model: {student_config_path}")
    shutil.copy(student_config_path, exported_config_path)
    print_rank_0(f"✅ Copied config.json to: {exported_config_path}")

    print_rank_0(f"\n{'=' * 80}")
    print_rank_0("Export complete!")
