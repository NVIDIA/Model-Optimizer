# SPDX-FileCopyrightText: Copyright (c) 2023-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

"""Export functions for NeMo Run."""

from pathlib import Path

from nemo.collections.llm.api import export_ckpt
from nemo.utils import logging


def export_most_recent_ckpt(directory: str, output_path: str):
    """Export most recent checkpoint from a NeMo Run experiment directory."""
    most_recent_ckpt = _get_most_recent_ckpt(directory)
    logging.info(f"Exporting most recent NeMo Run checkpoint: {most_recent_ckpt}")
    export_ckpt(
        most_recent_ckpt,
        "hf",
        output_path=output_path,
        overwrite=True,
    )


def _get_most_recent_subdir(directory: Path):
    # Get all subdirectories
    subdirs = [d for d in directory.iterdir() if d.is_dir()]
    if not subdirs:
        raise ValueError(f"No subdirectories found in {directory}")

    # Sort by modification time (most recent first)
    most_recent = max(subdirs, key=lambda x: x.stat().st_mtime)

    return most_recent


def _get_most_recent_ckpt(directory: str):
    """Find the most recent checkpoint subdirectory in a given NeMo Run experiment directory.

    Args:
        directory (str): Path to the directory to search in.

    Returns:
        str: Path to the most recent subdirectory.
    """
    exp_dir = Path(directory) / "default"
    if not exp_dir.exists():
        raise FileNotFoundError(f"Experiment directory {exp_dir} does not exist")

    checkpoint_dir = exp_dir / "checkpoints"
    if checkpoint_dir.exists():
        most_recent = _get_most_recent_subdir(checkpoint_dir)
    else:
        most_recent = _get_most_recent_subdir(exp_dir)
        checkpoint_dir = most_recent / "checkpoints"
        if not checkpoint_dir.exists():
            raise FileNotFoundError(f"Checkpoint directory {checkpoint_dir} does not exist")
        most_recent = _get_most_recent_subdir(checkpoint_dir)

    return str(most_recent)
