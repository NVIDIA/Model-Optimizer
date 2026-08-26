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

import os

import pytest
from _test_utils.examples.run_command import run_example_command
from _test_utils.torch.misc import minimum_gpu


@pytest.fixture
def imagenet_path():
    """Fixture to get IMAGENET_PATH from environment and skip if not valid."""
    path = os.getenv("IMAGENET_PATH")
    if not path or not os.path.isdir(path):
        pytest.skip(
            "IMAGENET_PATH environment variable is not set or does not point to a valid directory"
        )
    return path


def _build_common_command(imagenet_path):
    """Build common command arguments for CNN QAT training."""
    train_data_path = os.path.join(imagenet_path, "train")
    val_data_path = os.path.join(imagenet_path, "val")
    for p in (train_data_path, val_data_path):
        if not os.path.isdir(p):
            pytest.skip(f"Expected dataset folder '{p}' not found", allow_module_level=True)

    return [
        "--train-data-path",
        train_data_path,
        "--val-data-path",
        val_data_path,
        "--batch-size",
        "64",
        "--num-workers",
        "8",
        "--epochs",
        "5",
        "--lr",
        "1e-4",
        "--print-freq",
        "50",
    ]


def _run_qat_command(base_cmd, common_args, output_dir, example_dir="cnn_qat"):
    """Helper function to run QAT command with common arguments."""
    full_command = base_cmd + common_args + ["--output-dir", str(output_dir)]
    run_example_command(full_command, example_dir)


@minimum_gpu(1)
def test_cnn_qat_single_gpu(tmp_path, imagenet_path):
    """Test CNN QAT on single GPU."""
    common_args = _build_common_command(imagenet_path)
    base_command = ["python", "torchvision_qat.py", "--gpu", "0"]

    _run_qat_command(base_command, common_args, tmp_path)


@minimum_gpu(2)
def test_cnn_qat_multi_gpu(tmp_path, imagenet_path):
    """Test CNN QAT on multiple GPUs."""
    common_args = _build_common_command(imagenet_path)
    base_command = ["torchrun", "--nproc_per_node=2", "torchvision_qat.py"]

    _run_qat_command(base_command, common_args, tmp_path)
