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

"""Tests for skip-softmax sparse attention on Wan 2.2 (examples/diffusers/sparsity/).

Uses a tiny Wan 2.2 model (dual transformer, 2 layers, hidden_dim=24) created
from scratch. Tests run the wan22_skip_softmax.py example script in baseline,
triton-baseline, and raw-threshold modes.
"""

import pytest
from _test_utils.examples.run_command import run_example_command
from _test_utils.torch.diffusers_models import create_tiny_wan22_pipeline_dir

EXAMPLE_PATH = "diffusers/sparsity"

# Tiny inference settings — fast but exercises all code paths
_TINY_ARGS = [
    "--num-frames",
    "5",
    "--height",
    "16",
    "--width",
    "16",
    "--num-steps",
    "2",
    "--guidance-scale",
    "1.0",
    "--skip-first-last",
    "0",
    "--negative-prompt",
    "",
]


@pytest.fixture(scope="session")
def tiny_wan22_path(tmp_path_factory):
    """Create a tiny Wan 2.2 pipeline saved to disk (session-scoped)."""
    return str(create_tiny_wan22_pipeline_dir(tmp_path_factory.mktemp("tiny_wan22")))


def test_wan22_baseline(tiny_wan22_path, tmp_path):
    """Dense baseline — no sparsity, default diffusers attention backend."""
    cmd = [
        "python",
        "wan22_skip_softmax.py",
        "--model-path",
        tiny_wan22_path,
        "--baseline",
        "--prompt",
        "test",
        "--output",
        str(tmp_path / "baseline.mp4"),
        *_TINY_ARGS,
    ]
    run_example_command(cmd, EXAMPLE_PATH)


def test_wan22_triton_baseline(tiny_wan22_path, tmp_path):
    """Triton kernel without skip-softmax (threshold=0, apples-to-apples)."""
    cmd = [
        "python",
        "wan22_skip_softmax.py",
        "--model-path",
        tiny_wan22_path,
        "--triton-baseline",
        "--prompt",
        "test",
        "--output",
        str(tmp_path / "triton_baseline.mp4"),
        *_TINY_ARGS,
    ]
    run_example_command(cmd, EXAMPLE_PATH)


def test_wan22_raw_threshold(tiny_wan22_path, tmp_path):
    """Skip-softmax with a fixed raw threshold — no calibration needed."""
    cmd = [
        "python",
        "wan22_skip_softmax.py",
        "--model-path",
        tiny_wan22_path,
        "--raw-threshold",
        "-5.0",
        "--report-avg-sparsity",
        "--prompt",
        "test",
        "--output",
        str(tmp_path / "raw_threshold.mp4"),
        *_TINY_ARGS,
    ]
    run_example_command(cmd, EXAMPLE_PATH)
