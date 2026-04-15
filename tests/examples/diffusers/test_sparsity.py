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
triton-baseline, raw-threshold, and export modes. Also includes a Python API
test for calibration params + export (calibration can't succeed on tiny models
via the Triton kernel, so params are injected directly).
"""

import json
import math

import pytest
import torch
import yaml
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


def test_wan22_export_sparse_checkpoint(tiny_wan22_path, tmp_path):
    """Export sparsified Wan 2.2 checkpoint with raw threshold and verify config."""
    export_dir = tmp_path / "sparse_export"
    cmd = [
        "python",
        "wan22_skip_softmax.py",
        "--model-path",
        tiny_wan22_path,
        "--raw-threshold",
        "-5.0",
        "--export-dir",
        str(export_dir),
        *_TINY_ARGS,
    ]
    run_example_command(cmd, EXAMPLE_PATH)

    # Verify export directory structure
    assert export_dir.exists()

    # Both transformers should have sparse_attention_config in config.json and sparse.yaml
    for component in ["transformer", "transformer_2"]:
        component_dir = export_dir / component
        assert component_dir.exists(), f"Missing component dir: {component}"

        # Check config.json has sparse_attention_config
        config_path = component_dir / "config.json"
        assert config_path.exists(), f"Missing config.json for {component}"
        with open(config_path) as f:
            config_data = json.load(f)
        assert "sparse_attention_config" in config_data, (
            f"No sparse_attention_config in {component}/config.json"
        )

        sa_config = config_data["sparse_attention_config"]
        assert "config_groups" in sa_config
        group_0 = sa_config["config_groups"]["group_0"]
        assert group_0["sparse_algo"] == "softmax_skip"
        assert "raw_threshold" in group_0
        assert group_0["raw_threshold"] == -5.0
        assert "disabled_layers" in group_0  # cross-attn layers are disabled
        assert "producer" in sa_config

        # Check sparse.yaml exists and matches config.json
        yaml_path = component_dir / "sparse.yaml"
        assert yaml_path.exists(), f"Missing sparse.yaml for {component}"
        with open(yaml_path) as f:
            yaml_data = yaml.safe_load(f)
        assert yaml_data == sa_config, f"sparse.yaml does not match config.json for {component}"

        # Check model weights were saved
        weight_files = list(component_dir.glob("*.safetensors")) + list(component_dir.glob("*.bin"))
        assert len(weight_files) > 0, f"No weight files for {component}"


def test_wan22_calibrated_export(tmp_path):
    """Export sparsified Wan 2.2 with calibration params and verify log-space formula.

    Calibration can't succeed on tiny models via the Triton kernel (not enough
    data points in the 10%-90% sparsity range), so this test uses the Python API
    to sparsify, inject calibration params directly, and then export.
    This exercises the full calibration_params → export_sparse_attention_config path.
    """
    from diffusers import AutoencoderKLWan, WanPipeline

    import modelopt.torch.sparsity.attention_sparsity as mtsa
    from modelopt.torch.export import export_hf_checkpoint
    from modelopt.torch.sparsity.attention_sparsity.sparse_attention import SparseAttentionModule

    # Build tiny pipeline from scratch (not from disk — avoids session fixture dependency)
    pipe_dir = create_tiny_wan22_pipeline_dir(tmp_path / "model")
    vae = AutoencoderKLWan.from_pretrained(pipe_dir, subfolder="vae", torch_dtype=torch.float32)
    pipe = WanPipeline.from_pretrained(pipe_dir, vae=vae, torch_dtype=torch.bfloat16)
    pipe.to("cuda")

    # Sparsify both transformers with triton_skip_softmax (no calibration)
    sparse_cfg = {
        "*.attn1*": {
            "method": "triton_skip_softmax",
            "skip_softmax_threshold": 0.1,
            "backend": "triton",
            "is_causal": False,
            "enable": True,
        },
        "*.attn2*": {"enable": False},
        "default": {"enable": False},
    }
    config = {"sparse_cfg": sparse_cfg}

    for transformer in [pipe.transformer, pipe.transformer_2]:
        mtsa.sparsify(transformer, config)

    # Inject calibration params (simulating a successful log-space calibration)
    # These are the params that would come from DynamicThresholdCalibrator.calibrate()
    # with fit_logspace=True
    test_log_a = math.log(1.5)  # log_a ≈ 0.405
    test_b = 3.0
    calibration_params = {
        "prefill": {
            "a": math.exp(test_log_a),
            "b": test_b,
            "log_a": test_log_a,
            "fit_logspace": True,
            "min_observed_sparsity": 0.15,
            "max_observed_sparsity": 0.85,
        },
    }
    target_sparse_ratio = {"prefill": 0.5}

    for transformer in [pipe.transformer, pipe.transformer_2]:
        for module in transformer.modules():
            if isinstance(module, SparseAttentionModule) and module.is_enabled:
                module._sparse_method_instance.calibration_params = calibration_params
                module._sparse_method_instance.target_sparse_ratio = target_sparse_ratio

    # Export
    export_dir = tmp_path / "calibrated_export"
    export_hf_checkpoint(pipe, export_dir=export_dir)

    # Verify both transformers have correct calibrated config
    for component in ["transformer", "transformer_2"]:
        config_path = export_dir / component / "config.json"
        assert config_path.exists(), f"Missing config.json for {component}"
        with open(config_path) as f:
            config_data = json.load(f)
        assert "sparse_attention_config" in config_data, (
            f"No sparse_attention_config in {component}/config.json"
        )

        sa_config = config_data["sparse_attention_config"]
        group_0 = sa_config["config_groups"]["group_0"]

        # Verify algorithm and targets
        assert group_0["sparse_algo"] == "softmax_skip"
        assert "disabled_layers" in group_0

        # Verify log-space formula and params
        tsf = group_0["threshold_scale_factor"]
        assert tsf["formula"] == "log_a + b * target_sparsity"
        assert "prefill" in tsf
        assert tsf["prefill"]["log_a"] == pytest.approx(test_log_a)
        assert tsf["prefill"]["b"] == pytest.approx(test_b)
        assert "a" not in tsf["prefill"]  # log-space exports log_a, not a

        # No raw_threshold — this is calibrated mode
        assert "raw_threshold" not in group_0

        # Verify sparse.yaml matches
        yaml_path = export_dir / component / "sparse.yaml"
        assert yaml_path.exists()
        with open(yaml_path) as f:
            yaml_data = yaml.safe_load(f)
        assert yaml_data == sa_config
