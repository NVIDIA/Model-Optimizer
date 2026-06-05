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
triton-baseline, fixed-threshold, and export modes. Also includes a Python API
test for calibration params + export (calibration can't succeed on tiny models
via the Triton kernel, so params are injected directly).
"""

import json
import math

import pytest
import torch
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


def test_wan22_skip_softmax_threshold(tiny_wan22_path, tmp_path):
    """Skip-softmax with a fixed lambda threshold — no calibration needed."""
    cmd = [
        "python",
        "wan22_skip_softmax.py",
        "--model-path",
        tiny_wan22_path,
        "--skip-softmax-threshold",
        "0.03125",
        "--report-avg-sparsity",
        "--prompt",
        "test",
        "--output",
        str(tmp_path / "skip_softmax_threshold.mp4"),
        *_TINY_ARGS,
    ]
    run_example_command(cmd, EXAMPLE_PATH)


def test_wan22_export_sparse_checkpoint(tiny_wan22_path, tmp_path):
    """Export a fixed-threshold checkpoint and verify the structure.

    A fixed ``--skip-softmax-threshold`` run is uncalibrated, so
    ``export_sparse_attention_config`` returns ``None`` and no
    ``sparse_attention_config`` is written (the populated-config path is covered
    by ``test_wan22_calibrated_export``). Here we just verify the checkpoint is
    written and that no standalone ``sparse.yaml`` is produced.
    """
    export_dir = tmp_path / "sparse_export"
    cmd = [
        "python",
        "wan22_skip_softmax.py",
        "--model-path",
        tiny_wan22_path,
        "--skip-softmax-threshold",
        "0.03125",
        "--export-dir",
        str(export_dir),
        *_TINY_ARGS,
    ]
    run_example_command(cmd, EXAMPLE_PATH)

    assert export_dir.exists()
    for component in ["transformer", "transformer_2"]:
        component_dir = export_dir / component
        assert component_dir.exists(), f"Missing component dir: {component}"
        config_path = component_dir / "config.json"
        assert config_path.exists(), f"Missing config.json for {component}"
        with open(config_path) as f:
            config_data = json.load(f)
        # Fixed (uncalibrated) threshold has nothing to export.
        assert "sparse_attention_config" not in config_data, (
            f"Unexpected sparse_attention_config in {component}/config.json for a "
            "fixed-threshold (uncalibrated) export"
        )
        weight_files = list(component_dir.glob("*.safetensors")) + list(component_dir.glob("*.bin"))
        assert len(weight_files) > 0, f"No weight files for {component}"

    # Sparse config (when present) lives only in config.json — never a standalone yaml.
    assert not (export_dir / "sparse.yaml").exists(), "Unexpected top-level sparse.yaml"


def test_wan22_calibrated_export(tmp_path):
    """Inject calibration params via the Python API and verify the exported config.

    Calibration can't succeed on tiny models via the Triton kernel (not enough
    data points in the 10%-90% sparsity range), so this test sparsifies, injects
    calibration params directly, and exports. Verifies the calibrated config schema
    (top-level ``threshold_scale_factor`` of the form ``a * exp(b * target_sparsity)``)
    and that the dense (cross-attention) layers are recorded under ``ignore``.
    """
    from diffusers import AutoencoderKLWan, WanPipeline

    import modelopt.torch.sparsity.attention_sparsity as mtsa
    from modelopt.torch.export import export_hf_checkpoint
    from modelopt.torch.sparsity.attention_sparsity.sparse_attention import SparseAttentionModule

    pipe_dir = create_tiny_wan22_pipeline_dir(tmp_path / "model")
    vae = AutoencoderKLWan.from_pretrained(pipe_dir, subfolder="vae", torch_dtype=torch.float32)
    pipe = WanPipeline.from_pretrained(pipe_dir, vae=vae, torch_dtype=torch.bfloat16)
    pipe.to("cuda")

    # Sparsify self-attention only; cross-attention (attn2) stays dense.
    sparse_cfg = {
        "*.attn1*": {
            "method": "triton_skip_softmax",
            "skip_softmax_threshold": 0.1,
            "backend": "triton",
            "is_causal": False,
            "warmup": 5,
            "enable": True,
        },
        "*.attn2*": {"enable": False},
        "default": {"enable": False},
    }
    config = {"sparse_cfg": sparse_cfg}
    for transformer in [pipe.transformer, pipe.transformer_2]:
        mtsa.sparsify(transformer, config)

    # Inject calibration params (simulating a successful log-space calibration).
    test_log_a = math.log(1.5)
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

    export_dir = tmp_path / "calibrated_export"
    export_hf_checkpoint(pipe, export_dir=export_dir)

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
        assert group_0["algorithm"] == "skip_softmax"
        assert group_0["targets"]

        # Dense (uncalibrated) layers must be recorded so deployment skips them too.
        assert "ignore" in group_0
        assert any(".attn2" in name for name in group_0["ignore"])

        # Opt-in warmup metadata is carried through (exported only when > 0).
        assert group_0["warmup"] == 5

        # threshold_scale_factor is a top-level block: a * exp(b * target_sparsity).
        tsf = sa_config["threshold_scale_factor"]
        assert tsf["formula"] == "a * exp(b * target_sparsity)"
        assert tsf["prefill"]["a"] == pytest.approx(math.exp(test_log_a))
        assert tsf["prefill"]["b"] == pytest.approx(test_b)

        # Calibrated mode — no raw_threshold.
        assert "raw_threshold" not in group_0

    # Sparse config lives only in config.json — no standalone sparse.yaml.
    assert not (export_dir / "sparse.yaml").exists(), "Unexpected top-level sparse.yaml"
