# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

import sys
import tempfile
from pathlib import Path

import onnx
import pytest
from _test_utils.onnx.quantization.autotune import models as _test_models

import modelopt.onnx.quantization.autotune.__main__ as autotune_cli
import modelopt.onnx.quantization.autotune.workflows as workflows
from modelopt.onnx.quantization.autotune.autotuner import QDQAutotuner
from modelopt.onnx.quantization.autotune.common import Config


def _save_checkpoint(model, state_path: Path, quant_type: str, baseline_ms: float = 1.0) -> bytes:
    autotuner = QDQAutotuner(model)
    autotuner.initialize(Config(default_quant_type=quant_type))
    autotuner.submit(baseline_ms)
    autotuner.save_state(str(state_path))
    return state_path.read_bytes()


@pytest.mark.parametrize(
    ("saved_type", "requested_type", "mode"),
    [
        ("int8", "fp8", "explicit_dir"),
        ("fp8", "int8", "explicit_state"),
        ("fp8", None, "auto_discovered_default"),
        ("int8", "fp8", "parent_relative_dir"),
    ],
)
def test_workflow_rejects_incompatible_quant_type(
    monkeypatch, tmp_path, saved_type, requested_type, mode
):
    """Conflicting explicit or auto-discovered checkpoints fail without mutating files."""
    simple_conv_model = _test_models._create_simple_conv_onnx_model()
    results_dir = tmp_path / "results" if mode == "parent_relative_dir" else tmp_path
    results_dir.mkdir(exist_ok=True)
    state = results_dir / "autotuner_state.yaml"
    before = _save_checkpoint(simple_conv_model, state, saved_type)
    source = tmp_path / "source.onnx"
    onnx.save(simple_conv_model, source)
    mkdtemp_calls = []
    real_mkdtemp = tempfile.mkdtemp
    monkeypatch.setattr(
        tempfile, "mkdtemp", lambda *a, **kw: mkdtemp_calls.append(1) or real_mkdtemp(*a, **kw)
    )

    out_arg = (
        tmp_path / "missing_hop" / ".." / "results"
        if mode == "parent_relative_dir"
        else (None if mode == "explicit_state" else results_dir)
    )
    state_arg = str(state) if mode == "explicit_state" else None
    expected_req = requested_type or "int8"

    with pytest.raises(ValueError, match=rf"{saved_type}.*{expected_req}"):
        workflows.region_pattern_autotuning_workflow(
            str(source),
            out_arg,
            state_file=state_arg,
            num_schemes_per_region=1,
            quant_type=requested_type,
        )

    assert state.read_bytes() == before
    assert not (tmp_path / "missing_hop").exists()
    assert not mkdtemp_calls


def test_cli_and_workflow_inherit_saved_type_when_explicit_state_omits_quant_type(
    monkeypatch, tmp_path
):
    """An explicit --state_file without --quant_type inherits the checkpoint's saved type."""
    simple_conv_model = _test_models._create_simple_conv_onnx_model()
    model_path = tmp_path / "model.onnx"
    onnx.save(simple_conv_model, model_path)
    state_path = tmp_path / "fp8_state.yaml"
    _save_checkpoint(simple_conv_model, state_path, "fp8", baseline_ms=0.8)
    out_dir = tmp_path / "resume_out"

    monkeypatch.setattr(workflows, "benchmark_onnx_model", lambda *args, **kwargs: 0.7)
    monkeypatch.setattr(autotune_cli, "init_benchmark_instance", lambda **kwargs: object())
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "modelopt.onnx.quantization.autotune",
            "--onnx_path",
            str(model_path),
            "--output_dir",
            str(out_dir),
            "--state_file",
            str(state_path),
            "--schemes_per_region",
            "1",
        ],
    )

    assert autotune_cli.run_autotune() == 0
    resumed = QDQAutotuner(simple_conv_model)
    resumed.initialize(Config())
    resumed.load_state(str(state_path))
    assert resumed.config.default_quant_type == "fp8"


def test_workflow_transformed_cross_type_resume_remeasures_baseline(monkeypatch, tmp_path):
    """Calibrated PTQ resumes with model_transform replace baseline timing and adopt quant_type."""
    simple_conv_model = _test_models._create_simple_conv_onnx_model()
    state_path = tmp_path / "int8_state.yaml"
    _save_checkpoint(simple_conv_model, state_path, "int8", baseline_ms=0.5)
    measured = iter([1.25, 1.10, 1.05])
    monkeypatch.setattr(workflows, "benchmark_onnx_model", lambda *args, **kwargs: next(measured))

    autotuner = workflows.region_pattern_autotuning_workflow(
        simple_conv_model,
        output_dir=tmp_path / "transformed_out",
        state_file=str(state_path),
        num_schemes_per_region=1,
        quant_type="fp8",
        model_transform=lambda m: m,
    )

    assert autotuner.config.default_quant_type == "fp8"
    assert autotuner.baseline_latency_ms == pytest.approx(1.25)


def test_workflow_cleans_temporary_output_dir_when_omitted(monkeypatch, tmp_path):
    """Omitting output_dir allocates a temp workspace after validation and deletes it on exit."""
    simple_conv_model = _test_models._create_simple_conv_onnx_model()
    created_dirs = []
    real_mkdtemp = tempfile.mkdtemp

    def tracked_mkdtemp(*args, **kwargs):
        d = Path(real_mkdtemp(*args, **kwargs, dir=tmp_path))
        created_dirs.append(d)
        return str(d)

    monkeypatch.setattr(workflows.tempfile, "mkdtemp", tracked_mkdtemp)
    monkeypatch.setattr(workflows, "benchmark_onnx_model", lambda *args, **kwargs: 1.0)

    autotuner = workflows.region_pattern_autotuning_workflow(
        simple_conv_model,
        output_dir=None,
        state_file=None,
        num_schemes_per_region=1,
        quant_type="int8",
    )

    assert autotuner.config.default_quant_type == "int8"
    assert len(created_dirs) == 1
    assert not created_dirs[0].exists()
