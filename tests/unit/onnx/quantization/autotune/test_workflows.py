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

import tempfile
from pathlib import Path

import onnx
import pytest
from _test_utils.onnx.quantization.autotune import models as _test_models

import modelopt.onnx.quantization.autotune.workflows as workflows
from modelopt.onnx.quantization.autotune.autotuner import QDQAutotuner
from modelopt.onnx.quantization.autotune.common import Config


@pytest.mark.parametrize(
    ("saved_type", "requested_type", "explicit_output_dir"),
    [("int8", "fp8", True), ("fp8", "int8", False)],
)
def test_workflow_rejects_incompatible_quant_type(
    monkeypatch, tmp_path, saved_type, requested_type, explicit_output_dir
):
    """A checkpoint must not silently override the workflow's requested quantization type."""
    simple_conv_model = _test_models._create_simple_conv_onnx_model()
    autotuner = QDQAutotuner(simple_conv_model)
    autotuner.initialize(Config(default_quant_type=saved_type))
    autotuner.submit(1.0)
    state = tmp_path / "autotuner_state.yaml"
    autotuner.save_state(str(state))
    before = state.read_bytes()
    source = tmp_path / "source.onnx"
    onnx.save(simple_conv_model, source)
    mkdtemp_calls = []
    real_mkdtemp = tempfile.mkdtemp
    monkeypatch.setattr(
        tempfile, "mkdtemp", lambda *a, **kw: mkdtemp_calls.append(1) or real_mkdtemp(*a, **kw)
    )

    with pytest.raises(
        ValueError,
        match=rf"Checkpoint quant_type={saved_type} does not match requested quant_type={requested_type}",
    ):
        workflows.region_pattern_autotuning_workflow(
            str(source),
            tmp_path if explicit_output_dir else None,
            state_file=None if explicit_output_dir else str(state),
            num_schemes_per_region=1,
            quant_type=requested_type,
        )

    assert state.read_bytes() == before
    assert set(tmp_path.iterdir()) == {state, source}
    assert not mkdtemp_calls


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
