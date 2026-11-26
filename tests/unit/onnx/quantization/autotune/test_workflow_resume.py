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

"""Tests for resuming region_pattern_autotuning_workflow from a checkpoint."""

import pytest
from _test_utils.onnx.quantization.autotune.models import _create_simple_conv_onnx_model

from modelopt.onnx.quantization.autotune import Config, QDQAutotuner, workflows


class _BenchmarkReachedError(Exception):
    pass


def _write_checkpoint(output_dir, **config_kwargs):
    autotuner = QDQAutotuner(_create_simple_conv_onnx_model())
    autotuner.initialize(Config(**config_kwargs))
    autotuner.submit(10.0)
    output_dir.mkdir(parents=True, exist_ok=True)
    autotuner.save_state(str(output_dir / "autotuner_state.yaml"))


@pytest.fixture
def no_benchmark(monkeypatch):
    def _raise(*args, **kwargs):
        raise _BenchmarkReachedError

    monkeypatch.setattr(workflows, "benchmark_onnx_model", _raise)


@pytest.mark.parametrize(
    ("checkpoint", "requested"),
    [
        ({"default_quant_type": "int8"}, {"quant_type": "fp8"}),
        ({"default_quant_type": "fp8"}, {"quant_type": "int8"}),
        ({"default_dq_dtype": "float32"}, {"default_dq_dtype": "float16"}),
    ],
)
def test_resume_rejects_mismatched_quantization(tmp_path, no_benchmark, checkpoint, requested):
    _write_checkpoint(tmp_path, **checkpoint)
    with pytest.raises(ValueError, match="does not match the requested quantization settings"):
        workflows.region_pattern_autotuning_workflow(
            _create_simple_conv_onnx_model(), tmp_path, **requested
        )


def test_resume_accepts_matching_quantization(tmp_path, no_benchmark):
    _write_checkpoint(tmp_path, default_quant_type="fp8")
    with pytest.raises(_BenchmarkReachedError):
        workflows.region_pattern_autotuning_workflow(
            _create_simple_conv_onnx_model(), tmp_path, quant_type="fp8"
        )
