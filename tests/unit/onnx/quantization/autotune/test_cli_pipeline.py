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

"""Tests for the autotune CLI pipeline (``run_autotune`` in __main__.py).

Mocks ``init_benchmark_instance`` and ``region_pattern_autotuning_workflow`` so
the end-to-end argv → exit-code path can be exercised without TensorRT or a
real benchmark. Real ONNX files are used for ``--onnx_path`` / ``--qdq_baseline``
because ``validate_file_path`` exits the process on missing files.
"""

from unittest.mock import MagicMock, patch

import onnx
import pytest
from _test_utils.onnx.quantization.autotune.models import _create_simple_conv_onnx_model

# The autotune CLI transitively imports ``tensorrt``; in environments where the
# package is locatable but its shared libs are missing, collection fails. Match
# the soft-skip pattern used by ``test_region_inspect.py``.
try:
    from modelopt.onnx.quantization.autotune.__main__ import MODE_PRESETS, run_autotune
except ImportError:  # pragma: no cover — exercised only in TRT-less envs
    pytest.skip("Autotune CLI requires TensorRT", allow_module_level=True)


@pytest.fixture
def onnx_model_path(tmp_path):
    """A real ONNX file on disk so ``validate_file_path`` succeeds."""
    path = tmp_path / "model.onnx"
    onnx.save(_create_simple_conv_onnx_model(), str(path))
    return str(path)


@pytest.fixture
def mocked_pipeline():
    """Patch ``init_benchmark_instance`` and the autotuning workflow.

    Yields ``(init_mock, workflow_mock)`` so individual tests can inspect call
    args. ``init_mock`` returns a sentinel benchmark by default; tests that need
    the failure path can override ``init_mock.return_value = None``.
    """
    with (
        patch("modelopt.onnx.quantization.autotune.__main__.init_benchmark_instance") as init_mock,
        patch(
            "modelopt.onnx.quantization.autotune.__main__.region_pattern_autotuning_workflow"
        ) as workflow_mock,
    ):
        init_mock.return_value = MagicMock(name="benchmark_instance")
        yield init_mock, workflow_mock


def _run_with_argv(argv):
    """Invoke ``run_autotune`` with a patched ``sys.argv``."""
    with patch("sys.argv", ["modelopt.onnx.quantization.autotune", *argv]):
        return run_autotune()


# --- success paths ---


def test_run_autotune_minimal_argv_success(mocked_pipeline, onnx_model_path):
    """Minimal argv (just --onnx_path) drives the pipeline to a clean exit."""
    init_mock, workflow_mock = mocked_pipeline

    exit_code = _run_with_argv(["--onnx_path", onnx_model_path])

    assert exit_code == 0
    init_mock.assert_called_once()
    workflow_mock.assert_called_once()


def test_run_autotune_default_uses_tensorrt_python_api(mocked_pipeline, onnx_model_path):
    """Without --use_trtexec, the benchmark is the TensorRT-Python backend."""
    init_mock, _ = mocked_pipeline

    _run_with_argv(["--onnx_path", onnx_model_path])

    assert init_mock.call_args.kwargs["use_trtexec"] is False


def test_run_autotune_use_trtexec_flag_propagates(mocked_pipeline, onnx_model_path):
    """``--use_trtexec`` flips the backend selection passed to init."""
    init_mock, _ = mocked_pipeline

    _run_with_argv(["--onnx_path", onnx_model_path, "--use_trtexec"])

    assert init_mock.call_args.kwargs["use_trtexec"] is True


def test_run_autotune_trtexec_args_split_to_list(mocked_pipeline, onnx_model_path):
    """``--trtexec_benchmark_args`` is a quoted string at the CLI but a list at the API."""
    init_mock, _ = mocked_pipeline

    _run_with_argv(
        [
            "--onnx_path",
            onnx_model_path,
            "--use_trtexec",
            "--trtexec_benchmark_args",
            "--fp16 --workspace=4096 --verbose",
        ]
    )

    assert init_mock.call_args.kwargs["trtexec_args"] == [
        "--fp16",
        "--workspace=4096",
        "--verbose",
    ]


def test_run_autotune_workflow_receives_model_and_options(mocked_pipeline, onnx_model_path):
    """The workflow is called with model path, quant_type, default_dq_dtype, verbose."""
    _, workflow_mock = mocked_pipeline

    _run_with_argv(
        [
            "--onnx_path",
            onnx_model_path,
            "--quant_type",
            "fp8",
            "--default_dq_dtype",
            "float16",
            "--verbose",
        ]
    )

    kwargs = workflow_mock.call_args.kwargs
    assert kwargs["model_or_path"] == onnx_model_path
    assert kwargs["quant_type"] == "fp8"
    assert kwargs["default_dq_dtype"] == "float16"
    assert kwargs["verbose"] is True


def test_run_autotune_qdq_baseline_path_propagates(mocked_pipeline, onnx_model_path, tmp_path):
    """``--qdq_baseline`` is validated then forwarded to the workflow."""
    baseline_path = tmp_path / "baseline.onnx"
    onnx.save(_create_simple_conv_onnx_model(), str(baseline_path))
    _, workflow_mock = mocked_pipeline

    _run_with_argv(["--onnx_path", onnx_model_path, "--qdq_baseline", str(baseline_path)])

    assert workflow_mock.call_args.kwargs["qdq_baseline_model"] == str(baseline_path)


def test_run_autotune_mode_preset_propagates_to_init(mocked_pipeline, onnx_model_path):
    """``--mode quick`` overrides warmup/timing runs that init_benchmark_instance receives."""
    init_mock, workflow_mock = mocked_pipeline
    preset = MODE_PRESETS["quick"]

    _run_with_argv(["--onnx_path", onnx_model_path, "--mode", "quick"])

    assert init_mock.call_args.kwargs["warmup_runs"] == preset["warmup_runs"]
    assert init_mock.call_args.kwargs["timing_runs"] == preset["timing_runs"]
    assert workflow_mock.call_args.kwargs["num_schemes_per_region"] == preset["schemes_per_region"]


# --- failure paths ---


def test_run_autotune_returns_1_when_init_fails(mocked_pipeline, onnx_model_path):
    """Benchmark init returning None short-circuits before the workflow runs."""
    init_mock, workflow_mock = mocked_pipeline
    init_mock.return_value = None

    exit_code = _run_with_argv(["--onnx_path", onnx_model_path])

    assert exit_code == 1
    workflow_mock.assert_not_called()


def test_run_autotune_returns_130_on_keyboard_interrupt(mocked_pipeline, onnx_model_path):
    """``Ctrl+C`` during the workflow returns the conventional 130 exit code."""
    _, workflow_mock = mocked_pipeline
    workflow_mock.side_effect = KeyboardInterrupt

    exit_code = _run_with_argv(["--onnx_path", onnx_model_path])

    assert exit_code == 130


def test_run_autotune_returns_1_on_workflow_exception(mocked_pipeline, onnx_model_path):
    """Any other workflow exception is caught and reported as exit code 1."""
    _, workflow_mock = mocked_pipeline
    workflow_mock.side_effect = RuntimeError("boom")

    exit_code = _run_with_argv(["--onnx_path", onnx_model_path])

    assert exit_code == 1


def test_run_autotune_missing_model_exits(tmp_path):
    """``validate_file_path`` calls ``sys.exit(1)`` when --onnx_path does not exist."""
    missing = tmp_path / "does_not_exist.onnx"

    with pytest.raises(SystemExit) as exc_info:
        _run_with_argv(["--onnx_path", str(missing)])

    assert exc_info.value.code == 1
