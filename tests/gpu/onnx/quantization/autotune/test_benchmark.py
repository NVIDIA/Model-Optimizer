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

"""GPU tests for autotune Benchmark (TrtExecBenchmark, TensorRTPyBenchmark)."""

import contextlib
import os
import shutil
import tempfile

import pytest
from _test_utils.onnx.quantization.autotune.models import _create_simple_conv_onnx_model

from modelopt.onnx.quantization.autotune import TensorRTPyBenchmark, TrtExecBenchmark


@pytest.fixture
def simple_conv_model_bytes():
    """ONNX model bytes: Input -> Conv -> Relu -> Output (from _test_utils)."""
    model = _create_simple_conv_onnx_model()
    return model.SerializeToString()


@pytest.fixture
def simple_conv_model_path(simple_conv_model_bytes, tmp_path):
    """Path to ONNX model file (same graph as simple_conv_model_bytes)."""
    path = tmp_path / "simple_conv.onnx"
    path.write_bytes(simple_conv_model_bytes)
    return str(path)


class TestTensorRTPyBenchmark:
    """Tests for TensorRTPyBenchmark (TensorRT Python API + cudart)."""

    @pytest.fixture(autouse=True)
    def _require_tensorrt_and_cudart(self):
        pytest.importorskip("tensorrt")
        try:
            from cuda.bindings import runtime  # noqa: F401
        except ImportError:
            try:
                from cuda import cudart  # noqa: F401  # deprecated: prefer cuda.bindings.runtime
            except ImportError:
                pytest.skip("cuda-python (cudart) not available", allow_module_level=False)

    def test_run_with_bytes(self, simple_conv_model_bytes):
        """TensorRTPyBenchmark accepts model bytes and returns finite latency."""
        benchmark = TensorRTPyBenchmark(warmup_runs=1, timing_runs=2)
        latency_ms = benchmark.run(simple_conv_model_bytes)
        assert isinstance(latency_ms, float)
        assert latency_ms > 0
        assert latency_ms != float("inf")

    def test_run_with_path(self, simple_conv_model_path):
        """TensorRTPyBenchmark accepts model path and returns finite latency."""
        benchmark = TensorRTPyBenchmark(warmup_runs=1, timing_runs=2)
        latency_ms = benchmark.run(simple_conv_model_path)
        assert isinstance(latency_ms, float)
        assert latency_ms > 0
        assert latency_ms != float("inf")

    def test_callable(self, simple_conv_model_bytes):
        """Benchmark is callable and returns same as run()."""
        benchmark = TensorRTPyBenchmark(warmup_runs=1, timing_runs=2)
        latency_ms = benchmark(simple_conv_model_bytes)
        assert isinstance(latency_ms, float)
        assert latency_ms > 0


class TestTrtExecBenchmark:
    """Tests for TrtExecBenchmark (trtexec CLI)."""

    @pytest.fixture(autouse=True)
    def _require_trtexec(self):
        if shutil.which("trtexec") is None:
            pytest.skip("trtexec not found in PATH", allow_module_level=False)

    def test_run_with_path(self, simple_conv_model_path):
        """TrtExecBenchmark accepts model path and returns finite latency."""
        with tempfile.NamedTemporaryFile(suffix=".cache", delete=False) as f:
            cache_path = f.name
        try:
            benchmark = TrtExecBenchmark(
                timing_cache_file=cache_path,
                warmup_runs=1,
                timing_runs=2,
            )
            latency_ms = benchmark.run(simple_conv_model_path)
            assert isinstance(latency_ms, float)
            assert latency_ms > 0
            assert latency_ms != float("inf")
        finally:
            with contextlib.suppress(OSError):
                os.unlink(cache_path)

    def test_run_with_bytes(self, simple_conv_model_bytes):
        """TrtExecBenchmark accepts model bytes (writes temp file) and returns finite latency."""
        with tempfile.NamedTemporaryFile(suffix=".cache", delete=False) as f:
            cache_path = f.name
        try:
            benchmark = TrtExecBenchmark(
                timing_cache_file=cache_path,
                warmup_runs=1,
                timing_runs=2,
            )
            latency_ms = benchmark.run(simple_conv_model_bytes)
            assert isinstance(latency_ms, float)
            assert latency_ms > 0
            assert latency_ms != float("inf")
        finally:
            with contextlib.suppress(OSError):
                os.unlink(cache_path)
