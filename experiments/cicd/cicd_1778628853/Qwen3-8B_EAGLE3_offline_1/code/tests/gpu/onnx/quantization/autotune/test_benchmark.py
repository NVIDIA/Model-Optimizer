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

"""Tests for autotune benchmark (TrtExecBenchmark, TensorRTPyBenchmark).

Covers:
- Pure-Python logic: _validate_shape_range, _free_buffers, __init__ import guards.
- Mocked subprocess: TrtExecBenchmark.run error paths and latency parsing.
- PyTorch CUDA management: pinned host allocation, stream creation, H2D/D2H copy
  loops in _run_warmup and _run_timing (mocked TRT context). Requires CUDA.
- Full integration: TensorRTPyBenchmark and TrtExecBenchmark end-to-end.
  Requires TensorRT and trtexec respectively.
"""

import contextlib
import os
import shutil
import tempfile
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
import torch
from _test_utils.onnx.quantization.autotune.models import _create_simple_conv_onnx_model

import modelopt.onnx.quantization.autotune.benchmark as bm
from modelopt.onnx.quantization.autotune import TensorRTPyBenchmark, TrtExecBenchmark
from modelopt.onnx.quantization.autotune.benchmark import _validate_shape_range

# --- fixtures ---


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


@pytest.fixture
def trtexec_bench(tmp_path):
    return TrtExecBenchmark(
        timing_cache_file=str(tmp_path / "cache.bin"),
        warmup_runs=1,
        timing_runs=2,
    )


# --- helpers ---


def _make_bench(warmup_runs=2, timing_runs=3):
    """Instantiate TensorRTPyBenchmark without triggering __init__ guards."""
    bench = bm.TensorRTPyBenchmark.__new__(bm.TensorRTPyBenchmark)
    bench.warmup_runs = warmup_runs
    bench.timing_runs = timing_runs
    bench.logger = MagicMock()
    return bench


def _make_buffers(size=4):
    """Return (inputs, outputs) using real pinned host + CUDA device tensors."""
    host_in = torch.ones(size).pin_memory()
    device_in = torch.zeros(size, device="cuda")
    host_out = torch.zeros(size).pin_memory()
    device_out = torch.ones(size, device="cuda") * 2.0
    inputs = [{"host": host_in, "device": device_in, "name": "x"}]
    outputs = [{"host": host_out, "device": device_out, "name": "y"}]
    return inputs, outputs


# --- _validate_shape_range ---


def test_validate_shape_range_valid():
    _validate_shape_range([1, 1], [2, 2], [4, 4])


def test_validate_shape_range_equal_bounds():
    _validate_shape_range([2, 3], [2, 3], [2, 3])


@pytest.mark.parametrize(
    ("min_s", "opt_s", "max_s"),
    [
        ([1, 1], [2, 2], [4]),
        ([1], [2, 2], [4, 4]),
    ],
)
def test_validate_shape_range_length_mismatch(min_s, opt_s, max_s):
    with pytest.raises(ValueError, match="same length"):
        _validate_shape_range(min_s, opt_s, max_s)


@pytest.mark.parametrize(
    ("min_s", "opt_s", "max_s"),
    [
        ([3], [2], [4]),  # min > opt
        ([1], [5], [4]),  # opt > max
        ([5], [3], [2]),  # both violated
    ],
)
def test_validate_shape_range_invalid_order(min_s, opt_s, max_s):
    with pytest.raises(ValueError, match="min <= opt <= max"):
        _validate_shape_range(min_s, opt_s, max_s)


# --- TensorRTPyBenchmark._free_buffers ---


def test_free_buffers_clears_list():
    bufs = [{"host": object(), "device": object(), "name": "x"}]
    bm.TensorRTPyBenchmark._free_buffers(bufs)
    assert bufs == []


def test_free_buffers_empty_list():
    bufs = []
    bm.TensorRTPyBenchmark._free_buffers(bufs)
    assert bufs == []


# --- TensorRTPyBenchmark.__init__ import guards ---


def test_tensorrt_py_benchmark_raises_without_trt():
    with patch.object(bm, "TRT_AVAILABLE", False), pytest.raises(ImportError, match="TensorRT"):
        bm.TensorRTPyBenchmark()


def test_tensorrt_py_benchmark_raises_without_torch_cuda():
    # TRT guard passes; TORCH_CUDA guard fires before any trt symbol is used.
    with (
        patch.object(bm, "TRT_AVAILABLE", True),
        patch.object(bm, "TORCH_CUDA_AVAILABLE", False),
        pytest.raises(ImportError, match="PyTorch"),
    ):
        bm.TensorRTPyBenchmark()


# --- TrtExecBenchmark.run (mocked) ---


def test_trtexec_run_returns_inf_on_nonzero_returncode(trtexec_bench, tmp_path):
    model_path = str(tmp_path / "model.onnx")
    (tmp_path / "model.onnx").write_bytes(b"")

    mock_result = MagicMock()
    mock_result.returncode = 1
    mock_result.stderr = "engine build error"
    mock_result.stdout = ""

    with patch("subprocess.run", return_value=mock_result):
        assert trtexec_bench.run(model_path) == float("inf")


def test_trtexec_run_returns_inf_when_latency_not_parsed(trtexec_bench, tmp_path):
    model_path = str(tmp_path / "model.onnx")
    (tmp_path / "model.onnx").write_bytes(b"")

    mock_result = MagicMock()
    mock_result.returncode = 0
    mock_result.stdout = "Build complete. No latency line here."
    mock_result.stderr = ""

    with patch("subprocess.run", return_value=mock_result):
        assert trtexec_bench.run(model_path) == float("inf")


def test_trtexec_run_returns_parsed_latency(trtexec_bench, tmp_path):
    model_path = str(tmp_path / "model.onnx")
    (tmp_path / "model.onnx").write_bytes(b"")

    mock_result = MagicMock()
    mock_result.returncode = 0
    mock_result.stdout = "[I] Latency: min = 2.50 ms, max = 4.00 ms, median = 3.14 ms"
    mock_result.stderr = ""

    with patch("subprocess.run", return_value=mock_result):
        assert trtexec_bench.run(model_path) == pytest.approx(3.14)


def test_trtexec_run_returns_inf_when_binary_not_found(trtexec_bench, tmp_path):
    model_path = str(tmp_path / "model.onnx")
    (tmp_path / "model.onnx").write_bytes(b"")

    with patch("subprocess.run", side_effect=FileNotFoundError):
        assert trtexec_bench.run(model_path) == float("inf")


def test_trtexec_run_accepts_bytes_input(trtexec_bench):
    mock_result = MagicMock()
    mock_result.returncode = 0
    mock_result.stdout = "[I] Latency: min = 4.00 ms, max = 6.00 ms, median = 5.00 ms"
    mock_result.stderr = ""

    with patch("subprocess.run", return_value=mock_result):
        assert trtexec_bench.run(b"fake onnx bytes") == pytest.approx(5.0)


# --- TensorRTPyBenchmark._alloc_pinned_host ---


def test_alloc_pinned_host_returns_pinned_tensor_and_numpy_view():
    size = 16
    host_tensor, arr = bm.TensorRTPyBenchmark._alloc_pinned_host(size, np.float32)

    assert host_tensor.is_pinned()
    assert isinstance(arr, np.ndarray)
    assert arr.shape == (size,)
    assert arr.dtype == np.float32

    # Tensor and numpy array share the same memory.
    host_tensor[0] = 42.0
    assert arr[0] == pytest.approx(42.0)


@pytest.mark.parametrize("dtype", [np.float16, np.float32, np.int8, np.int32])
def test_alloc_pinned_host_dtype(dtype):
    host_tensor, arr = bm.TensorRTPyBenchmark._alloc_pinned_host(8, dtype)
    assert host_tensor.is_pinned()
    assert arr.dtype == dtype
    assert arr.shape == (8,)


# --- torch.cuda.Stream integration points ---


def test_cuda_stream_handle_is_integer():
    """stream.cuda_stream must be an int — this is passed directly to TRT execute_async_v3."""
    stream = torch.cuda.Stream()
    assert isinstance(stream.cuda_stream, int)


# --- TensorRTPyBenchmark._run_warmup ---


def test_run_warmup_calls_execute_async_v3_correct_times():
    bench = _make_bench(warmup_runs=3)
    inputs, outputs = _make_buffers()
    context = MagicMock()
    stream = torch.cuda.Stream()

    bench._run_warmup(context, inputs, outputs, stream)

    assert context.execute_async_v3.call_count == 3


def test_run_warmup_passes_stream_handle_to_trt():
    bench = _make_bench(warmup_runs=1)
    inputs, outputs = _make_buffers()
    context = MagicMock()
    stream = torch.cuda.Stream()

    bench._run_warmup(context, inputs, outputs, stream)

    context.execute_async_v3.assert_called_once_with(stream.cuda_stream)


def test_run_warmup_copies_host_to_device():
    bench = _make_bench(warmup_runs=1)
    inputs, outputs = _make_buffers()
    context = MagicMock()
    stream = torch.cuda.Stream()

    bench._run_warmup(context, inputs, outputs, stream)
    torch.cuda.synchronize()

    # host_in was all-ones; device_in should now match.
    assert torch.allclose(inputs[0]["device"].cpu(), inputs[0]["host"])


# --- TensorRTPyBenchmark._run_timing ---


def test_run_timing_returns_correct_number_of_latencies():
    bench = _make_bench(timing_runs=4)
    inputs, outputs = _make_buffers()
    context = MagicMock()
    stream = torch.cuda.Stream()

    latencies = bench._run_timing(context, inputs, outputs, stream)

    assert len(latencies) == 4


def test_run_timing_latencies_are_non_negative():
    bench = _make_bench(timing_runs=3)
    inputs, outputs = _make_buffers()
    context = MagicMock()
    stream = torch.cuda.Stream()

    latencies = bench._run_timing(context, inputs, outputs, stream)

    assert all(lat >= 0.0 for lat in latencies)


def test_run_timing_calls_execute_async_v3_correct_times():
    bench = _make_bench(timing_runs=3)
    inputs, outputs = _make_buffers()
    context = MagicMock()
    stream = torch.cuda.Stream()

    bench._run_timing(context, inputs, outputs, stream)

    assert context.execute_async_v3.call_count == 3


def test_run_timing_passes_stream_handle_to_trt():
    bench = _make_bench(timing_runs=1)
    inputs, outputs = _make_buffers()
    context = MagicMock()
    stream = torch.cuda.Stream()

    bench._run_timing(context, inputs, outputs, stream)

    context.execute_async_v3.assert_called_once_with(stream.cuda_stream)


# --- TensorRTPyBenchmark (integration) ---


class TestTensorRTPyBenchmark:
    """End-to-end tests for TensorRTPyBenchmark. Requires TensorRT and CUDA."""

    @pytest.fixture(autouse=True)
    def _require_tensorrt_and_torch_cuda(self):
        pytest.importorskip("tensorrt")
        if not bm.TORCH_CUDA_AVAILABLE:
            pytest.skip("CUDA not available")

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


# --- TrtExecBenchmark (integration) ---


class TestTrtExecBenchmark:
    """End-to-end tests for TrtExecBenchmark. Requires trtexec in PATH."""

    @pytest.fixture(autouse=True)
    def _require_trtexec(self):
        if shutil.which("trtexec") is None:
            pytest.skip("trtexec not found in PATH", allow_module_level=False)

    def test_run_with_path(self, simple_conv_model_path):
        """TrtExecBenchmark accepts model path and returns finite latency."""
        with tempfile.NamedTemporaryFile(suffix=".cache", delete=False) as f:
            cache_path = f.name
        try:
            benchmark = TrtExecBenchmark(timing_cache_file=cache_path, warmup_runs=1, timing_runs=2)
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
            benchmark = TrtExecBenchmark(timing_cache_file=cache_path, warmup_runs=1, timing_runs=2)
            latency_ms = benchmark.run(simple_conv_model_bytes)
            assert isinstance(latency_ms, float)
            assert latency_ms > 0
            assert latency_ms != float("inf")
        finally:
            with contextlib.suppress(OSError):
                os.unlink(cache_path)
