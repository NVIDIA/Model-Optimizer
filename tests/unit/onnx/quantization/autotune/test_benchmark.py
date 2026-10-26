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

"""Tests for the default timing-cache file in autotune/benchmark.py."""

from types import SimpleNamespace

import pytest

from modelopt.onnx.quantization.autotune import benchmark
from modelopt.onnx.quantization.autotune.benchmark import Benchmark, TrtExecBenchmark


class _NoOpBenchmark(Benchmark):
    def run(self, path_or_bytes, log_file=None):
        return 0.0


H100 = ("NVIDIA H100 80GB HBM3", (9, 0))
GB200 = ("NVIDIA GB200", (10, 0))
# The banner of a trtexec that sits in an image whose Python tensorrt is another version (10.16.1.11).
TRTEXEC_BANNER = "&&&& RUNNING TensorRT.trtexec [TensorRT v110200] [b87] # trtexec"


@pytest.mark.parametrize(
    ("gpu", "binding_version", "trtexec_banner", "benchmark_cls", "tag"),
    [
        # The Python API is keyed by the binding's version, whatever trtexec says.
        (H100, "10.13.0.35", None, _NoOpBenchmark, "nvidia_h100_80gb_hbm3_sm90_trt10.13.0.35"),
        (GB200, "10.16.1.11", TRTEXEC_BANNER, _NoOpBenchmark, "nvidia_gb200_sm100_trt10.16.1.11"),
        # trtexec is keyed by its own version, even if the binding is another one or missing.
        (GB200, "10.16.1.11", TRTEXEC_BANNER, TrtExecBenchmark, "nvidia_gb200_sm100_trt11.2.0"),
        (H100, None, TRTEXEC_BANNER, TrtExecBenchmark, "nvidia_h100_80gb_hbm3_sm90_trt11.2.0"),
        # A trtexec that cannot run leaves the version out: the binding's would be the wrong one.
        (H100, "10.13.0.35", None, TrtExecBenchmark, "nvidia_h100_80gb_hbm3_sm90"),
        (None, None, None, _NoOpBenchmark, "default"),
    ],
)
def test_default_timing_cache_is_per_gpu_and_builder_version(
    tmp_path, monkeypatch, gpu, binding_version, trtexec_banner, benchmark_cls, tag
):
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path))
    monkeypatch.setattr(benchmark, "TORCH_CUDA_AVAILABLE", gpu is not None)
    if gpu is not None:
        name, capability = gpu
        monkeypatch.setattr(benchmark.torch.cuda, "get_device_name", lambda *_: name)
        monkeypatch.setattr(benchmark.torch.cuda, "get_device_capability", lambda *_: capability)
    monkeypatch.setattr(benchmark, "TRT_AVAILABLE", binding_version is not None)
    monkeypatch.setattr(
        benchmark, "trt", SimpleNamespace(__version__=binding_version), raising=False
    )

    def run_trtexec(**_):
        if trtexec_banner is None:
            raise FileNotFoundError("trtexec")
        return SimpleNamespace(stdout=trtexec_banner, stderr="")

    monkeypatch.setattr(benchmark, "_run_trtexec_command", run_trtexec)

    expected = tmp_path / "modelopt" / f"trtexec_timing_{tag}.cache"
    assert benchmark_cls().timing_cache_file == str(expected)
