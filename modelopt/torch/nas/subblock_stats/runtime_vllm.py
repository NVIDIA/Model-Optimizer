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
"""vLLM Runtime Benchmark Integration for ModelOpt NAS Subblocks.

This module provides the integration logic to empirically benchmark subblock
runtime statistics within transformer architectures using the vLLM latency
benchmark. Each invocation is launched in a dedicated subprocess so that GPU
memory and CUDA state are fully reclaimed when the subprocess exits, allowing
many sequential benchmarks to run in a single Python session without leaking.

Usage:
    - Call `run_vllm_latency_benchmark` with a model path and a
      `RuntimeConfig` instance to run a latency benchmark and
      return the average latency for the configuration (in milliseconds).
"""

import json
import subprocess  # nosec B404
from pathlib import Path

from modelopt.torch.nas.subblock_stats.runtime_utils import RuntimeConfig


def run_vllm_latency_benchmark(model_path: Path, runtime_config: RuntimeConfig) -> float | None:
    """Run ``vllm bench latency`` in a fresh subprocess and return avg latency in ms.

    Spawning a subprocess per call gives OS-level isolation: GPU memory, CUDA
    context, and vLLM engine state are fully released on subprocess exit, so
    many calls in one parent process do not accumulate.
    """
    output_json_path = model_path / "vllm_latency_benchmark.json"
    max_model_len = runtime_config.prefill_seq_len + runtime_config.generation_seq_len

    cmd = [
        "vllm",
        "bench",
        "latency",
        "--model",
        str(model_path),
        "--input-len",
        str(runtime_config.prefill_seq_len),
        "--output-len",
        str(runtime_config.generation_seq_len),
        "--batch-size",
        "1",
        "--output-json",
        str(output_json_path),
        "--max-model-len",
        str(max_model_len),
        "--num-iters-warmup",
        str(runtime_config.num_warmup_iters),
        "--num-iters",
        str(runtime_config.num_iters),
        "--max-num-seqs",
        "1",
        "--tensor-parallel-size",
        "1",
        "--pipeline-parallel-size",
        "1",
        "--distributed-executor-backend",
        "external_launcher",
        # Required for accurate per-block runtime stats.
        "--optimization-level",
        "0",
    ]

    # cmd is a fixed list of strings (no shell, no untrusted input).
    vllm_results = None
    try:
        subprocess.run(
            cmd,
            check=True,
            capture_output=True,
            text=True,
            timeout=1800,  # 30 minutes
        )  # nosec B603
    except subprocess.TimeoutExpired as exc:
        raise TimeoutError("vLLM latency benchmark timed out") from exc
    except subprocess.CalledProcessError as exc:
        raise RuntimeError(exc.stderr or exc.stdout or "vLLM latency benchmark failed") from exc

    if output_json_path.exists():
        with open(output_json_path) as f:
            vllm_results = json.load(f)
            vllm_results = vllm_results["avg_latency"] * 1000  # seconds -> milliseconds

    return vllm_results
