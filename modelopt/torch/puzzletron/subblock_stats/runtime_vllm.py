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

import hashlib
import json
import os
import socket
import subprocess  # nosec B404
from pathlib import Path
from types import SimpleNamespace

from ..tools.logger import mprint
from ..utils.vllm_adapter import convert_block_configs_to_per_layer_config
from .runtime_utils import RuntimeConfig

# torchrun / torch.elastic environment variables that a child ``vllm`` process
# must NOT inherit: with ``--distributed-executor-backend external_launcher``
# vLLM reads its world size from these, so inheriting ``WORLD_SIZE=8`` from the
# parent makes every benchmark think it is one of 8 ranks (and grab/replicate
# GPU memory) even though we pass ``--tensor-parallel-size 1``. We rebuild a
# clean single-process rendezvous instead.
_ELASTIC_ENV_PREFIXES = ("TORCHELASTIC_",)
_ELASTIC_ENV_VARS = {
    "GROUP_WORLD_SIZE",
    "ROLE_RANK",
    "ROLE_WORLD_SIZE",
    "ROLE_NAME",
    "OMP_NUM_THREADS",
}


def _free_tcp_port() -> int:
    """Pick an unused localhost TCP port (so concurrent benchmarks don't collide)."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def _build_subprocess_env(gpu_id: str | int | None) -> dict[str, str]:
    """Clean env for a single-process vLLM benchmark, optionally pinned to one GPU.

    Mirrors the known-good single-GPU rendezvous (WORLD_SIZE=1, fresh MASTER_PORT)
    and, when ``gpu_id`` is given, restricts the subprocess to that physical device
    so several benchmarks can run concurrently on different GPUs.
    """
    env = dict(os.environ)
    for key in list(env):
        if key.startswith(_ELASTIC_ENV_PREFIXES) or key in _ELASTIC_ENV_VARS:
            env.pop(key, None)
    env["WORLD_SIZE"] = "1"
    env["RANK"] = "0"
    env["LOCAL_RANK"] = "0"
    env["LOCAL_WORLD_SIZE"] = "1"
    env["GROUP_RANK"] = "0"
    env["MASTER_ADDR"] = "127.0.0.1"
    env["MASTER_PORT"] = str(_free_tcp_port())
    if gpu_id is not None:
        env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    return env


def _benchmark_cache_key(config_dict: dict, bench_args: dict) -> str:
    """Stable content hash of a benchmark: model config + benchmark arguments.

    Volatile fields (e.g. the temp checkpoint path) are stripped so identical
    specs hash equally across runs, enabling on-disk resume.
    """
    config_for_key = dict(config_dict)
    for volatile in ("_name_or_path", "name_or_path"):
        config_for_key.pop(volatile, None)
    payload = json.dumps(config_for_key, sort_keys=True, default=str) + json.dumps(
        bench_args, sort_keys=True, default=str
    )
    return hashlib.sha256(payload.encode()).hexdigest()


def run_vllm_latency_benchmark(
    model_path: Path,
    runtime_config: RuntimeConfig,
    gpu_id: str | int | None = None,
    cache_dir: Path | None = None,
) -> float:
    """Run ``vllm bench latency`` in a fresh subprocess and return avg latency in ms.

    Spawning a subprocess per call gives OS-level isolation: GPU memory, CUDA
    context, and vLLM engine state are fully released on subprocess exit, so
    many calls in one parent process do not accumulate.

    Args:
        gpu_id: physical device id to pin this benchmark to via
            ``CUDA_VISIBLE_DEVICES`` (lets several benchmarks share a node);
            ``None`` leaves device selection to vLLM.
        cache_dir: if given, results are memoized to ``cache_dir/<hash>.json``
            keyed by model config + benchmark args, so a re-run skips already
            measured subblocks (resume).
    """
    output_json_path = model_path / "vllm_latency_benchmark.json"
    max_model_len = runtime_config.prefill_seq_len + runtime_config.generation_seq_len
    # Benchmark concurrency: None -> 1 (single-stream latency, the historical
    # default); set runtime_config.max_num_seqs (e.g. to batch_size) to run the
    # prompts concurrently and measure true batched throughput.
    max_num_seqs = runtime_config.max_num_seqs if runtime_config.max_num_seqs is not None else 1

    with open(model_path / "config.json") as f:
        config = json.load(f)

    config = SimpleNamespace(**config)
    if convert_block_configs_to_per_layer_config(config):
        mprint("Converted block configs to per-layer config")
        with open(model_path / "config.json", "w") as f:
            json.dump(vars(config), f, indent=2)
    else:
        mprint("No block configs to convert")

    # TEMP DEBUG: dump resolved attention dims + actual q/k/v tensor shapes so we
    # can see why vLLM's gated QKV loader disagrees with the saved checkpoint.
    try:
        _dbg_cfg = vars(config)
        _dbg_tc = _dbg_cfg.get("text_config", _dbg_cfg)
        if not isinstance(_dbg_tc, dict):
            _dbg_tc = vars(_dbg_tc)
        mprint(
            "[QKV-DEBUG] config: hidden_size=%s num_attention_heads=%s "
            "num_key_value_heads=%s head_dim=%s attn_output_gate=%s model_type=%s"
            % (
                _dbg_tc.get("hidden_size"),
                _dbg_tc.get("num_attention_heads"),
                _dbg_tc.get("num_key_value_heads"),
                _dbg_tc.get("head_dim"),
                _dbg_tc.get("attn_output_gate"),
                _dbg_cfg.get("model_type"),
            )
        )
        import json as _json
        from safetensors import safe_open as _safe_open

        _idx_path = model_path / "model.safetensors.index.json"
        if _idx_path.exists():
            with open(_idx_path) as _f:
                _wmap = _json.load(_f).get("weight_map", {})
            for _name, _file in _wmap.items():
                if any(
                    _k in _name
                    for _k in ("q_proj.weight", "k_proj.weight", "v_proj.weight", "qkv")
                ):
                    with _safe_open(str(model_path / _file), framework="pt") as _st:
                        _shape = _st.get_slice(_name).get_shape()
                    mprint("[QKV-DEBUG] tensor %s shape=%s" % (_name, _shape))
    except Exception as _exc:  # pragma: no cover - diagnostic only
        mprint("[QKV-DEBUG] failed to dump dims/shapes: %r" % (_exc,))

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
        str(runtime_config.batch_size),
        "--output-json",
        str(output_json_path),
        "--max-model-len",
        str(max_model_len),
        "--num-iters-warmup",
        str(runtime_config.num_warmup_iters),
        "--num-iters",
        str(runtime_config.num_iters),
        "--max-num-seqs",
        str(max_num_seqs),
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
    descriptor_args = list(runtime_config.descriptor.runtime_vllm_benchmark_args(config))
    cmd.extend(descriptor_args)

    # On-disk resume: skip the (expensive) benchmark if this exact spec was
    # already measured. Key off the finalized config + benchmark arguments.
    cache_file = None
    if cache_dir is not None:
        bench_args = {
            "input_len": runtime_config.prefill_seq_len,
            "output_len": runtime_config.generation_seq_len,
            "batch_size": runtime_config.batch_size,
            "max_num_seqs": max_num_seqs,
            "max_model_len": max_model_len,
            "num_iters_warmup": runtime_config.num_warmup_iters,
            "num_iters": runtime_config.num_iters,
            "descriptor_args": descriptor_args,
        }
        cache_file = Path(cache_dir) / f"{_benchmark_cache_key(vars(config), bench_args)}.json"
        if cache_file.exists():
            with open(cache_file) as f:
                return json.load(f)["avg_latency_ms"]

    # cmd is a fixed list of strings (no shell, no untrusted input). The env is
    # scrubbed of inherited torch.elastic vars and pinned to one GPU so several
    # benchmarks can run concurrently without colliding (see _build_subprocess_env).
    try:
        subprocess.run(
            cmd,
            check=True,
            capture_output=True,
            text=True,
            timeout=1800,  # 30 minutes
            env=_build_subprocess_env(gpu_id),
        )  # nosec B603
    except subprocess.TimeoutExpired as exc:
        raise TimeoutError("vLLM latency benchmark timed out") from exc
    except subprocess.CalledProcessError as exc:
        raise RuntimeError(exc.stderr or exc.stdout or "vLLM latency benchmark failed") from exc

    if output_json_path.exists():
        with open(output_json_path) as f:
            vllm_results = json.load(f)
        if "avg_latency" in vllm_results:
            avg_latency_ms = vllm_results["avg_latency"] * 1000  # seconds -> milliseconds
            if cache_file is not None:
                cache_file.parent.mkdir(parents=True, exist_ok=True)
                # Atomic write so a concurrent reader never sees a partial file.
                tmp_file = cache_file.with_name(cache_file.name + ".tmp")
                with open(tmp_file, "w") as f:
                    json.dump({"avg_latency_ms": avg_latency_ms, "gpu_id": str(gpu_id)}, f)
                tmp_file.replace(cache_file)
            return avg_latency_ms

    raise RuntimeError(f"vLLM benchmark output not found at {output_json_path}")
