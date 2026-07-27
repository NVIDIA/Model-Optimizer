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
import importlib.metadata
import json
import os
import socket
import subprocess  # nosec B404
import threading
from dataclasses import dataclass, replace
from pathlib import Path
from types import SimpleNamespace

from ..tools.logger import mprint
from ..utils.vllm_adapter import convert_block_configs_to_per_layer_config
from .runtime_utils import RuntimeConfig
from .topology import RuntimeTopology

# torchrun / torch.elastic environment variables that a child ``vllm`` process
# must NOT inherit: with ``--distributed-executor-backend external_launcher``
# vLLM reads its world size from these, so inheriting ``WORLD_SIZE=8`` from the
# parent makes every benchmark think it is one of 8 ranks (and grab/replicate
# GPU memory) even though we pass ``--tensor-parallel-size 1``. We rebuild a
# clean single-process rendezvous instead.
_ELASTIC_ENV_PREFIXES = ("TORCHELASTIC_",)
_ELASTIC_ENV_VARS = {
    "WORLD_SIZE",
    "RANK",
    "LOCAL_RANK",
    "LOCAL_WORLD_SIZE",
    "GROUP_RANK",
    "MASTER_ADDR",
    "MASTER_PORT",
    "GROUP_WORLD_SIZE",
    "ROLE_RANK",
    "ROLE_WORLD_SIZE",
    "ROLE_NAME",
    "OMP_NUM_THREADS",
}
_INHERITED_VLLM_RENDEZVOUS_ENV_VARS = {"VLLM_PORT", "VLLM_DP_MASTER_PORT"}

_RUNTIME_CACHE_SCHEMA_VERSION = 5


@dataclass(frozen=True)
class RuntimeMeasurement:
    """Additive phase timing carried through repeated-block differencing."""

    total_ms: float
    prefill_ms: float

    @classmethod
    def zero(cls) -> "RuntimeMeasurement":
        return cls(total_ms=0.0, prefill_ms=0.0)

    @classmethod
    def mean(cls, values) -> "RuntimeMeasurement":
        values = tuple(values)
        if not values:
            raise ValueError("RuntimeMeasurement.mean requires at least one value")
        return sum(values, cls.zero()) / len(values)

    @property
    def decode_ms(self) -> float:
        return self.total_ms - self.prefill_ms

    def decode_ms_per_token(self, generation_seq_len: int) -> float:
        decode_tokens = max(1, int(generation_seq_len) - 1)
        return self.decode_ms / decode_tokens

    def to_dict(self) -> dict[str, float]:
        return {"total_ms": float(self.total_ms), "prefill_ms": float(self.prefill_ms)}

    @classmethod
    def from_dict(cls, value: dict[str, float]) -> "RuntimeMeasurement":
        return cls(total_ms=float(value["total_ms"]), prefill_ms=float(value["prefill_ms"]))

    def __add__(self, other):
        if other == 0:
            return self
        if not isinstance(other, RuntimeMeasurement):
            return NotImplemented
        return RuntimeMeasurement(
            total_ms=self.total_ms + other.total_ms,
            prefill_ms=self.prefill_ms + other.prefill_ms,
        )

    __radd__ = __add__

    def __sub__(self, other):
        if not isinstance(other, RuntimeMeasurement):
            return NotImplemented
        return RuntimeMeasurement(
            total_ms=self.total_ms - other.total_ms,
            prefill_ms=self.prefill_ms - other.prefill_ms,
        )

    def __truediv__(self, divisor: float):
        return RuntimeMeasurement(
            total_ms=self.total_ms / divisor,
            prefill_ms=self.prefill_ms / divisor,
        )


def _free_tcp_port() -> int:
    """Pick an unused localhost TCP port (so concurrent benchmarks don't collide)."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def _build_subprocess_env(
    gpu_id: str | int | None,
    topology,
    overrides: tuple[tuple[str, str], ...] = (),
) -> dict[str, str]:
    """Clean env for vLLM-owned multiprocessing on one ordered GPU group.

    Mirrors the known-good single-GPU rendezvous (WORLD_SIZE=1, fresh MASTER_PORT)
    and, when ``gpu_id`` is given, restricts the subprocess to that physical device
    so several benchmarks can run concurrently on different GPUs.
    """
    env = dict(os.environ)
    for key in list(env):
        if (
            key.startswith(_ELASTIC_ENV_PREFIXES)
            or key in _ELASTIC_ENV_VARS
            or key in _INHERITED_VLLM_RENDEZVOUS_ENV_VARS
        ):
            env.pop(key, None)
    env.update({str(key): str(value) for key, value in overrides})
    # Every independent local vLLM engine needs its own rendezvous range. If
    # VLLM_PORT is absent, vLLM starts scanning at its process-wide default;
    # concurrent one-GPU engines can then all observe the same port as free and
    # race to bind it. A fresh base port gives each subprocess an independent
    # scan range while remaining intentionally absent from the cache identity.
    env.setdefault("VLLM_PORT", str(_free_tcp_port()))
    if topology.distributed_executor_backend == "external_launcher":
        if topology.world_size != 1:
            raise ValueError(
                "external_launcher runtime stats require an external multi-rank launcher; "
                "use distributed_executor_backend=mp for local GPU groups"
            )
        env.update(
            {
                "WORLD_SIZE": "1",
                "RANK": "0",
                "LOCAL_RANK": "0",
                "LOCAL_WORLD_SIZE": "1",
                "GROUP_RANK": "0",
                "MASTER_ADDR": "127.0.0.1",
                "MASTER_PORT": str(_free_tcp_port()),
            }
        )
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
    payload = json.dumps(
        {
            "schema_version": _RUNTIME_CACHE_SCHEMA_VERSION,
            "model_config": config_for_key,
            "benchmark_args": bench_args,
        },
        sort_keys=True,
        default=str,
    )
    return hashlib.sha256(payload.encode()).hexdigest()


def _runtime_estimator_metadata(runtime_config: RuntimeConfig) -> dict[str, object]:
    """Return cache provenance for the repeated-candidate estimator."""

    return {
        "schema": runtime_config.estimator_schema,
        "mode": runtime_config.estimator_mode,
        "effective_repeat_count": runtime_config.effective_repeat_count,
        "scaffold_policy": runtime_config.scaffold_policy,
    }


def _runtime_environment_metadata(runtime_config: RuntimeConfig) -> dict[str, str]:
    """Return explicit vLLM environment overrides for provenance and caching."""

    return dict(runtime_config.vllm_env)


def _cacheable_command(
    cmd: list[str], model_path: Path, output_json_path: Path
) -> list[str]:
    """Return the effective command with per-invocation paths normalized.

    The temporary checkpoint directory is deliberately different for every
    benchmark invocation. Keeping it in the cache identity makes every rerun a
    miss, defeating resume across allocations. Preserve the full command shape
    while replacing only the two volatile paths.
    """
    replacements = {
        str(model_path): "<MODEL_PATH>",
        str(output_json_path): "<OUTPUT_JSON>",
    }
    return [replacements.get(arg, arg) for arg in cmd]


def _has_attention_arg(args: list[str], key: str) -> bool:
    prefix = f"--attention-config.{key}"
    return any(arg == prefix or arg.startswith(f"{prefix}=") for arg in args)


def _scheduler_metadata_failure(output: str) -> bool:
    return "scheduler_metadata must have shape (metadata_size)" in output


def _mamba_align_block_size_failure(output: str) -> bool:
    return "In Mamba cache align mode, block_size" in output and "max_num_batched_tokens" in output


def _mamba_max_num_batched_tokens(max_model_len: int) -> int:
    """Safe scheduler budget for aligned hybrid-cache preflights.

    Aligned Mamba cache blocks are architecture/state-size dependent and can be
    hundreds of tokens even when a smoke prompt is only a few tokens long.
    Production long-context workloads naturally exceed that floor; keep a
    modest minimum only for short integration probes.
    """
    return max(2048, 2 * int(max_model_len))


def _transient_distributed_startup_failure(output: str) -> bool:
    transient_tcp = "TCP client failed to connect/validate" in output and (
        "Interrupted system call" in output or "Connection reset by peer" in output
    )
    # Editable vLLM is loaded from the shared workspace. Under a 16-worker
    # cold start, a spawned engine worker can very occasionally fail to resolve
    # an existing nested package while the other workers initialize normally.
    # Treat only this exact vLLM-internal import signature as transient; an
    # arbitrary missing dependency must still fail immediately.
    transient_editable_import = (
        "ModuleNotFoundError: No module named 'vllm.v1.engine'" in output
    )
    return transient_tcp or transient_editable_import


def _has_cli_arg(args: list[str], key: str) -> bool:
    return any(arg == key or arg.startswith(f"{key}=") for arg in args)


def _run_latency_cmd(
    cmd: list[str],
    *,
    gpu_id: str | int | None,
    topology,
    vllm_env: tuple[tuple[str, str], ...] = (),
) -> subprocess.CompletedProcess:
    return subprocess.run(
        cmd,
        check=True,
        capture_output=True,
        text=True,
        timeout=1800,  # 30 minutes
        env=_build_subprocess_env(gpu_id, topology, vllm_env),
    )  # nosec B603


def _called_process_failure_output(exc) -> str:
    """Preserve stdout root causes alongside stderr loader warnings."""
    pieces = []
    if getattr(exc, "stdout", None):
        pieces.append(f"--- stdout ---\n{exc.stdout}")
    if getattr(exc, "stderr", None):
        pieces.append(f"--- stderr ---\n{exc.stderr}")
    return "\n".join(pieces) or "vLLM latency benchmark failed"


def _topology_vllm_args(topology: RuntimeTopology) -> list[str]:
    """Translate one runtime-statistics topology to vLLM latency flags."""
    args = [
        "--tensor-parallel-size",
        str(topology.tensor_parallel_size),
        "--pipeline-parallel-size",
        str(topology.pipeline_parallel_size),
    ]
    args.extend(
        (
            "--prefill-context-parallel-size",
            str(topology.prefill_context_parallel_size),
            "--decode-context-parallel-size",
            str(topology.decode_context_parallel_size),
            "--distributed-executor-backend",
            topology.distributed_executor_backend,
        )
    )
    if topology.enable_expert_parallel:
        args.append("--enable-expert-parallel")
    return args


def _run_vllm_latency_phase(
    model_path: Path,
    runtime_config: RuntimeConfig,
    gpu_id: str | int | None = None,
    cache_dir: Path | None = None,
) -> float:
    """Run one ``vllm bench latency`` phase and return average latency in ms.

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
        # Required for accurate per-block runtime stats.
        "--optimization-level",
        "0",
    ]
    cmd.extend(_topology_vllm_args(runtime_config.topology))
    cmd.extend(runtime_config.extra_vllm_args)
    descriptor_args = list(runtime_config.descriptor.runtime_vllm_benchmark_args(config))
    cmd.extend(descriptor_args)
    # Align-mode hybrid models require a cache block larger than the exact
    # prompt+generation length. Supplying the known-safe value up front avoids
    # paying for a complete engine initialization that can only fail and be
    # retried for every GDN candidate.
    if not _has_cli_arg(cmd, "--max-num-batched-tokens"):
        cmd.append(
            f"--max-num-batched-tokens={_mamba_max_num_batched_tokens(max_model_len)}"
        )

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
            "extra_vllm_args": runtime_config.extra_vllm_args,
            "descriptor_args": descriptor_args,
            "topology": runtime_config.topology.to_dict(),
            "estimator": _runtime_estimator_metadata(runtime_config),
            "vllm_env": _runtime_environment_metadata(runtime_config),
            "effective_command": _cacheable_command(
                cmd, model_path, output_json_path
            ),
            "gpu_name": _gpu_name(),
            "vllm_version": _package_version("vllm"),
            "torch_version": _package_version("torch"),
        }
        cache_identity = {
            "schema_version": _RUNTIME_CACHE_SCHEMA_VERSION,
            "model_config": {
                key: value
                for key, value in vars(config).items()
                if key not in {"_name_or_path", "name_or_path"}
            },
            "benchmark_args": bench_args,
        }
        cache_file = Path(cache_dir) / f"{_benchmark_cache_key(vars(config), bench_args)}.json"
        if cache_file.exists():
            with open(cache_file) as f:
                return json.load(f)["avg_latency_ms"]
    else:
        cache_identity = None

    # cmd is a fixed list of strings (no shell, no untrusted input). The env is
    # scrubbed of inherited torch.elastic vars and pinned to one GPU so several
    # benchmarks can run concurrently without colliding (see _build_subprocess_env).
    effective_cmd = cmd
    fallback_args: tuple[str, ...] = ()
    try:
        _run_latency_cmd(
            effective_cmd,
            gpu_id=gpu_id,
            topology=runtime_config.topology,
            vllm_env=runtime_config.vllm_env,
        )
    except subprocess.TimeoutExpired as exc:
        raise TimeoutError("vLLM latency benchmark timed out") from exc
    except subprocess.CalledProcessError as exc:
        failure_output = _called_process_failure_output(exc)
        if _transient_distributed_startup_failure(failure_output):
            if output_json_path.exists():
                output_json_path.unlink()
            mprint("Transient vLLM distributed startup failure; retrying once")
            try:
                _run_latency_cmd(
                    effective_cmd,
                    gpu_id=gpu_id,
                    topology=runtime_config.topology,
                    vllm_env=runtime_config.vllm_env,
                )
            except subprocess.TimeoutExpired as retry_exc:
                raise TimeoutError("vLLM latency benchmark timed out") from retry_exc
            except subprocess.CalledProcessError as retry_exc:
                raise RuntimeError(_called_process_failure_output(retry_exc)) from retry_exc
            failure_output = ""
        can_retry_mamba_tokens = (
            _mamba_align_block_size_failure(failure_output)
            and not _has_cli_arg(cmd, "--max-num-batched-tokens")
        )
        can_retry_fa2 = (
            _scheduler_metadata_failure(failure_output)
            and not _has_attention_arg(cmd, "flash_attn_version")
            and not _has_attention_arg(cmd, "backend")
        )
        if not failure_output:
            pass
        elif not can_retry_mamba_tokens and not can_retry_fa2:
            raise RuntimeError(failure_output) from exc
        elif can_retry_mamba_tokens:
            fallback_args = (
                f"--max-num-batched-tokens={_mamba_max_num_batched_tokens(max_model_len)}",
            )
            retry_reason = "vLLM Mamba align block-size validation failure"
        else:
            fallback_args = ("--attention-config.flash_attn_version=2",)
            retry_reason = "vLLM FA3 scheduler metadata failure"
        if failure_output:
            effective_cmd = [*cmd, *fallback_args]
            if output_json_path.exists():
                output_json_path.unlink()
            mprint(f"{retry_reason}; retrying latency benchmark with {' '.join(fallback_args)}")
            try:
                _run_latency_cmd(
                    effective_cmd,
                    gpu_id=gpu_id,
                    topology=runtime_config.topology,
                    vllm_env=runtime_config.vllm_env,
                )
            except subprocess.TimeoutExpired as retry_exc:
                raise TimeoutError("vLLM latency benchmark timed out") from retry_exc
            except subprocess.CalledProcessError as retry_exc:
                raise RuntimeError(_called_process_failure_output(retry_exc)) from retry_exc

    if output_json_path.exists():
        with open(output_json_path) as f:
            vllm_results = json.load(f)
        if "avg_latency" in vllm_results:
            avg_latency_ms = vllm_results["avg_latency"] * 1000  # seconds -> milliseconds
            if cache_file is not None:
                cache_file.parent.mkdir(parents=True, exist_ok=True)
                # Atomic write so a concurrent reader never sees a partial file.
                tmp_file = cache_file.with_name(
                    f"{cache_file.name}.{os.getpid()}.{threading.get_ident()}.tmp"
                )
                with open(tmp_file, "w") as f:
                    json.dump(
                        {
                            "avg_latency_ms": avg_latency_ms,
                            "cache_identity": cache_identity,
                            "gpu_id": str(gpu_id),
                            "topology": runtime_config.topology.to_dict(),
                            "effective_command": effective_cmd,
                            "fallback_args": fallback_args,
                            "raw_result": vllm_results,
                            "gpu_name": _gpu_name(),
                            "vllm_version": _package_version("vllm"),
                        },
                        f,
                    )
                tmp_file.replace(cache_file)
            return avg_latency_ms

    raise RuntimeError(f"vLLM benchmark output not found at {output_json_path}")


def run_vllm_latency_benchmark(
    model_path: Path,
    runtime_config: RuntimeConfig,
    gpu_id: str | int | None = None,
    cache_dir: Path | None = None,
) -> RuntimeMeasurement:
    """Measure combined and prefill-only latency for one serialized model.

    The prefill workload generates one token.  Decode latency is intentionally
    derived later as ``combined - prefill`` so repeated-block differencing is
    performed component-wise before the phase split is exposed to consumers.
    """
    total_ms = _run_vllm_latency_phase(
        model_path,
        runtime_config,
        gpu_id=gpu_id,
        cache_dir=cache_dir,
    )
    prefill_ms = _run_vllm_latency_phase(
        model_path,
        replace(runtime_config, generation_seq_len=1),
        gpu_id=gpu_id,
        cache_dir=cache_dir,
    )
    return RuntimeMeasurement(total_ms=total_ms, prefill_ms=prefill_ms)


def _package_version(name: str) -> str:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return "unknown"


def _gpu_name() -> str:
    try:
        import torch

        return torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu"
    except Exception:  # pragma: no cover - metadata only
        return "unknown"
