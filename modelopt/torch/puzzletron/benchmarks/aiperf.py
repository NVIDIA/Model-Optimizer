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

"""AIPerf subprocess adapter with owned vLLM server lifecycle."""

from __future__ import annotations

import importlib.metadata
import importlib.util
import json
import os
import shutil
import signal
import socket
import subprocess  # nosec B404
import tempfile
import time
import urllib.error
import urllib.request
from datetime import datetime, timezone
from pathlib import Path
from threading import Lock
from typing import TYPE_CHECKING, Any, Iterable, cast

from ..identity import stable_hash
from ..orchestration.mesh import normalize_vllm_topology
from .schema import BenchmarkResult

if TYPE_CHECKING:
    from ..anymodel.model_descriptor import ModelDescriptor

__all__ = ["run_aiperf_benchmark", "run_aiperf_sweep"]

_CHECKPOINT_PREPARE_LOCK = Lock()


def _prepare_vllm_checkpoint(
    checkpoint_dir: Path,
    *,
    trust_remote_code: bool = False,
) -> bool:
    """Restore AnyModel metadata lost by generic HF checkpoint consolidation."""
    with _CHECKPOINT_PREPARE_LOCK:
        config = json.loads((checkpoint_dir / "config.json").read_text())
        text_config = config.get("text_config") or config
        per_layer_config = text_config.get("per_layer_config") or {}
        architectures = config.get("architectures") or []
        if not per_layer_config or architectures == ["AnyModel"]:
            return False
        from ..utils.vllm_adapter import refresh_realized_checkpoint_config

        refresh_realized_checkpoint_config(
            checkpoint_dir,
            trust_remote_code=trust_remote_code,
        )
        return True


def _descriptor_vllm_args(checkpoint_dir: Path, *, multimodal: bool = False) -> list[str]:
    """Resolve the same model-specific vLLM contract used by runtime stats."""
    from ..anymodel.registry import resolve_descriptor

    config = json.loads((checkpoint_dir / "config.json").read_text())
    resolution = resolve_descriptor(config)
    descriptor = cast("type[ModelDescriptor]", resolution.descriptor)
    method = (
        descriptor.runtime_vllm_multimodal_benchmark_args
        if multimodal
        else descriptor.runtime_vllm_benchmark_args
    )
    args = [str(arg) for arg in method(config)]
    if multimodal and "--language-model-only" in args:
        raise ValueError("multimodal vLLM serving cannot use --language-model-only")
    return args


def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def _short_tokenizer_alias(checkpoint_dir: Path, artifact_dir: Path) -> Path:
    """Return a node-local alias that stays below HF cache component limits."""
    root = Path(
        os.environ.get(
            "AIPERF_TOKENIZER_ALIAS_DIR",
            str(Path(tempfile.gettempdir()) / "puzzletron-aiperf-tokenizers"),
        )
    )
    root.mkdir(parents=True, exist_ok=True)
    alias_hash = stable_hash(
        {"checkpoint": str(checkpoint_dir), "artifact": str(artifact_dir)},
        prefix="aiperf_tokenizer_alias",
    )
    alias_id = alias_hash.removeprefix("aiperf_tokenizer_alias_")[:24]
    alias = root / alias_id
    try:
        alias.symlink_to(checkpoint_dir, target_is_directory=True)
    except FileExistsError:
        if alias.resolve() != checkpoint_dir:
            raise RuntimeError(f"AIPerf tokenizer alias collision at {alias}")
    return alias


def _server_max_model_len(
    input_tokens: int,
    output_tokens: int,
    topology: dict[str, Any],
    *,
    image_batch_size: int = 0,
) -> int:
    """Reserve capacity for endpoint-specific chat/control tokens."""
    if image_batch_size > 0 and "server_context_overhead_tokens" not in topology:
        raise ValueError(
            "multimodal AIPerf workloads require an explicit "
            "server_context_overhead_tokens visual-token budget"
        )
    overhead = int(topology.get("server_context_overhead_tokens", 64))
    if overhead < 0:
        raise ValueError("server_context_overhead_tokens must be nonnegative")
    return input_tokens + output_tokens + overhead


def _canonical_topology(topology: dict[str, Any]) -> dict[str, Any]:
    """Normalize serving topology names for cache and report identities."""

    return normalize_vllm_topology(topology)


def _topology_vllm_args(topology: dict[str, Any]) -> list[str]:
    """Translate the canonical TP/PP/DP/EP/CP contract to vLLM CLI arguments."""

    canonical = _canonical_topology(topology)
    args = [
        "--tensor-parallel-size",
        str(canonical["tp"]),
        "--pipeline-parallel-size",
        str(canonical["pp"]),
        "--prefill-context-parallel-size",
        str(canonical["prefill_cp"]),
        "--decode-context-parallel-size",
        str(canonical["decode_cp"]),
        "--distributed-executor-backend",
        str(canonical["distributed_executor_backend"]),
    ]
    if canonical["dp"] > 1:
        args.extend(
            (
                "--data-parallel-size",
                str(canonical["dp"]),
                "--data-parallel-size-local",
                str(canonical["dp"]),
            )
        )
    if canonical["enable_expert_parallel"]:
        args.append("--enable-expert-parallel")
    return args


def _exact_length_extra_inputs(
    extra_inputs: dict[str, Any] | None, output_tokens: int
) -> dict[str, Any]:
    """Guarantee the measured OSL unless the caller chose an explicit policy."""
    resolved = dict(extra_inputs or {})
    if "ignore_eos" not in resolved and "min_tokens" not in resolved:
        resolved["ignore_eos"] = True
    return resolved


def _wait_for_health(url: str, process: subprocess.Popen, timeout: float) -> None:
    deadline = time.monotonic() + timeout
    last_error: BaseException | None = None
    while time.monotonic() < deadline:
        if process.poll() is not None:
            raise RuntimeError(
                f"vLLM server exited before readiness with code {process.returncode}"
            )
        try:
            with urllib.request.urlopen(url, timeout=5) as response:  # nosec B310
                if response.status == 200:
                    return
        except (urllib.error.URLError, TimeoutError, ConnectionError) as exc:
            last_error = exc
        time.sleep(2)
    raise TimeoutError(f"vLLM server did not become ready at {url}: {last_error}")


def _stop_process_group(process: subprocess.Popen) -> None:
    if process.poll() is not None:
        return
    os.killpg(process.pid, signal.SIGTERM)
    try:
        process.wait(timeout=30)
    except subprocess.TimeoutExpired:
        os.killpg(process.pid, signal.SIGKILL)
        process.wait(timeout=10)


def _metric(raw: dict[str, Any], name: str, stat: str = "avg") -> float | None:
    value = raw.get(name)
    if not isinstance(value, dict) or value.get(stat) is None:
        return None
    return float(value[stat])


def _parse_export(path: Path) -> tuple[dict[str, float], int]:
    raw = json.loads(path.read_text())
    mapping = {
        "request_throughput": ("request_throughput", "avg"),
        "output_token_throughput": ("output_token_throughput", "avg"),
        "output_token_throughput_per_user_mean": (
            "output_token_throughput_per_user",
            "avg",
        ),
        "output_token_throughput_per_user_p95": (
            "output_token_throughput_per_user",
            "p95",
        ),
        "ttft_mean_ms": ("time_to_first_token", "avg"),
        "ttft_p95_ms": ("time_to_first_token", "p95"),
        "ttft_p99_ms": ("time_to_first_token", "p99"),
        "tpot_mean_ms": ("inter_token_latency", "avg"),
        "tpot_p95_ms": ("inter_token_latency", "p95"),
        "tpot_p99_ms": ("inter_token_latency", "p99"),
        "request_latency_mean_ms": ("request_latency", "avg"),
        "request_latency_p95_ms": ("request_latency", "p95"),
        "request_latency_p99_ms": ("request_latency", "p99"),
        "input_sequence_length": ("input_sequence_length", "avg"),
        "output_sequence_length": ("output_sequence_length", "avg"),
        "goodput": ("goodput", "avg"),
        "total_gpu_power_w": ("total_gpu_power", "avg"),
        "total_gpu_energy_j": ("total_gpu_energy", "avg"),
        "output_tokens_per_joule": ("output_tokens_per_joule", "avg"),
        "energy_per_user_j": ("energy_per_user", "avg"),
        "num_images": ("num_images", "avg"),
        "image_throughput": ("image_throughput", "avg"),
        "image_latency_mean_ms": ("image_latency", "avg"),
        "image_latency_p95_ms": ("image_latency", "p95"),
        "image_latency_p99_ms": ("image_latency", "p99"),
    }
    metrics = {}
    for output_name, (raw_name, stat) in mapping.items():
        value = _metric(raw, raw_name, stat)
        if value is not None:
            metrics[output_name] = value
    failures = int(_metric(raw, "error_request_count") or 0)
    return metrics, failures


def _package_version(name: str) -> str:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return "unknown"


def _resolve_executable(executable: str | Path) -> Path:
    """Resolve AIPerf without requiring it in Puzzletron's training venv."""
    value = Path(executable)
    if value.name != str(executable) or value.is_absolute():
        return value.resolve()
    discovered = shutil.which(str(executable))
    if discovered:
        return Path(discovered).resolve()
    if str(executable) == "aiperf":
        engineering_root = Path(__file__).resolve().parents[5]
        sibling = engineering_root / "aiperf" / ".venv" / "bin" / "aiperf"
        if sibling.is_file():
            return sibling.resolve()
    raise FileNotFoundError(f"Cannot find AIPerf executable {executable!s}; set AIPERF_EXECUTABLE")


def _profile_command(
    *,
    executable: str | Path,
    model_name: str,
    port: int,
    endpoint_type: str,
    concurrency: int,
    request_count: int,
    input_tokens: int,
    output_tokens: int,
    tokenizer_dir: Path,
    artifact_dir: Path,
    seed: int,
    extra_inputs: dict[str, Any] | None,
    use_server_token_count: bool,
    gpu_telemetry: str | None,
    image_batch_size: int,
    image_width_mean: int,
    image_height_mean: int,
) -> list[str]:
    if image_batch_size < 0:
        raise ValueError("image_batch_size must be nonnegative")
    if image_batch_size > 0:
        if endpoint_type != "chat":
            raise ValueError("multimodal AIPerf workloads require endpoint_type='chat'")
        if image_width_mean <= 0 or image_height_mean <= 0:
            raise ValueError(
                "multimodal AIPerf workloads require positive image_width_mean "
                "and image_height_mean"
            )
    elif image_width_mean != 0 or image_height_mean != 0:
        raise ValueError("image dimensions require image_batch_size > 0")

    command = [
        str(executable),
        "profile",
        "--model",
        model_name,
        "--url",
        f"http://127.0.0.1:{port}",
        "--endpoint-type",
        endpoint_type,
        "--streaming",
        "--concurrency",
        str(concurrency),
        "--request-count",
        str(request_count),
        "--synthetic-input-tokens-mean",
        str(input_tokens),
        "--synthetic-input-tokens-stddev",
        "0",
        "--output-tokens-mean",
        str(output_tokens),
        "--output-tokens-stddev",
        "0",
        "--tokenizer",
        str(tokenizer_dir),
        "--artifact-dir",
        str(artifact_dir),
        "--random-seed",
        str(seed),
        "--image-batch-size",
        str(image_batch_size),
        "--ui-type",
        "none",
    ]
    if image_batch_size > 0:
        command.extend(
            (
                "--image-width-mean",
                str(image_width_mean),
                "--image-width-stddev",
                "0",
                "--image-height-mean",
                str(image_height_mean),
                "--image-height-stddev",
                "0",
            )
        )
    if gpu_telemetry:
        command.extend(("--gpu-telemetry", str(gpu_telemetry)))
    resolved_extra_inputs = _exact_length_extra_inputs(extra_inputs, output_tokens)
    if resolved_extra_inputs:
        command.extend(("--extra-inputs", json.dumps(resolved_extra_inputs, sort_keys=True)))
    if use_server_token_count:
        command.append("--use-server-token-count")
    return command


def _vllm_server_command(
    *,
    checkpoint_dir: Path,
    port: int,
    model_name: str,
    input_tokens: int,
    output_tokens: int,
    topology: dict[str, Any],
    trust_remote_code: bool,
    image_batch_size: int = 0,
    image_width_mean: int = 0,
    image_height_mean: int = 0,
) -> list[str]:
    """Build the vLLM command under the caller's explicit code-trust policy."""

    command = [
        "vllm",
        "serve",
        str(checkpoint_dir),
        "--host",
        "127.0.0.1",
        "--port",
        str(port),
        "--served-model-name",
        model_name,
        "--max-model-len",
        str(
            _server_max_model_len(
                input_tokens,
                output_tokens,
                topology,
                image_batch_size=image_batch_size,
            )
        ),
    ]
    if trust_remote_code:
        command.append("--trust-remote-code")
    command.extend(_topology_vllm_args(topology))
    command.extend(_descriptor_vllm_args(checkpoint_dir, multimodal=image_batch_size > 0))
    if image_batch_size > 0:
        config = json.loads((checkpoint_dir / "config.json").read_text())
        architectures = tuple(config.get("architectures") or ()) + (
            str(config.get("base_architecture") or ""),
        )
        if "vision_config" not in config or not any(
            "ConditionalGeneration" in str(value) for value in architectures
        ):
            raise ValueError("multimodal AIPerf requires a full VLM checkpoint")
        processor_assets = ("preprocessor_config.json", "video_preprocessor_config.json")
        if not any((checkpoint_dir / name).is_file() for name in processor_assets):
            raise ValueError("multimodal AIPerf requires local processor assets")
        max_pixels = max(1, image_width_mean * image_height_mean)
        command.extend(
            (
                "--limit-mm-per-prompt",
                json.dumps({"image": image_batch_size, "video": 0}, sort_keys=True),
                "--mm-processor-kwargs",
                json.dumps(
                    {
                        "max_pixels": max_pixels,
                        "min_pixels": min(262144, max_pixels),
                        "max_num_frames": 1,
                    },
                    sort_keys=True,
                ),
            )
        )
    extra_vllm_args = tuple(str(arg) for arg in topology.get("extra_vllm_args", ()))
    reserved_options = {
        "--config",
        "--trust-remote-code",
        "--limit-mm-per-prompt",
        "--mm-processor-kwargs",
    }
    normalized_options = {arg.partition("=")[0].replace("_", "-") for arg in extra_vllm_args}
    overridden_options = {
        reserved
        for option in normalized_options
        for reserved in reserved_options
        if reserved.startswith(option)
    }
    if overridden_options:
        raise ValueError(
            "topology.extra_vllm_args cannot set policy-owned vLLM options: "
            + ", ".join(sorted(overridden_options))
        )
    command.extend(extra_vllm_args)
    return command


def _clean_subprocess_environment(
    gpu_ids: str, *, architecture_id: str, topology_id: str
) -> dict[str, str]:
    env = dict(os.environ)
    for key in list(env):
        if key.startswith("TORCHELASTIC_") or key in {
            "WORLD_SIZE",
            "RANK",
            "LOCAL_RANK",
            "LOCAL_WORLD_SIZE",
            "MASTER_ADDR",
            "MASTER_PORT",
        }:
            env.pop(key, None)
    env["CUDA_VISIBLE_DEVICES"] = gpu_ids
    # The FlashInfer sampler lazily JIT-compiles a tiny top-k/top-p extension.
    # Serving does not need that backend, and disabling it avoids a dependency
    # on TVM-FFI development headers that are intentionally absent from the
    # runtime container.
    env.setdefault("VLLM_USE_FLASHINFER_SAMPLER", "0")
    env.setdefault("VLLM_USE_LAYERNAME", "0")

    # Concurrent servers must not populate the same vLLM Python compile-cache
    # package.  Namespace the persistent cache by the two identities that
    # determine compiled serving code, while retaining reuse across reruns and
    # concurrencies of the same sweep.
    cache_root = Path(env.get("VLLM_CACHE_ROOT", Path.home() / ".cache" / "vllm"))
    env["VLLM_CACHE_ROOT"] = str(cache_root / "puzzletron-aiperf" / architecture_id / topology_id)

    # Editable vLLM installs rely on an import hook in the parent process.  An
    # explicit path for that active package keeps spawned engine workers on the
    # same source tree and compiled extensions as the CLI entry point.
    spec = importlib.util.find_spec("vllm")
    locations = tuple(spec.submodule_search_locations or ()) if spec is not None else ()
    vllm_source = Path(locations[0]).parent if len(locations) == 1 else None
    compatibility = Path(__file__).with_name("vllm_compat")
    python_paths = [str(compatibility)]
    if vllm_source is not None:
        python_paths.append(str(vllm_source))
    existing = env.get("PYTHONPATH", "")
    if existing:
        python_paths.append(existing)
    env["PYTHONPATH"] = os.pathsep.join(python_paths)
    return env


def _aiperf_subprocess_environment(
    env: dict[str, str],
    *,
    allow_aiperf_v011_online_tokenizer_resolution: bool = False,
) -> dict[str, str]:
    """Work around AIPerf v0.11's broken offline local-tokenizer resolution.

    Remove this compatibility option after the pinned AIPerf resolver accepts
    absolute local tokenizer directories while offline.
    """

    resolved = dict(env)
    if allow_aiperf_v011_online_tokenizer_resolution:
        resolved.pop("HF_HUB_OFFLINE", None)
        resolved.pop("TRANSFORMERS_OFFLINE", None)
    return resolved


def run_aiperf_sweep(
    checkpoint_dir: str | Path,
    *,
    artifact_dir: str | Path,
    concurrencies: Iterable[int],
    input_tokens: int,
    output_tokens: int,
    gpu_ids: str,
    topology: dict[str, Any],
    request_counts: dict[int, int] | None = None,
    solution_id: str = "unknown",
    profile_id: str = "unknown",
    topology_id: str | None = None,
    executable: str | Path = "aiperf",
    endpoint_type: str = "chat",
    extra_inputs: dict[str, Any] | None = None,
    use_server_token_count: bool = True,
    seed: int = 42,
    readiness_timeout: float = 1200,
    benchmark_timeout: float = 600,
    gpu_telemetry: str | None = "pynvml",
    image_batch_sizes: Iterable[int] | None = None,
    image_width_mean: int = 0,
    image_height_mean: int = 0,
    trust_remote_code: bool = False,
    allow_aiperf_v011_online_tokenizer_resolution: bool = False,
) -> list[BenchmarkResult]:
    """Run a serving sweep while preserving offline and remote-code policy by default."""

    checkpoint_dir = Path(checkpoint_dir).resolve()
    artifact_dir = Path(artifact_dir).resolve()
    concurrency_values = tuple(int(value) for value in concurrencies)
    if not concurrency_values or len(set(concurrency_values)) != len(concurrency_values):
        raise ValueError("AIPerf concurrencies must be non-empty and unique")
    if any(value < 1 for value in concurrency_values):
        raise ValueError(f"AIPerf concurrencies must be positive: {concurrency_values}")
    image_batch_values = (
        (0,) if image_batch_sizes is None else tuple(int(value) for value in image_batch_sizes)
    )
    if not image_batch_values or len(set(image_batch_values)) != len(image_batch_values):
        raise ValueError("AIPerf image_batch_sizes must be non-empty and unique")
    if any(value < 0 for value in image_batch_values):
        raise ValueError(f"AIPerf image_batch_sizes must be nonnegative: {image_batch_values}")
    multimodal = any(value > 0 for value in image_batch_values)
    if multimodal and any(value == 0 for value in image_batch_values):
        raise ValueError("AIPerf image_batch_sizes cannot mix text-only and multimodal workloads")
    if multimodal and (image_width_mean <= 0 or image_height_mean <= 0):
        raise ValueError("multimodal AIPerf requires positive image dimensions")
    artifact_dir.mkdir(parents=True, exist_ok=True)
    _prepare_vllm_checkpoint(
        checkpoint_dir,
        trust_remote_code=trust_remote_code,
    )
    request_counts = {
        value: int((request_counts or {}).get(value, max(32, 4 * value)))
        for value in concurrency_values
    }
    architecture_id = stable_hash(
        json.loads((checkpoint_dir / "config.json").read_text()),
        prefix="aiperf_architecture",
    )
    canonical_topology = _canonical_topology(topology)
    descriptor_args = _descriptor_vllm_args(checkpoint_dir, multimodal=multimodal)
    server_contract = {
        "topology": canonical_topology,
        "server_context_overhead_tokens": topology.get("server_context_overhead_tokens"),
        "descriptor_args": descriptor_args,
        "extra_vllm_args": tuple(str(arg) for arg in topology.get("extra_vllm_args", ())),
        "max_image_batch_size": max(image_batch_values),
        "image_width_mean": image_width_mean,
        "image_height_mean": image_height_mean,
    }
    topology_id = topology_id or stable_hash(server_contract, prefix="aiperf_topology")
    revisions = {"aiperf": _package_version("aiperf"), "vllm": _package_version("vllm")}
    port = _free_port()
    model_name = f"puzzletron-{architecture_id[:16]}"
    tokenizer_dir = _short_tokenizer_alias(checkpoint_dir, artifact_dir)
    server_log = artifact_dir / "vllm_server.log"
    server_cmd = _vllm_server_command(
        checkpoint_dir=checkpoint_dir,
        port=port,
        model_name=model_name,
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        topology=topology,
        trust_remote_code=trust_remote_code,
        image_batch_size=max(image_batch_values),
        image_width_mean=image_width_mean,
        image_height_mean=image_height_mean,
    )
    executable = _resolve_executable(executable)
    env = _clean_subprocess_environment(
        gpu_ids,
        architecture_id=architecture_id,
        topology_id=topology_id,
    )
    for key, value in (topology.get("env") or {}).items():
        env[str(key)] = str(value)
    aiperf_env = _aiperf_subprocess_environment(
        env,
        allow_aiperf_v011_online_tokenizer_resolution=(
            allow_aiperf_v011_online_tokenizer_resolution
        ),
    )
    cached: dict[tuple[int, int], BenchmarkResult] = {}
    missing: list[tuple[int, int, dict[str, int], str, Path, list[str], str]] = []
    for image_batch_size in image_batch_values:
        for concurrency in concurrency_values:
            workload = {
                "input_tokens": input_tokens,
                "output_tokens": output_tokens,
                "image_batch_size": image_batch_size,
                "image_width_mean": image_width_mean if image_batch_size else 0,
                "image_height_mean": image_height_mean if image_batch_size else 0,
            }
            workload_id = stable_hash(workload, prefix="aiperf_workload")
            run_dir = artifact_dir / f"concurrency_{concurrency}"
            if image_batch_size > 0:
                run_dir = artifact_dir / f"images_{image_batch_size}" / f"concurrency_{concurrency}"
            run_dir.mkdir(parents=True, exist_ok=True)
            command = _profile_command(
                executable=executable,
                model_name=model_name,
                port=port,
                endpoint_type=endpoint_type,
                concurrency=concurrency,
                request_count=request_counts[concurrency],
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                tokenizer_dir=tokenizer_dir,
                artifact_dir=run_dir,
                seed=seed,
                extra_inputs=extra_inputs,
                use_server_token_count=use_server_token_count,
                gpu_telemetry=gpu_telemetry,
                image_batch_size=image_batch_size,
                image_width_mean=image_width_mean if image_batch_size else 0,
                image_height_mean=image_height_mean if image_batch_size else 0,
            )
            cache_identity = stable_hash(
                {
                    "architecture_id": architecture_id,
                    "solution_id": solution_id,
                    "profile_id": profile_id,
                    "server_contract": server_contract,
                    "workload": workload,
                    "concurrency": concurrency,
                    "request_count": request_counts[concurrency],
                    "endpoint_type": endpoint_type,
                    "extra_inputs": _exact_length_extra_inputs(extra_inputs, output_tokens),
                    "use_server_token_count": use_server_token_count,
                    "trust_remote_code": trust_remote_code,
                    "allow_aiperf_v011_online_tokenizer_resolution": (
                        allow_aiperf_v011_online_tokenizer_resolution
                    ),
                    "revisions": revisions,
                },
                prefix="aiperf_result",
            )
            metadata_path = run_dir / "puzzletron_aiperf_result.json"
            export = run_dir / "profile_export_aiperf.json"
            if metadata_path.is_file() and export.is_file():
                result = BenchmarkResult(**json.loads(metadata_path.read_text()))
                if result.cache_identity == cache_identity:
                    cached[(image_batch_size, concurrency)] = result
                    continue
            missing.append(
                (
                    image_batch_size,
                    concurrency,
                    workload,
                    workload_id,
                    run_dir,
                    command,
                    cache_identity,
                )
            )

    if not missing:
        return [
            cached[(image_batch_size, concurrency)]
            for image_batch_size in image_batch_values
            for concurrency in concurrency_values
        ]

    with server_log.open("a", encoding="utf-8") as log:
        server = subprocess.Popen(  # nosec B603
            server_cmd,
            stdout=log,
            stderr=subprocess.STDOUT,
            env=env,
            start_new_session=True,
            text=True,
        )
        try:
            _wait_for_health(f"http://127.0.0.1:{port}/health", server, readiness_timeout)
            for (
                image_batch_size,
                concurrency,
                workload,
                workload_id,
                run_dir,
                command,
                cache_identity,
            ) in missing:
                started_at = datetime.now(timezone.utc)
                subprocess.run(  # nosec B603
                    command,
                    check=True,
                    timeout=benchmark_timeout,
                    env=aiperf_env,
                )
                export = run_dir / "profile_export_aiperf.json"
                if not export.is_file():
                    raise FileNotFoundError(f"AIPerf did not produce {export}")
                metrics, failures = _parse_export(export)
                # Server-reported chat prompt counts include the rendered chat
                # template, while input_tokens describes the synthetic message.
                exact_input_length = endpoint_type != "chat" or not use_server_token_count
                if (
                    exact_input_length
                    and round(metrics.get("input_sequence_length", -1)) != input_tokens
                ):
                    raise RuntimeError(
                        "AIPerf input length drift: "
                        f"{metrics.get('input_sequence_length')} != {input_tokens}"
                    )
                if round(metrics.get("output_sequence_length", -1)) != output_tokens:
                    raise RuntimeError(
                        "AIPerf output length drift: "
                        f"{metrics.get('output_sequence_length')} != {output_tokens}"
                    )
                if failures:
                    raise RuntimeError(f"AIPerf recorded {failures} failed requests in {export}")
                raw_artifacts = {"profile": str(export), "server_log": str(server_log)}
                jsonl = run_dir / "profile_export.jsonl"
                if jsonl.is_file():
                    raw_artifacts["requests"] = str(jsonl)
                result = BenchmarkResult(
                    architecture_id=architecture_id,
                    checkpoint_dir=str(checkpoint_dir),
                    solution_id=solution_id,
                    profile_id=profile_id,
                    topology_id=topology_id,
                    workload_id=workload_id,
                    gpu_count=int(canonical_topology["gpu_count"]),
                    cache_identity=cache_identity,
                    topology=canonical_topology,
                    workload=workload,
                    concurrency=concurrency,
                    metrics=metrics,
                    failures=failures,
                    raw_artifacts=raw_artifacts,
                    command=tuple(command),
                    started_at=started_at,
                )
                result_payload = (
                    result.model_dump(mode="json")
                    if hasattr(result, "model_dump")
                    else json.loads(result.json())
                )
                (run_dir / "puzzletron_aiperf_result.json").write_text(
                    json.dumps(result_payload, indent=2, sort_keys=True) + "\n"
                )
                cached[(image_batch_size, concurrency)] = result
        finally:
            _stop_process_group(server)
    return [
        cached[(image_batch_size, concurrency)]
        for image_batch_size in image_batch_values
        for concurrency in concurrency_values
    ]


def run_aiperf_benchmark(
    checkpoint_dir: str | Path,
    *,
    artifact_dir: str | Path,
    concurrency: int,
    input_tokens: int,
    output_tokens: int,
    gpu_ids: str,
    topology: dict[str, Any],
    request_count: int,
    solution_id: str = "unknown",
    profile_id: str = "unknown",
    topology_id: str | None = None,
    **kwargs,
) -> BenchmarkResult:
    """Compatibility wrapper for callers that request one concurrency."""

    root = Path(artifact_dir)
    if root.name == f"concurrency_{concurrency}":
        root = root.parent
    return run_aiperf_sweep(
        checkpoint_dir,
        artifact_dir=root,
        concurrencies=(concurrency,),
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        gpu_ids=gpu_ids,
        topology=topology,
        request_counts={concurrency: request_count},
        solution_id=solution_id,
        profile_id=profile_id,
        topology_id=topology_id,
        **kwargs,
    )[0]
