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

"""Reproducibility metadata for Puzzletron serving measurements."""

from __future__ import annotations

import ast
import csv
import hashlib
import importlib.metadata
import json
import platform
import shlex
import shutil
import sys
from pathlib import Path
from typing import Any, Iterable

from ..identity import stable_hash

__all__ = [
    "artifact_sha256",
    "benchmark_result_fingerprint",
    "checkpoint_identity",
    "executable_identity",
    "hardware_identity",
    "software_identity",
]


def artifact_sha256(path: str | Path) -> str:
    """Hash one retained benchmark artifact."""

    path = Path(path)
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _safetensors_inventory(path: Path) -> tuple[int, int]:
    """Return tensor and parameter counts without loading tensor payloads."""

    with path.open("rb") as stream:
        prefix = stream.read(8)
        if len(prefix) != 8:
            raise ValueError(f"invalid safetensors header in {path}")
        header_size = int.from_bytes(prefix, "little")
        if not 0 < header_size <= 100 * 1024 * 1024:
            raise ValueError(f"invalid safetensors header size in {path}: {header_size}")
        header_bytes = stream.read(header_size)
        if len(header_bytes) != header_size:
            raise ValueError(f"truncated safetensors header in {path}")
        header = json.loads(header_bytes)
    tensors = [metadata for name, metadata in header.items() if name != "__metadata__"]
    parameter_count = 0
    for metadata in tensors:
        elements = 1
        for dimension in metadata["shape"]:
            elements *= int(dimension)
        parameter_count += elements
    return len(tensors), parameter_count


def checkpoint_identity(checkpoint_dir: str | Path) -> dict[str, Any]:
    """Describe and content-address a serialized checkpoint."""

    root = Path(checkpoint_dir).resolve()
    files = tuple(path for path in sorted(root.rglob("*")) if path.is_file())
    safetensors = tuple(path for path in files if path.suffix == ".safetensors")
    tensor_count = 0
    parameter_count = 0
    for path in safetensors:
        tensors, parameters = _safetensors_inventory(path)
        tensor_count += tensors
        parameter_count += parameters
    config = json.loads((root / "config.json").read_text())
    manifest = [
        {
            "path": path.relative_to(root).as_posix(),
            "size_bytes": path.stat().st_size,
            "sha256": artifact_sha256(path),
        }
        for path in files
    ]
    return {
        "architecture_id": stable_hash(config, prefix="aiperf_architecture"),
        "file_count": len(files),
        "serialized_size_bytes": sum(path.stat().st_size for path in files),
        "tensor_count": tensor_count,
        "parameter_count": parameter_count,
        "content_manifest_sha256": hashlib.sha256(
            json.dumps(manifest, separators=(",", ":"), sort_keys=True).encode()
        ).hexdigest(),
    }


def executable_identity(
    executable: str | Path, *, distribution_name: str | None = "aiperf"
) -> dict[str, Any]:
    """Content-address a resolved executable and its supported distribution."""

    path = Path(executable).resolve()
    if not path.is_file():
        return {"path": str(path), "size_bytes": None, "sha256": "unavailable"}
    identity = {
        "path": str(path),
        "size_bytes": path.stat().st_size,
        "sha256": artifact_sha256(path),
    }
    if distribution_name == "aiperf":
        distribution = _aiperf_distribution_identity(path)
        if distribution is not None:
            identity["aiperf_distribution"] = distribution
    return identity


def _aiperf_distribution_identity(executable: Path) -> dict[str, Any] | None:
    """Identify the AIPerf installation selected by a console-script shebang."""

    try:
        shebang = executable.open("rb").readline(4096).decode(errors="replace").strip()
        command = shlex.split(shebang[2:]) if shebang.startswith("#!") else []
    except (OSError, ValueError):
        return None
    if not command:
        return None
    interpreter = command[0]
    if Path(interpreter).name == "env" and len(command) > 1:
        interpreter = shutil.which(command[1]) or command[1]
    interpreter_path = Path(interpreter).expanduser().resolve()
    prefix = interpreter_path.parent.parent
    site_packages = sorted(prefix.glob("lib/python*/site-packages"))
    windows_site_packages = prefix / "Lib" / "site-packages"
    if windows_site_packages.is_dir():
        site_packages.append(windows_site_packages)
    for site_root in site_packages:
        for dist_info in sorted(site_root.glob("aiperf-*.dist-info")):
            metadata_path = dist_info / "METADATA"
            record_path = dist_info / "RECORD"
            if not metadata_path.is_file() or not record_path.is_file():
                continue
            version = next(
                (
                    line.partition(":")[2].strip()
                    for line in metadata_path.read_text(errors="replace").splitlines()
                    if line.startswith("Version:")
                ),
                "unknown",
            )
            manifest = []
            record_entries = []
            with record_path.open(newline="", encoding="utf-8") as stream:
                for row in csv.reader(stream):
                    if not row:
                        continue
                    record_entries.append(row[0])
                    installed_path = (site_root / row[0]).resolve()
                    manifest.append(
                        {
                            "path": row[0],
                            "sha256": (
                                artifact_sha256(installed_path)
                                if installed_path.is_file()
                                else "missing"
                            ),
                        }
                    )
            identity = {
                "version": version,
                "content_manifest_sha256": hashlib.sha256(
                    json.dumps(manifest, separators=(",", ":"), sort_keys=True).encode()
                ).hexdigest(),
            }
            source_roots = _editable_aiperf_source_roots(site_root, record_entries)
            if source_roots:
                identity["editable_source_manifest_sha256"] = _source_manifest_sha256(source_roots)
            return identity
    return None


def _editable_aiperf_source_roots(site_root: Path, record_entries: list[str]) -> list[Path]:
    """Resolve source roots recorded by common editable-install mechanisms."""

    roots: set[Path] = set()
    for entry in record_entries:
        installed_path = (site_root / entry).resolve()
        name = installed_path.name.lower()
        if name.endswith(".pth") and "editable" in name and installed_path.is_file():
            for line in installed_path.read_text(errors="replace").splitlines():
                candidate = line.strip()
                if not candidate or candidate.startswith(("#", "import ")):
                    continue
                package_root = (installed_path.parent / candidate).resolve() / "aiperf"
                if package_root.is_dir():
                    roots.add(package_root)
        if name.endswith("_finder.py") and "editable" in name and installed_path.is_file():
            try:
                tree = ast.parse(installed_path.read_text(errors="replace"))
            except (OSError, SyntaxError):
                continue
            for node in tree.body:
                if not isinstance(node, (ast.Assign, ast.AnnAssign)):
                    continue
                targets = node.targets if isinstance(node, ast.Assign) else [node.target]
                if not any(
                    isinstance(target, ast.Name) and target.id == "MAPPING" for target in targets
                ):
                    continue
                if node.value is None:
                    continue
                try:
                    mapping = ast.literal_eval(node.value)
                except (TypeError, ValueError):
                    continue
                if isinstance(mapping, dict) and isinstance(mapping.get("aiperf"), str):
                    package_root = Path(mapping["aiperf"]).expanduser().resolve()
                    if package_root.is_dir():
                        roots.add(package_root)
    return sorted(roots)


def _source_manifest_sha256(roots: list[Path]) -> str:
    manifest = []
    for root in roots:
        paths = (root,) if root.is_file() else sorted(root.rglob("*"))
        for path in paths:
            if not path.is_file() or "__pycache__" in path.parts:
                continue
            manifest.append(
                {
                    "root": root.name,
                    "path": path.name if root.is_file() else path.relative_to(root).as_posix(),
                    "sha256": artifact_sha256(path),
                }
            )
    return hashlib.sha256(
        json.dumps(manifest, separators=(",", ":"), sort_keys=True).encode()
    ).hexdigest()


def _package_versions(names: Iterable[str]) -> dict[str, str]:
    versions = {}
    for name in names:
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = "unknown"
    return versions


def software_identity() -> dict[str, Any]:
    """Return the runtime versions that can affect serving measurements."""

    benchmark_root = Path(__file__).resolve().parent
    benchmark_sources = [
        benchmark_root / "aiperf.py",
        benchmark_root / "provenance.py",
        benchmark_root / "schema.py",
        benchmark_root / "vllm_compat",
    ]
    vllm_spec = importlib.util.find_spec("vllm")
    vllm_locations = (
        tuple(Path(path).resolve() for path in (vllm_spec.submodule_search_locations or ()))
        if vllm_spec is not None
        else ()
    )
    return {
        "python": platform.python_version(),
        "packages": _package_versions(("aiperf", "nvidia-ml-py", "torch", "vllm")),
        "source_manifests": {
            "modelopt_benchmarks": _source_manifest_sha256(benchmark_sources),
            "vllm": (
                _source_manifest_sha256(list(vllm_locations)) if vllm_locations else "unavailable"
            ),
        },
    }


def hardware_identity(gpu_ids: str) -> dict[str, Any]:
    """Return host and visible-accelerator identity without requiring a GPU locally."""

    identity: dict[str, Any] = {
        "machine": platform.machine(),
        "platform": platform.platform(),
        "python_executable": sys.executable,
        "visible_gpu_ids": gpu_ids,
        "gpus": [],
    }
    try:
        import pynvml  # Optional GPU dependency; Mac-side contract tests remain dependency-light.
    except ImportError:
        identity["nvml"] = "unavailable"
        return identity

    initialized = False
    try:
        pynvml.nvmlInit()
        initialized = True
        identity["driver_version"] = str(pynvml.nvmlSystemGetDriverVersion())
        for raw_id in (value.strip() for value in gpu_ids.split(",") if value.strip()):
            handle = (
                pynvml.nvmlDeviceGetHandleByIndex(int(raw_id))
                if raw_id.isdigit()
                else pynvml.nvmlDeviceGetHandleByUUID(raw_id)
            )
            memory = pynvml.nvmlDeviceGetMemoryInfo(handle)
            identity["gpus"].append(
                {
                    "id": raw_id,
                    "name": str(pynvml.nvmlDeviceGetName(handle)),
                    "uuid": str(pynvml.nvmlDeviceGetUUID(handle)),
                    "total_memory_bytes": int(memory.total),
                }
            )
    except Exception:
        identity["nvml"] = "unavailable"
        identity["gpus"] = []
    finally:
        if initialized:
            pynvml.nvmlShutdown()
    return identity


def benchmark_result_fingerprint(payload: dict[str, Any]) -> str:
    """Return an integrity fingerprint for one self-contained result row."""

    canonical = dict(payload)
    canonical.pop("result_fingerprint", None)
    return stable_hash(canonical, prefix="aiperf_measurement", length=64)
