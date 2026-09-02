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

"""Tests for serving-measurement checkpoint identity and fingerprints."""

import json
import sys
from types import SimpleNamespace

from modelopt.torch.puzzletron.benchmarks.provenance import (
    artifact_sha256,
    benchmark_result_fingerprint,
    checkpoint_identity,
    executable_identity,
    hardware_identity,
    software_identity,
)


def test_checkpoint_identity_counts_serialized_tensors_without_loading_weights(tmp_path):
    checkpoint = tmp_path / "checkpoint"
    checkpoint.mkdir()
    (checkpoint / "config.json").write_text(json.dumps({"architectures": ["TestModel"]}))
    header = json.dumps(
        {
            "weight": {"dtype": "F32", "shape": [2, 3], "data_offsets": [0, 24]},
            "bias": {"dtype": "F32", "shape": [3], "data_offsets": [24, 36]},
        }
    ).encode()
    (checkpoint / "model.safetensors").write_bytes(
        len(header).to_bytes(8, "little") + header + bytes(36)
    )

    identity = checkpoint_identity(checkpoint)

    assert identity["file_count"] == 2
    assert identity["serialized_size_bytes"] == sum(
        path.stat().st_size for path in checkpoint.iterdir()
    )
    assert identity["tensor_count"] == 2
    assert identity["parameter_count"] == 9
    assert len(identity["content_manifest_sha256"]) == 64

    (checkpoint / "model.safetensors").write_bytes(
        len(header).to_bytes(8, "little") + header + bytes([1]) + bytes(35)
    )
    changed_identity = checkpoint_identity(checkpoint)

    assert changed_identity["serialized_size_bytes"] == identity["serialized_size_bytes"]
    assert changed_identity["content_manifest_sha256"] != identity["content_manifest_sha256"]


def test_executable_identity_covers_resolved_file_contents(tmp_path):
    executable = tmp_path / "aiperf"
    executable.write_text(f"#!{sys.executable}\nfirst")

    identity = executable_identity(executable)
    executable.write_text(f"#!{sys.executable}\nother")

    assert identity["path"] == str(executable.resolve())
    assert identity["size_bytes"] == executable.stat().st_size
    assert identity["sha256"] != executable_identity(executable)["sha256"]


def test_artifact_sha256_covers_retained_evidence_contents(tmp_path):
    artifact = tmp_path / "profile.json"
    artifact.write_text("first")
    first = artifact_sha256(artifact)

    artifact.write_text("other")

    assert artifact_sha256(artifact) != first


def test_software_identity_covers_modelopt_and_vllm_source(monkeypatch, tmp_path):
    vllm = tmp_path / "vllm"
    vllm.mkdir()
    source = vllm / "engine.py"
    source.write_text("first")
    monkeypatch.setattr(
        "modelopt.torch.puzzletron.benchmarks.provenance.importlib.util.find_spec",
        lambda _name: SimpleNamespace(submodule_search_locations=[str(vllm)]),
    )

    first = software_identity()
    source.write_text("other")
    second = software_identity()

    assert first["source_manifests"]["modelopt_benchmarks"]
    assert first["source_manifests"]["vllm"] != second["source_manifests"]["vllm"]


def test_executable_identity_covers_shebang_environment_aiperf_distribution(tmp_path):
    environment = tmp_path / "environment"
    interpreter = environment / "bin" / "python"
    executable = environment / "bin" / "aiperf"
    site_packages = environment / "lib" / "python3.12" / "site-packages"
    dist_info = site_packages / "aiperf-0.12.0.dist-info"
    package = site_packages / "aiperf" / "client.py"
    interpreter.parent.mkdir(parents=True)
    dist_info.mkdir(parents=True)
    package.parent.mkdir()
    interpreter.write_text("")
    executable.write_text(f"#!{interpreter}\n")
    package.write_text("first")
    (dist_info / "METADATA").write_text("Name: aiperf\nVersion: 0.12.0\n")
    (dist_info / "RECORD").write_text("aiperf/client.py,,\n")

    identity = executable_identity(executable)
    package.write_text("other")
    changed_identity = executable_identity(executable)

    assert identity["aiperf_distribution"]["version"] == "0.12.0"
    assert (
        identity["aiperf_distribution"]["content_manifest_sha256"]
        != changed_identity["aiperf_distribution"]["content_manifest_sha256"]
    )


def test_executable_identity_covers_editable_aiperf_source(tmp_path):
    environment = tmp_path / "environment"
    source_package = tmp_path / "source" / "aiperf"
    interpreter = environment / "bin" / "python"
    executable = environment / "bin" / "aiperf"
    site_packages = environment / "lib" / "python3.12" / "site-packages"
    dist_info = site_packages / "aiperf-0.12.0.dist-info"
    finder = site_packages / "__editable___aiperf_finder.py"
    interpreter.parent.mkdir(parents=True)
    dist_info.mkdir(parents=True)
    source_package.mkdir(parents=True)
    interpreter.write_text("")
    executable.write_text(f"#!{interpreter}\n")
    source_file = source_package / "client.py"
    source_file.write_text("first")
    finder.write_text(f"MAPPING = {{'aiperf': {str(source_package)!r}}}\n")
    (dist_info / "METADATA").write_text("Name: aiperf\nVersion: 0.12.0\n")
    (dist_info / "RECORD").write_text("__editable___aiperf_finder.py,,\n")

    identity = executable_identity(executable)
    source_file.write_text("other")
    changed_identity = executable_identity(executable)

    assert (
        identity["aiperf_distribution"]["editable_source_manifest_sha256"]
        != changed_identity["aiperf_distribution"]["editable_source_manifest_sha256"]
    )


def test_hardware_identity_records_index_and_uuid_devices_and_releases_nvml(monkeypatch):
    shutdowns = []
    pynvml = SimpleNamespace(
        nvmlInit=lambda: None,
        nvmlShutdown=lambda: shutdowns.append(True),
        nvmlSystemGetDriverVersion=lambda: "driver",
        nvmlDeviceGetHandleByIndex=lambda index: f"index-{index}",
        nvmlDeviceGetHandleByUUID=lambda uuid: f"uuid-{uuid}",
        nvmlDeviceGetMemoryInfo=lambda handle: SimpleNamespace(total=len(handle) * 100),
        nvmlDeviceGetName=lambda handle: f"name-{handle}",
        nvmlDeviceGetUUID=lambda handle: f"resolved-{handle}",
    )
    monkeypatch.setitem(sys.modules, "pynvml", pynvml)

    identity = hardware_identity("0,GPU-test")

    assert identity["driver_version"] == "driver"
    assert identity["gpus"] == [
        {
            "id": "0",
            "name": "name-index-0",
            "uuid": "resolved-index-0",
            "total_memory_bytes": 700,
        },
        {
            "id": "GPU-test",
            "name": "name-uuid-GPU-test",
            "uuid": "resolved-uuid-GPU-test",
            "total_memory_bytes": 1300,
        },
    ]
    assert shutdowns == [True]


def test_benchmark_result_fingerprint_covers_metrics_but_not_itself():
    payload = {"metrics": {"throughput": 10.0}, "result_fingerprint": "stale"}

    fingerprint = benchmark_result_fingerprint(payload)

    assert fingerprint == benchmark_result_fingerprint(
        {**payload, "result_fingerprint": "another-value"}
    )
    assert fingerprint != benchmark_result_fingerprint({"metrics": {"throughput": 11.0}})
