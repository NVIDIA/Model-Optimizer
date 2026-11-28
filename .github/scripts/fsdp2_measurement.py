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

"""Run only the two fixed actual FSDP2 measurements; reject incomplete evidence."""

import hashlib
import json
import math
import os
import shutil
import subprocess
import sys
import xml.etree.ElementTree as ET
from importlib.metadata import version
from pathlib import Path

NODEIDS = (
    "tests/gpu/torch/quantization/test_fsdp2.py::test_nested_fsdp2_read_cuda_peak",
    "tests/gpu/torch/export/test_fsdp2_export.py::test_fsdp2_streaming_export_matches_reference[fp8]",
)
PATTERNS = ("nested-read-*.json", "cuda-fsdp-export-*.json")


def check_devices(cuda):
    """Reject any route that cannot execute a real two-rank CUDA measurement."""
    if not cuda.is_available() or cuda.device_count() != 2:
        raise ValueError("FSDP2 measurements require exactly two visible CUDA devices")


def check_source():
    """Require the actual checkout to match the dispatch event source."""
    workspace = Path(os.environ.get("GITHUB_WORKSPACE", ""))
    if not workspace.is_absolute():
        raise ValueError("Measurement requires an absolute GITHUB_WORKSPACE checkout")
    workspace = workspace.resolve()
    if workspace != Path.cwd().resolve() or not (workspace / ".git").exists():
        raise ValueError("Measurement workspace must be the current checkout root")
    # Checkout's temporary HOME trust may not reach this child; trust only this verified
    # checkout for this read-only command, without changing persistent Git configuration.
    root, sha = subprocess.check_output(
        ["git", "-c", f"safe.directory={workspace}", "rev-parse", "--show-toplevel", "HEAD"],
        cwd=workspace,
        text=True,
        timeout=5,
    ).splitlines()
    if Path(root).resolve() != workspace:
        raise ValueError("Measurement Git checkout root differs from workspace")
    if not os.environ.get("GITHUB_SHA") or sha != os.environ["GITHUB_SHA"]:
        raise ValueError("Measurement checkout does not match event SHA")
    return sha


def capture_metadata():
    """Capture only bounded source, run and actual runtime identity."""
    import torch

    check_devices(torch.cuda)
    return {
        "sha": check_source(),
        **{
            key: os.environ[key]
            for key in ("GITHUB_SHA", "GITHUB_REF", "GITHUB_RUN_ID", "GITHUB_RUN_ATTEMPT")
        },
        "runner": os.environ["FSDP2_MEASUREMENT_RUNNER"],
        "container": os.environ["FSDP2_MEASUREMENT_IMAGE"],
        "python": sys.version,
        "torch": torch.__version__,
        "transformers": version("transformers"),
        "cuda": torch.version.cuda,
        "devices": [str(torch.cuda.get_device_properties(rank)) for rank in range(2)],
        "nodeids": NODEIDS,
        "source_sha256": {
            name: hashlib.sha256(Path(name).read_bytes()).hexdigest()
            for name in (
                *(node.split("::")[0] for node in NODEIDS),
                ".github/scripts/fsdp2_measurement.py",
                "modelopt/torch/quantization/utils/core_utils.py",
                "noxfile.py",
                ".github/workflows/gpu_tests.yml",
            )
        },
    }


def _memory(record):
    for key in ("baseline_bytes", "peak_bytes", "incremental_peak_bytes", "seconds"):
        value = record[key]
        if (
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(value)
            or value < 0
        ):
            raise ValueError(f"Invalid measured {key}")
    if record["peak_bytes"] - record["baseline_bytes"] != record["incremental_peak_bytes"]:
        raise ValueError("Inconsistent measured peak")


def validate_results(scratch, junit):
    """Accept only both successful calls and four complete actual rank receipts."""
    cases = ET.parse(junit).getroot().findall(".//testcase")
    expected = {
        (node.split("::")[0][:-3].replace("/", "."), node.split("::")[1]) for node in NODEIDS
    }
    if (
        len(cases) != 2
        or {(c.get("classname"), c.get("name")) for c in cases} != expected
        or any(len(c) for c in cases)
    ):
        raise ValueError("Measurement requires exactly two passed, non-skipped test calls")
    records = []
    for pattern, kind, lifecycle in zip(
        PATTERNS,
        ("nested_fsdp_shared_read", "native_fsdp_fp8_export"),
        ("resharded_unchanged", "resharded"),
        strict=True,
    ):
        files = sorted(scratch.rglob(pattern))
        if len(files) != 2:
            raise ValueError("Missing or duplicate per-rank measurement receipts")
        pair = [json.loads(path.read_text()) for path in files]
        if {r["rank"] for r in pair} != {0, 1}:
            raise ValueError("Invalid measurement ranks")
        for record in pair:
            if (
                record["world"] != 2
                or record["measurement"] != kind
                or record["lifecycle"] != lifecycle
            ):
                raise ValueError("Invalid measurement world, kind or lifecycle")
            if not all(record.get(key) for key in ("torch", "transformers", "cuda", "device")):
                raise ValueError("Missing per-rank runtime metadata")
            if kind == "nested_fsdp_shared_read":
                if (
                    record["exporter"] is not False
                    or record["full_groups"] != 5
                    or record["full_bytes"] != 40 * 1024**2
                    or record["persistent_payload_bytes"] != 20 * 1024**2
                    or record["shape"] != [2048, 2048]
                    or record["dtype"] != "bfloat16"
                ):
                    raise ValueError("Not the actual nested shared-reader measurement")
                _memory(record)
            else:
                if (
                    record["actual_writer"] is not True
                    or record["false_windows"] != 0
                    or len(record["windows"]) != 4
                    or {w["layer"] for w in record["windows"]} != set(range(4))
                ):
                    raise ValueError("Not the actual four-layer native FP8 writer measurement")
                for window in record["windows"]:
                    if window["writeback"] is not True:
                        raise ValueError("Invalid writer access window")
                    _memory(window)
        records.extend(pair)
    return records


def run(output):
    """Run fixed selectors, preserve failure status and publish validated evidence."""
    output.mkdir(parents=True, exist_ok=False)
    metadata = capture_metadata()
    (output / "metadata.json").write_text(json.dumps(metadata, indent=2))
    scratch, junit = output.parent / "fsdp2-measurement-pytest", output / "junit.xml"
    env = dict(
        os.environ,
        PYTEST_ADDOPTS="",
        COVERAGE_PROCESS_START="",
        HF_HUB_OFFLINE="1",
        HF_DATASETS_OFFLINE="1",
        TRANSFORMERS_OFFLINE="1",
    )
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "pytest",
            *NODEIDS,
            "--no-cov",
            "--force-reruns",
            "0",
            "--basetemp",
            str(scratch),
            "--junitxml",
            str(junit),
        ],
        env=env,
        check=False,
    )
    raw = output / "raw"
    raw.mkdir(exist_ok=True)
    for pattern in PATTERNS:
        for index, path in enumerate(sorted(scratch.rglob(pattern))):
            shutil.copyfile(path, raw / f"{index}-{path.name}")
    if result.returncode:
        return result.returncode
    records = validate_results(scratch, junit)
    (output / "receipt.json").write_text(
        json.dumps({"status": "passed", "metadata": metadata, "measurements": records}, indent=2)
    )
    print((output / "receipt.json").read_text(), flush=True)
    return 0


if __name__ == "__main__":
    if len(sys.argv) == 2 and sys.argv[1] == "preflight":
        import torch

        check_devices(torch.cuda)
        check_source()
    elif len(sys.argv) == 3 and sys.argv[1] == "run":
        raise SystemExit(run(Path(sys.argv[2])))
    else:
        raise SystemExit("Only preflight or run <output-directory> is supported")
