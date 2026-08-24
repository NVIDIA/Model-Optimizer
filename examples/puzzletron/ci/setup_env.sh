#!/usr/bin/env bash
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

set -euo pipefail

MODE="${1:---deps}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MODEL_OPT_ROOT="${MODEL_OPT_ROOT:-$(cd "${SCRIPT_DIR}/../../.." && pwd)}"
VIRTUAL_ENV="${VIRTUAL_ENV:-/venv}"
PUZZLETRON_CI_ENVIRONMENT="${PUZZLETRON_CI_ENVIRONMENT:-${MODEL_OPT_ROOT}/examples/puzzletron/ci_environment.json}"
PUZZLETRON_REQUIREMENTS="${PUZZLETRON_REQUIREMENTS:-${MODEL_OPT_ROOT}/examples/puzzletron/requirements.txt}"

export GIT_TERMINAL_PROMPT="${GIT_TERMINAL_PROMPT:-0}"
export PIP_NO_CACHE_DIR="${PIP_NO_CACHE_DIR:-1}"
export PYTHONUNBUFFERED="${PYTHONUNBUFFERED:-1}"
export VIRTUAL_ENV
export PATH="${VIRTUAL_ENV}/bin:${PATH}"

environment_value() {
    python3 - "${PUZZLETRON_CI_ENVIRONMENT}" "$1" <<'PY'
import json
import sys

value = json.load(open(sys.argv[1], encoding="utf-8"))
for part in sys.argv[2].split("."):
    value = value[part]
print(value)
PY
}

verify_environment() {
    python - "${PUZZLETRON_CI_ENVIRONMENT}" <<'PY'
import json
import sys
from importlib import metadata

import torch
from packaging.version import Version

environment = json.load(open(sys.argv[1], encoding="utf-8"))
expected = {
    "python": environment["python"],
    "torch": environment["torch"],
    "torchvision": environment["torchvision"],
    "transformers": environment["transformers"],
    "lmms-eval": environment["lmms_eval"]["base_version"],
    "nemo-automodel": environment["nemo_automodel"]["base_version"],
    "aiperf": environment["gpu_image"]["aiperf"],
    "nox": environment["gpu_image"]["nox"],
}
actual = {
    "python": f"{sys.version_info.major}.{sys.version_info.minor}",
    **{
        package: Version(metadata.version(package)).base_version
        for package in expected
        if package != "python"
    },
}
mismatches = {
    package: (actual[package], expected_version)
    for package, expected_version in expected.items()
    if actual[package] != expected_version
}
if mismatches:
    raise RuntimeError(f"Pinned Puzzletron image mismatch: {mismatches}")

for package, key in (("lmms-eval", "lmms_eval"), ("nemo-automodel", "nemo_automodel")):
    direct_url = json.loads(metadata.distribution(package).read_text("direct_url.json") or "{}")
    vcs_info = direct_url.get("vcs_info") or {}
    actual_source = (
        str(direct_url.get("url") or "").removesuffix(".git").rstrip("/"),
        vcs_info.get("commit_id"),
    )
    expected_source = (
        environment[key]["repository"].removesuffix(".git").rstrip("/"),
        environment[key]["commit"],
    )
    if actual_source != expected_source:
        raise RuntimeError(
            f"Pinned Puzzletron dependency {package!r} source mismatch: "
            f"actual={actual_source!r}, expected={expected_source!r}"
        )

if torch.version.cuda != environment["gpu_image"]["torch_cuda"]:
    raise RuntimeError(
        "Pinned Puzzletron CUDA mismatch: "
        f"actual={torch.version.cuda!r}, expected={environment['gpu_image']['torch_cuda']!r}"
    )
PY
}

install_dependencies() {
    apt-get update
    apt-get install -y --no-install-recommends \
        build-essential ca-certificates cmake git ninja-build \
        python3 python3-dev python3-pip python3-venv
    apt-get clean

    python3 -m venv "${VIRTUAL_ENV}"
    python -m pip install --upgrade \
        pip "setuptools>=80,<81" "setuptools-scm>=8,<10" setuptools-rust wheel \
        "packaging>=24.2" "cmake>=3.26.1" ninja jinja2

    python -m pip install \
        "torch==$(environment_value torch)" \
        "torchvision==$(environment_value torchvision)" \
        "torchaudio==$(environment_value torch)" \
        --index-url https://download.pytorch.org/whl/cu129
    python -m pip install \
        -r "${PUZZLETRON_REQUIREMENTS}" \
        "nemo-automodel @ git+$(environment_value nemo_automodel.repository)@$(environment_value nemo_automodel.commit)" \
        "aiperf==$(environment_value gpu_image.aiperf)" \
        "nox==$(environment_value gpu_image.nox)"
    python -m pip install "transformers==$(environment_value transformers)"
    python -m pip check
    verify_environment
}

install_modelopt() {
    if [[ ! -x "${VIRTUAL_ENV}/bin/python" ]] || \
        [[ "$(python -c 'import sys; print(sys.prefix)')" != "${VIRTUAL_ENV}" ]]; then
        echo "Puzzletron CI must install ModelOpt inside ${VIRTUAL_ENV}." >&2
        exit 1
    fi
    python -m pip uninstall -y nvidia-modelopt
    python -m pip install --no-build-isolation --no-deps -e \
        "${MODEL_OPT_ROOT}[hf,puzzletron,dev-test]"
    python -m pip check
    verify_environment
}

case "${MODE}" in
    --deps)
        install_dependencies
        ;;
    --modelopt)
        install_modelopt
        ;;
    --verify)
        verify_environment
        ;;
    *)
        echo "Usage: $0 [--deps | --modelopt | --verify]" >&2
        exit 2
        ;;
esac
