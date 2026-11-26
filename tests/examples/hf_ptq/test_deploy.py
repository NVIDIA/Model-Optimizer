# SPDX-FileCopyrightText: Copyright (c) 2023-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

import os
from functools import cache
from pathlib import Path

import pytest
import yaml
from _test_utils.deploy_config import load_deploy_cases, load_deploy_config, resolve_deploy_config
from _test_utils.deploy_utils import ModelDeployer, get_available_backends
from _test_utils.image_deploy_utils import ImageModelDeployer

pytestmark = pytest.mark.release


@cache
def _deployment_settings(config):
    groups = load_deploy_cases(Path(__file__).with_name("deploy_cases"))
    expected = {name for name in globals() if name.startswith("test_")}
    if unknown := groups.keys() - expected:
        raise ValueError(f"Unknown deployment test groups: {sorted(unknown)}")
    if missing := expected - groups.keys():
        raise ValueError(f"Missing deployment test groups: {sorted(missing)}")
    return groups, load_deploy_config(config.getoption("--deploy-config"))


def pytest_generate_tests(metafunc):
    if "command" not in metafunc.fixturenames:
        return
    try:
        groups, overrides = _deployment_settings(metafunc.config)
        tp = metafunc.config.getoption("--deploy-tp")
        if tp is not None and tp < 1:
            raise ValueError("--deploy-tp must be a positive integer")
        requested_backend = metafunc.config.getoption("--deploy-backend") or overrides["run"].get(
            "backend"
        )
        # Explicit selection must reach execution even if an engine import is broken.
        available = {requested_backend} if requested_backend else get_available_backends()
        commands = [
            pytest.param((case, backend), id=f"{case['model_id']}_{backend}")
            for case in groups[metafunc.function.__name__]
            for backend in case["backends"]
            if backend in available
        ]
    except (OSError, ValueError, yaml.YAMLError) as exc:
        raise pytest.UsageError(f"Invalid deployment configuration: {exc}") from exc
    if requested_backend and not commands:
        commands = [pytest.param(None, marks=pytest.mark.deploy_empty_group, id="no-backend-cases")]
    metafunc.parametrize("command", commands, indirect=True)


@pytest.fixture
def command(request):
    """Apply run overrides only after pytest has selected this case."""
    case, backend = request.param
    _, overrides = _deployment_settings(request.config)
    params = resolve_deploy_config(
        case, backend, overrides, tensor_parallel_size=request.config.getoption("--deploy-tp")
    )
    deployer = ImageModelDeployer if case.get("task") == "image-generation" else ModelDeployer
    return deployer(**params)


def test_deepseek(command):
    command.run()


def test_llama(command):
    command.run()


def test_qwen(command):
    command.run()


@pytest.mark.timeout(7200)
def test_qwen_image_nvfp4(command, tmp_path):
    command.run(tmp_path)


def test_gemma(command):
    command.run()


# test phi
def test_phi(command):
    command.run()


def test_kimi(command):
    if command.model_id == "nvidia/Kimi-K2.6-Eagle3" and command.backend in {"trtllm", "sglang"}:
        draft = _local_eagle_draft(command.model_id)
        if draft is not None and draft.is_dir():
            command.model_id = str(draft)
    command.run()


def test_glm(command):
    command.run()


def test_minimax(command):
    command.run()


def test_llama_nemotron(command):
    command.run()


def test_medusa(command):
    command.run()


def _local_eagle_draft(model_id):
    local_root = os.getenv("MODELOPT_LOCAL_EAGLE_MODEL")
    return Path(local_root) / model_id if local_root else None


def test_eagle(command):
    draft = _local_eagle_draft(command.model_id)
    if draft is None:
        pytest.skip("MODELOPT_LOCAL_EAGLE_MODEL is not set")
    if not draft.is_dir():
        pytest.skip(f"Local model not found: {draft}")
    command.model_id = str(draft)
    command.run()


def test_nvidia_nemotron_3_ultra_550b_a55b_nvfp4(command):
    command.run()


def test_wan2_2_t2v_a14b_diffusers_nvfp4(command):
    command.run()


def test_wan2_2_t2v_a14b_diffusers_fp8(command):
    command.run()


def test_diffusiongemma_26b_a4b_it_nvfp4(command):
    command.run()


def test_nemotron(command):
    command.run()


def test_nemotron_lilicorr(command):
    command.run()


def test_muse_spark_dflash2(command):
    command.run()


def test_step5_preview(command):
    command.run()


def test_nvidia_nemotron_3_ultra_550b_a55b_bf16(command):
    command.run()
