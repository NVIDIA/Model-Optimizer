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
import shutil
import tempfile
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


@pytest.fixture(scope="module")
def _deployment_cache(request):
    # Legacy Transformers cache overrides take precedence over the Hub cache in some images.
    transformers_caches = (
        "TRANSFORMERS_CACHE",
        "PYTORCH_TRANSFORMERS_CACHE",
        "PYTORCH_PRETRAINED_BERT_CACHE",
    )
    offline = any(
        os.getenv(key, "").upper() in {"1", "ON", "YES", "TRUE"}
        for key in ("HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE")
    )
    if (
        offline
        or os.getenv("HF_HUB_CACHE")
        or hasattr(request.config, "workerinput")
        or any(os.getenv(key) for key in transformers_caches)
    ):
        yield None
        return

    # Hub is optional during collection. Use its configured volume for large downloads.
    from huggingface_hub.constants import HF_HUB_CACHE

    hub = Path(HF_HUB_CACHE).expanduser().resolve()
    hub.mkdir(parents=True, exist_ok=True)
    last_consumer = {}
    for item in request.session.items:
        if item.path != request.node.path:
            continue
        parameter = item.callspec.params.get("command")
        if not isinstance(parameter, tuple):  # pytest placeholder for a group without a backend
            continue
        case, _ = parameter
        for repo_id in (case["model_id"], case.get("base_model")):
            if repo_id:
                last_consumer["models--" + repo_id.replace("/", "--")] = item
    with tempfile.TemporaryDirectory(prefix=".modelopt-deploy-", dir=hub) as owned:
        yield Path(owned), last_consumer


def _native_overrides(overrides, backend):
    """Whether ``--deploy-config`` supplies native engine, sampling or chat options."""
    return any(
        overrides.get("backends", {}).get(backend, {}).get(name)
        for name in ("engine_kwargs", "sampling_kwargs", "chat_kwargs")
    )


@pytest.fixture
def command(request, _deployment_cache, monkeypatch):
    """Apply run overrides only after pytest has selected this case."""
    case, backend = request.param
    _, overrides = _deployment_settings(request.config)
    params = resolve_deploy_config(
        case, backend, overrides, tensor_parallel_size=request.config.getoption("--deploy-tp")
    )
    deployer = ImageModelDeployer if case.get("task") == "image-generation" else ModelDeployer
    if deployer is ModelDeployer:
        # Override values are caller-supplied and may carry credentials; recipe values are public.
        params["private_output"] = _native_overrides(overrides, backend)
    if _deployment_cache is None:
        if cache_root := os.getenv("HF_HUB_CACHE"):
            monkeypatch.setenv("HF_HUB_CACHE", str(Path(cache_root).expanduser().resolve()))
        yield deployer(**params)
        return
    hub, last_consumer = _deployment_cache
    # Every engine starts in a fresh process and reads this before importing the Hub.
    monkeypatch.setenv("HF_HUB_CACHE", str(hub))
    try:
        yield deployer(**params)
    finally:
        for entry in hub.glob("models--*"):
            if entry.name not in last_consumer or last_consumer[entry.name] is request.node:
                last_consumer.pop(entry.name, None)
                if entry.is_symlink():
                    entry.unlink()
                elif entry.is_dir():
                    shutil.rmtree(entry)


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
        if draft is not None:
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
    if local_root:
        draft = (Path(local_root).expanduser() / model_id).resolve()
        if not draft.is_dir():
            pytest.skip(f"Local model not found: {draft}")
        return draft
    return None


def test_eagle(command):
    draft = _local_eagle_draft(command.model_id)
    if draft is None:
        pytest.skip("MODELOPT_LOCAL_EAGLE_MODEL is not set")
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
