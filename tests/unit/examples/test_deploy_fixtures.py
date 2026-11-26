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

import importlib.util
import os
from pathlib import Path
from types import SimpleNamespace

import pytest
from _test_utils.deploy_config import load_deploy_config
from _test_utils.deploy_utils import ModelDeployer
from huggingface_hub import constants

_SPEC = importlib.util.spec_from_file_location(
    "hf_ptq_deploy", Path(__file__).resolve().parents[2] / "examples/hf_ptq/test_deploy.py"
)
deploy = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(deploy)


@pytest.mark.parametrize(
    ("env_var", "env_value", "hub_subpath"),
    [
        (None, None, None),
        ("HF_HUB_CACHE", "hub", "hub"),
        ("HUGGINGFACE_HUB_CACHE", "hub", "hub"),
        ("HF_HOME", "home", "home/hub"),
        ("TRANSFORMERS_CACHE", "legacy", "hub"),
    ],
    ids=["owned", "HF_HUB_CACHE", "HUGGINGFACE_HUB_CACHE", "HF_HOME", "TRANSFORMERS_CACHE"],
)
def test_deployment_cache(monkeypatch, tmp_path, env_var, env_value, hub_subpath):
    monkeypatch.chdir(tmp_path)
    for key in (
        "HF_HUB_CACHE",
        "HUGGINGFACE_HUB_CACHE",
        "HF_HOME",
        "HF_HUB_OFFLINE",
        "TRANSFORMERS_OFFLINE",
        "TRANSFORMERS_CACHE",
        "PYTORCH_TRANSFORMERS_CACHE",
        "PYTORCH_PRETRAINED_BERT_CACHE",
    ):
        monkeypatch.delenv(key, raising=False)
    monkeypatch.setattr(constants, "HF_HUB_CACHE", "hub")
    if env_var:
        monkeypatch.setenv(env_var, env_value)
    case = {"model_id": "test/model", "backends": {"vllm": {}}}
    request = SimpleNamespace(
        config=SimpleNamespace(getoption=lambda name: None),
        node=SimpleNamespace(path=Path(__file__)),
        session=SimpleNamespace(items=[]),
        param=(case, "vllm"),
    )
    monkeypatch.setattr(deploy, "_deployment_settings", lambda config: ({}, load_deploy_config()))
    expected = (tmp_path / hub_subpath).resolve() if env_var else None
    snapshot = (expected or tmp_path / "hub") / "models--test--model/snapshots/staged/config.json"
    snapshot.parent.mkdir(parents=True)
    snapshot.write_text("{}", encoding="utf-8")
    cache_fixture = deploy._deployment_cache.__wrapped__(request)
    cache = next(cache_fixture)
    command_fixture = deploy.command.__wrapped__(request, cache, monkeypatch)
    command = next(command_fixture)
    assert command.model_id == "test/model"
    worker_cache = Path(os.environ["HF_HUB_CACHE"])
    assert worker_cache.is_absolute()
    if env_var:
        # Relative inherited paths are resolved before workers change to tests/.
        assert Path(os.environ[env_var]) == (tmp_path / env_value).resolve()
        assert worker_cache == expected
        assert (worker_cache / snapshot.relative_to(expected)).read_text() == "{}"
    else:
        assert worker_cache != snapshot.parents[3]
        assert worker_cache.is_dir()
    command_fixture.close()
    cache_fixture.close()
    assert snapshot.read_text() == "{}"
    if not env_var:
        assert not worker_cache.exists()


@pytest.mark.parametrize("backend", ["trtllm", "sglang"])
def test_local_eagle_draft(monkeypatch, tmp_path, backend):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("MODELOPT_LOCAL_EAGLE_MODEL", "drafts")
    model_id = "nvidia/Kimi-K2.6-Eagle3"
    draft = tmp_path / "drafts" / model_id
    draft.mkdir(parents=True)
    kwargs = (
        {"speculative_config": {"decoding_type": "Eagle"}}
        if backend == "trtllm"
        else {"speculative_algorithm": "EAGLE3"}
    )
    command = ModelDeployer(
        backend=backend, model_id=model_id, base_model="test/target", engine_kwargs=kwargs
    )
    arguments = {}
    monkeypatch.setattr(command, "run", lambda: arguments.update(command._engine_arguments()))
    deploy.test_kimi(command)
    actual = (
        arguments["speculative_config"]["speculative_model_dir"]
        if backend == "trtllm"
        else arguments["speculative_draft_model_path"]
    )
    assert Path(actual) == draft.resolve()
    draft.rmdir()
    command.model_id = model_id
    with pytest.raises(pytest.skip.Exception):
        deploy.test_kimi(command)
