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

import os
import subprocess

import pytest
from _test_utils.deploy_utils import ModelDeployer, _run_deploy_via_subprocess


@pytest.mark.parametrize(
    ("backend", "field", "value", "rejected"),
    [
        ("vllm", "pipeline_parallel_size", 2, True),
        ("vllm", "pipeline_parallel_size", 1, False),
        ("sglang", "dp_size", 2, True),
        ("trtllm", "context_parallel_size", 4, True),
        ("trtllm", "gpus_per_node", 8, True),
        ("sglang", "ep_size", 4, False),  # expert parallel shares the TP ranks
    ],
)
def test_text_deploy_topology_overrides(backend, field, value, rejected):
    kwargs = {"engine_kwargs": {field: value}}
    if rejected:
        with pytest.raises(ValueError, match="Rank-multiplying"):
            ModelDeployer(backend=backend, model_id="test/model", **kwargs)
    else:
        ModelDeployer(backend=backend, model_id="test/model", **kwargs)


@pytest.mark.parametrize("private", [True, False], ids=["private", "diagnostic"])
def test_text_deploy_private_output(monkeypatch, tmp_path, capsys, private):
    # A stand-in engine echoes an unknown credential field before failing in the real worker.
    secret = "placeholder-private-authorization"
    reached = tmp_path / "engine-received.txt"
    (tmp_path / "vllm.py").write_text(
        "import pathlib\n"
        "class SamplingParams: pass\n"
        "class LLM:\n"
        "    def __init__(self, **kwargs):\n"
        f"        pathlib.Path({str(reached)!r}).write_text(kwargs['extra_headers']['Authorization'])\n"
        "        print(kwargs, flush=True)\n"
        "        raise ValueError(str(kwargs))\n",
        encoding="utf-8",
    )
    monkeypatch.setenv("PYTHONPATH", str(tmp_path) + os.pathsep + os.getenv("PYTHONPATH", ""))
    with pytest.raises(subprocess.CalledProcessError) as error:
        _run_deploy_via_subprocess(
            "vllm",
            model_id="test/model",
            engine_kwargs={"extra_headers": {"Authorization": secret}},
            private_output=private,
        )
    assert reached.read_text() == secret
    # Worker argv only carries the config path in either mode.
    assert secret not in repr(error.value.cmd)
    captured = capsys.readouterr().out
    if private:
        assert not error.value.output
        assert secret not in str(error.value)
        assert secret not in captured
    else:
        # Public recipe values keep their diagnostic output on failure.
        assert secret in error.value.output
        assert secret in captured
