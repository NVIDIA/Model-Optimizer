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

import traceback

import pytest
from _test_utils.deploy_config import (
    engine_tp_kwargs,
    load_deploy_cases,
    load_deploy_config,
    resolve_deploy_config,
)


@pytest.mark.parametrize("backend", ["vllm", "sglang", "trtllm"])
@pytest.mark.parametrize("scenario", ["defaults", "yaml", "cli", "conflicting_tp"])
def test_deploy_config_resolution(tmp_path, backend, scenario):
    context_key = {"vllm": "max_model_len", "sglang": "context_length", "trtllm": "max_seq_len"}[
        backend
    ]
    nullable_key = "quant_config" if backend == "trtllm" else "quantization"
    catalog = tmp_path / "cases"
    (catalog / "family").mkdir(parents=True)
    (catalog / "family" / "model.yaml").write_text(
        f"""test_deploy:
  - model_id: test/model
    tensor_parallel_size: 8
    mini_sm: 100
    backends:
      {backend}:
        engine_kwargs:
          {context_key}: 4096
""",
        encoding="utf-8",
    )
    (catalog / "family" / "other.yaml").write_text(
        f"test_deploy:\n  - model_id: test/other\n    backends:\n      {backend}: {{}}\n",
        encoding="utf-8",
    )
    cases = load_deploy_cases(catalog)["test_deploy"]
    assert [case["model_id"] for case in cases] == ["test/model", "test/other"]
    case = cases[0]
    overrides = load_deploy_config()
    if scenario != "defaults":
        run = tmp_path / "run.yaml"
        tp_key = "tp_size" if backend == "sglang" else "tensor_parallel_size"
        conflicting_tp = f"      {tp_key}: 2\n" if scenario == "conflicting_tp" else ""
        run.write_text(
            f"""run:
  backend: {backend}
  tensor_parallel_size: 4
backends:
  {backend}:
    engine_kwargs:
      {context_key}: 2048
      {nullable_key}: null
{conflicting_tp}    sampling_kwargs:
      temperature: 0
""",
            encoding="utf-8",
        )
        if scenario == "conflicting_tp":
            with pytest.raises(ValueError, match="outside engine_kwargs"):
                load_deploy_config(run)
            return
        overrides = load_deploy_config(run)
    cli_tp = 2 if scenario == "cli" else None
    resolved = resolve_deploy_config(case, backend, overrides, tensor_parallel_size=cli_tp)
    expected_tp = {"defaults": 8, "yaml": 4, "cli": 2}[scenario]
    assert resolved["tensor_parallel_size"] == expected_tp
    assert engine_tp_kwargs(backend, resolved["tensor_parallel_size"]) == {
        "tp_size" if backend == "sglang" else "tensor_parallel_size": expected_tp
    }
    assert resolved["model_id"] == "test/model"
    assert resolved["mini_sm"] == 100
    expected_engine = {context_key: 4096}
    if scenario != "defaults":
        expected_engine = {context_key: 2048, nullable_key: None}
    assert resolved["engine_kwargs"] == expected_engine
    assert resolved["sampling_kwargs"] == ({} if scenario == "defaults" else {"temperature": 0})
    assert case["tensor_parallel_size"] == 8


@pytest.mark.parametrize(
    ("mapping", "expected_tp"),
    [
        ("{<<: {tensor_parallel_size: 2, tensor_parallel_size: 4}}", None),
        ("{<<: {tensor_parallel_size: 2}, <<: {tensor_parallel_size: 4}}", None),
        ("{<<: [{backend: vllm}, {tensor_parallel_size: 4}]}", 4),
        ("{<<: {tensor_parallel_size: 2}, tensor_parallel_size: 4}", 4),
    ],
)
def test_yaml_merge_keys(tmp_path, mapping, expected_tp):
    config = tmp_path / "run.yaml"
    config.write_text(f"run: {mapping}\n", encoding="utf-8")
    if expected_tp is None:
        with pytest.raises(ValueError, match="Duplicate YAML"):
            load_deploy_config(config)
    else:
        assert load_deploy_config(config)["run"]["tensor_parallel_size"] == expected_tp


@pytest.mark.parametrize("variant", ["invalid-integer", "malformed-scalar", "duplicate-key"])
def test_yaml_errors_report_location_not_values(tmp_path, variant):
    secret = "placeholder-private-authorization"
    bodies = {
        "malformed-scalar": f'run:\n  hf_token: "{secret}" {{unclosed\n',
        "invalid-integer": f"backends:\n  vllm:\n    engine_kwargs:\n      hf_token: !!int {secret}\n",
        "duplicate-key": f'run:\n  backend: vllm\n  hf_token: "{secret}"\n  hf_token: "{secret}"\n',
    }
    config = tmp_path / "run.yaml"
    config.write_text(bodies[variant], encoding="utf-8")
    with pytest.raises(ValueError) as error:
        load_deploy_config(config)
    message = str(error.value)
    assert secret not in message
    assert str(config) in message
    assert secret not in "".join(traceback.format_exception(error.value))
    assert "line" in message and "column" in message
