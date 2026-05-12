# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

import pytest
import torch

from modelopt.torch.opt.conversion import _validate_modelopt_state, load_modelopt_state


def test_validate_accepts_empty_state_dict():
    _validate_modelopt_state({"modelopt_state_dict": [], "modelopt_version": "0.0.0"})


def test_validate_accepts_populated_state_dict():
    _validate_modelopt_state(
        {
            "modelopt_state_dict": [("some_mode", {"config": {}, "metadata": {}})],
            "modelopt_version": "1.2.3",
        }
    )


@pytest.mark.parametrize("bad", [[], None, 42, "state", (1, 2)])
def test_validate_rejects_non_dict(bad):
    with pytest.raises(TypeError, match="expected a dict"):
        _validate_modelopt_state(bad)


def test_validate_rejects_full_checkpoint():
    ckpt = {"modelopt_state": {}, "model_state_dict": {}}
    with pytest.raises(ValueError, match="full checkpoint"):
        _validate_modelopt_state(ckpt)


def test_validate_rejects_missing_version():
    with pytest.raises(ValueError, match="missing required key"):
        _validate_modelopt_state({"modelopt_state_dict": []})


def test_validate_rejects_missing_state_dict():
    with pytest.raises(ValueError, match="missing required key"):
        _validate_modelopt_state({"modelopt_version": "1.0.0"})


def test_validate_rejects_non_list_state_dict():
    with pytest.raises(TypeError, match="'modelopt_state_dict' must be a list"):
        _validate_modelopt_state(
            {"modelopt_state_dict": {"not": "a list"}, "modelopt_version": "1.0.0"}
        )


def test_validate_rejects_non_str_version():
    with pytest.raises(TypeError, match="'modelopt_version' must be a str"):
        _validate_modelopt_state({"modelopt_state_dict": [], "modelopt_version": 1.0})


def test_load_modelopt_state_valid(tmp_path):
    path = tmp_path / "state.pt"
    state = {"modelopt_state_dict": [], "modelopt_version": "1.0.0"}
    torch.save(state, path)
    loaded = load_modelopt_state(path)
    assert loaded == state


def test_load_modelopt_state_invalid_type(tmp_path):
    path = tmp_path / "bad.pt"
    torch.save([1, 2, 3], path)
    with pytest.raises(TypeError, match="expected a dict"):
        load_modelopt_state(path)


def test_load_modelopt_state_missing_keys(tmp_path):
    path = tmp_path / "bad.pt"
    torch.save({"foo": "bar"}, path)
    with pytest.raises(ValueError, match="missing required key"):
        load_modelopt_state(path)


def test_load_modelopt_state_full_checkpoint(tmp_path):
    path = tmp_path / "ckpt.pt"
    torch.save({"modelopt_state": {}, "model_state_dict": {}}, path)
    with pytest.raises(ValueError, match="full checkpoint"):
        load_modelopt_state(path)
