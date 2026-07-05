# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

import argparse
import dataclasses
import datetime
import json
from enum import Enum
from pathlib import Path

import torch
from omegaconf import OmegaConf

from modelopt.torch.utils.robust_json import json_dump, json_dumps, json_load


@dataclasses.dataclass
class _Point:
    x: int
    y: int


@dataclasses.dataclass
class _Config:
    name: str
    point: _Point
    out_dir: Path


class _Color(Enum):
    RED = "red"
    GREEN = "green"


class _Widget:
    """Arbitrary object without any special JSON support."""

    def render(self):
        """Regular method used for the bound-method encoding test."""


def _dumps_loads(obj):
    return json.loads(json_dumps(obj))


def test_plain_json_types_round_trip():
    obj = {"a": 1, "b": 2.5, "c": "str", "d": None, "e": True, "f": [1, "x", False]}
    assert _dumps_loads(obj) == obj


def test_output_is_indented_by_two_spaces():
    assert json_dumps({"a": 1}) == '{\n  "a": 1\n}'


def test_dataclass_encoded_as_dict_recursively():
    cfg = _Config(name="run", point=_Point(x=1, y=2), out_dir=Path("out") / "dir")
    assert _dumps_loads(cfg) == {
        "name": "run",
        "point": {"x": 1, "y": 2},
        "out_dir": str(Path("out") / "dir"),
    }


def test_path_encoded_as_string():
    path = Path("some") / "nested" / "file.txt"
    assert _dumps_loads({"p": path}) == {"p": str(path)}


def test_enum_encoded_as_name_not_value():
    assert _dumps_loads({"color": _Color.RED}) == {"color": "RED"}


def test_argparse_namespace_encoded_as_dict():
    ns = argparse.Namespace(lr=0.1, name="exp")
    assert _dumps_loads(ns) == {"lr": 0.1, "name": "exp"}


def test_torch_dtype_encoded_as_string():
    assert _dumps_loads({"dtype": torch.float32}) == {"dtype": "torch.float32"}
    assert _dumps_loads({"dtype": torch.bfloat16}) == {"dtype": "torch.bfloat16"}


def test_omegaconf_containers_resolved():
    dict_cfg = OmegaConf.create({"a": 1, "b": {"c": [1, 2]}})
    list_cfg = OmegaConf.create([1, "x"])
    assert _dumps_loads(dict_cfg) == {"a": 1, "b": {"c": [1, 2]}}
    assert _dumps_loads(list_cfg) == [1, "x"]


def test_omegaconf_interpolation_resolved():
    cfg = OmegaConf.create({"a": 5, "b": "${a}"})
    assert _dumps_loads(cfg) == {"a": 5, "b": 5}


def test_function_encoded_as_qualified_name():
    assert _dumps_loads({"fn": json.dumps}) == {"fn": "json.dumps"}


def test_method_encoded_as_qualified_name():
    method = _Widget().render  # bound method
    expected = f"{method.__module__}.{method.__qualname__}"
    assert _dumps_loads({"m": method}) == {"m": expected}


def test_class_encoded_as_qualified_name():
    expected = f"{Path.__module__}.{Path.__qualname__}"
    assert _dumps_loads({"cls": Path}) == {"cls": expected}


def test_timedelta_encoded_as_string():
    td = datetime.timedelta(hours=1, minutes=2, seconds=3)
    assert _dumps_loads({"td": td}) == {"td": "1:02:03"}


def test_arbitrary_object_falls_back_to_class_path():
    expected = f"{_Widget.__module__}.{_Widget.__qualname__}"
    assert _dumps_loads({"obj": _Widget()}) == {"obj": expected}


def test_json_dump_creates_parent_dirs_and_json_load_round_trips(tmp_path):
    path = tmp_path / "a" / "b" / "result.json"  # parents do not exist yet
    obj = {"name": "test", "values": [1, 2, 3], "nested": {"ok": True}}
    json_dump(obj, path)
    assert path.is_file()
    assert json_load(path) == obj


def test_json_dump_accepts_string_path(tmp_path):
    path = str(tmp_path / "out.json")
    json_dump({"k": 1}, path)
    assert json_load(path) == {"k": 1}
