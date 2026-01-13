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
# mypy: ignore-errors

"""
Provides a robust JSON encoder that can handle various types of objects,
including dataclasses, paths, enums, namespaces, and functions.
"""

import argparse
import dataclasses
import datetime
import inspect
import json
from enum import Enum
from pathlib import Path
from typing import Any

from omegaconf import DictConfig, ListConfig, OmegaConf


class RobustJSONEncoder(json.JSONEncoder):
    def default(self, o):
        if dataclasses.is_dataclass(o):
            return dataclasses.asdict(o)
        if isinstance(o, Path):
            return str(o)
        if isinstance(o, Enum):
            return o.name
        if isinstance(o, argparse.Namespace):
            return vars(o)
        if type(o).__name__ == "dtype":
            return str(o)
        if isinstance(o, (DictConfig, ListConfig)):
            return OmegaConf.to_container(o, resolve=True)
        if inspect.isfunction(o) or inspect.ismethod(o):
            if o.__module__ == "__main__":
                # User-defined function in main — fallback to just the name
                return o.__name__
            return f"{o.__module__}.{o.__qualname__}"
        if isinstance(o, datetime.timedelta):
            return str(o)
        return super().default(o)


def json_dumps(obj: Any) -> str:
    return json.dumps(obj, cls=RobustJSONEncoder, indent=2)


def json_dump(obj: Any, path: Path | str) -> None:
    path = Path(path)
    path.parent.mkdir(exist_ok=True, parents=True)
    json_text = json_dumps(obj)
    path.write_text(json_text)


def json_load(path: Path | str) -> dict:
    path = Path(path)
    text = path.read_text()
    return json.loads(text)
