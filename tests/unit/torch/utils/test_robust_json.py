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
import datetime
import json
from enum import Enum

import pytest
import torch

from modelopt.torch.utils.robust_json import json_dumps


class _Color(Enum):
    RED = "red"
    GREEN = "green"


@pytest.mark.parametrize(
    ("obj", "expected"),
    [
        (_Color.RED, "RED"),  # Enum encoded as name, not value
        (argparse.Namespace(lr=0.1, name="exp"), {"lr": 0.1, "name": "exp"}),
        (torch.bfloat16, "torch.bfloat16"),  # torch dtype encoded as string
        (datetime.timedelta(hours=1, minutes=2, seconds=3), "1:02:03"),
    ],
)
def test_special_types_encoded(obj, expected):
    assert json.loads(json_dumps({"k": obj})) == {"k": expected}


def test_main_module_function_encoded_as_bare_name():
    def user_fn():
        pass

    # user-defined functions in __main__ fall back to just the name
    user_fn.__module__ = "__main__"
    assert json.loads(json_dumps({"f": user_fn})) == {"f": "user_fn"}
