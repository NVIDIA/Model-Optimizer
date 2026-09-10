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

"""Tests for modelopt.torch.puzzletron.tools.common utilities."""

import pytest
import torch

import modelopt.torch.puzzletron as mtpz


@pytest.mark.parametrize(
    ("input_dtype", "expected"),
    [
        ("torch.bfloat16", torch.bfloat16),
        ("torch.float16", torch.float16),
        ("torch.float32", torch.float32),
        ("bfloat16", torch.bfloat16),
        ("float16", torch.float16),
        ("float32", torch.float32),
        (torch.bfloat16, torch.bfloat16),
        (torch.float32, torch.float32),
    ],
    ids=[
        "str-bf16",
        "str-fp16",
        "str-fp32",
        "bare-bf16",
        "bare-fp16",
        "bare-fp32",
        "dtype-bf16",
        "dtype-fp32",
    ],
)
def test_resolve_torch_dtype(input_dtype, expected):
    assert mtpz.tools.resolve_torch_dtype(input_dtype) is expected


def test_resolve_torch_dtype_unknown_name():
    with pytest.raises(ValueError, match="Unknown torch dtype"):
        mtpz.tools.resolve_torch_dtype("not_a_real_dtype")


def test_resolve_torch_dtype_non_dtype_attr():
    with pytest.raises(ValueError, match="is not a dtype"):
        mtpz.tools.resolve_torch_dtype("torch.nn")
