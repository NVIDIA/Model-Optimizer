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

"""Unit tests for :mod:`modelopt.torch.export.trtllm.quant_utils`.

These helpers are reachable only from the TensorRT-LLM checkpoint export path
(``trtllm/postprocess.py``), so they are tested alongside it rather than with the
backend-agnostic helpers in ``modelopt.torch.export.quant_utils``.

Test modules in this directory carry a ``trtllm_`` prefix because pytest runs without
``__init__.py`` here and derives the module name from the bare filename, so a plain
``test_quant_utils.py`` would collide with the one a directory up.
"""

import contextlib
import inspect
import warnings

import pytest
import torch
import torch.nn as nn

from modelopt.torch.export.trtllm import (
    export_tensorrt_llm_checkpoint,
    torch_to_tensorrt_llm_checkpoint,
)
from modelopt.torch.export.trtllm.quant_utils import get_scaling_factor_from_weight


@pytest.mark.parametrize(
    ("weight", "group_size", "expected"),
    [
        (
            torch.tensor([[0.0, 0.35, 0.28, 7.0], [0.49, 0.84, -0.77, 0.07]]),
            2,
            torch.tensor([[0.05, 1.0], [0.12, 0.11]]),
        ),  # group_size != 0 and divides weight.shape[1]
        (
            torch.tensor([[0.127, 0.0, 1.27, -12.7], [0.0, 127.0, 0.254, 2.54]]),
            0,
            torch.tensor([0.1, 1.0]),
        ),  # group_size = 0
        (
            torch.tensor([[0.0, 0.0, 0.0, 0.0], [0.0, -0.127, 0.254, 2.54]]),
            0,
            torch.tensor([1.0, 0.02]),
        ),  # zero replaced with 1.0
        (
            torch.tensor([[0.0, 0.84, -0.77, 0.07], [0.0, 0.0, 0.0, 0.0]]),
            2,
            torch.tensor([[0.12, 0.11], [1.0, 1.0]]),
        ),  # zero replaced with 1.0
    ],
)
def test_get_scaling_factor_from_weight(weight, group_size, expected):
    scaling_factor = get_scaling_factor_from_weight(weight, group_size)
    # Check if shapes match
    if group_size != 0:
        assert list(scaling_factor.shape) == [weight.shape[0], weight.shape[1] // group_size]
    else:
        assert list(scaling_factor.shape) == [weight.shape[0]]

    assert torch.allclose(scaling_factor, expected, rtol=0.0, atol=0.0)


def test_old_top_level_import_still_works():
    """The pre-0.48 import path stays importable for the migration period.

    A DeprecationWarning is only useful if callers can still reach the code it warns about,
    so removing this re-export before 0.49.0 would silently skip the migration window.
    """
    import modelopt.torch.export as mte

    assert mte.export_tensorrt_llm_checkpoint is export_tensorrt_llm_checkpoint
    assert mte.torch_to_tensorrt_llm_checkpoint is torch_to_tensorrt_llm_checkpoint


def test_torch_to_tensorrt_llm_checkpoint_warns_at_call_time():
    """The warning must fire on call, not on first ``next()``.

    ``torch_to_tensorrt_llm_checkpoint`` hands back a generator. A bare ``warnings.warn`` in a
    generator body does not run until the first item is pulled, so a caller that builds the
    generator and abandons it would never be warned. Hence the public name is a plain function
    wrapping a private generator.
    """
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        generator = torch_to_tensorrt_llm_checkpoint(nn.Linear(4, 4), "llama")

    assert inspect.isgenerator(generator)
    deprecations = [w for w in caught if issubclass(w.category, DeprecationWarning)]
    assert len(deprecations) == 1, "expected exactly one DeprecationWarning before iteration"
    assert "torch_to_tensorrt_llm_checkpoint" in str(deprecations[0].message)


def test_export_tensorrt_llm_checkpoint_warns_exactly_once(tmp_path):
    """One user call yields one warning, not two.

    ``export_tensorrt_llm_checkpoint`` drives the same generator, so it must call the private
    form; calling the public wrapper instead would emit a second, redundant warning.
    """
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        with contextlib.suppress(Exception):  # a toy model cannot complete a real export
            export_tensorrt_llm_checkpoint(nn.Linear(4, 4), "llama", export_dir=tmp_path)

    deprecations = [w for w in caught if issubclass(w.category, DeprecationWarning)]
    assert len(deprecations) == 1, f"expected 1 DeprecationWarning, got {len(deprecations)}"
    assert "export_tensorrt_llm_checkpoint" in str(deprecations[0].message)
