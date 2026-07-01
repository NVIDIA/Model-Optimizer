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

"""Unit tests for the self-implemented final norm and the pre-norm re-application helper."""

import pytest
import torch

pytest.importorskip("transformers")

from modelopt.torch.speculative.plugins.modeling_final_norm import (
    _FinalRMSNorm,
    _maybe_apply_base_final_norm,
    _select_final_norm_type,
)

_HIDDEN_SIZE = 16


@pytest.mark.parametrize(
    ("model_type", "expected"),
    [
        ("llama", "rmsnorm"),
        ("qwen3", "rmsnorm"),
        ("deepseek_v3", "rmsnorm"),
        ("kimi_k25", "rmsnorm"),
        # gpt_oss is intentionally DISABLED: its RMSNorm forward/weight dtype differs from the
        # Llama variant we reuse, so it must not resolve to a norm type until a matching class
        # is added.
        ("gpt_oss", None),
        ("gemma", None),  # unlisted model
        ("", None),
        (None, None),
    ],
)
def test_select_final_norm_type(model_type, expected):
    assert _select_final_norm_type(model_type) == expected


def test_final_rmsnorm_dtype_and_shape():
    """_FinalRMSNorm builds its weight in the requested dtype and preserves input dtype/shape."""
    norm = _FinalRMSNorm(_HIDDEN_SIZE, eps=1e-6, dtype=torch.bfloat16)
    assert norm.weight.dtype == torch.bfloat16
    x = torch.randn(2, 4, _HIDDEN_SIZE, dtype=torch.bfloat16)
    out = norm(x)
    assert out.dtype == torch.bfloat16
    assert out.shape == x.shape


def test_maybe_apply_norm_postnorm_is_noop():
    """base_hidden_prenorm falsy or absent -> hidden returned unchanged, norm never touched."""
    hidden = torch.randn(2, 4, _HIDDEN_SIZE)
    # Missing key.
    assert _maybe_apply_base_final_norm(hidden, {}, None) is hidden
    # Explicit False, even when a norm is available.
    norm = _FinalRMSNorm(_HIDDEN_SIZE, dtype=torch.float32)
    assert _maybe_apply_base_final_norm(hidden, {"base_hidden_prenorm": False}, norm) is hidden


def test_maybe_apply_norm_prenorm_applies():
    """base_hidden_prenorm=True with a norm available -> the norm is applied."""
    norm = _FinalRMSNorm(_HIDDEN_SIZE, dtype=torch.float32)
    hidden = torch.randn(2, 4, _HIDDEN_SIZE)
    out = _maybe_apply_base_final_norm(hidden, {"base_hidden_prenorm": True}, norm)
    assert out.shape == hidden.shape
    torch.testing.assert_close(out, norm(hidden))


def test_maybe_apply_norm_prenorm_without_norm_raises():
    """base_hidden_prenorm=True but no norm located -> fail loud, never silently skip."""
    hidden = torch.randn(2, 4, _HIDDEN_SIZE)
    with pytest.raises(RuntimeError, match="_FINAL_NORM_TYPE_BY_MODEL_TYPE"):
        _maybe_apply_base_final_norm(hidden, {"base_hidden_prenorm": True}, None)
