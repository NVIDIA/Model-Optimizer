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

"""Unit tests for final norms and the pre-norm re-application helper."""

import pytest
import safetensors.torch
import torch

pytest.importorskip("transformers")
from transformers import GptOssConfig, GptOssForCausalLM

from modelopt.torch.speculative.plugins.modeling_fakebase import FakeBaseModel
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
        ("kimi_k2", "rmsnorm"),
        ("kimi_k25", "rmsnorm"),
        ("gpt_oss", "gpt_oss_rmsnorm"),
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


@pytest.mark.parametrize("checkpoint_dtype", [torch.bfloat16, torch.float32])
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16, torch.float16])
def test_gpt_oss_fakebase_final_norm_matches_transformers(tmp_path, dtype, checkpoint_dtype):
    config = GptOssConfig(
        hidden_size=_HIDDEN_SIZE,
        vocab_size=32,
        num_hidden_layers=0,
        num_attention_heads=2,
        num_key_value_heads=1,
        head_dim=8,
        dtype=dtype,
    )
    config.save_pretrained(tmp_path)
    generator = torch.Generator().manual_seed(0)
    weight = (torch.linspace(0.8, 1.2, _HIDDEN_SIZE) + 0.00137).to(checkpoint_dtype)
    lm_head = torch.randn(32, _HIDDEN_SIZE, generator=generator).to(checkpoint_dtype)
    safetensors.torch.save_file(
        {
            "model.embed_tokens.weight": torch.zeros_like(lm_head),
            "lm_head.weight": lm_head,
            "model.norm.weight": weight,
        },
        tmp_path / "model.safetensors",
    )
    model = FakeBaseModel.from_source(str(tmp_path))
    reference = GptOssForCausalLM.from_pretrained(tmp_path, dtype=dtype, local_files_only=True)
    assert model.norm.weight.dtype == reference.model.norm.weight.dtype
    torch.testing.assert_close(model.norm.weight, reference.model.norm.weight, rtol=0, atol=0)
    hidden = torch.randn(2, 4, _HIDDEN_SIZE, generator=generator).to(dtype).requires_grad_()
    reference_hidden = hidden.detach().clone().requires_grad_()

    actual = _maybe_apply_base_final_norm(hidden, {"base_hidden_prenorm": True}, model.norm)
    expected = reference.model.norm(reference_hidden)

    assert actual.dtype == dtype
    actual_logits = model.lm_head(actual)
    expected_logits = reference.lm_head(expected)
    torch.testing.assert_close(actual_logits, expected_logits, rtol=0, atol=0)
    actual_logits.float().sum().backward()
    expected_logits.float().sum().backward()
    torch.testing.assert_close(hidden.grad, reference_hidden.grad, rtol=0, atol=0)
    torch.testing.assert_close(
        model.norm.weight.grad, reference.model.norm.weight.grad, rtol=0, atol=0
    )

    resume_path = tmp_path / "resume"
    model.save_pretrained(resume_path)
    restored = FakeBaseModel.from_pretrained(resume_path, dtype="auto", local_files_only=True)
    assert restored.norm.weight.dtype == reference.model.norm.weight.dtype
    restored_logits = restored.lm_head(restored.norm(hidden.detach()))
    torch.testing.assert_close(restored_logits, expected_logits.detach(), rtol=0, atol=0)


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
