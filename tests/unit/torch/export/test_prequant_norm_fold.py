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

"""Regression tests for zero-centered gamma handling in ``fuse_prequant_layernorm``.

A norm whose forward computes ``x * (1 + weight)`` (zero-centered gamma) must have its
``pre_quant_scale`` folded as ``(1 + weight) * scale - 1``.  Folding it with the plain
``weight * scale`` formula rewrites the norm so that every consumer receives activations
that no stored weight was scaled for, silently destroying the checkpoint.  Measured on
MiMo-V2.6-Distill-Qwen-9B (Qwen3.5 hybrid, INT4_AWQ): PPL 1,241,095 with the plain formula
vs 10.28 bf16, 11.34 with the formula fixed.

Detection is by class name (plus an explicit ``zero_centered_gamma`` attribute), so the
tests (a) check every allow-listed class name against the forward formula that name claims
and (b) check the same names against the real ``transformers`` implementations when the
installed version ships them.
"""

import pytest
import torch

from modelopt.torch.export.quant_utils import (
    ZERO_CENTERED_NORM_CLASS_EXCLUSIONS,
    ZERO_CENTERED_NORM_CLASS_NAMES,
    _layernorm_uses_weight_plus_one,
    fuse_prequant_layernorm,
)

DIM = 8


def _make_norm_class(class_name: str, weight_plus_one: bool):
    """A synthetic norm whose forward matches the convention ``class_name`` claims."""

    class _SyntheticNorm(torch.nn.Module):
        def __init__(self, dim: int = DIM):
            super().__init__()
            self.weight = torch.nn.Parameter(torch.randn(dim))

        def forward(self, x):
            if weight_plus_one:
                return x * (1.0 + self.weight)
            return x * self.weight

    return type(class_name, (_SyntheticNorm,), {})


def _forward_uses_weight_plus_one(norm) -> bool:
    """Probe ``norm``'s forward: zero-centered gamma or plain ``x * weight``?"""
    with torch.no_grad():
        norm.weight.copy_(torch.randn(norm.weight.shape))
        x = torch.randn(3, norm.weight.shape[0])
        out = norm(x)
        plus_one = torch.allclose(out, x * (1.0 + norm.weight), rtol=1e-5, atol=1e-6)
        plain = torch.allclose(out, x * norm.weight, rtol=1e-5, atol=1e-6)
    assert plus_one != plain, "synthetic norm does not discriminate the two formulas"
    return plus_one


def _linear_with_pre_quant_scale(scale: torch.Tensor, dim: int = DIM):
    from modelopt.torch.quantization.nn import TensorQuantizer

    linear = torch.nn.Linear(dim, dim, bias=False)
    linear.input_quantizer = TensorQuantizer()
    linear.input_quantizer._pre_quant_scale = scale.clone()
    return linear


@pytest.mark.parametrize("class_name", sorted(ZERO_CENTERED_NORM_CLASS_NAMES))
def test_detector_accepts_zero_centered_names(class_name):
    """Every allow-listed name must describe a norm whose forward computes ``1 + weight``."""
    norm = _make_norm_class(class_name, weight_plus_one=True)()

    assert _forward_uses_weight_plus_one(norm)
    assert _layernorm_uses_weight_plus_one(norm)


@pytest.mark.parametrize("class_name", sorted(ZERO_CENTERED_NORM_CLASS_EXCLUSIONS))
def test_detector_rejects_lookalike_plain_norms(class_name):
    """Names containing an allow-listed name but using ``x * weight`` must be rejected."""
    norm = _make_norm_class(class_name, weight_plus_one=False)()

    assert not _forward_uses_weight_plus_one(norm)
    assert not _layernorm_uses_weight_plus_one(norm)


def test_detector_rejects_unrelated_norms_and_honors_explicit_flag():
    """Plain norms are rejected; an explicit ``zero_centered_gamma`` attribute wins either way."""
    plain = _make_norm_class("LlamaRMSNorm", weight_plus_one=False)()
    assert not _layernorm_uses_weight_plus_one(plain)

    plain.zero_centered_gamma = True
    assert _layernorm_uses_weight_plus_one(plain)

    centered = _make_norm_class("Qwen3_5RMSNorm", weight_plus_one=True)()
    centered.zero_centered_gamma = False
    assert not _layernorm_uses_weight_plus_one(centered)


@pytest.mark.parametrize(
    ("class_name", "weight_plus_one"),
    [("Qwen3_5RMSNorm", True), ("LlamaRMSNorm", False)],
)
def test_fuse_prequant_layernorm_inverts_the_norm_formula(class_name, weight_plus_one):
    """The folded weight must invert the norm's own forward formula, not a fixed one."""
    norm = _make_norm_class(class_name, weight_plus_one)()
    scale = torch.linspace(0.5, 2.0, DIM)
    modules = [_linear_with_pre_quant_scale(scale) for _ in range(2)]
    original = norm.weight.detach().clone()

    fuse_prequant_layernorm(norm, modules)

    expected = (original + 1.0) * scale - 1.0 if weight_plus_one else original * scale
    assert torch.allclose(norm.weight, expected, rtol=1e-6, atol=1e-6)
    for module in modules:
        assert not hasattr(module.input_quantizer, "_pre_quant_scale")
        assert module.fused_with_prequant


def _find_transformers_class(class_name: str):
    """Return the installed transformers class with this name, loading model modules lazily."""
    pytest.importorskip("transformers")

    def _search():
        import sys

        for module in list(sys.modules.values()):
            obj = getattr(module, class_name, None)
            if isinstance(obj, type) and issubclass(obj, torch.nn.Module):
                return obj
        return None

    found = _search()
    if found is not None:
        return found
    for model_name in ("qwen3_5", "qwen3_next", "gemma3", "gemma", "gemma2"):
        try:
            __import__(f"transformers.models.{model_name}.modeling_{model_name}")
        except Exception:
            continue
        found = _search()
        if found is not None:
            return found
    return None


def _call_norm(norm, x, gate):
    """Call a real norm class: some take a gate tensor, and they raise on ``gate=None``."""
    try:
        return norm(x, gate)
    except (TypeError, AttributeError):
        pass
    try:
        return norm(x)
    except (TypeError, AttributeError):
        pytest.skip(f"cannot call {type(norm).__name__} with generic arguments")


@pytest.mark.parametrize(
    ("class_name", "weight_plus_one"),
    [
        ("Qwen3_5RMSNorm", True),
        ("Qwen3_5RMSNormGated", False),
        ("Qwen3NextRMSNormGated", False),
        ("Gemma3RMSNorm", True),
    ],
)
def test_fold_contract_holds_for_installed_transformers(class_name, weight_plus_one):
    """Folding the real implementation must be transparent: ``out`` becomes ``out * scale``.

    This validates detection and formula together, without assuming how the norm normalizes
    its input: whatever ``forward`` computes, folding a pre_quant_scale has to leave the
    product of the two unchanged.
    """
    cls = _find_transformers_class(class_name)
    if cls is None:
        pytest.skip(f"{class_name} not present in the installed transformers")

    try:
        norm = cls(DIM)
    except TypeError:
        pytest.skip(f"cannot instantiate {class_name} with generic arguments")

    assert _layernorm_uses_weight_plus_one(norm) is weight_plus_one

    x, gate = torch.randn(3, DIM), torch.randn(3, DIM)
    with torch.no_grad():
        before = _call_norm(norm, x, gate)

    scale = torch.linspace(0.5, 2.0, DIM)
    fuse_prequant_layernorm(norm, [_linear_with_pre_quant_scale(scale) for _ in range(2)])

    with torch.no_grad():
        after = _call_norm(norm, x, gate)

    assert torch.allclose(after, before * scale, rtol=1e-4, atol=1e-4)
