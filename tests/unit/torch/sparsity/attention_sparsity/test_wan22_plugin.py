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

"""CPU-only unit tests for the Wan 2.2 VSA plugin.

The Wan 2.2 plugin matches by class name ``WanTransformer3DModel``, so tests
build mock modules with that class name to exercise detection, hook
installation, idempotency, and shape-extraction behavior without requiring a
diffusers/Wan install.
"""

import types

import pytest

pytest.importorskip("transformers")

import torch
import torch.nn as nn

from modelopt.torch.sparsity.attention_sparsity.plugins.wan22 import (
    _find_wan22_transformers,
    _get_patch_size,
    _is_wan22_model,
    _make_wan22_video_shape_hook,
    register_wan22_vsa,
)
from modelopt.torch.sparsity.attention_sparsity.sparse_attention import SparseAttentionModule


def _make_wan_transformer(patch_size=(1, 2, 2)) -> nn.Module:
    """Return a mock module whose class name is ``WanTransformer3DModel``.

    A real ``WanTransformer3DModel`` exposes ``config.patch_size``; we
    replicate that on a freshly-created class with the matching name so the
    plugin's name-based detection fires.
    """
    cls = type("WanTransformer3DModel", (nn.Module,), {})
    transformer = cls()
    nn.Module.__init__(transformer)
    if patch_size is not None:
        transformer.config = types.SimpleNamespace(patch_size=patch_size)
    return transformer


def _fake_vsa_method(initial_shape=None):
    """Tiny stand-in for ``VSA`` exposing only the surface the hook touches."""

    class _FakeMethod:
        name = "vsa"

        def __init__(self, initial_shape):
            self.video_shape = initial_shape
            self.set_calls: list[tuple[int, int, int]] = []

        def set_video_shape(self, shape):
            self.video_shape = shape
            self.set_calls.append(shape)

    return _FakeMethod(initial_shape)


def _attach_sparse_module(parent: nn.Module, name: str, method) -> nn.Module:
    """Attach a ``SparseAttentionModule``-typed shell carrying ``method``.

    The hook only checks ``isinstance(sub, SparseAttentionModule)`` and reads
    ``_sparse_method_instance`` — we bypass the registry/conversion machinery
    by allocating an empty instance via ``__new__`` and wiring the attribute
    directly.
    """
    shell = SparseAttentionModule.__new__(SparseAttentionModule)
    nn.Module.__init__(shell)
    shell._sparse_method_instance = method
    parent.add_module(name, shell)
    return shell


class TestWan22Detection:
    def test_is_wan22_model_by_class_name(self):
        transformer = _make_wan_transformer()
        assert _is_wan22_model(transformer)

    def test_is_wan22_model_via_submodule(self):
        wrapper = nn.Module()
        wrapper.transformer = _make_wan_transformer()
        assert _is_wan22_model(wrapper)

    def test_non_wan22_model_rejected(self):
        assert not _is_wan22_model(nn.Linear(8, 8))

    def test_find_returns_every_match(self):
        wrapper = nn.Module()
        wrapper.transformer = _make_wan_transformer()
        wrapper.transformer_2 = _make_wan_transformer()
        found = _find_wan22_transformers(wrapper)
        assert len(found) == 2

    def test_get_patch_size_missing_config(self):
        transformer = _make_wan_transformer(patch_size=None)
        assert _get_patch_size(transformer) is None

    def test_get_patch_size_invalid_value(self):
        transformer = _make_wan_transformer(patch_size=(1, 2))  # too short
        assert _get_patch_size(transformer) is None


class TestWan22HookRegistration:
    def test_register_installs_hook_once(self):
        transformer = _make_wan_transformer()
        n1 = register_wan22_vsa(transformer)
        n2 = register_wan22_vsa(transformer)
        assert n1 == 1
        assert n2 == 0  # idempotent: the second call must not re-register
        assert getattr(transformer, "_vsa_hook_registered", False) is True

    def test_register_skips_non_wan22(self):
        assert register_wan22_vsa(nn.Linear(4, 4)) == 0

    def test_register_handles_pipeline_with_two_transformers(self):
        wrapper = nn.Module()
        wrapper.transformer = _make_wan_transformer()
        wrapper.transformer_2 = _make_wan_transformer()
        assert register_wan22_vsa(wrapper) == 2


class TestWan22HookBehavior:
    def test_hook_extracts_video_shape_from_5d_input(self):
        transformer = _make_wan_transformer(patch_size=(1, 2, 2))
        method = _fake_vsa_method()
        _attach_sparse_module(transformer, "attn", method)

        hook = _make_wan22_video_shape_hook(transformer)
        hidden = torch.zeros(1, 4, 8, 16, 16)  # B, C, T, H, W
        hook(transformer, args=(hidden,), kwargs={})

        assert method.video_shape == (8, 8, 8)
        assert getattr(method, "_wan22_auto_video_shape", False) is True
        assert transformer._vsa_video_shape == (8, 8, 8)

    def test_hook_returns_noop_when_patch_size_missing(self):
        transformer = _make_wan_transformer(patch_size=None)
        method = _fake_vsa_method()
        _attach_sparse_module(transformer, "attn", method)

        hook = _make_wan22_video_shape_hook(transformer)
        hook(transformer, args=(torch.zeros(1, 4, 8, 16, 16),), kwargs={})

        # Inert hook: must not touch the method or stamp _vsa_video_shape.
        assert method.video_shape is None
        assert method.set_calls == []
        assert not hasattr(transformer, "_vsa_video_shape")

    def test_hook_skips_non_5d_input(self):
        transformer = _make_wan_transformer()
        method = _fake_vsa_method()
        _attach_sparse_module(transformer, "attn", method)

        hook = _make_wan22_video_shape_hook(transformer)
        hook(transformer, args=(torch.zeros(1, 8, 16),), kwargs={})

        assert method.set_calls == []

    def test_hook_skips_invalid_shape(self):
        transformer = _make_wan_transformer(patch_size=(4, 4, 4))
        method = _fake_vsa_method()
        _attach_sparse_module(transformer, "attn", method)

        hook = _make_wan22_video_shape_hook(transformer)
        # T=2 < p_t=4 -> derived T-dim collapses to zero.
        hook(transformer, args=(torch.zeros(1, 4, 2, 16, 16),), kwargs={})

        assert method.set_calls == []

    def test_hook_only_touches_vsa_methods(self):
        transformer = _make_wan_transformer()
        vsa_method = _fake_vsa_method()
        non_vsa_method = _fake_vsa_method()
        non_vsa_method.name = "flash_skip_softmax"
        _attach_sparse_module(transformer, "attn_vsa", vsa_method)
        _attach_sparse_module(transformer, "attn_skip", non_vsa_method)

        hook = _make_wan22_video_shape_hook(transformer)
        hook(transformer, args=(torch.zeros(1, 4, 8, 16, 16),), kwargs={})

        assert vsa_method.set_calls == [(8, 8, 8)]
        assert non_vsa_method.set_calls == []

    def test_hook_preserves_user_supplied_video_shape(self):
        """Explicit user shapes must survive every forward pass."""
        transformer = _make_wan_transformer(patch_size=(1, 2, 2))
        user_shape = (4, 4, 4)
        method = _fake_vsa_method(initial_shape=user_shape)
        _attach_sparse_module(transformer, "attn", method)

        hook = _make_wan22_video_shape_hook(transformer)
        hook(transformer, args=(torch.zeros(1, 4, 8, 16, 16),), kwargs={})
        hook(transformer, args=(torch.zeros(1, 4, 8, 16, 16),), kwargs={})

        assert method.video_shape == user_shape
        assert method.set_calls == []
        assert getattr(method, "_wan22_auto_video_shape", False) is False

    def test_hook_refreshes_auto_set_shape_across_calls(self):
        """Hook-owned slots keep updating when input dims change."""
        transformer = _make_wan_transformer(patch_size=(1, 2, 2))
        method = _fake_vsa_method()
        _attach_sparse_module(transformer, "attn", method)

        hook = _make_wan22_video_shape_hook(transformer)
        hook(transformer, args=(torch.zeros(1, 4, 8, 16, 16),), kwargs={})
        hook(transformer, args=(torch.zeros(1, 4, 16, 16, 16),), kwargs={})

        assert method.set_calls == [(8, 8, 8), (16, 8, 8)]
        assert method.video_shape == (16, 8, 8)
