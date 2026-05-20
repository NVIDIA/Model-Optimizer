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

"""CPU-only unit tests for the LTX-2 VSA plugin.

The LTX-2 plugin keys off the class names ``LTXModel`` /
``LTXSelfAttention`` (or a structural duck-test on the attention module),
so tests build mock modules with those class names rather than depending
on the third-party ``ltx_core`` package.
"""

import types
import weakref

import pytest

pytest.importorskip("transformers")

import torch
import torch.nn as nn

from modelopt.torch.sparsity.attention_sparsity.plugins.ltx2 import (
    _extract_video_shape_hook,
    _is_ltx2_attention_module,
    _is_ltx2_model,
    _LTX2SparseAttention,
)


def _make_named_module(class_name: str, **attrs) -> nn.Module:
    """Return an nn.Module instance whose class name is ``class_name``."""
    cls = type(class_name, (nn.Module,), {})
    obj = cls()
    nn.Module.__init__(obj)
    for k, v in attrs.items():
        setattr(obj, k, v)
    return obj


def _build_positions(t: int, h: int, w: int) -> torch.Tensor:
    """Return a ``(1, 3, T*H*W)`` positions tensor with the given dim sizes.

    Mirrors the layout that ``Modality.positions`` carries: row 0 holds the
    T-coordinate of each token, row 1 the H-coordinate, row 2 the W-coordinate.
    """
    coords = torch.cartesian_prod(torch.arange(t), torch.arange(h), torch.arange(w))
    return coords.t().unsqueeze(0)  # (1, 3, T*H*W)


class TestLTX2Detection:
    def test_is_ltx2_model_by_class_name(self):
        assert _is_ltx2_model(_make_named_module("LTXModel"))

    def test_is_ltx2_model_by_self_attention_submodule(self):
        wrapper = nn.Module()
        wrapper.attn = _make_named_module("LTXSelfAttention")
        assert _is_ltx2_model(wrapper)

    def test_non_ltx2_model_rejected(self):
        assert not _is_ltx2_model(nn.Linear(8, 8))

    def test_attention_detection_by_class_name(self):
        assert _is_ltx2_attention_module(_make_named_module("LTXSelfAttention"))

    def test_attention_detection_by_structure(self):
        """Falls back to duck-typing when class name doesn't match."""
        attn = _make_named_module(
            "SomeOtherAttention",
            to_q=nn.Linear(4, 4),
            to_k=nn.Linear(4, 4),
            to_v=nn.Linear(4, 4),
            q_norm=nn.Identity(),
            k_norm=nn.Identity(),
            rope_type="default",
        )
        assert _is_ltx2_attention_module(attn)

    def test_attention_detection_rejects_partial_match(self):
        """Missing rope_type / q_norm should fail the structural check."""
        attn = _make_named_module(
            "Attention",
            to_q=nn.Linear(4, 4),
            to_k=nn.Linear(4, 4),
            to_v=nn.Linear(4, 4),
        )
        assert not _is_ltx2_attention_module(attn)


class TestExtractVideoShapeHook:
    def test_extracts_3d_positions(self):
        model = nn.Module()
        modality = types.SimpleNamespace(positions=_build_positions(2, 3, 4))
        _extract_video_shape_hook(model, args=(modality,))
        assert model._vsa_video_shape == (2, 3, 4)

    def test_extracts_4d_positions_taking_start_coord(self):
        model = nn.Module()
        # (B, 3, T, 2) — _extract_video_shape_hook drops the trailing dim.
        positions_3d = _build_positions(2, 2, 2)  # (1, 3, 8)
        positions_4d = positions_3d.unsqueeze(-1).expand(-1, -1, -1, 2).contiguous()
        modality = types.SimpleNamespace(positions=positions_4d)
        _extract_video_shape_hook(model, args=(modality,))
        assert model._vsa_video_shape == (2, 2, 2)

    def test_skips_when_video_is_none(self):
        model = nn.Module()
        _extract_video_shape_hook(model, args=(None,))
        assert not hasattr(model, "_vsa_video_shape")

    def test_skips_when_positions_is_none(self):
        model = nn.Module()
        modality = types.SimpleNamespace(positions=None)
        _extract_video_shape_hook(model, args=(modality,))
        assert not hasattr(model, "_vsa_video_shape")

    def test_skips_when_product_mismatches_seq_len(self):
        """Defensive guard: if unique-counts don't multiply to seq_len, bail."""
        model = nn.Module()
        # Both T-dim and H-dim share the same single value, so unique counts
        # collapse and the product no longer equals seq_len.
        positions = torch.zeros(1, 3, 4, dtype=torch.long)
        positions[0, 2] = torch.arange(4)  # only W varies
        modality = types.SimpleNamespace(positions=positions)
        _extract_video_shape_hook(model, args=(modality,))
        # 1 * 1 * 4 == 4 still matches seq_len, so this should succeed.
        assert model._vsa_video_shape == (1, 1, 4)

    def test_skips_unsupported_ndim(self):
        model = nn.Module()
        modality = types.SimpleNamespace(positions=torch.zeros(1, 3))  # 2D
        _extract_video_shape_hook(model, args=(modality,))
        assert not hasattr(model, "_vsa_video_shape")


class _StubMethod:
    """Tiny stand-in for the ``VSA`` method instance used by _resolve_video_shape."""

    def __init__(self, video_shape=None):
        self.video_shape = video_shape


class TestResolveVideoShape:
    """Cover the resolution order: root_ref -> method.video_shape -> None."""

    def _make_attn(self, *, root=None, method_shape=None):
        attn = _LTX2SparseAttention.__new__(_LTX2SparseAttention)
        nn.Module.__init__(attn)
        if root is not None:
            object.__setattr__(attn, "_vsa_root_model_ref", weakref.ref(root))
        attn._sparse_method_instance = _StubMethod(method_shape)
        return attn

    def test_returns_root_shape_when_set(self):
        root = nn.Module()
        root._vsa_video_shape = (2, 3, 4)
        attn = self._make_attn(root=root, method_shape=(9, 9, 9))
        # root takes priority over method.video_shape when both match seq_len
        assert attn._resolve_video_shape(seq_len=24) == (2, 3, 4)

    def test_falls_back_to_method_shape(self):
        root = nn.Module()  # no _vsa_video_shape attribute
        attn = self._make_attn(root=root, method_shape=(2, 3, 4))
        assert attn._resolve_video_shape(seq_len=24) == (2, 3, 4)

    def test_returns_none_when_seq_len_mismatch(self):
        root = nn.Module()
        root._vsa_video_shape = (2, 3, 4)  # product=24
        attn = self._make_attn(root=root, method_shape=(5, 5, 5))  # product=125
        assert attn._resolve_video_shape(seq_len=99) is None

    def test_returns_none_when_no_source_available(self):
        attn = self._make_attn(root=None, method_shape=None)
        assert attn._resolve_video_shape(seq_len=24) is None

    def test_root_ref_can_be_garbage_collected_safely(self):
        """A dead weakref must not crash resolution — fall through to method."""
        attn = _LTX2SparseAttention.__new__(_LTX2SparseAttention)
        nn.Module.__init__(attn)
        attn._sparse_method_instance = _StubMethod((2, 3, 4))
        # Dead weakref: the ref's target was never alive.
        attn._vsa_root_model_ref = lambda: None
        assert attn._resolve_video_shape(seq_len=24) == (2, 3, 4)
