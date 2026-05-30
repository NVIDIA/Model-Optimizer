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

"""Unit tests for ``modelopt.torch.fastgen.plugins.qwen_image``.

Ports the checklist §1 bullets (pack/unpack + adapter parity + _call_model wiring)
into pytest form so they live in-repo and run under ``pytest tests/unit/torch/fastgen``.
Adds the §6-specific bullet ``num_train_timesteps != None`` constructor error.

Parity comparisons against the AutoModel adapter and the FastGen reference
extract are bit-exact (``torch.equal``) — both are pure permute+reshape
operations with no floating-point arithmetic.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch
from torch import nn

from modelopt.torch.fastgen import DMDConfig
from modelopt.torch.fastgen.plugins.qwen_image import (
    QwenImageDMDPipeline,
    build_img_shapes,
    pack_latents,
    unpack_latents,
)

# ---------------------------------------------------------------------------- #
# §1.1 — pack / unpack inverse for representative latent sizes                 #
# ---------------------------------------------------------------------------- #


@pytest.mark.parametrize(
    "shape",
    [
        (1, 16, 16, 16),
        (2, 16, 32, 32),
        (1, 16, 64, 64),
        (1, 16, 128, 128),
    ],
)
def test_pack_unpack_inverse(shape):
    """Round-trip is bit-exact; dtype/device/contiguity preserved."""
    x = torch.randn(*shape)
    p = pack_latents(x)
    y = unpack_latents(p, shape[2], shape[3])
    assert torch.equal(x, y)
    assert x.dtype == p.dtype == y.dtype
    assert x.device == p.device == y.device
    assert p.is_contiguous()
    assert y.is_contiguous()


def test_pack_unpack_inverse_bf16():
    """bf16 dtype is preserved through pack/unpack (spot check)."""
    x = torch.randn(2, 16, 32, 32, dtype=torch.bfloat16)
    p = pack_latents(x)
    y = unpack_latents(p, 32, 32)
    assert p.dtype == torch.bfloat16
    assert y.dtype == torch.bfloat16
    assert torch.equal(x, y)


# ---------------------------------------------------------------------------- #
# §1.2 — odd spatial dims raise a clear ValueError                             #
# ---------------------------------------------------------------------------- #


@pytest.mark.parametrize("shape", [(1, 16, 31, 32), (1, 16, 32, 31)])
def test_pack_rejects_odd_spatial(shape):
    with pytest.raises(ValueError, match="even"):
        pack_latents(torch.randn(*shape))


@pytest.mark.parametrize("hw", [(31, 32), (32, 31)])
def test_unpack_rejects_odd_target(hw):
    h, w = hw
    with pytest.raises(ValueError, match="even"):
        unpack_latents(torch.randn(1, 256, 64), h, w)


# ---------------------------------------------------------------------------- #
# §1.3 / §1.4 — parity vs AutoModel ``QwenImageAdapter``                        #
#                                                                               #
# Soft-skipped if AutoModel isn't on PYTHONPATH (e.g. CPU-only env without it). #
# ---------------------------------------------------------------------------- #


def test_pack_parity_vs_automodel():
    adapters = pytest.importorskip(
        "nemo_automodel.components.flow_matching.adapters.qwen_image",
    )
    adapter = adapters.QwenImageAdapter()
    x = torch.randn(2, 16, 32, 32)
    am = adapter._pack_latents(x)
    mo = pack_latents(x)
    assert torch.equal(am, mo)


def test_unpack_parity_vs_automodel():
    """AutoModel's ``_unpack_latents`` takes pixel dims (H*8, W*8); modelopt takes latent dims.

    Both produce the same tensor for even latents — this test calls them with
    matched API conventions and asserts bit-exact equality.
    """
    adapters = pytest.importorskip(
        "nemo_automodel.components.flow_matching.adapters.qwen_image",
    )
    adapter = adapters.QwenImageAdapter()
    h_lat, w_lat = 32, 32
    x = torch.randn(2, 16, h_lat, w_lat)
    p = adapter._pack_latents(x)
    am_un = adapter._unpack_latents(p, h_lat * 8, w_lat * 8, vae_scale_factor=8)
    mo_un = unpack_latents(p, h_lat, w_lat)
    assert torch.equal(am_un, mo_un)


# ---------------------------------------------------------------------------- #
# §1.5 — parity vs the FastGen reference                                       #
#                                                                               #
# FastGen's QwenImage class pulls heavy deps; we inline the two methods         #
# verbatim from ``source/FastGen/fastgen/networks/QwenImage/network.py:527-560`` #
# so the parity check is hermetic.                                              #
# ---------------------------------------------------------------------------- #


def _fastgen_pack(latents: torch.Tensor) -> torch.Tensor:
    batch_size, channels, height, width = latents.shape
    latents = latents.view(batch_size, channels, height // 2, 2, width // 2, 2)
    latents = latents.permute(0, 2, 4, 1, 3, 5)
    return latents.reshape(batch_size, (height // 2) * (width // 2), channels * 4)


def _fastgen_unpack(latents: torch.Tensor, height: int, width: int) -> torch.Tensor:
    batch_size = latents.shape[0]
    channels = latents.shape[2] // 4
    latents = latents.reshape(batch_size, height // 2, width // 2, channels, 2, 2)
    latents = latents.permute(0, 3, 1, 4, 2, 5)
    return latents.reshape(batch_size, channels, height, width)


def test_pack_unpack_parity_vs_fastgen():
    x = torch.randn(2, 16, 32, 32)
    fg_p = _fastgen_pack(x)
    mo_p = pack_latents(x)
    assert torch.equal(fg_p, mo_p)

    fg_u = _fastgen_unpack(fg_p, 32, 32)
    mo_u = unpack_latents(mo_p, 32, 32)
    assert torch.equal(fg_u, mo_u)
    assert torch.equal(fg_u, x)  # FastGen unpack round-trips the input


# ---------------------------------------------------------------------------- #
# §1.6 — build_img_shapes structural equality + aliasing                       #
# ---------------------------------------------------------------------------- #


def test_build_img_shapes_structure():
    out = build_img_shapes(batch_size=2, h_lat=32, w_lat=32)
    assert out == [[(1, 16, 16)], [(1, 16, 16)]]


def test_build_img_shapes_inner_list_aliased():
    """Implementation uses ``[[...]] * B`` so the inner list is shared across
    batch entries. Safe as long as no caller mutates the nested list/tuple.
    This test documents the aliasing — if a future change makes the lists
    independent, the assertion will fail and this test should be updated.
    """
    out = build_img_shapes(batch_size=2, h_lat=32, w_lat=32)
    assert out[0] is out[1]


# ---------------------------------------------------------------------------- #
# §1.7 / §1.8 — _call_model kwarg forwarding + unpack return styles            #
# ---------------------------------------------------------------------------- #


class _CapturingModel(nn.Module):
    """Stub transformer that records the kwargs it was called with and emits
    a zero tensor of the requested shape in one of three return styles."""

    def __init__(self, out_shape: tuple[int, ...], style: str = "tensor") -> None:
        super().__init__()
        self.out_shape = out_shape
        self.style = style
        self.last_kwargs: dict[str, object] = {}

    def forward(self, **kwargs):
        self.last_kwargs = dict(kwargs)
        out = torch.zeros(*self.out_shape, dtype=kwargs["hidden_states"].dtype)
        if self.style == "tensor":
            return out
        if self.style == "tuple":
            return (out, "extra")
        if self.style == "sample":
            return SimpleNamespace(sample=out)
        raise ValueError(self.style)


def _make_pipeline(student: nn.Module) -> QwenImageDMDPipeline:
    return QwenImageDMDPipeline(
        student=student,
        teacher=nn.Identity(),
        fake_score=nn.Identity(),
        config=DMDConfig(num_train_timesteps=None),
        discriminator=None,
    )


def test_call_model_forwards_qwen_kwargs():
    """``_call_model`` must forward the exact Qwen signature (hidden_states packed
    to ``[B, num_patches, 64]``, encoder_hidden_states verbatim,
    encoder_hidden_states_mask verbatim, txt_seq_lens derived from the mask,
    img_shapes as ``[[(1, h//2, w//2)]] * B``, guidance=None, return_dict=False,
    timestep verbatim with no /1000 rescale)."""
    b, c, h, w = 2, 16, 32, 32
    student = _CapturingModel(out_shape=(b, (h // 2) * (w // 2), c * 4), style="tensor")
    pipe = _make_pipeline(student)

    hidden = torch.randn(b, c, h, w)
    t = torch.full((b,), 0.5, dtype=torch.float32)
    enc = torch.randn(b, 512, 3584)
    mask = torch.zeros(b, 512, dtype=torch.long)
    mask[0, :37] = 1
    mask[1, :42] = 1

    out = pipe._call_model(
        student,
        hidden,
        t,
        encoder_hidden_states=enc,
        encoder_hidden_states_mask=mask,
    )

    kw = student.last_kwargs
    assert tuple(kw["hidden_states"].shape) == (b, (h // 2) * (w // 2), c * 4)
    assert tuple(kw["encoder_hidden_states"].shape) == (b, 512, 3584)
    assert torch.equal(kw["encoder_hidden_states_mask"], mask)
    assert kw["txt_seq_lens"] == [37, 42]
    assert kw["img_shapes"] == [[(1, h // 2, w // 2)]] * b
    assert kw["guidance"] is None
    assert kw["return_dict"] is False
    assert torch.equal(kw["timestep"], t)  # no /1000 rescale when num_train_timesteps=None
    assert tuple(out.shape) == (b, c, h, w)


@pytest.mark.parametrize("style", ["tensor", "tuple", "sample"])
def test_call_model_unpacks_return_styles(style):
    """``_call_model`` must unpack ``tensor`` / ``tuple`` / ``.sample`` return
    styles into ``[B, C, H, W]`` of the input's dtype."""
    b, c, h, w = 1, 16, 32, 32
    model = _CapturingModel(out_shape=(b, (h // 2) * (w // 2), c * 4), style=style)
    pipe = _make_pipeline(model)
    hidden = torch.randn(b, c, h, w)
    t = torch.full((b,), 0.5, dtype=torch.float32)
    enc = torch.randn(b, 512, 3584)
    out = pipe._call_model(model, hidden, t, encoder_hidden_states=enc)
    assert tuple(out.shape) == (b, c, h, w)
    assert out.dtype == hidden.dtype


# ---------------------------------------------------------------------------- #
# §6.X — QwenImageDMDPipeline constructor rejects num_train_timesteps != None  #
# ---------------------------------------------------------------------------- #


def test_constructor_rejects_non_null_num_train_timesteps():
    """The pipeline normalizes ``t ∈ [0, 1]`` internally and forwards continuous
    ``t`` to the Qwen transformer. ``num_train_timesteps`` is a discretization
    knob that doesn't apply — the constructor must refuse it loudly."""
    cfg = DMDConfig(num_train_timesteps=1000)
    with pytest.raises(ValueError, match="num_train_timesteps"):
        QwenImageDMDPipeline(
            student=nn.Identity(),
            teacher=nn.Identity(),
            fake_score=nn.Identity(),
            config=cfg,
            discriminator=None,
        )
