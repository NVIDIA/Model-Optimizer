# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Qwen-Image plumbing for the DMD2 pipeline.

Qwen-Image's ``QwenImageTransformer2DModel`` does not accept the diffusers-standard
``(hidden_states[B,C,H,W], timestep, encoder_hidden_states)`` triple. Instead it expects
*packed* latents ``[B, (H//2)*(W//2), C*4]`` plus three extra kwargs
(``encoder_hidden_states_mask``, ``img_shapes``, ``guidance``) and returns its prediction
in the same packed layout. The packing / unpacking step mirrors
``QwenImagePipeline._pack_latents`` in diffusers.

DMD2's internal math (noise injection, VSD / DSM losses, EMA updates, fake-score updates)
all operates on the *unpacked* latent ``[B, C, H, W]``, so we keep that as the
:class:`DMDPipeline` external contract and push the pack / call / unpack triple into a
single override of :meth:`DMDPipeline._call_model` on :class:`QwenImageDMDPipeline`.

Usage from a training recipe::

    from modelopt.torch.fastgen.plugins.qwen_image import QwenImageDMDPipeline

    pipeline = QwenImageDMDPipeline(
        student=student_transformer,
        teacher=teacher_transformer,
        fake_score=fake_score_transformer,
        config=dmd_config,
        discriminator=None,
    )

The companion ``modelopt_recipes/general/distillation/dmd2_qwen_image.yaml`` must keep
``num_train_timesteps: null`` so the continuous RF time ``t ∈ [0, 1]`` is forwarded
verbatim to the transformer (Qwen-Image normalises timesteps to ``[0, 1]`` internally;
the diffusers ``[0, 1000]`` scale used for Wan / SD3 / Flux does NOT apply here).
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import torch
from torch import nn

from ..methods.dmd import DMDPipeline

if TYPE_CHECKING:
    from ..config import DMDConfig

__all__ = [
    "QwenImageDMDPipeline",
    "attach_feature_capture",
    "build_img_shapes",
    "pack_latents",
    "remove_feature_capture",
    "unpack_latents",
]


# ---------------------------------------------------------------------------- #
#  Latent pack / unpack helpers (2x2 patch grouping)                           #
# ---------------------------------------------------------------------------- #


def pack_latents(latents: torch.Tensor) -> torch.Tensor:
    """Pack ``[B, C, H, W]`` latents into ``[B, (H//2)*(W//2), C*4]`` for Qwen-Image.

    Mirrors ``QwenImagePipeline._pack_latents`` (diffusers): groups every ``2x2`` spatial
    block on each channel and lays the four values out along the channel axis so the
    transformer's ``in_channels = 4 * out_channels`` patch embedding sees them as a
    single token. ``H`` and ``W`` must both be even.
    """
    if latents.ndim != 4:
        raise ValueError(
            f"pack_latents expects [B, C, H, W] (got {latents.ndim}D tensor of shape "
            f"{tuple(latents.shape)})."
        )
    b, c, h, w = latents.shape
    if h % 2 or w % 2:
        raise ValueError(
            f"pack_latents requires even spatial dims, got H={h}, W={w}. "
            "Increase the latent resolution or pad before packing."
        )
    x = latents.view(b, c, h // 2, 2, w // 2, 2)
    x = x.permute(0, 2, 4, 1, 3, 5)
    x = x.reshape(b, (h // 2) * (w // 2), c * 4)
    return x


def unpack_latents(packed: torch.Tensor, height: int, width: int) -> torch.Tensor:
    """Inverse of :func:`pack_latents`. ``height`` / ``width`` are the unpacked latent dims."""
    if packed.ndim != 3:
        raise ValueError(
            f"unpack_latents expects [B, num_patches, C*4] (got {packed.ndim}D tensor of "
            f"shape {tuple(packed.shape)})."
        )
    b, num_patches, c4 = packed.shape
    if c4 % 4:
        raise ValueError(
            f"unpack_latents expects last dim divisible by 4, got {c4}."
        )
    c = c4 // 4
    if height % 2 or width % 2:
        raise ValueError(
            f"unpack_latents requires even target spatial dims, got H={height}, W={width}."
        )
    h2, w2 = height // 2, width // 2
    if num_patches != h2 * w2:
        raise ValueError(
            f"num_patches ({num_patches}) does not match H//2 * W//2 ({h2 * w2}) for "
            f"target shape H={height}, W={width}."
        )
    x = packed.view(b, h2, w2, c, 2, 2)
    x = x.permute(0, 3, 1, 4, 2, 5)
    x = x.reshape(b, c, height, width)
    return x


def build_img_shapes(batch_size: int, h_lat: int, w_lat: int) -> list[list[tuple[int, int, int]]]:
    """Build the ``img_shapes`` kwarg expected by ``QwenImageTransformer2DModel``.

    Each entry is ``[(1, h_lat // 2, w_lat // 2)]`` — one tuple per sample in the batch.
    The leading ``1`` is the temporal dim (single frame for T2I).
    """
    if h_lat % 2 or w_lat % 2:
        raise ValueError(
            f"build_img_shapes requires even latent dims, got h_lat={h_lat}, w_lat={w_lat}."
        )
    return [[(1, h_lat // 2, w_lat // 2)]] * batch_size


# ---------------------------------------------------------------------------- #
#  Pipeline subclass                                                           #
# ---------------------------------------------------------------------------- #


class QwenImageDMDPipeline(DMDPipeline):
    """DMD2 pipeline that targets Qwen-Image's packed transformer signature.

    Drops in for :class:`DMDPipeline` and overrides :meth:`_call_model` only. All other
    behaviour (noise schedules, VSD / DSM losses, EMA, GAN paths) is inherited unchanged.

    The student / teacher / fake_score modules must be the raw diffusers
    ``QwenImageTransformer2DModel`` (or FSDP-sharded copy thereof). The pipeline handles
    pack / call / unpack on every internal forward.

    Args:
        guidance: Optional scalar guidance value forwarded to the transformer's
            ``guidance`` kwarg every call. Only used when the transformer was built with
            ``guidance_embeds=true`` (off by default for ``Qwen/Qwen-Image``). Leave
            ``None`` to skip the guidance embedding entirely — this is independent of
            :attr:`DMDConfig.guidance_scale`, which controls classifier-free guidance on
            the teacher.
    """

    def __init__(
        self,
        student: nn.Module,
        teacher: nn.Module,
        fake_score: nn.Module,
        config: DMDConfig,
        *,
        discriminator: nn.Module | None = None,
        guidance: float | None = None,
    ) -> None:
        super().__init__(
            student=student,
            teacher=teacher,
            fake_score=fake_score,
            config=config,
            discriminator=discriminator,
        )
        if config.num_train_timesteps is not None:
            raise ValueError(
                "QwenImageDMDPipeline requires DMDConfig.num_train_timesteps=None — "
                f"Qwen-Image normalises timesteps to [0, 1] internally (got "
                f"num_train_timesteps={config.num_train_timesteps})."
            )
        self._guidance_value = guidance

    def _call_model(
        self,
        model: nn.Module,
        hidden_states: torch.Tensor,
        timestep: torch.Tensor,
        encoder_hidden_states: torch.Tensor | None = None,
        **model_kwargs: Any,
    ) -> torch.Tensor:
        """Pack [B, C, H, W] -> packed -> call transformer -> unpack -> [B, C, H, W]."""
        if hidden_states.ndim != 4:
            raise ValueError(
                f"QwenImageDMDPipeline._call_model expects 4D hidden_states "
                f"[B, C, H, W] (got {hidden_states.ndim}D)."
            )
        b, _c, h, w = hidden_states.shape

        packed = pack_latents(hidden_states)
        img_shapes = build_img_shapes(b, h, w)

        call_kwargs: dict[str, Any] = dict(model_kwargs)
        call_kwargs.pop("hidden_states", None)
        call_kwargs.pop("encoder_hidden_states_mask", None)
        call_kwargs.pop("img_shapes", None)
        call_kwargs.pop("guidance", None)
        call_kwargs.pop("return_dict", None)

        guidance = None
        if self._guidance_value is not None:
            guidance = torch.full(
                (b,),
                float(self._guidance_value),
                device=hidden_states.device,
                dtype=hidden_states.dtype,
            )

        out = model(
            hidden_states=packed,
            timestep=timestep,
            encoder_hidden_states=encoder_hidden_states,
            encoder_hidden_states_mask=None,
            img_shapes=img_shapes,
            guidance=guidance,
            return_dict=False,
            **call_kwargs,
        )

        if isinstance(out, tuple):
            raw_packed = out[0]
        elif isinstance(out, torch.Tensor):
            raw_packed = out
        elif hasattr(out, "sample"):
            raw_packed = out.sample
        else:
            raise TypeError(
                "QwenImageDMDPipeline._call_model could not extract a tensor from output "
                f"of type {type(out).__name__!r}."
            )

        return unpack_latents(raw_packed, h, w)


# ---------------------------------------------------------------------------- #
#  GAN feature capture                                                          #
# ---------------------------------------------------------------------------- #

# Attribute names match :mod:`modelopt.torch.fastgen.plugins.wan22` so the shared
# :func:`~modelopt.torch.fastgen.methods.dmd._drain_if_hooked` /
# :func:`~modelopt.torch.fastgen.methods.dmd._require_hooked` helpers work
# without modification.
_CAPTURED_ATTR = "_fastgen_captured"
_HANDLES_ATTR = "_fastgen_capture_handles"
_INDICES_ATTR = "_fastgen_capture_indices"
_SHAPE_ATTR = "_fastgen_capture_shape"


def attach_feature_capture(
    teacher: nn.Module,
    feature_indices: list[int],
    h_lat: int,
    w_lat: int,
    *,
    blocks_attr: str = "transformer_blocks",
) -> None:
    """Install forward hooks on ``teacher.transformer_blocks[i]`` for each ``i`` in ``feature_indices``.

    Qwen-Image's ``QwenImageTransformerBlock.forward`` returns
    ``(encoder_hidden_states, hidden_states)`` where ``hidden_states`` has shape
    ``[B, num_image_patches, 3072]`` with ``num_image_patches == (h_lat // 2) * (w_lat // 2)``
    (joint dual-stream attention; the text branch is the first tuple element and
    is discarded). The hook unpacks the image branch and reshapes it to
    ``[B, 3072, h_lat // 2, w_lat // 2]`` so the discriminator can consume it
    as standard NCHW spatial features.

    Captured tensors land in ``teacher._fastgen_captured`` (a list) for the
    DMD2 ``_drain_if_hooked`` / ``_require_hooked`` helpers to pop after each
    teacher forward.

    Args:
        teacher: The teacher transformer module.
        feature_indices: Block indices to capture (e.g. ``[30]`` or
            ``[15, 30, 45]`` for the 60-block Qwen-Image teacher).
        h_lat: Latent height passed to the transformer this step. Must be even.
        w_lat: Latent width passed to the transformer this step. Must be even.
        blocks_attr: Attribute under which the teacher exposes its block stack.
            Default ``"transformer_blocks"`` matches diffusers'
            ``QwenImageTransformer2DModel``.

    Raises:
        AttributeError: ``teacher`` does not expose ``blocks_attr``.
        IndexError: An entry of ``feature_indices`` is out of range.
        ValueError: ``h_lat`` or ``w_lat`` is odd.
    """
    if h_lat % 2 != 0 or w_lat % 2 != 0:
        raise ValueError(
            f"attach_feature_capture requires even latent dims, got h_lat={h_lat}, w_lat={w_lat}."
        )

    remove_feature_capture(teacher)

    blocks = getattr(teacher, blocks_attr, None)
    if blocks is None:
        raise AttributeError(
            f"Teacher {type(teacher).__name__!r} does not expose a ``{blocks_attr}`` attribute; "
            f"pass blocks_attr=... if the block stack is named differently."
        )
    try:
        num_blocks = len(blocks)
    except TypeError as exc:
        raise TypeError(
            f"Teacher ``{blocks_attr}`` is not a sequence (got {type(blocks).__name__!r})."
        ) from exc

    sorted_indices = sorted(set(feature_indices))
    for idx in sorted_indices:
        if not (0 <= idx < num_blocks):
            raise IndexError(
                f"feature_indices entry {idx} is out of range for teacher with {num_blocks} blocks."
            )

    captured: list[torch.Tensor] = []
    setattr(teacher, _CAPTURED_ATTR, captured)
    setattr(teacher, _INDICES_ATTR, list(sorted_indices))
    setattr(teacher, _SHAPE_ATTR, (h_lat // 2, w_lat // 2))

    handles: list[Any] = []
    h_half = h_lat // 2
    w_half = w_lat // 2
    for idx in sorted_indices:
        block = blocks[idx]

        def _hook(_module: nn.Module, _inputs: Any, output: Any) -> None:
            # Qwen-Image block.forward returns (encoder_hidden_states, hidden_states).
            if isinstance(output, tuple) and len(output) == 2:
                hidden = output[1]
            elif isinstance(output, torch.Tensor):
                hidden = output
            else:
                raise TypeError(
                    f"Unexpected QwenImage block output type {type(output).__name__!r}; "
                    "expected (encoder_hidden_states, hidden_states) tuple or Tensor."
                )
            # hidden: [B, num_image_patches, C] -> [B, C, H_half, W_half].
            b, s, c = hidden.shape
            expected_s = h_half * w_half
            if s != expected_s:
                raise RuntimeError(
                    f"QwenImage feature-capture got hidden_states seq_len={s} but expected "
                    f"{expected_s} = (h_lat // 2) * (w_lat // 2). Did the input resolution "
                    f"drift from the attach_feature_capture-time setting?"
                )
            feat = hidden.permute(0, 2, 1).reshape(b, c, h_half, w_half)
            captured.append(feat)

        handles.append(block.register_forward_hook(_hook))

    setattr(teacher, _HANDLES_ATTR, handles)


def remove_feature_capture(teacher: nn.Module) -> None:
    """Remove previously installed feature-capture hooks (no-op if none are installed)."""
    handles = getattr(teacher, _HANDLES_ATTR, None)
    if handles:
        for h in handles:
            h.remove()
    for attr in (_HANDLES_ATTR, _CAPTURED_ATTR, _INDICES_ATTR, _SHAPE_ATTR):
        if hasattr(teacher, attr):
            try:
                delattr(teacher, attr)
            except AttributeError:
                pass
