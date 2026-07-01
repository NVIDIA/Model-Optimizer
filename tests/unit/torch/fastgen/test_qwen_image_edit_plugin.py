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

"""Unit tests for the Qwen-Image-Edit DMD2 target/reference-token wrapper."""

from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch
from torch import nn

from modelopt.torch.fastgen import DMDConfig
from modelopt.torch.fastgen.config import SampleTimestepConfig
from modelopt.torch.fastgen.plugins.qwen_image import (
    attach_feature_capture as attach_t2i_feature_capture,
)
from modelopt.torch.fastgen.plugins.qwen_image import (
    pack_latents,
    remove_feature_capture,
    update_feature_capture_shape,
)
from modelopt.torch.fastgen.plugins.qwen_image_edit import QwenImageEditDMDPipeline
from modelopt.torch.fastgen.plugins.qwen_image_edit import (
    attach_feature_capture as attach_edit_feature_capture,
)


class _CapturingModel(nn.Module):
    """Record Qwen kwargs and return the packed input in a requested output style."""

    def __init__(self, style: str = "tensor") -> None:
        super().__init__()
        self.style = style
        self.last_kwargs: dict[str, object] = {}

    def forward(self, **kwargs):
        self.last_kwargs = dict(kwargs)
        output = kwargs["hidden_states"]
        if self.style == "tensor":
            return output
        if self.style == "tuple":
            return (output,)
        if self.style == "sample":
            return SimpleNamespace(sample=output)
        raise ValueError(self.style)


class _CurrentDiffusersSignatureModel(nn.Module):
    """Approximate current Diffusers Qwen forward, which removed ``txt_seq_lens``."""

    def __init__(self) -> None:
        super().__init__()
        self.called = False

    def forward(
        self,
        hidden_states,
        encoder_hidden_states,
        encoder_hidden_states_mask,
        timestep,
        img_shapes,
        guidance=None,
        attention_kwargs=None,
        return_dict=True,
    ):
        self.called = True
        return hidden_states


class _GenericForwardWrapper(_CurrentDiffusersSignatureModel):
    """Mimic a distributed wrapper that exposes ``**kwargs`` around an explicit model API."""

    def forward(self, *args, **kwargs):
        return super().forward(*args, **kwargs)


def _make_pipeline(student: nn.Module, *, guidance: float | None = None):
    return QwenImageEditDMDPipeline(
        student=student,
        teacher=nn.Identity(),
        fake_score=nn.Identity(),
        config=DMDConfig(num_train_timesteps=None),
        discriminator=None,
        guidance=guidance,
    )


@pytest.mark.parametrize("style", ["tensor", "tuple", "sample"])
def test_call_model_appends_multiple_references_and_crops_target_prefix(style):
    """Packed references follow the target, while only target output tokens are unpacked."""
    b, c, h, w = 2, 4, 8, 10
    target = torch.arange(b * c * h * w, dtype=torch.float32).reshape(b, c, h, w)
    references = [
        torch.full((b, c, 6, 8), 10_000.0),
        torch.full((b, c, 4, 12), 20_000.0),
    ]
    model = _CapturingModel(style)
    pipe = _make_pipeline(model, guidance=1.25)
    timestep = torch.tensor([0.25, 0.75])
    text = torch.randn(b, 7, 16)
    text_mask = torch.tensor([[1, 1, 1, 0, 0, 0, 0], [1, 1, 1, 1, 1, 0, 0]])

    output = pipe._call_model(
        model,
        target,
        timestep,
        encoder_hidden_states=text,
        encoder_hidden_states_mask=text_mask,
        conditioning_latents=references,
    )

    expected_packed = torch.cat(
        [pack_latents(target), *(pack_latents(x) for x in references)], dim=1
    )
    kwargs = model.last_kwargs
    assert torch.equal(kwargs["hidden_states"], expected_packed)
    assert kwargs["img_shapes"] == [
        [(1, 4, 5), (1, 3, 4), (1, 2, 6)],
        [(1, 4, 5), (1, 3, 4), (1, 2, 6)],
    ]
    assert "txt_seq_lens" not in kwargs
    assert torch.equal(kwargs["timestep"], timestep)
    assert torch.equal(kwargs["guidance"], torch.full((b,), 1.25))
    assert kwargs["return_dict"] is False
    assert "conditioning_latents" not in kwargs

    # The capturing model echoes target+reference tokens. Cropping the prefix before
    # unpacking must recover the target bit-exactly rather than trying to unpack all tokens.
    assert torch.equal(output, target)


def test_conditioning_latent_validation_errors_are_actionable():
    b, c, h, w = 1, 4, 8, 8
    target = torch.randn(b, c, h, w)
    text = torch.randn(b, 2, 8)
    timestep = torch.tensor([0.5])
    pipe = _make_pipeline(_CapturingModel())

    def call(conditioning_latents):
        return pipe._call_model(
            pipe.student,
            target,
            timestep,
            encoder_hidden_states=text,
            conditioning_latents=conditioning_latents,
        )

    with pytest.raises(ValueError, match="non-empty.*conditioning_latents"):
        call(None)
    with pytest.raises(ValueError, match="list or tuple"):
        call(torch.randn_like(target))
    with pytest.raises(TypeError, match=r"conditioning_latents\[0\].*Tensor"):
        call(["not-a-tensor"])
    with pytest.raises(ValueError, match=r"conditioning_latents\[0\].*\[B, C, H, W\]"):
        call([torch.randn(c, h, w)])
    with pytest.raises(ValueError, match="batch/channels"):
        call([torch.randn(2, c, h, w)])
    with pytest.raises(ValueError, match="even spatial"):
        call([torch.randn(b, c, h - 1, w)])
    with pytest.raises(ValueError, match="dtype"):
        call([torch.randn(b, c, h, w, dtype=torch.bfloat16)])


def test_current_diffusers_signature_does_not_receive_removed_txt_seq_lens():
    model = _GenericForwardWrapper()
    pipe = _make_pipeline(model)
    target = torch.randn(1, 4, 8, 8)
    reference = torch.randn(1, 4, 8, 8)
    text = torch.randn(1, 3, 8)
    mask = torch.ones(1, 3, dtype=torch.long)

    output = pipe._call_model(
        model,
        target,
        torch.tensor([0.5]),
        encoder_hidden_states=text,
        encoder_hidden_states_mask=mask,
        conditioning_latents=[reference],
    )

    assert model.called
    assert output.shape == target.shape


class _TinyEditTransformer(nn.Module):
    """Grad-capable packed-token transformer used for an end-to-end DMD loss call."""

    def __init__(self, packed_dim: int = 16) -> None:
        super().__init__()
        self.proj = nn.Linear(packed_dim, packed_dim)
        self.seen_token_counts: list[int] = []

    def forward(self, hidden_states, **_kwargs):
        self.seen_token_counts.append(hidden_states.shape[1])
        return self.proj(hidden_states)


def test_shared_dmd_losses_forward_references_to_student_teacher_and_fake_score():
    """All DMD branches receive the fixed reference suffix through ``model_kwargs``."""
    torch.manual_seed(0)
    student = _TinyEditTransformer()
    teacher = _TinyEditTransformer()
    fake_score = _TinyEditTransformer()
    config = DMDConfig(
        pred_type="flow",
        num_train_timesteps=None,
        student_sample_steps=1,
        guidance_scale=None,
        gan_loss_weight_gen=0.0,
        sample_t_cfg=SampleTimestepConfig(time_dist_type="uniform", min_t=0.001, max_t=0.999),
        ema=None,
    )
    pipe = QwenImageEditDMDPipeline(student, teacher, fake_score, config)
    target = torch.randn(1, 4, 8, 8)  # 16 target patches
    noise = torch.randn_like(target)
    reference = torch.randn(1, 4, 4, 8)  # 8 reference patches
    text = torch.randn(1, 3, 8)

    student_losses = pipe.compute_student_loss(
        target,
        noise,
        encoder_hidden_states=text,
        conditioning_latents=[reference],
    )
    assert torch.isfinite(student_losses["total"])
    student_losses["total"].backward()
    assert any(p.grad is not None for p in student.parameters())
    assert student.seen_token_counts == [24]
    assert teacher.seen_token_counts == [24]
    assert fake_score.seen_token_counts == [24]

    fake_score.zero_grad(set_to_none=True)
    fake_losses = pipe.compute_fake_score_loss(
        target,
        noise,
        encoder_hidden_states=text,
        conditioning_latents=(reference,),
    )
    assert torch.isfinite(fake_losses["total"])
    fake_losses["total"].backward()
    assert any(p.grad is not None for p in fake_score.parameters())
    assert student.seen_token_counts[-1] == 24
    assert fake_score.seen_token_counts[-1] == 24


class _TupleBlock(nn.Module):
    def forward(self, hidden_states):
        return torch.empty(0), hidden_states


class _TeacherWithBlocks(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.transformer_blocks = nn.ModuleList([_TupleBlock()])


def test_edit_feature_capture_keeps_target_prefix_and_t2i_remains_strict():
    b, target_h, target_w, channels = 1, 8, 6, 5
    target_tokens = (target_h // 2) * (target_w // 2)
    hidden = torch.arange(b * (target_tokens + 7) * channels, dtype=torch.float32).reshape(
        b, target_tokens + 7, channels
    )

    edit_teacher = _TeacherWithBlocks()
    attach_edit_feature_capture(edit_teacher, [0], target_h, target_w)
    edit_teacher.transformer_blocks[0](hidden)
    captured = edit_teacher._fastgen_captured
    assert len(captured) == 1
    expected = (
        hidden[:, :target_tokens]
        .permute(0, 2, 1)
        .reshape(b, channels, target_h // 2, target_w // 2)
    )
    assert torch.equal(captured[0], expected)

    # The installed hooks must follow later multiresolution batches instead of retaining
    # the base-resolution shape present at hook registration time.
    captured.clear()
    dynamic_h, dynamic_w = 4, 8
    dynamic_tokens = (dynamic_h // 2) * (dynamic_w // 2)
    dynamic_hidden = hidden[:, : dynamic_tokens + 3]
    update_feature_capture_shape(edit_teacher, dynamic_h, dynamic_w)
    edit_teacher.transformer_blocks[0](dynamic_hidden)
    dynamic_expected = (
        dynamic_hidden[:, :dynamic_tokens]
        .permute(0, 2, 1)
        .reshape(b, channels, dynamic_h // 2, dynamic_w // 2)
    )
    assert torch.equal(captured[0], dynamic_expected)
    remove_feature_capture(edit_teacher)

    # The text-to-image helper keeps exact-length validation by default, preventing
    # accidental resolution drift from being silently interpreted as edit references.
    t2i_teacher = _TeacherWithBlocks()
    attach_t2i_feature_capture(t2i_teacher, [0], target_h, target_w)
    with pytest.raises(RuntimeError, match="seq_len"):
        t2i_teacher.transformer_blocks[0](hidden)
    remove_feature_capture(t2i_teacher)
