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

"""Hermetic Qwen-like tests for the ModelOpt PDD adapter and conversion."""

from __future__ import annotations

import copy
from types import SimpleNamespace

import pytest
import torch
import torch.nn.functional as F
from torch import nn

from modelopt.torch.fastgen import PDDConfig, PDDOutputProjection
from modelopt.torch.fastgen.flow_matching import fusion_coefficients
from modelopt.torch.fastgen.plugins import QwenImagePDDAdapter
from modelopt.torch.fastgen.plugins.qwen_image import build_img_shapes, pack_latents, unpack_latents
from modelopt.torch.fastgen.plugins.qwen_image_pdd import (
    QWEN_IMAGE_PDD_LAYER_SPEC,
    convert_qwen_image_to_pdd,
)


class _TinyQwenTransformer(nn.Module):
    """Qwen-shaped packed transformer with a real registered final linear."""

    def __init__(self, *, packed_channels: int = 4, hidden_width: int = 5) -> None:
        super().__init__()
        self.config = SimpleNamespace(guidance_embeds=False)
        self.backbone = nn.Linear(packed_channels, hidden_width, dtype=torch.bfloat16)
        self.proj_out = nn.Linear(hidden_width, packed_channels, dtype=torch.bfloat16)
        self.calls: list[dict[str, object]] = []

    def forward(
        self,
        *,
        hidden_states,
        timestep,
        encoder_hidden_states,
        encoder_hidden_states_mask,
        img_shapes,
        guidance,
        return_dict,
        **kwargs,
    ):
        condition_value = encoder_hidden_states.mean(dim=(1, 2), keepdim=True)
        condition_value = condition_value + 0.01 * encoder_hidden_states_mask.sum(
            dim=1, keepdim=True
        ).unsqueeze(-1)
        hidden = torch.tanh(self.backbone(hidden_states))
        hidden = hidden + condition_value.to(hidden.dtype)
        hidden = hidden + (0.1 * timestep[:, None, None]).to(hidden.dtype)
        output = self.proj_out(hidden)
        self.calls.append(
            {
                "hidden_states": hidden_states.detach().clone(),
                "timestep": timestep.detach().clone(),
                "encoder_hidden_states": encoder_hidden_states.detach().clone(),
                "encoder_hidden_states_mask": encoder_hidden_states_mask.detach().clone(),
                "img_shapes": img_shapes,
                "guidance": guidance,
                "projection_input": hidden.detach().clone(),
                "return_dict": return_dict,
                "kwargs": kwargs,
                "output": output.detach().clone(),
            }
        )
        return (output,)


def _config(*, guidance_scale: float | None = 4.0, grid_size: int = 4) -> PDDConfig:
    return PDDConfig(
        grid_size=grid_size,
        grid_max_t=0.999,
        flow_shift=5.0,
        block_size_min=1,
        block_size_max=grid_size,
        inference_blocks=[2, 2] if grid_size == 4 else [grid_size],
        student_sample_steps=2 if grid_size == 4 else 1,
        guidance_scale=guidance_scale,
        num_train_timesteps=None,
    )


def _inputs(batch_size: int = 2):
    torch.manual_seed(7)
    state = torch.randn(batch_size, 1, 4, 4)
    time = torch.tensor([0.875, 0.25])[:batch_size]
    embeddings = torch.randn(batch_size, 3, 2, dtype=torch.bfloat16)
    mask = torch.tensor([[1, 1, 1], [1, 0, 0]], dtype=torch.long)[:batch_size]
    embeddings = embeddings * mask.unsqueeze(-1)
    negative_embeddings = torch.randn(batch_size, 3, 2, dtype=torch.bfloat16)
    negative_mask = torch.tensor([[1, 0, 0], [1, 1, 1]], dtype=torch.long)[:batch_size]
    negative_embeddings = negative_embeddings * negative_mask.unsqueeze(-1)
    return state, time, (embeddings, mask), (negative_embeddings, negative_mask)


def _call_base_packed(
    model: _TinyQwenTransformer,
    state: torch.Tensor,
    time: torch.Tensor,
    condition: tuple[torch.Tensor, torch.Tensor],
) -> torch.Tensor:
    embeddings, mask = condition
    return model(
        hidden_states=pack_latents(state).to(torch.bfloat16),
        timestep=time,
        encoder_hidden_states=embeddings.to(torch.bfloat16),
        encoder_hidden_states_mask=mask,
        img_shapes=build_img_shapes(state.shape[0], state.shape[2], state.shape[3]),
        guidance=None,
        return_dict=False,
    )[0]


def test_conversion_is_idempotent_and_every_initialized_head_matches_base() -> None:
    base = _TinyQwenTransformer()
    student = copy.deepcopy(base)
    state, time, condition, _ = _inputs()
    base_packed = _call_base_packed(base, state, time, condition)
    base_velocity = unpack_latents(base_packed, 4, 4)
    config = _config()

    projection = convert_qwen_image_to_pdd(student, config)
    repeated = convert_qwen_image_to_pdd(student, config)
    adapter = QwenImagePDDAdapter(config)
    actual = adapter.student_all_heads(student, state, time, condition=condition)

    assert projection is repeated
    assert student.proj_out is projection
    assert isinstance(projection, PDDOutputProjection)
    assert projection.layer_spec == QWEN_IMAGE_PDD_LAYER_SPEC
    assert projection.layer_spec.projection_path == "transformer.proj_out"
    assert actual.shape == (2, 4, 1, 4, 4)
    torch.testing.assert_close(actual, base_velocity[:, None].expand_as(actual))
    assert len(student.calls) == 1
    torch.testing.assert_close(student.calls[0]["timestep"], time)
    assert student.calls[0]["img_shapes"] == [[(1, 2, 2)], [(1, 2, 2)]]
    assert "txt_seq_lens" not in student.calls[0]["kwargs"]
    assert student.calls[0]["guidance"] is None


def test_unfused_channel_major_output_maps_each_packed_head_in_order() -> None:
    student = _TinyQwenTransformer()
    config = _config()
    projection = convert_qwen_image_to_pdd(student, config)
    with torch.no_grad():
        projection.weight.zero_()
        head_bias = (torch.arange(16, dtype=torch.float32).reshape(4, 4) / 5).to(torch.bfloat16)
        projection.bias.copy_(head_bias.reshape(-1))
    state, time, condition, _ = _inputs(batch_size=1)

    actual = QwenImagePDDAdapter(config).student_all_heads(
        student,
        state,
        time,
        condition=condition,
    )
    expected = torch.stack(
        [
            unpack_latents(
                head_bias[index].reshape(1, 1, 4).expand(1, 4, 4),
                4,
                4,
            )
            for index in range(4)
        ],
        dim=1,
    )

    torch.testing.assert_close(actual, expected)


def test_fused_student_matches_explicit_packed_weight_fusion() -> None:
    student = _TinyQwenTransformer()
    config = _config()
    projection = convert_qwen_image_to_pdd(student, config)
    generator = torch.Generator().manual_seed(91)
    with torch.no_grad():
        projection.weight.copy_(torch.randn(projection.weight.shape, generator=generator) / 3)
        projection.bias.copy_(torch.randn(projection.bias.shape, generator=generator) / 7)
    adapter = QwenImagePDDAdapter(config)
    state, time, condition, _ = _inputs()
    grid = torch.tensor([1.0, 0.85, 0.55, 0.2, 0.0])

    actual = adapter.student_fused_block(
        student,
        state,
        time,
        start=1,
        end=4,
        grid=grid,
        condition=condition,
    )
    coefficients = fusion_coefficients(grid, 1, 4).float()
    head_weights = projection.weight.reshape(4, 4, 5)
    head_bias = projection.bias.reshape(4, 4)
    fused_weight = torch.einsum("n,n...->...", coefficients, head_weights[1:4].float()).to(
        torch.bfloat16
    )
    fused_bias = torch.einsum("n,n...->...", coefficients, head_bias[1:4].float()).to(
        torch.bfloat16
    )
    expected_packed = F.linear(student.calls[0]["projection_input"], fused_weight, fused_bias)
    expected = unpack_latents(expected_packed, 4, 4)

    torch.testing.assert_close(actual, expected, rtol=2e-6, atol=2e-6)
    assert len(student.calls) == 1
    assert student.proj_out is projection
    assert student.proj_out(projection.weight.new_zeros(1, 5)).shape[-1] == 16


def test_teacher_cfg_uses_canonical_packed_per_token_norm_rescale() -> None:
    teacher = _TinyQwenTransformer()
    config = _config(guidance_scale=4.0)
    adapter = QwenImagePDDAdapter(config)
    state, time, condition, negative_condition = _inputs()

    actual = adapter.teacher_velocity(
        teacher,
        state,
        time,
        condition=condition,
        negative_condition=negative_condition,
    )

    assert len(teacher.calls) == 2
    conditional = teacher.calls[0]["output"]
    unconditional = teacher.calls[1]["output"]
    guided = unconditional + 4.0 * (conditional - unconditional)
    factor = torch.linalg.vector_norm(
        conditional,
        dim=-1,
        keepdim=True,
    ) / torch.linalg.vector_norm(guided, dim=-1, keepdim=True)
    expected = unpack_latents(guided * factor, 4, 4)

    assert actual.dtype == torch.bfloat16
    torch.testing.assert_close(actual, expected)
    torch.testing.assert_close(teacher.calls[0]["encoder_hidden_states"], condition[0])
    torch.testing.assert_close(teacher.calls[1]["encoder_hidden_states"], negative_condition[0])
    assert all("txt_seq_lens" not in call["kwargs"] for call in teacher.calls)
    assert all(call["guidance"] is None for call in teacher.calls)


def test_teacher_cfg_stays_in_model_output_dtype() -> None:
    class LowPrecisionTeacher(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.config = SimpleNamespace(guidance_embeds=False)
            self.anchor = nn.Parameter(torch.zeros((), dtype=torch.bfloat16), requires_grad=False)
            self.outputs: list[torch.Tensor] = []

        def forward(self, *, hidden_states, encoder_hidden_states, **kwargs):
            value = encoder_hidden_states.mean(dim=(1, 2), keepdim=True).to(torch.bfloat16)
            output = hidden_states.to(torch.bfloat16) + value
            self.outputs.append(output.detach().clone())
            return (output,)

    teacher = LowPrecisionTeacher()
    state, time, condition, negative_condition = _inputs()
    actual = QwenImagePDDAdapter(_config(guidance_scale=4.0)).teacher_velocity(
        teacher,
        state,
        time,
        condition=condition,
        negative_condition=negative_condition,
    )

    assert actual.dtype == torch.bfloat16
    conditional, unconditional = teacher.outputs
    guided = unconditional + 4.0 * (conditional - unconditional)
    factor = torch.linalg.vector_norm(conditional, dim=-1, keepdim=True) / torch.linalg.vector_norm(
        guided, dim=-1, keepdim=True
    )
    expected = unpack_latents(guided * factor, 4, 4)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)


def test_guidance_disabled_teacher_is_one_conditional_call_without_negative_condition() -> None:
    teacher = _TinyQwenTransformer()
    config = _config(guidance_scale=None)
    adapter = QwenImagePDDAdapter(config)
    state, time, condition, _ = _inputs()

    actual = adapter.teacher_velocity(
        teacher,
        state,
        time,
        condition=condition,
    )

    assert len(teacher.calls) == 1
    expected = unpack_latents(teacher.calls[0]["output"], 4, 4)
    torch.testing.assert_close(actual, expected)


def test_conversion_preserves_requires_grad_mode_and_rejects_conflicts() -> None:
    transformer = _TinyQwenTransformer()
    transformer.eval()
    transformer.proj_out.weight.requires_grad_(False)
    transformer.proj_out.bias.requires_grad_(False)
    projection = convert_qwen_image_to_pdd(transformer, _config())

    assert transformer.training is False
    assert projection.training is False
    assert projection.weight.requires_grad is False
    assert projection.bias.requires_grad is False
    with pytest.raises(ValueError, match="incompatible"):
        convert_qwen_image_to_pdd(transformer, _config(grid_size=2))


def _tiny_diffusers_qwen():
    diffusers = pytest.importorskip("diffusers")
    return diffusers.QwenImageTransformer2DModel(
        patch_size=2,
        in_channels=8,
        out_channels=2,
        num_layers=1,
        attention_head_dim=8,
        num_attention_heads=2,
        joint_attention_dim=12,
        guidance_embeds=False,
        axes_dims_rope=(2, 2, 4),
    )


def test_conversion_preserves_the_ordinary_diffusers_qwen_root() -> None:
    student = _tiny_diffusers_qwen().eval()
    root_type = type(student)
    config = dict(student.config)

    convert_qwen_image_to_pdd(student, _config())

    assert type(student) is root_type
    assert isinstance(student.proj_out, PDDOutputProjection)
    assert dict(student.config) == config


def test_canonical_qwen_conversion_preserves_every_initialized_head() -> None:
    base = _tiny_diffusers_qwen().eval()
    student = copy.deepcopy(base)
    config = _config()
    generator = torch.Generator().manual_seed(20260715)
    state = torch.randn(2, 2, 4, 4, generator=generator)
    time = torch.tensor([0.875, 0.25], dtype=torch.float32)
    embeddings = torch.randn(2, 3, 12, generator=generator)
    mask = torch.tensor([[1, 1, 1], [1, 0, 1]], dtype=torch.long)
    model_kwargs = {
        "hidden_states": pack_latents(state),
        "timestep": time,
        "encoder_hidden_states": embeddings,
        "encoder_hidden_states_mask": mask,
        "img_shapes": build_img_shapes(2, 4, 4),
        "guidance": None,
        "return_dict": False,
    }

    with torch.no_grad():
        expected = unpack_latents(base(**model_kwargs)[0], 4, 4)
        convert_qwen_image_to_pdd(student, config)
        actual = QwenImagePDDAdapter(config).student_all_heads(
            student,
            state,
            time,
            condition=(embeddings, mask),
        )

    torch.testing.assert_close(actual, expected[:, None].expand_as(actual), rtol=0, atol=0)


def test_canonical_qwen_mask_makes_masked_padding_numerically_inert() -> None:
    student = _tiny_diffusers_qwen().eval()
    config = _config()
    convert_qwen_image_to_pdd(student, config)
    adapter = QwenImagePDDAdapter(config)
    generator = torch.Generator().manual_seed(20260715)
    state = torch.randn(2, 2, 4, 4, generator=generator)
    time = torch.tensor([0.875, 0.25], dtype=torch.float32)
    encoder_hidden_states = torch.randn(2, 3, 12, generator=generator)
    mask = torch.tensor([[1, 1, 1], [1, 0, 1]], dtype=torch.long)
    poisoned = encoder_hidden_states.clone()
    poisoned[~mask.bool()] = (
        torch.randn(
            poisoned[~mask.bool()].shape,
            generator=generator,
        )
        * 100
    )
    with torch.no_grad():
        baseline = adapter.student_all_heads(
            student,
            state,
            time,
            condition=(encoder_hidden_states, mask),
        )
        actual = adapter.student_all_heads(
            student,
            state,
            time,
            condition=(poisoned, mask),
        )

    torch.testing.assert_close(actual, baseline, rtol=0, atol=0)


def test_canonical_qwen_teacher_cfg_matches_the_pipeline_formula() -> None:
    teacher = _tiny_diffusers_qwen().eval()
    config = _config(guidance_scale=4.0)
    adapter = QwenImagePDDAdapter(config)
    generator = torch.Generator().manual_seed(20260716)
    state = torch.randn(2, 2, 4, 4, generator=generator)
    time = torch.tensor([0.75, 0.125], dtype=torch.float32)
    condition = (
        torch.randn(2, 3, 12, generator=generator),
        torch.tensor([[1, 1, 0], [1, 0, 1]], dtype=torch.long),
    )
    negative_condition = (
        torch.randn(2, 2, 12, generator=generator),
        torch.tensor([[1, 0], [1, 1]], dtype=torch.long),
    )

    def direct_packed(current_condition):
        embeddings, mask = current_condition
        return teacher(
            hidden_states=pack_latents(state),
            timestep=time,
            encoder_hidden_states=embeddings,
            encoder_hidden_states_mask=mask,
            img_shapes=build_img_shapes(2, 4, 4),
            guidance=None,
            return_dict=False,
        )[0]

    with torch.no_grad():
        conditional = direct_packed(condition)
        unconditional = direct_packed(negative_condition)
        guided = unconditional + 4.0 * (conditional - unconditional)
        expected = unpack_latents(
            guided
            * (
                torch.linalg.vector_norm(conditional, dim=-1, keepdim=True)
                / torch.linalg.vector_norm(guided, dim=-1, keepdim=True)
            ),
            4,
            4,
        )
        actual = adapter.teacher_velocity(
            teacher,
            state,
            time,
            condition=condition,
            negative_condition=negative_condition,
        )

    torch.testing.assert_close(actual, expected, rtol=0, atol=0)


def test_adapter_accepts_arbitrary_binary_masks_nonzero_padding_and_floating_time() -> None:
    student = _TinyQwenTransformer()
    config = _config()
    convert_qwen_image_to_pdd(student, config)
    state, time, condition, _ = _inputs()
    embeddings = condition[0].clone()
    mask = torch.tensor([[1, 0, 1], [0, 0, 0]], dtype=torch.long)
    embeddings[~mask.bool()] = 17

    actual = QwenImagePDDAdapter(config).student_all_heads(
        student,
        state,
        time.to(torch.bfloat16),
        condition=(embeddings, mask),
    )

    assert actual.shape == (2, 4, 1, 4, 4)
    torch.testing.assert_close(student.calls[0]["encoder_hidden_states_mask"], mask)
    assert student.calls[0]["timestep"].dtype == torch.bfloat16


def test_qwen_pdd_rejects_unsupported_config_condition_and_call_contracts() -> None:
    with pytest.raises(ValueError, match="num_train_timesteps=None"):
        QwenImagePDDAdapter(_config().model_copy(update={"num_train_timesteps": 1000}))
    with pytest.raises(TypeError, match="compute_dtype"):
        QwenImagePDDAdapter(_config(), compute_dtype=torch.long)

    transformer = _TinyQwenTransformer()
    transformer.config.guidance_embeds = True
    with pytest.raises(ValueError, match="guidance embeddings"):
        convert_qwen_image_to_pdd(transformer, _config())

    transformer.config.guidance_embeds = False
    config = _config()
    adapter = QwenImagePDDAdapter(config)
    state, time, condition, _ = _inputs()
    with pytest.raises(ValueError, match="negative_condition is required"):
        adapter.teacher_velocity(transformer, state, time, condition=condition)
    with pytest.raises(TypeError, match="negative_condition must be a tuple"):
        adapter.teacher_velocity(
            transformer,
            state,
            time,
            condition=condition,
            negative_condition=condition[0],
        )
    assert transformer.calls == []
    with pytest.raises(TypeError, match="converted to PDDOutputProjection"):
        adapter.student_all_heads(transformer, state, time, condition=condition)

    convert_qwen_image_to_pdd(transformer, config)
    with pytest.raises(TypeError, match="tuple"):
        adapter.student_all_heads(transformer, state, time, condition=condition[0])
    with pytest.raises(ValueError, match="requires batched embeddings"):
        adapter.student_all_heads(
            transformer,
            state,
            time,
            condition=(condition[0][..., 0], condition[1]),
        )
    with pytest.raises(ValueError, match="controlled keys"):
        adapter.student_all_heads(
            transformer,
            state,
            time,
            condition=condition,
            guidance=torch.ones(state.shape[0]),
        )
    with pytest.raises(ValueError, match="zero and one"):
        adapter.student_all_heads(
            transformer,
            state,
            time,
            condition=(condition[0], torch.tensor([[1, 2, 0], [1, 0, 0]])),
        )
    with pytest.raises(ValueError, match="integer/bool"):
        adapter.student_all_heads(
            transformer,
            state,
            time,
            condition=(condition[0], condition[1].float()),
        )


def test_raw_head_reference_uses_independent_linear_outputs() -> None:
    """Pin the widened storage order without calling adapter reshape helpers."""
    student = _TinyQwenTransformer()
    config = _config()
    projection = convert_qwen_image_to_pdd(student, config)
    state, time, condition, _ = _inputs(batch_size=1)
    embeddings, mask = condition
    packed = pack_latents(state).to(torch.bfloat16)
    hidden = torch.tanh(student.backbone(packed))
    condition_value = embeddings.mean(dim=(1, 2), keepdim=True)
    condition_value = condition_value + 0.01 * mask.sum(dim=1, keepdim=True).unsqueeze(-1)
    hidden = hidden + condition_value.to(hidden.dtype)
    hidden = hidden + (0.1 * time[:, None, None]).to(hidden.dtype)
    head_weights = projection.weight.reshape(4, 4, 5)
    head_bias = projection.bias.reshape(4, 4)
    expected_packed = torch.stack(
        [F.linear(hidden, head_weights[index], head_bias[index]) for index in range(4)],
        dim=1,
    )
    expected = torch.stack(
        [unpack_latents(expected_packed[:, index], 4, 4) for index in range(4)],
        dim=1,
    )

    actual = QwenImagePDDAdapter(config).student_all_heads(
        student,
        state,
        time,
        condition=condition,
    )

    torch.testing.assert_close(actual, expected)
