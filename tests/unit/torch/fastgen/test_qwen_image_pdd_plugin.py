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
from types import MethodType, SimpleNamespace

import pytest
import torch
import torch.nn.functional as F
from torch import nn

import modelopt.torch.fastgen.plugins.qwen_image_pdd as qwen_image_pdd_plugin
from modelopt.torch.fastgen import PDDConfig, PDDPipeline
from modelopt.torch.fastgen.flow_matching import fusion_coefficients
from modelopt.torch.fastgen.plugins import QwenImagePDDAdapter
from modelopt.torch.fastgen.plugins.qwen_image import build_img_shapes, pack_latents, unpack_latents
from modelopt.torch.fastgen.plugins.qwen_image_pdd import (
    QWEN_IMAGE_PDD_EXECUTION,
    convert_qwen_image_to_pdd,
    enable_qwen_image_pdd_forward,
    freeze_qwen_image_pdd_unused_parameters,
    require_qwen_image_pdd_forward,
)


class _QwenImageTestDouble(nn.Module):
    """Explicit unit-test scope for adapter protocol doubles."""


@pytest.fixture(autouse=True)
def _allow_qwen_image_test_doubles(monkeypatch):
    require_production_forward = qwen_image_pdd_plugin.require_qwen_image_pdd_forward

    def require_forward(model: nn.Module) -> str:
        if isinstance(model, _QwenImageTestDouble):
            if (
                getattr(model, "_modelopt_qwen_image_pdd_execution", None)
                != QWEN_IMAGE_PDD_EXECUTION
            ):
                raise RuntimeError("Qwen-Image PDD requires its masked joint-attention forward.")
            return QWEN_IMAGE_PDD_EXECUTION
        return require_production_forward(model)

    monkeypatch.setattr(qwen_image_pdd_plugin, "require_qwen_image_pdd_forward", require_forward)


class _TinyQwenTransformer(_QwenImageTestDouble):
    """Qwen-shaped packed transformer with a real registered final linear."""

    def __init__(self, *, packed_channels: int = 4, hidden_width: int = 5) -> None:
        super().__init__()
        self.config = SimpleNamespace(guidance_embeds=False)
        self._modelopt_qwen_image_pdd_execution = QWEN_IMAGE_PDD_EXECUTION
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
        max_txt_seq_len,
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
                "max_txt_seq_len": max_txt_seq_len,
                "projection_input": hidden.detach().clone(),
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


def _pack_oracle(latents: torch.Tensor) -> torch.Tensor:
    batch, channels, height, width = latents.shape
    return (
        latents.reshape(batch, channels, height // 2, 2, width // 2, 2)
        .permute(0, 2, 4, 1, 3, 5)
        .reshape(batch, (height // 2) * (width // 2), channels * 4)
    )


def _unpack_oracle(packed: torch.Tensor, height: int, width: int) -> torch.Tensor:
    batch, _patches, packed_channels = packed.shape
    channels = packed_channels // 4
    return (
        packed.reshape(batch, height // 2, width // 2, channels, 2, 2)
        .permute(0, 3, 1, 4, 2, 5)
        .reshape(batch, channels, height, width)
    )


def _mr210_rollout_oracle(
    state: torch.Tensor,
    heads: torch.Tensor,
    grid: torch.Tensor,
    n: torch.Tensor,
    k: torch.Tensor,
) -> torch.Tensor:
    interval_ids = torch.arange(grid.numel() - 1, device=state.device)
    velocity_mask = (interval_ids[None] >= n[:, None]) & (interval_ids[None] < k[:, None])
    weighted_intervals = velocity_mask.to(torch.float32) * torch.diff(grid.float())[None]
    return state.float() + torch.einsum("bn,bn...->b...", weighted_intervals, heads.float())


def test_all_head_training_accepts_a_serialized_widened_linear() -> None:
    student = _TinyQwenTransformer()
    config = _config()
    converted = convert_qwen_image_to_pdd(student, config)
    serialized = nn.Linear(
        converted.in_features,
        converted.out_features,
        bias=converted.bias is not None,
        dtype=converted.weight.dtype,
    )
    serialized.load_state_dict(converted.state_dict())
    student.proj_out = serialized
    state, time, condition, _ = _inputs(batch_size=1)

    heads = QwenImagePDDAdapter(config).student_all_heads(
        student,
        state,
        time,
        condition=condition,
    )
    heads.float().square().mean().backward()

    assert heads.shape == (1, 4, 1, 4, 4)
    assert serialized.weight.grad is not None


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


def test_teacher_cfg_remote_preflight_failure_stops_both_model_calls(monkeypatch) -> None:
    teacher = _TinyQwenTransformer()
    adapter = QwenImagePDDAdapter(_config(guidance_scale=4.0))
    state, time, condition, negative_condition = _inputs()

    monkeypatch.setattr(qwen_image_pdd_plugin.dist, "is_available", lambda: True)
    monkeypatch.setattr(qwen_image_pdd_plugin.dist, "is_initialized", lambda: True)

    def report_remote_failure(failed, *, op):
        assert not bool(failed)
        assert op is torch.distributed.ReduceOp.MAX
        failed.fill_(1)

    monkeypatch.setattr(qwen_image_pdd_plugin.dist, "all_reduce", report_remote_failure)

    with pytest.raises(RuntimeError, match="preflight failed on another rank"):
        adapter.teacher_velocity(
            teacher,
            state,
            time,
            condition=condition,
            negative_condition=negative_condition,
        )

    assert teacher.calls == []


def test_teacher_cfg_zero_guided_norm_uses_qwen_clamp() -> None:
    class ZeroGuidedTeacher(_QwenImageTestDouble):
        def __init__(self) -> None:
            super().__init__()
            self.config = SimpleNamespace(guidance_embeds=False)
            self._modelopt_qwen_image_pdd_execution = QWEN_IMAGE_PDD_EXECUTION
            self.anchor = nn.Parameter(torch.zeros((), dtype=torch.bfloat16), requires_grad=False)
            self.calls = 0

        def forward(self, *, hidden_states, **_kwargs):
            self.calls += 1
            value = 3.0 if self.calls == 1 else 4.0
            return torch.full_like(hidden_states, value, dtype=torch.bfloat16)

    teacher = ZeroGuidedTeacher()
    state, time, condition, negative_condition = _inputs(batch_size=1)
    actual = QwenImagePDDAdapter(_config(guidance_scale=4.0)).teacher_velocity(
        teacher,
        state,
        time,
        condition=condition,
        negative_condition=negative_condition,
    )

    assert torch.isfinite(actual).all()
    torch.testing.assert_close(actual, torch.zeros_like(actual), rtol=0, atol=0)


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


def test_mr210_freezes_exactly_the_structurally_unused_parameters() -> None:
    student = _tiny_diffusers_qwen()

    frozen_names = freeze_qwen_image_pdd_unused_parameters(student)

    assert set(frozen_names) == {
        "transformer_blocks.0.attn.to_add_out.weight",
        "transformer_blocks.0.attn.to_add_out.bias",
        "transformer_blocks.0.txt_mlp.net.0.proj.weight",
        "transformer_blocks.0.txt_mlp.net.0.proj.bias",
        "transformer_blocks.0.txt_mlp.net.2.weight",
        "transformer_blocks.0.txt_mlp.net.2.bias",
    }
    assert {
        name for name, parameter in student.named_parameters() if not parameter.requires_grad
    } == set(frozen_names)
    optimizer = torch.optim.AdamW(
        [parameter for parameter in student.parameters() if parameter.requires_grad]
    )
    optimized_parameter_ids = {
        id(parameter) for group in optimizer.param_groups for parameter in group["params"]
    }
    assert all(
        id(student.get_parameter(name)) not in optimized_parameter_ids for name in frozen_names
    )


def _mr210_qwen_forward_oracle(
    model: nn.Module,
    hidden_states: torch.Tensor,
    encoder_hidden_states: torch.Tensor,
    encoder_hidden_states_mask: torch.Tensor,
    timestep: torch.Tensor,
    img_shapes: list,
    max_txt_seq_len: int,
) -> torch.Tensor:
    """Test-local MR210 operation order; intentionally independent of production binding."""
    hidden_states = model.img_in(hidden_states)
    encoder_hidden_states = model.txt_in(model.txt_norm(encoder_hidden_states))
    temb = model.time_text_embed(timestep, hidden_states)
    image_rotary_emb = model.pos_embed(
        img_shapes,
        max_txt_seq_len=max_txt_seq_len,
        device=hidden_states.device,
    )
    image_mask = torch.ones(
        (hidden_states.shape[0], hidden_states.shape[1]),
        dtype=torch.bool,
        device=hidden_states.device,
    )
    joint_attention_mask = torch.cat(
        (encoder_hidden_states_mask.to(torch.bool), image_mask),
        dim=1,
    )[:, None, None, :]
    for block in model.transformer_blocks:
        encoder_hidden_states, hidden_states = block(
            hidden_states=hidden_states,
            encoder_hidden_states=encoder_hidden_states,
            encoder_hidden_states_mask=None,
            temb=temb,
            image_rotary_emb=image_rotary_emb,
            joint_attention_kwargs={"attention_mask": joint_attention_mask},
        )
    return model.proj_out(model.norm_out(hidden_states, temb))


def test_mr210_real_qwen_loss_and_backward_match_independent_graph() -> None:
    class CapturingAdapter(QwenImagePDDAdapter):
        def _call_packed(self, *args, **kwargs):
            with torch.autocast(device_type="cpu", dtype=torch.bfloat16):
                return super()._call_packed(*args, **kwargs)

        def student_all_heads(self, *args, **kwargs):
            value = super().student_all_heads(*args, **kwargs)
            self.captured_heads = value.detach().clone()
            return value

        def teacher_velocity(self, model, state, time, **kwargs):
            self.captured_teacher_state = state.detach().clone()
            value = super().teacher_velocity(model, state, time, **kwargs)
            self.captured_teacher = value.detach().clone()
            return value

    torch.manual_seed(20260716)
    base = _tiny_diffusers_qwen().eval()
    actual_student = enable_qwen_image_pdd_forward(copy.deepcopy(base))
    actual_student.enable_gradient_checkpointing()
    actual_teacher = copy.deepcopy(actual_student).eval().requires_grad_(False)
    oracle_student = copy.deepcopy(base)
    oracle_teacher = copy.deepcopy(base).eval().requires_grad_(False)
    config = _config(guidance_scale=4.0)
    convert_qwen_image_to_pdd(actual_student, config)
    convert_qwen_image_to_pdd(oracle_student, config)
    adapter = CapturingAdapter(config, compute_dtype=torch.bfloat16)
    pipeline = PDDPipeline(actual_student, actual_teacher, config, adapter)

    generator = torch.Generator().manual_seed(20260716)
    data = torch.randn(1, 2, 4, 4, generator=generator)
    noise = torch.randn(1, 2, 4, 4, generator=generator)
    condition = (
        torch.randn(1, 3, 12, generator=generator).to(torch.bfloat16),
        torch.tensor([[1, 1, 1]], dtype=torch.long),
    )
    negative_condition = (
        torch.randn(1, 2, 12, generator=generator).to(torch.bfloat16),
        torch.tensor([[1, 1]], dtype=torch.long),
    )
    n = torch.tensor([1], dtype=torch.long)
    k = torch.tensor([3], dtype=torch.long)

    actual_loss, _ = pipeline.compute_loss(
        data,
        noise=noise,
        condition=condition,
        negative_condition=negative_condition,
        n=n,
        k=k,
    )
    actual_loss.backward()

    unshifted = torch.linspace(0.999, 0.0, 5, dtype=torch.float64)
    grid = (5.0 * unshifted / (1.0 + 4.0 * unshifted)).clamp_max(0.999).float()
    time_n = grid[n]
    broadcast_time = time_n.to(torch.float64).reshape(1, 1, 1, 1)
    x_n = (
        data.float().to(torch.float64) * (1.0 - broadcast_time)
        + noise.float().to(torch.float64) * broadcast_time
    ).float()

    def oracle_forward(model, state, time, current_condition):
        embeddings, mask = current_condition
        with torch.autocast(device_type="cpu", dtype=torch.bfloat16):
            return _mr210_qwen_forward_oracle(
                model,
                hidden_states=_pack_oracle(state).to(torch.bfloat16),
                encoder_hidden_states=embeddings,
                encoder_hidden_states_mask=mask,
                timestep=time,
                img_shapes=build_img_shapes(state.shape[0], state.shape[2], state.shape[3]),
                max_txt_seq_len=int(mask.sum(dim=1).max().to(torch.int32).item()),
            )

    packed_heads = oracle_forward(oracle_student, x_n, time_n, condition)
    batch, patches, _features = packed_heads.shape
    packed_heads = packed_heads.reshape(batch, patches, 4, 8).permute(0, 2, 1, 3)
    oracle_heads = _unpack_oracle(
        packed_heads.reshape(4, patches, 8),
        4,
        4,
    ).reshape(1, 4, 2, 4, 4)
    oracle_heads_fp32 = oracle_heads.float()
    with torch.no_grad():
        x_bar_k = _mr210_rollout_oracle(x_n, oracle_heads_fp32, grid, n, k)
    student_target = oracle_heads_fp32[:, int(k.item())]
    time_k = grid[k]
    conditional = oracle_forward(oracle_teacher, x_bar_k, time_k, condition)
    unconditional = oracle_forward(oracle_teacher, x_bar_k, time_k, negative_condition)
    guided_low_precision = conditional + 3.0 * (conditional - unconditional)
    conditional_fp32 = conditional.float()
    guided_fp32 = guided_low_precision.float()
    teacher_target_packed = (
        guided_fp32
        * (
            torch.linalg.vector_norm(conditional_fp32, dim=-1, keepdim=True)
            / torch.linalg.vector_norm(guided_fp32, dim=-1, keepdim=True).clamp_min(1e-5)
        )
    ).to(torch.bfloat16)
    teacher_target_low_precision = _unpack_oracle(teacher_target_packed, 4, 4)
    teacher_target = teacher_target_low_precision.float().detach()
    oracle_loss = (student_target - teacher_target).square().mean()
    oracle_loss.backward()

    torch.testing.assert_close(
        adapter.captured_heads[:, int(k.item())],
        oracle_heads[:, int(k.item())],
        rtol=0,
        atol=0,
    )
    torch.testing.assert_close(
        adapter.captured_teacher, teacher_target_low_precision, rtol=0, atol=0
    )
    torch.testing.assert_close(adapter.captured_teacher_state, x_bar_k, rtol=1e-6, atol=1e-7)
    torch.testing.assert_close(actual_loss, oracle_loss, rtol=1e-6, atol=1e-7)
    for actual_parameter, oracle_parameter in (
        (actual_student.proj_out.weight, oracle_student.proj_out.weight),
        (actual_student.img_in.weight, oracle_student.img_in.weight),
    ):
        assert actual_parameter.grad is not None and oracle_parameter.grad is not None
        assert actual_parameter.grad.dtype == torch.float32
        assert oracle_parameter.grad.dtype == torch.float32
        torch.testing.assert_close(
            actual_parameter.grad,
            oracle_parameter.grad,
            rtol=1e-6,
            atol=1e-7,
        )


def test_qwen_pdd_forward_binding_preserves_root_state_and_deepcopy() -> None:
    source = _tiny_diffusers_qwen().eval()
    source_type = type(source)
    source_state = {name: value.detach().clone() for name, value in source.state_dict().items()}

    adopted = enable_qwen_image_pdd_forward(source)

    assert adopted is source
    assert type(adopted) is source_type
    assert enable_qwen_image_pdd_forward(adopted) is adopted
    assert require_qwen_image_pdd_forward(adopted) == QWEN_IMAGE_PDD_EXECUTION
    for name, value in adopted.state_dict().items():
        torch.testing.assert_close(value, source_state[name], rtol=0, atol=0)

    round_trip = copy.deepcopy(adopted)
    assert round_trip.forward.__self__ is round_trip
    require_qwen_image_pdd_forward(round_trip)

    conflicting = _tiny_diffusers_qwen()
    conflicting.forward = MethodType(lambda self, **_kwargs: self, conflicting)
    with pytest.raises(RuntimeError, match="instance-level forward override"):
        enable_qwen_image_pdd_forward(conflicting)


def test_mr210_qwen_conversion_preserves_every_initialized_head() -> None:
    base = _tiny_diffusers_qwen().eval().to(torch.bfloat16)
    student = copy.deepcopy(base)
    student = enable_qwen_image_pdd_forward(student)
    config = _config()
    generator = torch.Generator().manual_seed(20260715)
    state = torch.randn(2, 2, 4, 4, generator=generator)
    time = torch.tensor([0.875, 0.25], dtype=torch.float32)
    embeddings = torch.randn(2, 3, 12, generator=generator).to(torch.bfloat16)
    mask = torch.tensor([[1, 1, 1], [1, 0, 0]], dtype=torch.long)
    model_kwargs = {
        "hidden_states": pack_latents(state).to(torch.bfloat16),
        "timestep": time,
        "encoder_hidden_states": embeddings,
        "encoder_hidden_states_mask": mask,
        "img_shapes": build_img_shapes(2, 4, 4),
        "max_txt_seq_len": 3,
    }

    with torch.no_grad():
        expected = unpack_latents(
            _mr210_qwen_forward_oracle(base, **model_kwargs),
            4,
            4,
        )
        convert_qwen_image_to_pdd(student, config)
        actual = QwenImagePDDAdapter(config).student_all_heads(
            student,
            state,
            time,
            condition=(embeddings, mask),
        )

    torch.testing.assert_close(actual, expected[:, None].expand_as(actual), rtol=0, atol=0)


def test_mr210_joint_mask_ignores_padded_token_values() -> None:
    canonical = _tiny_diffusers_qwen().eval().to(torch.bfloat16)
    student = copy.deepcopy(canonical)
    student = enable_qwen_image_pdd_forward(student)
    config = _config()
    convert_qwen_image_to_pdd(student, config)
    adapter = QwenImagePDDAdapter(config)
    generator = torch.Generator().manual_seed(20260715)
    state = torch.randn(2, 2, 4, 4, generator=generator)
    time = torch.tensor([0.875, 0.25], dtype=torch.float32)
    encoder_hidden_states = torch.randn(2, 3, 12, generator=generator).to(torch.bfloat16)
    mask = torch.tensor([[1, 1, 1], [1, 0, 0]], dtype=torch.long)
    poisoned = encoder_hidden_states.clone()
    poisoned[~mask.bool()] = (
        torch.randn(
            poisoned[~mask.bool()].shape,
            generator=generator,
        )
        * 100
    ).to(torch.bfloat16)
    canonical_kwargs = {
        "hidden_states": pack_latents(state).to(torch.bfloat16),
        "timestep": time,
        "encoder_hidden_states_mask": mask,
        "img_shapes": build_img_shapes(2, 4, 4),
        "guidance": None,
        "return_dict": False,
    }
    captured_masks: list[torch.Tensor] = []

    def capture_block_mask(_module, _args, kwargs):
        assert kwargs["encoder_hidden_states_mask"] is None
        captured_masks.append(kwargs["joint_attention_kwargs"]["attention_mask"].detach().clone())

    hook = student.transformer_blocks[0].register_forward_pre_hook(
        capture_block_mask,
        with_kwargs=True,
    )
    with torch.no_grad():
        canonical_baseline = canonical(
            encoder_hidden_states=encoder_hidden_states,
            **canonical_kwargs,
        )[0]
        canonical_poisoned = canonical(
            encoder_hidden_states=poisoned,
            **canonical_kwargs,
        )[0]
        strict_baseline = adapter.student_all_heads(
            student,
            state,
            time,
            condition=(encoder_hidden_states, mask),
        )
        strict_poisoned = adapter.student_all_heads(
            student,
            state,
            time,
            condition=(poisoned, mask),
        )
    hook.remove()

    torch.testing.assert_close(canonical_poisoned, canonical_baseline, rtol=0, atol=0)
    torch.testing.assert_close(strict_poisoned, strict_baseline, rtol=0, atol=0)
    assert len(captured_masks) == 2
    expected_mask = torch.cat((mask.bool(), torch.ones(2, 4, dtype=torch.bool)), dim=1)
    expected_mask = expected_mask[:, None, None, :]
    assert all(torch.equal(captured, expected_mask) for captured in captured_masks)


def test_mr210_preserves_diffusers_output_and_harmless_call_contract() -> None:
    student = enable_qwen_image_pdd_forward(_tiny_diffusers_qwen().eval().to(torch.bfloat16))
    generator = torch.Generator().manual_seed(20260716)
    kwargs = {
        "hidden_states": pack_latents(torch.randn(2, 2, 4, 4, generator=generator)).to(
            torch.bfloat16
        ),
        "encoder_hidden_states": torch.randn(2, 3, 12, generator=generator).to(torch.bfloat16),
        "encoder_hidden_states_mask": torch.tensor([[1, 1, 1], [1, 0, 0]], dtype=torch.long),
        "timestep": torch.tensor([0.875, 0.25], dtype=torch.float32),
        "img_shapes": build_img_shapes(2, 4, 4),
        "txt_seq_lens": [3, 1],
        "guidance": None,
    }

    with torch.no_grad():
        tuple_output = student(**kwargs, return_dict=False)
        model_output = student(**kwargs, return_dict=True)

    assert isinstance(tuple_output, tuple) and len(tuple_output) == 1
    assert hasattr(model_output, "sample")
    torch.testing.assert_close(model_output.sample, tuple_output[0], rtol=0, atol=0)


def test_mr210_time_embed_receives_fp32_grid_value() -> None:
    student = enable_qwen_image_pdd_forward(_tiny_diffusers_qwen().eval().to(torch.bfloat16))
    config = _config()
    convert_qwen_image_to_pdd(student, config)
    captured: list[torch.Tensor] = []

    def capture_time(_module, args):
        captured.append(args[0].detach().clone())

    hook = student.time_text_embed.register_forward_pre_hook(capture_time)
    generator = torch.Generator().manual_seed(20260715)
    state = torch.randn(1, 2, 4, 4, generator=generator)
    time = torch.tensor([0.999], dtype=torch.float32)
    embeddings = torch.randn(1, 3, 12, generator=generator).to(torch.bfloat16)
    mask = torch.ones(1, 3, dtype=torch.long)
    with torch.no_grad():
        QwenImagePDDAdapter(config).student_all_heads(
            student,
            state,
            time,
            condition=(embeddings, mask),
        )
    hook.remove()

    assert len(captured) == 1
    assert captured[0].dtype == torch.float32
    torch.testing.assert_close(captured[0], time, rtol=0, atol=0)
    assert captured[0].item() != time.to(torch.bfloat16).float().item()


def test_mr210_qwen_teacher_cfg_matches_per_token_reference() -> None:
    teacher = enable_qwen_image_pdd_forward(_tiny_diffusers_qwen().eval().to(torch.bfloat16))
    config = _config(guidance_scale=4.0)
    adapter = QwenImagePDDAdapter(config)
    generator = torch.Generator().manual_seed(20260716)
    state = torch.randn(2, 2, 4, 4, generator=generator)
    time = torch.tensor([0.75, 0.125], dtype=torch.float32)
    condition = (
        torch.randn(2, 3, 12, generator=generator).to(torch.bfloat16),
        torch.tensor([[1, 1, 1], [1, 0, 0]], dtype=torch.long),
    )
    negative_condition = (
        torch.randn(2, 2, 12, generator=generator).to(torch.bfloat16),
        torch.tensor([[1, 0], [1, 1]], dtype=torch.long),
    )

    def direct_packed(current_condition):
        embeddings, mask = current_condition
        return teacher(
            hidden_states=pack_latents(state).to(torch.bfloat16),
            timestep=time,
            encoder_hidden_states=embeddings,
            encoder_hidden_states_mask=mask,
            img_shapes=build_img_shapes(2, 4, 4),
            max_txt_seq_len=int(mask.sum(dim=1).max().item()),
            return_dict=False,
        )[0]

    with torch.no_grad():
        conditional = direct_packed(condition)
        unconditional = direct_packed(negative_condition)
        guided_low_precision = conditional + 3.0 * (conditional - unconditional)
        conditional_fp32 = conditional.float()
        guided_fp32 = guided_low_precision.float()
        factor = torch.linalg.vector_norm(
            conditional_fp32, dim=-1, keepdim=True
        ) / torch.linalg.vector_norm(guided_fp32, dim=-1, keepdim=True).clamp_min(1e-5)
        expected = unpack_latents((guided_fp32 * factor).to(torch.bfloat16), 4, 4)
        global_factor = torch.linalg.vector_norm(
            conditional_fp32, dim=(1, 2), keepdim=True
        ) / torch.linalg.vector_norm(guided_fp32, dim=(1, 2), keepdim=True).clamp_min(1e-5)
        global_expected = unpack_latents((guided_fp32 * global_factor).to(torch.bfloat16), 4, 4)
        actual = adapter.teacher_velocity(
            teacher,
            state,
            time,
            condition=condition,
            negative_condition=negative_condition,
        )

    assert not torch.equal(expected, global_expected)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)


def test_qwen_pdd_rejects_unsupported_configuration_and_inputs() -> None:
    with pytest.raises(ValueError, match="num_train_timesteps=None"):
        QwenImagePDDAdapter(_config().model_copy(update={"num_train_timesteps": 1000}))

    transformer = _TinyQwenTransformer()
    config = _config()
    adapter = QwenImagePDDAdapter(config)
    state, time, condition, _ = _inputs()
    with pytest.raises(TypeError, match="negative_condition must be a tuple"):
        adapter.teacher_velocity(transformer, state, time, condition=condition)
    assert transformer.calls == []

    convert_qwen_image_to_pdd(transformer, config)
    with pytest.raises(TypeError, match="FP32 time"):
        adapter.student_all_heads(
            transformer,
            state,
            time.to(torch.bfloat16),
            condition=condition,
        )

    unmarked = _TinyQwenTransformer()
    delattr(unmarked, "_modelopt_qwen_image_pdd_execution")
    convert_qwen_image_to_pdd(unmarked, config)
    with pytest.raises(RuntimeError, match="masked joint-attention forward"):
        adapter.student_all_heads(unmarked, state, time, condition=condition)
