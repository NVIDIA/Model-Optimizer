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

"""Qwen-Image integration tests for PDD conversion, training, and execution."""

from __future__ import annotations

import copy
from types import MethodType

import pytest
import torch
from torch import nn

from modelopt.torch.fastgen import PDDConfig, PDDPipeline
from modelopt.torch.fastgen.plugins import QwenImagePDDAdapter
from modelopt.torch.fastgen.plugins.qwen_image import build_img_shapes, pack_latents, unpack_latents
from modelopt.torch.fastgen.plugins.qwen_image_pdd import (
    QWEN_IMAGE_PDD_EXECUTION,
    convert_qwen_image_to_pdd,
    enable_qwen_image_pdd_forward,
    freeze_qwen_image_pdd_unused_parameters,
    require_qwen_image_pdd_forward,
)


def _config(*, guidance_scale: float | None = 4.0, grid_size: int = 4) -> PDDConfig:
    return PDDConfig(
        grid_size=grid_size,
        grid_max_t=0.999,
        flow_shift=5.0,
        block_size_min=1,
        block_size_max=grid_size,
        inference_blocks=[2, 2] if grid_size == 4 else [grid_size],
        guidance_scale=guidance_scale,
    )


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
    encoder_hidden_states_mask = encoder_hidden_states_mask.to(torch.bool)
    hidden_states = model.img_in(hidden_states)
    encoder_hidden_states = model.txt_in(model.txt_norm(encoder_hidden_states))
    temb = model.time_text_embed(timestep, hidden_states)
    image_rotary_emb = model.pos_embed(
        img_shapes,
        max_txt_seq_len=max_txt_seq_len,
        device=hidden_states.device,
    )
    for block in model.transformer_blocks:
        encoder_hidden_states, hidden_states = block(
            hidden_states=hidden_states,
            encoder_hidden_states=encoder_hidden_states,
            encoder_hidden_states_mask=encoder_hidden_states_mask,
            temb=temb,
            image_rotary_emb=image_rotary_emb,
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
    encoder_hidden_states = torch.randn(2, 4, 12, generator=generator).to(torch.bfloat16)
    mask = torch.tensor([[1, 1, 0, 0], [1, 0, 1, 0]], dtype=torch.long)
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
        assert kwargs.get("joint_attention_kwargs") is None
        captured_masks.append(kwargs["encoder_hidden_states_mask"].detach().clone())

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
    assert all(torch.equal(captured, mask.bool()) for captured in captured_masks)


def test_mr210_preserves_diffusers_output_contract() -> None:
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
