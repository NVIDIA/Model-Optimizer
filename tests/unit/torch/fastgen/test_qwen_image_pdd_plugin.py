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
    QWEN_IMAGE_PDD_FORWARD_SUBSTRATE,
    QWEN_IMAGE_PDD_FORWARD_SUBSTRATE_ID,
    QWEN_IMAGE_PDD_LAYER_SPEC,
    adopt_qwen_image_mr210_forward,
    convert_qwen_image_to_pdd,
    require_qwen_image_pdd_forward_substrate,
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


def test_teacher_cfg_and_global_norm_rescale_match_mr210_reference() -> None:
    teacher = _TinyQwenTransformer()
    config = _config(guidance_scale=4.0)
    adapter = QwenImagePDDAdapter(config, guidance_rescale=1.0, guidance_eps=1e-5)
    state, time, condition, negative_condition = _inputs()

    actual = adapter.teacher_velocity(
        teacher,
        state,
        time,
        condition=condition,
        negative_condition=negative_condition,
    )

    assert len(teacher.calls) == 2
    conditional_bf16 = teacher.calls[0]["output"]
    unconditional_bf16 = teacher.calls[1]["output"]
    guided_bf16 = conditional_bf16 + 3.0 * (conditional_bf16 - unconditional_bf16)
    conditional = conditional_bf16.float()
    guided = guided_bf16.float()
    factor = torch.linalg.vector_norm(
        conditional,
        dim=(1, 2),
        keepdim=True,
    ) / torch.linalg.vector_norm(guided, dim=(1, 2), keepdim=True).clamp_min(1e-5)
    expected = unpack_latents((guided * factor).to(teacher.calls[0]["output"].dtype), 4, 4)

    assert actual.dtype == torch.bfloat16
    torch.testing.assert_close(actual, expected)
    torch.testing.assert_close(teacher.calls[0]["encoder_hidden_states"], condition[0])
    torch.testing.assert_close(teacher.calls[1]["encoder_hidden_states"], negative_condition[0])
    assert all("txt_seq_lens" not in call["kwargs"] for call in teacher.calls)
    assert all(call["guidance"] is None for call in teacher.calls)


def test_teacher_cfg_returns_to_low_precision_model_output_dtype() -> None:
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
    guided_bf16 = conditional + 3.0 * (conditional - unconditional)
    conditional_fp32 = conditional.float()
    guided_fp32 = guided_bf16.float()
    factor = torch.linalg.vector_norm(
        conditional_fp32, dim=(1, 2), keepdim=True
    ) / torch.linalg.vector_norm(guided_fp32, dim=(1, 2), keepdim=True).clamp_min(1e-5)
    expected = unpack_latents((guided_fp32 * factor).to(torch.bfloat16), 4, 4)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)

    unrounded_guided = conditional_fp32 + 3.0 * (conditional_fp32 - unconditional.float())
    unrounded_factor = torch.linalg.vector_norm(
        conditional_fp32, dim=(1, 2), keepdim=True
    ) / torch.linalg.vector_norm(unrounded_guided, dim=(1, 2), keepdim=True).clamp_min(1e-5)
    unrounded = unpack_latents((unrounded_guided * unrounded_factor).to(torch.bfloat16), 4, 4)
    assert not torch.equal(actual, unrounded)


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


def test_forward_substrate_identity_is_exact() -> None:
    assert QWEN_IMAGE_PDD_FORWARD_SUBSTRATE_ID == (
        "pdd_qwen_mr210_c8100b1347b278511336dccfc074a461457216ec_"
        "qwen_33706683487ba16d133b99b73be27b21164c53335441d77b1dcabbfca970f70e"
    )
    assert require_qwen_image_pdd_forward_substrate(QWEN_IMAGE_PDD_FORWARD_SUBSTRATE) == dict(
        QWEN_IMAGE_PDD_FORWARD_SUBSTRATE
    )
    mismatched = dict(QWEN_IMAGE_PDD_FORWARD_SUBSTRATE)
    mismatched["id"] = "canonical-diffusers"
    with pytest.raises(ValueError, match="authenticated MR210"):
        require_qwen_image_pdd_forward_substrate(mismatched)
    with pytest.raises(ValueError, match="authenticated MR210"):
        require_qwen_image_pdd_forward_substrate(None)


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


def test_adoption_preserves_diffusers_interface_state_keys_and_parameter_identity(
    monkeypatch,
) -> None:
    qwen_pdd = pytest.importorskip("modelopt.torch.fastgen.plugins.qwen_image_pdd")
    source = _tiny_diffusers_qwen()
    source.eval()
    source_type = type(source)
    source_keys = tuple(source.state_dict())
    source_parameters = tuple(source.parameters())
    source_config = dict(source.config)
    monkeypatch.setattr(qwen_pdd, "_require_qwen_source_identity", lambda _model: None)

    adopted = adopt_qwen_image_mr210_forward(source)

    assert adopted is not source
    assert isinstance(adopted, source_type)
    assert type(source) is source_type
    assert tuple(adopted.state_dict()) == source_keys
    assert all(
        actual is expected
        for actual, expected in zip(adopted.parameters(), source_parameters, strict=True)
    )
    assert dict(adopted.config) == source_config
    assert adopted.device == source.device
    assert adopted.dtype == source.dtype
    assert adopted.training is False
    assert adopt_qwen_image_mr210_forward(adopted) is adopted


def test_adoption_rejects_unpinned_qwen_source(monkeypatch) -> None:
    qwen_pdd = pytest.importorskip("modelopt.torch.fastgen.plugins.qwen_image_pdd")
    source = _tiny_diffusers_qwen()
    monkeypatch.setattr(qwen_pdd, "_sha256_source", lambda _owner: "0" * 64)

    with pytest.raises(RuntimeError, match="transformer source"):
        adopt_qwen_image_mr210_forward(source)


def test_adoption_rejects_unpinned_timestep_embedding_and_diffusers_version(monkeypatch) -> None:
    diffusers = pytest.importorskip("diffusers")
    qwen_pdd = pytest.importorskip("modelopt.torch.fastgen.plugins.qwen_image_pdd")
    source = _tiny_diffusers_qwen()

    def mismatched_embedding_hash(owner):
        if owner is type(source):
            return QWEN_IMAGE_PDD_FORWARD_SUBSTRATE["diffusers_qwen_source_sha256"]
        return "0" * 64

    monkeypatch.setattr(qwen_pdd, "_sha256_source", mismatched_embedding_hash)
    with pytest.raises(RuntimeError, match="timestep embedding source"):
        adopt_qwen_image_mr210_forward(source)

    def pinned_source_hash(owner):
        if owner is type(source):
            return QWEN_IMAGE_PDD_FORWARD_SUBSTRATE["diffusers_qwen_source_sha256"]
        return QWEN_IMAGE_PDD_FORWARD_SUBSTRATE["diffusers_embeddings_source_sha256"]

    monkeypatch.setattr(qwen_pdd, "_sha256_source", pinned_source_hash)
    monkeypatch.setattr(diffusers, "__version__", "0.0.0")
    with pytest.raises(RuntimeError, match="Diffusers version"):
        adopt_qwen_image_mr210_forward(source)


def test_adoption_rejects_every_material_root_invariant(monkeypatch) -> None:
    qwen_pdd = pytest.importorskip("modelopt.torch.fastgen.plugins.qwen_image_pdd")
    baseline = _tiny_diffusers_qwen()
    monkeypatch.setattr(qwen_pdd, "_require_qwen_source_identity", lambda _model: None)

    def set_config_flag(source, name):
        source._internal_dict = type(source._internal_dict)({**dict(source.config), name: True})

    def check(mutator, match):
        source = copy.deepcopy(baseline)
        mutator(source)
        with pytest.raises((RuntimeError, ValueError), match=match):
            adopt_qwen_image_mr210_forward(source)

    check(lambda source: source.add_module("unexpected", nn.Identity()), "root child layout")
    check(
        lambda source: setattr(
            source,
            "_modules",
            dict(reversed(tuple(source._modules.items()))),
        ),
        "root child layout",
    )
    check(
        lambda source: source.register_parameter("root_parameter", nn.Parameter(torch.zeros(1))),
        "direct parameters or buffers",
    )
    check(
        lambda source: source.register_buffer("root_buffer", torch.zeros(1)),
        "direct parameters or buffers",
    )
    check(lambda source: set_config_flag(source, "guidance_embeds"), "guidance embeddings")
    check(lambda source: setattr(source, "peft_config", {"active": True}), "PEFT")
    check(
        lambda source: setattr(source.transformer_blocks[0], "fused_projections", True),
        "fused QKV",
    )
    for name in ("zero_cond_t", "use_additional_t_cond", "use_layer3d_rope"):
        check(lambda source, name=name: set_config_flag(source, name), rf"{name}=False")
    check(lambda source: source.register_forward_hook(lambda *_args: None), "hooks must be empty")


def test_adopted_forward_rejects_every_unsupported_input_contract(monkeypatch) -> None:
    qwen_pdd = pytest.importorskip("modelopt.torch.fastgen.plugins.qwen_image_pdd")
    source = _tiny_diffusers_qwen().eval().to(dtype=torch.bfloat16)
    monkeypatch.setattr(qwen_pdd, "_require_qwen_source_identity", lambda _model: None)
    adopted = adopt_qwen_image_mr210_forward(source)

    generator = torch.Generator().manual_seed(20260715)
    hidden_states = torch.randn(2, 4, 8, generator=generator).to(torch.bfloat16)
    encoder_hidden_states = torch.randn(2, 3, 12, generator=generator).to(torch.bfloat16)
    mask = torch.tensor([[1, 1, 1], [1, 0, 0]], dtype=torch.long)
    encoder_hidden_states[~mask.bool()] = 0
    base = {
        "hidden_states": hidden_states,
        "encoder_hidden_states": encoder_hidden_states,
        "encoder_hidden_states_mask": mask,
        "timestep": torch.tensor([0.875, 0.25], dtype=torch.float32),
        "img_shapes": [[(1, 2, 2)], [(1, 2, 2)]],
        "return_dict": False,
    }

    def condition_for(candidate_mask):
        candidate_embeddings = encoder_hidden_states.clone()
        candidate_embeddings[~candidate_mask.bool()] = 0
        return candidate_embeddings

    nonbinary_mask = torch.tensor([[1, 1, 1], [1, 2, 0]], dtype=torch.long)
    nonprefix_mask = torch.tensor([[1, 1, 1], [1, 0, 1]], dtype=torch.long)
    empty_mask = torch.tensor([[1, 1, 1], [0, 0, 0]], dtype=torch.long)
    no_full_row_mask = torch.tensor([[1, 1, 0], [1, 0, 0]], dtype=torch.long)
    nonzero_padding = encoder_hidden_states.clone()
    nonzero_padding[1, 1] = 1
    cases = (
        ({"hidden_states": hidden_states.float()}, TypeError, "hidden_states"),
        ({"encoder_hidden_states": encoder_hidden_states.float()}, TypeError, "encoder_hidden"),
        ({"timestep": base["timestep"].to(torch.bfloat16)}, TypeError, "timestep"),
        (
            {
                "encoder_hidden_states": condition_for(nonbinary_mask),
                "encoder_hidden_states_mask": nonbinary_mask,
            },
            ValueError,
            "binary prefix masks",
        ),
        (
            {
                "encoder_hidden_states": condition_for(nonprefix_mask),
                "encoder_hidden_states_mask": nonprefix_mask,
            },
            ValueError,
            "binary prefix masks",
        ),
        (
            {
                "encoder_hidden_states": condition_for(empty_mask),
                "encoder_hidden_states_mask": empty_mask,
            },
            ValueError,
            "binary prefix masks",
        ),
        (
            {
                "encoder_hidden_states": condition_for(no_full_row_mask),
                "encoder_hidden_states_mask": no_full_row_mask,
            },
            ValueError,
            "binary prefix masks",
        ),
        ({"encoder_hidden_states": nonzero_padding}, ValueError, "zero padding"),
        ({"max_txt_seq_len": 2}, ValueError, "max_txt_seq_len"),
        ({"txt_seq_lens": [3, 1]}, ValueError, "txt_seq_lens"),
        ({"guidance": torch.ones(2)}, ValueError, "guidance embeddings"),
        ({"attention_kwargs": {"scale": 1.0}}, ValueError, "attention_kwargs"),
        ({"controlnet_block_samples": ()}, ValueError, "ControlNet"),
        ({"additional_t_cond": torch.ones(2)}, ValueError, "additional time"),
    )
    for override, error_type, match in cases:
        with pytest.raises(error_type, match=match):
            adopted(**(base | override))


def test_qwen_pdd_rejects_unsupported_config_condition_and_call_contracts() -> None:
    with pytest.raises(ValueError, match="num_train_timesteps=None"):
        QwenImagePDDAdapter(_config().model_copy(update={"num_train_timesteps": 1000}))
    with pytest.raises(ValueError, match="guidance_rescale"):
        QwenImagePDDAdapter(_config(), guidance_rescale=1.1)
    with pytest.raises(ValueError, match="guidance_eps"):
        QwenImagePDDAdapter(_config(), guidance_eps=0.0)
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
