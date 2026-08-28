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

"""Tests for Puzzletron cross-model descriptor selection and integration."""

from __future__ import annotations

import dataclasses
from types import SimpleNamespace

import pytest
import torch

from modelopt.torch.puzzletron.anymodel.capabilities import (
    AxisCapabilities,
    CapabilityValidationError,
    default_capabilities,
    validate_capabilities,
)
from modelopt.torch.puzzletron.anymodel.models.gpt_oss.gpt_oss_model_descriptor import (
    GptOssModelDescriptor,
)
from modelopt.torch.puzzletron.anymodel.models.nemotron_h.nemotron_h_model_descriptor import (
    NemotronHModelDescriptor,
)
from modelopt.torch.puzzletron.anymodel.registry import infer_descriptor_name, resolve_descriptor
from modelopt.torch.puzzletron.block_config import AttentionConfig, BlockConfig, MoEConfig
from modelopt.torch.puzzletron.stage_runner import StageResult, _preflight, run_stage


@pytest.mark.parametrize(
    ("model_type", "architecture", "expected"),
    [
        ("qwen3_5_moe", "Qwen3_5MoeForConditionalGeneration", "qwen3_5_moe"),
        ("qwen3_5_moe_text", "Qwen3_5MoeForCausalLM", "qwen3_5_moe_text"),
    ],
)
def test_cross_model_registry_resolution(model_type, architecture, expected) -> None:
    config = SimpleNamespace(model_type=model_type, architectures=[architecture])

    assert infer_descriptor_name(config)[0] == expected
    assert resolve_descriptor(config).name == expected


def _text_config(**overrides):
    values = {
        "model_type": "fixture",
        "hidden_size": 2048,
        "num_hidden_layers": 40,
        "num_attention_heads": 16,
        "num_key_value_heads": 2,
        "head_dim": 128,
        "intermediate_size": 8192,
        "num_experts": 128,
        "moe_intermediate_size": 512,
        "shared_expert_intermediate_size": 512,
        "tie_word_embeddings": False,
        "enable_moe_block": False,
    }
    values.update(overrides)
    return SimpleNamespace(**values)


def test_qwen_moe_contract_combines_moe_gdn_attention_vlm_and_mtp() -> None:
    text = _text_config(
        model_type="qwen3_5_moe_text",
        layer_types=["linear_attention", "full_attention"],
        linear_num_key_heads=16,
        linear_num_value_heads=32,
        linear_key_head_dim=128,
        linear_value_head_dim=128,
    )
    config = SimpleNamespace(
        model_type="qwen3_5_moe",
        architectures=["Qwen3_5MoeForConditionalGeneration"],
        text_config=text,
    )
    resolution = resolve_descriptor(config)
    contract = resolution.descriptor.generic_decoder_contract(config)
    capabilities = resolution.descriptor.puzzletron_capabilities(config)

    assert contract.routed_moe is not None
    assert contract.vision is not None
    assert contract.mtp is not None
    assert {
        "gdn_key_groups",
        "gdn_value_heads_per_group",
        "gdn_key_head_dim",
        "gdn_value_head_dim",
        "moe_experts",
        "moe_expert_intermediate",
        "moe_shared_expert_intermediate",
    } <= set(capabilities.axes)
    assert resolution.descriptor.sorted_teacher_layout_kwargs(text) == {
        "q_gate_row_group": 1,
        "mamba_module": "linear_attn",
        "gated_delta_net": True,
        "moe_fused_expert_subnames": (
            "experts.gate_up_proj",
            "experts.down_proj",
        ),
        "moe_fused_gate_up_subnames": ("experts.gate_up_proj",),
        "moe_fused_down_subnames": ("experts.down_proj",),
        "moe_shared_expert_subname": "shared_expert",
        "moe_shared_gate_subname": "gate_proj",
    }


def test_gpt_oss_hf_and_native_share_contract_field_mapping() -> None:
    block = BlockConfig(
        subblock_configs=(
            AttentionConfig(num_query_heads=8, num_kv_heads=2),
            MoEConfig(num_experts=16, expert_intermediate_size=1440, top_k=2),
        )
    )

    assert GptOssModelDescriptor.block_config_to_layer_overrides(block) == {
        "num_attention_heads": 8,
        "num_key_value_heads": 2,
        "num_local_experts": 16,
        "intermediate_size": 1440,
        "num_experts_per_tok": 2,
    }


def test_nemotron_hidden_width_spec_covers_hf_and_native_hybrid_tensors() -> None:
    config = SimpleNamespace(
        hidden_size=2688,
        tie_word_embeddings=False,
        moe_latent_size=640,
    )
    latent_size = 640
    spec = NemotronHModelDescriptor.embedding_pruning_spec(
        config, widths=(2688, 1344), alignment=64
    )
    state = {
        "backbone.embeddings.weight": torch.zeros(8, 2688),
        "backbone.layers.0.norm.weight": torch.zeros(2688),
        "backbone.layers.0.mixer.in_proj.weight": torch.zeros(16, 2688),
        "backbone.layers.0.mixer.out_proj.weight": torch.zeros(2688, 16),
        "backbone.layers.1.mixer.q_proj.weight": torch.zeros(8, 2688),
        "backbone.layers.1.mixer.o_proj.weight": torch.zeros(2688, 8),
        "backbone.layers.2.mixer.gate.weight": torch.zeros(4, 2688),
        "backbone.layers.2.mixer.fc1_latent_proj.weight": torch.zeros(latent_size, 2688),
        "backbone.layers.2.mixer.fc2_latent_proj.weight": torch.zeros(2688, latent_size),
        "backbone.layers.2.mixer.experts.0.up_proj.weight": torch.zeros(6, latent_size),
        "backbone.layers.2.mixer.experts.0.down_proj.weight": torch.zeros(latent_size, 6),
        "backbone.layers.2.mixer.shared_experts.up_proj.weight": torch.zeros(12, 2688),
        "backbone.layers.2.mixer.shared_experts.down_proj.weight": torch.zeros(2688, 12),
        "backbone.norm_f.weight": torch.zeros(2688),
        "lm_head.weight": torch.zeros(8, 2688),
        "model.layers.3.mixer.fc1_latent_proj.weight": torch.zeros(latent_size, 2688),
        "model.layers.3.mixer.fc2_latent_proj.weight": torch.zeros(2688, latent_size),
        "model.layers.3.mixer.experts.gate_and_up_projs": torch.zeros(4, latent_size, 6),
        "model.layers.3.mixer.experts.down_projs": torch.zeros(4, 6, latent_size),
        "mtp.layers.0.eh_proj.weight": torch.zeros(2688, 2 * 2688),
        "mtp.layers.0.enorm.weight": torch.zeros(2688),
        "mtp.layers.0.hnorm.weight": torch.zeros(2688),
        "mtp.layers.0.norm.weight": torch.zeros(2688),
        "mtp.layers.0.mixer.q_proj.weight": torch.zeros(2688, 2688),
        "mtp.layers.0.mixer.k_proj.weight": torch.zeros(4, 2688),
        "mtp.layers.0.mixer.v_proj.weight": torch.zeros(4, 2688),
        "mtp.layers.0.mixer.o_proj.weight": torch.zeros(2688, 2688),
        "mtp.layers.1.norm.weight": torch.zeros(2688),
        "mtp.layers.1.final_layernorm.weight": torch.zeros(2688),
        "mtp.layers.1.mixer.gate.weight": torch.zeros(4, 2688),
        "mtp.layers.1.mixer.fc1_latent_proj.weight": torch.zeros(latent_size, 2688),
        "mtp.layers.1.mixer.fc2_latent_proj.weight": torch.zeros(2688, latent_size),
        "mtp.layers.1.mixer.experts.0.up_proj.weight": torch.zeros(6, latent_size),
        "mtp.layers.1.mixer.experts.0.down_proj.weight": torch.zeros(latent_size, 6),
        "mtp.layers.1.mixer.shared_experts.up_proj.weight": torch.zeros(12, 2688),
        "mtp.layers.1.mixer.shared_experts.down_proj.weight": torch.zeros(2688, 12),
    }

    audit = spec.audit_state_dict(state)
    sliced = spec.slice_state_dict(state, 1344)

    assert set(state) - set(audit["handled"]) == {
        "backbone.layers.2.mixer.experts.0.up_proj.weight",
        "backbone.layers.2.mixer.experts.0.down_proj.weight",
        "model.layers.3.mixer.experts.gate_and_up_projs",
        "model.layers.3.mixer.experts.down_projs",
        "mtp.layers.1.mixer.experts.0.up_proj.weight",
        "mtp.layers.1.mixer.experts.0.down_proj.weight",
    }
    assert sliced["backbone.layers.0.mixer.in_proj.weight"].shape == (16, 1344)
    assert sliced["backbone.layers.0.mixer.out_proj.weight"].shape == (1344, 16)
    assert sliced["backbone.layers.2.mixer.fc1_latent_proj.weight"].shape == (
        latent_size,
        1344,
    )
    assert sliced["backbone.layers.2.mixer.fc2_latent_proj.weight"].shape == (
        1344,
        latent_size,
    )
    assert sliced["model.layers.3.mixer.experts.gate_and_up_projs"].shape == (
        4,
        latent_size,
        6,
    )
    assert sliced["model.layers.3.mixer.experts.down_projs"].shape == (
        4,
        6,
        latent_size,
    )
    assert sliced["mtp.layers.0.eh_proj.weight"].shape == (1344, 2 * 1344)
    assert sliced["mtp.layers.0.mixer.q_proj.weight"].shape == (2688, 1344)
    assert sliced["mtp.layers.0.mixer.o_proj.weight"].shape == (1344, 2688)
    assert sliced["mtp.layers.1.mixer.fc1_latent_proj.weight"].shape == (
        latent_size,
        1344,
    )
    assert sliced["mtp.layers.1.mixer.fc2_latent_proj.weight"].shape == (
        1344,
        latent_size,
    )
    assert spec.validate_width(1344, tp_size=2) == 1344


def test_nemotron_nano_capabilities_exclude_absent_latent_axis() -> None:
    nano_axes = NemotronHModelDescriptor.puzzletron_capabilities(
        SimpleNamespace(hidden_size=2688, moe_latent_size=None)
    ).axes
    super_axes = NemotronHModelDescriptor.puzzletron_capabilities(
        SimpleNamespace(hidden_size=2688, moe_latent_size=640)
    ).axes

    assert "moe_latent_dim" not in nano_axes
    assert "moe_latent_dim" in super_axes


def test_stage_runtime_receives_inferred_descriptor_without_persisting_override(
    monkeypatch, tmp_path
) -> None:
    resolution = resolve_descriptor(
        SimpleNamespace(model_type="llama", architectures=["LlamaForCausalLM"])
    )
    monkeypatch.setattr(
        "modelopt.torch.puzzletron.stage_runner._resolve_capabilities",
        lambda _config: resolution,
    )
    observed = {}

    def handler(config, manifest):
        observed["runtime_descriptor"] = config["_runtime"]["descriptor"]
        observed["manifest_config"] = manifest.config
        observed["manifest_effective_config"] = manifest.effective_config
        observed["resolution"] = manifest.inputs["descriptor_resolution"]
        config["_runtime"]["descriptor"] = "mutated-by-handler"
        return StageResult(
            stage="width_importance",
            status="success",
            manifest_path=tmp_path / "width_importance.json",
            message="ok",
        )

    input_config = {
        "experiment": {"dir": str(tmp_path)},
        "model": {"source": "/checkpoint", "force_hf": False},
        "bypass": {"backend": "automodel"},
        "search_space": {"axes": {}},
    }

    run_stage(
        input_config,
        "width_importance",
        handlers={"width_importance": handler},
    )

    assert observed["runtime_descriptor"] == "llama"
    assert "_runtime" not in observed["manifest_config"]
    assert observed["manifest_effective_config"]["_runtime"]["descriptor"] == "llama"
    assert "descriptor_override" not in observed["manifest_config"]["model"]
    assert observed["resolution"]["name"] == "llama"


def test_stage_preflight_requires_vllm_control_for_runtime_stats(monkeypatch) -> None:
    captured = {}
    monkeypatch.setattr(
        "modelopt.torch.puzzletron.stage_runner.validate_capabilities",
        lambda capabilities, **kwargs: captured.update(kwargs),
    )

    _preflight(
        {
            "model": {"force_hf": False},
            "parallel": {"ep": 1},
            "search_space": {"axes": {"mla_q_lora_rank": {"enabled": True, "values": [384]}}},
            "calc_subblock_stats": {"runtime_stats": {"enabled": True, "backend": "vllm"}},
        },
        SimpleNamespace(capabilities=object()),
        stage="library",
    )

    assert captured["require_vllm"] is True


def test_complete_pipeline_preflight_names_every_missing_axis_consumer() -> None:
    base = default_capabilities(descriptor_name="fixture")
    broken = AxisCapabilities(
        axis_id="broken_width",
        subblock_kind="ffn",
        field="intermediate_size",
    )
    capabilities = dataclasses.replace(base, axes={"broken_width": broken})

    with pytest.raises(CapabilityValidationError) as error:
        validate_capabilities(
            capabilities,
            enabled_axes=("broken_width",),
            force_hf=False,
            require_vllm=True,
            require_complete_pipeline=True,
        )

    message = str(error.value)
    assert "activation scorer" in message
    assert "sort_impl" in message
    assert "materialize_impl" in message
    assert "runtime_slice_impl" in message
    assert "vLLM export" in message


def test_nemotron_runtime_proxy_uses_native_layer_types_for_new_config() -> None:
    class _NewConfig:
        def __init__(self, layers_block_type=None, **kwargs):
            pass

    assert NemotronHModelDescriptor._runtime_layer_config_kwargs(
        _NewConfig, ["attention", "moe"]
    ) == {"layers_block_type": ["attention", "moe"]}


def test_nemotron_runtime_uses_exact_moe_dimensions_by_default() -> None:
    values = {
        "n_routed_experts": 128,
        "moe_intermediate_size": 1856,
        "moe_shared_expert_intermediate_size": 3712,
        "moe_latent_size": None,
        "num_experts_per_tok": 6,
        "runtime_proxy_max_experts": 16,
        "runtime_proxy_max_expert_intermediate": 512,
        "runtime_proxy_max_shared_expert_intermediate": 512,
        "runtime_proxy_max_latent": 256,
        "runtime_proxy_max_top_k": 4,
    }
    runtime = SimpleNamespace(
        model_config_value=lambda name, default=None: values.get(name, default),
    )
    block = BlockConfig(
        subblock_configs=(
            MoEConfig(
                num_experts=96,
                expert_intermediate_size=1344,
                shared_expert_intermediate_size=2560,
                top_k=4,
                latent_dim=None,
            ),
        )
    )

    assert NemotronHModelDescriptor._runtime_proxy_block_config(runtime, block) == block
