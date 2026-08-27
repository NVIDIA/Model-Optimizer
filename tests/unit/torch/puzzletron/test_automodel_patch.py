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

"""Unit tests for the AutoModel load helpers (no NeMo / GPU required).

Covers the pure-Python block-config helpers and the force_hf/EP guard. The
actual NeMo ``from_pretrained`` patching and forward-parity are exercised by an
in-container GPU integration check (see the plan, P2).
"""

import json
from types import SimpleNamespace

import pytest
import torch

from modelopt.torch.puzzletron.anymodel.automodel import (
    AutoModelDescriptor,
    AutoModelDescriptorFactory,
    automodel_patcher,
)
from modelopt.torch.puzzletron.block_config import (
    AttentionConfig,
    BlockConfig,
    FFNConfig,
    MoEConfig,
)
from modelopt.torch.puzzletron.plugins.automodel.load import validate_force_hf_ep
from modelopt.torch.puzzletron.plugins.automodel.patch import (
    _cast_stage_local_model_to_dtype,
    _native_checkpoint_requires_heterogeneous_adapter,
    auto_detect_block_configs,
    load_block_configs,
)


def test_stage_local_dtype_cast_preserves_protected_fp32_submodules():
    class Model(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.projection = torch.nn.Linear(2, 2, bias=False)
            self._fp32_params = torch.nn.Linear(2, 2, bias=False)

    model = Model()

    _cast_stage_local_model_to_dtype(model, torch.bfloat16)

    assert model.projection.weight.dtype is torch.bfloat16
    assert model._fp32_params.weight.dtype is torch.float32


def test_load_block_configs_wrapped_dict(tmp_path):
    expected = [BlockConfig(subblock_configs=(FFNConfig(intermediate_size=64),))]
    cfgs = [config.to_dict() for config in expected]
    path = tmp_path / "bc.json"
    path.write_text(json.dumps({"block_configs": cfgs}))
    assert load_block_configs(path) == expected


def test_auto_detect_block_configs_present(tmp_path):
    expected = [BlockConfig(subblock_configs=(FFNConfig(intermediate_size=32),))]
    cfgs = [config.to_dict() for config in expected]
    (tmp_path / "block_configs.json").write_text(json.dumps(cfgs))
    assert auto_detect_block_configs(tmp_path) == expected


def test_native_realized_or_explicit_override_requires_heterogeneous_adapter(tmp_path):
    (tmp_path / "config.json").write_text(
        json.dumps({"architectures": ["AnyModel"], "base_architecture": "NativeModel"})
    )

    assert _native_checkpoint_requires_heterogeneous_adapter(tmp_path, block_configs_override=None)
    assert _native_checkpoint_requires_heterogeneous_adapter(
        tmp_path, block_configs_override=[BlockConfig(subblock_configs=())]
    )


def test_validate_force_hf_ep_rejects_force_hf_with_ep():
    with pytest.raises(ValueError, match="ep_size"):
        validate_force_hf_ep(force_hf=True, ep_size=2)


def test_native_descriptor_applies_per_layer_window_for_layer_idx_first_constructor():
    class NativeBlock:
        def __init__(self, layer_idx, config, backend=None):
            del backend
            self.layer_idx = layer_idx
            self.config = config
            self.sliding_window = (
                config.sliding_window
                if config.layer_types[layer_idx] == "sliding_attention"
                else None
            )

    class Descriptor(AutoModelDescriptor):
        @staticmethod
        def decoder_layer_cls():
            return NativeBlock

    config = SimpleNamespace(
        layer_types=["sliding_attention", "full_attention"],
        sliding_window=512,
    )
    block_configs = [
        BlockConfig(subblock_configs=(AttentionConfig(sliding_window_size="full"),)),
        BlockConfig(subblock_configs=(AttentionConfig(sliding_window_size=256),)),
    ]

    with automodel_patcher(Descriptor(), block_configs):
        first = NativeBlock(0, config)
        second = NativeBlock(1, config)

    assert first.sliding_window is None
    assert second.sliding_window == 256
    assert config.layer_types == ["sliding_attention", "full_attention"]
    assert config.sliding_window == 512


def test_qwen_native_constructor_preserves_mtp_layer_types_outside_decoder_configs():
    descriptor = AutoModelDescriptorFactory.get("qwen3_5_moe")
    config = SimpleNamespace(layer_types=["full_attention", "full_attention"])

    def original_init(self, layer_idx, config, moe_config, backend):
        del self, moe_config, backend
        assert config.layer_types[layer_idx] == "full_attention"

    patched = descriptor.make_patched_init(
        original_init,
        [BlockConfig(subblock_configs=(AttentionConfig(num_kv_heads=2),))],
    )

    patched(SimpleNamespace(), 1, config, None, None)
    assert config.layer_types == ["full_attention", "full_attention"]


def test_nemotron_native_state_dict_adapter_uses_per_layer_expert_geometry():
    from nemo_automodel.components.models.nemotron_v3.state_dict_adapter import (
        NemotronV3StateDictAdapter,
    )

    descriptor = AutoModelDescriptorFactory.get("nemotron_v3")
    block_config = BlockConfig(
        subblock_configs=(
            MoEConfig(
                num_experts=2,
                expert_intermediate_size=2,
                top_k=1,
            ),
        )
    )
    adapter = NemotronV3StateDictAdapter(
        config=SimpleNamespace(num_hidden_layers=1),
        moe_config=SimpleNamespace(
            n_routed_experts=4,
            n_activated_experts=2,
            moe_inter_dim=3,
            expert_activation="relu2",
        ),
        backend=SimpleNamespace(experts="grouped"),
        dtype=torch.float32,
    )
    native_up = torch.zeros(2, 4, 2)

    with descriptor.native_state_dict_adapter_context([block_config]):
        split = adapter.convert_single_tensor_to_hf(
            "model.layers.0.mixer.experts.gate_and_up_projs",
            native_up,
        )
        assert len(split) == 2

        hf_state = {}
        for expert_idx in range(2):
            hf_state[f"model.layers.0.mixer.experts.{expert_idx}.up_proj.weight"] = torch.zeros(
                2, 4
            )
            hf_state[f"model.layers.0.mixer.experts.{expert_idx}.down_proj.weight"] = torch.zeros(
                4, 2
            )
        merged = adapter._from_hf_w_merged_experts(hf_state)

    assert merged["model.layers.0.mixer.experts.gate_and_up_projs"].shape == (
        2,
        4,
        2,
    )
    assert merged["model.layers.0.mixer.experts.down_projs"].shape == (2, 2, 4)
    assert adapter.moe_config.n_routed_experts == 4
    assert adapter.moe_config.moe_inter_dim == 3
