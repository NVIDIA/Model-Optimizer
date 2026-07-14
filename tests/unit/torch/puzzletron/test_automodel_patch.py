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

from modelopt.torch.puzzletron.block_config import (
    AttentionConfig,
    BlockConfig,
    FFNConfig,
    MoEConfig,
)
from modelopt.torch.puzzletron.anymodel.automodel import (
    AutoModelDescriptor,
    AutoModelDescriptorFactory,
    automodel_patcher,
)
from modelopt.torch.puzzletron.plugins.automodel.load import validate_force_hf_ep
from modelopt.torch.puzzletron.plugins.automodel.patch import (
    _native_checkpoint_requires_heterogeneous_adapter,
    auto_detect_block_configs,
    load_block_configs,
)


def test_load_block_configs_bare_list(tmp_path):
    expected = [
        BlockConfig(subblock_configs=(FFNConfig(intermediate_size=128),)),
        BlockConfig(subblock_configs=(AttentionConfig(no_op=True),)),
    ]
    cfgs = [config.to_dict() for config in expected]
    path = tmp_path / "bc.json"
    path.write_text(json.dumps(cfgs))
    assert load_block_configs(path) == expected


def test_load_block_configs_wrapped_dict(tmp_path):
    expected = [BlockConfig(subblock_configs=(FFNConfig(intermediate_size=64),))]
    cfgs = [config.to_dict() for config in expected]
    path = tmp_path / "bc.json"
    path.write_text(json.dumps({"block_configs": cfgs}))
    assert load_block_configs(path) == expected


def test_load_block_configs_missing(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_block_configs(tmp_path / "does_not_exist.json")


def test_auto_detect_block_configs_present(tmp_path):
    expected = [BlockConfig(subblock_configs=(FFNConfig(intermediate_size=32),))]
    cfgs = [config.to_dict() for config in expected]
    (tmp_path / "block_configs.json").write_text(json.dumps(cfgs))
    assert auto_detect_block_configs(tmp_path) == expected


def test_auto_detect_block_configs_absent(tmp_path):
    assert auto_detect_block_configs(tmp_path) is None


def test_native_converted_teacher_does_not_require_heterogeneous_adapter(tmp_path):
    (tmp_path / "config.json").write_text(
        json.dumps({"architectures": ["LlamaForCausalLM"], "block_configs": [{}]})
    )

    assert not _native_checkpoint_requires_heterogeneous_adapter(
        tmp_path, block_configs_override=None
    )


def test_native_realized_or_explicit_override_requires_heterogeneous_adapter(tmp_path):
    (tmp_path / "config.json").write_text(
        json.dumps({"architectures": ["AnyModel"], "base_architecture": "NativeModel"})
    )

    assert _native_checkpoint_requires_heterogeneous_adapter(
        tmp_path, block_configs_override=None
    )
    assert _native_checkpoint_requires_heterogeneous_adapter(
        tmp_path, block_configs_override=[BlockConfig(subblock_configs=())]
    )


def test_validate_force_hf_ep_allows_valid_combos():
    # force_hf with no expert parallel, and the custom path with EP, are both fine.
    validate_force_hf_ep(force_hf=True, ep_size=1)
    validate_force_hf_ep(force_hf=True, ep_size=None)
    validate_force_hf_ep(force_hf=False, ep_size=4)


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
        BlockConfig(
            subblock_configs=(AttentionConfig(sliding_window_size="full"),)
        ),
        BlockConfig(
            subblock_configs=(AttentionConfig(sliding_window_size=256),)
        ),
    ]

    with automodel_patcher(Descriptor(), block_configs):
        first = NativeBlock(0, config)
        second = NativeBlock(1, config)

    assert first.sliding_window is None
    assert second.sliding_window == 256
    assert config.layer_types == ["sliding_attention", "full_attention"]
    assert config.sliding_window == 512


def test_gpt_oss_native_descriptor_reuses_generic_window_and_maps_moe_fields():
    descriptor = AutoModelDescriptorFactory.get("gpt_oss")
    assert descriptor is not None
    config = SimpleNamespace(
        layer_types=["sliding_attention", "full_attention"],
        sliding_window=512,
        num_local_experts=32,
        intermediate_size=2880,
        num_experts_per_tok=4,
    )
    block_config = BlockConfig(
        subblock_configs=(
            AttentionConfig(sliding_window_size=256),
            MoEConfig(
                num_experts=16,
                expert_intermediate_size=1440,
                top_k=2,
            ),
        )
    )

    updated = descriptor._apply_overrides(config, block_config, layer_idx=1)

    assert updated.layer_types == ["sliding_attention", "sliding_attention"]
    assert updated.sliding_window == 256
    assert updated.num_local_experts == 16
    assert updated.intermediate_size == 1440
    assert updated.num_experts_per_tok == 2


def test_qwen35_moe_has_native_descriptor_and_maps_moe_fields():
    descriptor = AutoModelDescriptorFactory.get("qwen3_5_moe")

    assert descriptor is not None
    overrides = descriptor.block_config_to_config_overrides(
        BlockConfig(
            subblock_configs=(
                MoEConfig(
                    num_experts=64,
                    expert_intermediate_size=1024,
                    top_k=4,
                ),
            )
        )
    )
    assert overrides["num_experts"] == 64
    assert overrides["moe_intermediate_size"] == 1024
    assert overrides["num_experts_per_tok"] == 4


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
