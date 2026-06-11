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

"""Tests for model-descriptor pruning mixin registries.

Bypass child initialization resolves pruning behavior by descriptor key. These
tests pin the public keys and layer prefixes that external configs depend on,
without instantiating full transformer models.
"""

from importlib import import_module

import pytest

from modelopt.torch.puzzletron.pruning.expert_removal_pruning_mixin import ExpertRemovalPruningMixIn
from modelopt.torch.puzzletron.pruning.ffn_intermediate_pruning_mixin import (
    FFNIntermediatePruningMixIn,
)
from modelopt.torch.puzzletron.pruning.kv_heads_pruning_mixin import KVHeadsPruningMixIn


def test_gpt_oss_descriptor_exposes_canonical_alias_and_kv_heads_mixins():
    pytest.importorskip("transformers.models.gpt_oss.modeling_gpt_oss")

    from modelopt.torch.puzzletron.anymodel.models.gpt_oss.gpt_oss_model_descriptor import (
        GptOssModelDescriptor,
    )

    mixins = GptOssModelDescriptor.pruning_mixins()

    assert set(mixins) == {"experts_removal", "expert_removal", "kv_heads"}
    assert mixins["experts_removal"] is mixins["expert_removal"]
    assert isinstance(mixins["experts_removal"], ExpertRemovalPruningMixIn)
    assert isinstance(mixins["kv_heads"], KVHeadsPruningMixIn)
    assert mixins["kv_heads"].layer_descriptor.attn_prefix(3) == "model.layers.3.self_attn"


def test_nemotron_h_descriptor_exposes_expert_removal_and_kv_heads_mixins():
    from modelopt.torch.puzzletron.anymodel.models.nemotron_h.nemotron_h_model_descriptor import (
        NemotronHModelDescriptor,
    )

    mixins = NemotronHModelDescriptor.pruning_mixins()

    assert set(mixins) == {"experts_removal", "kv_heads"}
    assert isinstance(mixins["experts_removal"], ExpertRemovalPruningMixIn)
    assert isinstance(mixins["kv_heads"], KVHeadsPruningMixIn)
    assert mixins["kv_heads"].layer_descriptor.attn_prefix(2) == "backbone.layers.2.mixer"


def test_nemotron_h_v2_descriptor_exposes_ffn_and_kv_heads_mixins():
    module = import_module(
        "modelopt.torch.puzzletron.anymodel.models.nemotron_h_v2.nemotron_h_v2_model_descriptor"
    )

    mixins = module.NemotronHV2ModelDescriptor.pruning_mixins()

    assert set(mixins) == {"ffn_intermediate", "kv_heads"}
    assert isinstance(mixins["ffn_intermediate"], FFNIntermediatePruningMixIn)
    assert isinstance(mixins["kv_heads"], KVHeadsPruningMixIn)
    assert mixins["kv_heads"].layer_descriptor.attn_prefix(2) == "backbone.layers.2.mixer"


def test_qwen3_vl_descriptor_exposes_expert_removal_and_kv_heads_mixins():
    pytest.importorskip("transformers.models.qwen3_vl_moe.modeling_qwen3_vl_moe")

    from modelopt.torch.puzzletron.anymodel.models.qwen3_vl.qwen3_vl_model_descriptor import (
        Qwen3VLModelDescriptor,
    )

    mixins = Qwen3VLModelDescriptor.pruning_mixins()

    assert set(mixins) == {"experts_removal", "kv_heads"}
    assert isinstance(mixins["experts_removal"], ExpertRemovalPruningMixIn)
    assert isinstance(mixins["kv_heads"], KVHeadsPruningMixIn)
    assert (
        mixins["kv_heads"].layer_descriptor.attn_prefix(2)
        == "model.language_model.layers.2.self_attn"
    )


def test_qwen3_5_descriptors_expose_ffn_and_kv_heads_mixins():
    pytest.importorskip("transformers.models.qwen3_5.modeling_qwen3_5")

    from modelopt.torch.puzzletron.anymodel.models.qwen3_5.qwen3_5_model_descriptor import (
        Qwen3P5TextModelDescriptor,
        Qwen3P5VLModelDescriptor,
    )

    text_mixins = Qwen3P5TextModelDescriptor.pruning_mixins()
    vl_mixins = Qwen3P5VLModelDescriptor.pruning_mixins()

    assert set(text_mixins) == {"ffn_intermediate", "kv_heads"}
    assert set(vl_mixins) == {"ffn_intermediate", "kv_heads"}
    assert isinstance(text_mixins["ffn_intermediate"], FFNIntermediatePruningMixIn)
    assert isinstance(vl_mixins["ffn_intermediate"], FFNIntermediatePruningMixIn)
    assert isinstance(text_mixins["kv_heads"], KVHeadsPruningMixIn)
    assert isinstance(vl_mixins["kv_heads"], KVHeadsPruningMixIn)
    assert text_mixins["ffn_intermediate"].layer_descriptor.ffn_prefix(2) == "model.layers.2.mlp"
    assert (
        vl_mixins["ffn_intermediate"].layer_descriptor.ffn_prefix(2)
        == "model.language_model.layers.2.mlp"
    )
    assert text_mixins["kv_heads"].layer_descriptor.attn_prefix(2) == "model.layers.2.self_attn"
    assert (
        vl_mixins["kv_heads"].layer_descriptor.attn_prefix(2)
        == "model.language_model.layers.2.self_attn"
    )
