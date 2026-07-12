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

from functools import partial

import torch
import transformers
from _test_utils.torch.megatron.models import get_mcore_mamba_hybrid_model
from _test_utils.torch.transformers_models import create_tiny_nemotron_h_dir
from safetensors import safe_open

from modelopt.torch.export import export_mcore_gpt_to_hf, import_mcore_gpt_from_hf
from modelopt.torch.export.plugins.megatron_importer import _get_mamba_conv1d


class _Mixer:
    pass


def test_get_mamba_conv1d_returns_legacy_module():
    mixer = _Mixer()
    mixer.conv1d = torch.nn.Conv1d(4, 4, 3)

    assert _get_mamba_conv1d(mixer) is mixer.conv1d


def test_get_mamba_conv1d_wraps_direct_params():
    mixer = _Mixer()
    mixer.conv1d_weight = torch.nn.Parameter(torch.zeros(4, 1, 3))
    mixer.conv1d_bias = torch.nn.Parameter(torch.zeros(4))

    conv1d = _get_mamba_conv1d(mixer)
    new_weight = torch.ones_like(mixer.conv1d_weight)
    new_bias = torch.ones_like(mixer.conv1d_bias)
    conv1d.load_state_dict({"weight": new_weight, "bias": new_bias})

    assert set(conv1d.state_dict()) == {"weight", "bias"}
    assert conv1d.weight is mixer.conv1d_weight
    assert conv1d.bias is mixer.conv1d_bias
    torch.testing.assert_close(mixer.conv1d_weight, new_weight)
    torch.testing.assert_close(mixer.conv1d_bias, new_bias)


# NemotronH-style MoE + Mamba hybrid: exercise import/export of a model with Mamba
# (conv1d/in_proj/out_proj), attention, MLP, and MoE expert layers.


def _build_mcore_nemotron_h(config, size, initialize=True):
    return get_mcore_mamba_hybrid_model(
        tensor_model_parallel_size=size,
        pipeline_model_parallel_size=1,
        initialize_megatron=initialize,
        num_layers=config.num_hidden_layers,
        hybrid_override_pattern=config.hybrid_override_pattern,
        hidden_size=config.hidden_size,
        num_attention_heads=config.num_attention_heads,
        num_query_groups=config.num_key_value_heads,
        ffn_hidden_size=config.intermediate_size,
        max_sequence_length=config.max_position_embeddings,
        vocab_size=config.vocab_size,
        mamba_state_dim=config.ssm_state_size,
        mamba_num_heads=config.mamba_num_heads,
        mamba_head_dim=config.mamba_head_dim,
        mamba_num_groups=config.n_groups,
        num_moe_experts=config.n_routed_experts,
        moe_ffn_hidden_size=config.moe_intermediate_size,
        moe_shared_expert_intermediate_size=config.moe_shared_expert_intermediate_size,
    ).cuda()


def _test_nemotron_h_export_import(tmp_path, model_dir, rank, size):
    config = transformers.AutoConfig.from_pretrained(model_dir)

    # Export a Mamba + MoE hybrid mcore model to HF safetensors.
    model = _build_mcore_nemotron_h(config, size)
    export_dir = tmp_path / "export"
    export_mcore_gpt_to_hf(model, str(model_dir), dtype=torch.bfloat16, export_dir=str(export_dir))

    if rank == 0:
        keys = []
        for sf in export_dir.glob("*.safetensors"):
            with safe_open(str(sf), framework="pt", device="cpu") as f:
                keys.extend(f.keys())
        # Mamba mixer (conv1d is the layer fixed by _get_mamba_conv1d), plus MoE experts.
        assert any("mixer.conv1d" in k for k in keys), "mamba conv1d weights missing from export"
        assert any("mixer.in_proj" in k for k in keys), "mamba in_proj weights missing from export"
        assert any("mixer.experts" in k for k in keys), "moe expert weights missing from export"

    # Import the exported checkpoint back into a fresh mcore model
    # (megatron is already initialized from the first build).
    imported = _build_mcore_nemotron_h(config, size, initialize=False)
    import_mcore_gpt_from_hf(imported, str(export_dir))


def test_nemotron_h_export_import(dist_workers_size_1, tmp_path):
    model_dir = create_tiny_nemotron_h_dir(tmp_path)
    dist_workers_size_1.run(partial(_test_nemotron_h_export_import, tmp_path, model_dir))
