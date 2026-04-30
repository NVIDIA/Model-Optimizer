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


import torch
from _test_utils.import_helper import skip_if_no_mamba

skip_if_no_mamba()

from _test_utils.torch.megatron.models import get_mcore_mamba_hybrid_model
from _test_utils.torch.megatron.utils import run_mcore_inference
from megatron.core.parallel_state import (
    get_pipeline_model_parallel_group,
    is_pipeline_first_stage,
    is_pipeline_last_stage,
)

import modelopt.torch.nas as mtn
import modelopt.torch.utils.distributed as dist
from modelopt.torch.nas.modules.conv import _DynamicConvNd
from modelopt.torch.nas.plugins.megatron import (
    MambaDInnerHp,
    MambaNumHeadsHp,
    _DynamicColumnParallelLinear,
    _DynamicEmbedding,
    _DynamicExtendedRMSNorm,
    _DynamicMambaLayer,
    _DynamicMambaMixer,
    _DynamicMCoreLanguageModel,
    _DynamicTELayerNormColumnParallelLinear,
    _DynamicTENorm,
    _DynamicTERowParallelLinear,
)
from modelopt.torch.nas.plugins.megatron_model_stats import mcore_param_count
from modelopt.torch.nas.traced_hp import TracedHp
from modelopt.torch.opt.utils import named_dynamic_modules, named_hparams, search_space_size
from modelopt.torch.prune.plugins.mcore_minitron import (
    _param_num_dynamic,
    get_mcore_minitron_config,
)
from modelopt.torch.utils.random import centroid

SEED = 1234


def _test_mamba_search_space(rank, size):
    channel_divisor = 4
    mamba_head_dim_divisor = 4

    num_layers = size
    hybrid_override_pattern = "M" * size  # all layers are Mamba layers
    hidden_size = channel_divisor * 4
    mamba_state_dim = channel_divisor
    mamba_head_dim = mamba_head_dim_divisor * 2
    mamba_num_groups = 2
    max_sequence_length = 8
    vocab_size = 32
    batch_size = 2

    model = get_mcore_mamba_hybrid_model(
        tensor_model_parallel_size=1,
        pipeline_model_parallel_size=size,
        initialize_megatron=True,
        num_layers=num_layers,
        hybrid_override_pattern=hybrid_override_pattern,
        hidden_size=hidden_size,
        mamba_state_dim=mamba_state_dim,
        mamba_head_dim=mamba_head_dim,
        mamba_num_groups=mamba_num_groups,
        max_sequence_length=max_sequence_length,
        vocab_size=vocab_size,
        transformer_impl="transformer_engine",
        bf16=False,
    ).cuda()
    mamba_num_heads = model.decoder.layers[0].mixer.nheads

    mtn.convert(
        model,
        [
            (
                "mcore_minitron",
                get_mcore_minitron_config(
                    hidden_size_divisor=channel_divisor,
                    ffn_hidden_size_divisor=channel_divisor,
                    mamba_head_dim_divisor=mamba_head_dim_divisor,
                    num_layers_divisor=1,
                ),
            )
        ],
    )

    assert isinstance(model, _DynamicMCoreLanguageModel)
    if is_pipeline_first_stage():
        assert isinstance(model.embedding.word_embeddings, _DynamicEmbedding)
    for layer in model.decoder.layers:
        assert isinstance(layer, _DynamicMambaLayer)
        assert isinstance(layer.mixer, _DynamicMambaMixer)
        assert isinstance(layer.mixer.in_proj, _DynamicTELayerNormColumnParallelLinear)
        assert isinstance(layer.mixer.out_proj, _DynamicTERowParallelLinear)
        assert isinstance(layer.mixer.conv1d, _DynamicConvNd)
        if layer.mixer.rmsnorm:
            assert isinstance(layer.mixer.norm, _DynamicExtendedRMSNorm)
    if is_pipeline_last_stage():
        assert isinstance(model.decoder.final_norm, _DynamicTENorm)
        assert isinstance(model.output_layer, _DynamicColumnParallelLinear)

    # NOTE: `search_space_size` does not reduce across TP/PP groups
    ss_size_per_pp = search_space_size(model)
    num_heads_choices = mamba_num_heads // mamba_num_groups
    head_dim_choices = mamba_head_dim // mamba_head_dim_divisor
    hidden_size_choices = hidden_size // channel_divisor
    num_layers_per_pp = num_layers // size
    assert (
        ss_size_per_pp
        == (num_heads_choices * head_dim_choices) ** num_layers_per_pp
        * num_layers
        * hidden_size_choices
    )

    # Make sure forward pass works on min and centroid subnets
    prompt_tokens = torch.randint(0, vocab_size, (batch_size, max_sequence_length)).cuda()
    for sample_func in [min, max, centroid]:
        mtn.sample(model, sample_func)
        output = run_mcore_inference(model, prompt_tokens, model.hidden_size)
        assert output.shape == (batch_size, max_sequence_length, vocab_size)

    # Make sure export and forward pass works on centroid model
    mtn.export(model)
    _ = run_mcore_inference(model, prompt_tokens, model.hidden_size)
    assert not any(named_dynamic_modules(model))


def test_mamba_search_space(dist_workers):
    dist_workers.run(_test_mamba_search_space)


def _test_param_num_dynamic_matches_formula(rank, size):
    """Sample min-width subnet and assert _param_num_dynamic matches the analytical formula.

    Uses "ME*-" to exercise all four block types (Mamba, MoE, Attention, dense MLP).
    Depth pruning is excluded from the formula override because _param_num_dynamic counts all
    physical layers on each PP rank (actual depth pruning requires drop_mcore_language_model_layers).
    """
    assert size <= 4, "test_param_num_dynamic_matches_formula only configured for upto 4 GPUs"
    channel_divisor = 4
    mamba_head_dim_divisor = 4

    # 4-layer hybrid covering all block types
    num_layers = 4
    hybrid_override_pattern = "ME*-"
    hidden_size = 16
    ffn_hidden_size = 32
    num_attention_heads = 16
    num_query_groups = 4
    mamba_state_dim = 4
    mamba_num_heads = 8
    mamba_head_dim = 16
    mamba_num_groups = 2
    num_moe_experts = 8
    moe_ffn_hidden_size = 16
    moe_shared_expert_intermediate_size = 16
    vocab_size = 32

    model = get_mcore_mamba_hybrid_model(
        tensor_model_parallel_size=1,
        pipeline_model_parallel_size=size,
        initialize_megatron=True,
        num_layers=num_layers,
        hybrid_override_pattern=hybrid_override_pattern,
        hidden_size=hidden_size,
        num_attention_heads=num_attention_heads,
        num_query_groups=num_query_groups,
        ffn_hidden_size=ffn_hidden_size,
        mamba_state_dim=mamba_state_dim,
        mamba_num_heads=mamba_num_heads,
        mamba_head_dim=mamba_head_dim,
        mamba_num_groups=mamba_num_groups,
        moe_ffn_hidden_size=moe_ffn_hidden_size,
        moe_shared_expert_intermediate_size=moe_shared_expert_intermediate_size,
        num_moe_experts=num_moe_experts,
        vocab_size=vocab_size,
        transformer_impl="transformer_engine",
        bf16=False,
    ).cuda()

    mtn.convert(
        model,
        [
            (
                "mcore_minitron",
                get_mcore_minitron_config(
                    hidden_size_divisor=channel_divisor,
                    ffn_hidden_size_divisor=channel_divisor,
                    mamba_head_dim_divisor=mamba_head_dim_divisor,
                    num_moe_experts_divisor=1,
                    num_layers_divisor=1,
                ),
            )
        ],
    )

    mtn.sample(model, min)

    hybrid_key = (
        "hybrid_override_pattern"
        if hasattr(model, "hybrid_override_pattern")
        else "hybrid_layer_pattern"
    )
    hybrid_layer_pattern = getattr(model, hybrid_key)

    # Build a flat {hparam_name: active_value} dict (same convention as the searcher's ss_config).
    # get_subnet_config() returns full-path keys, which mcore_param_count does not understand.
    # With PP > 1 each rank only holds a subset of layer types, so gather across the PP group
    # to get the complete set of hparam overrides for the global formula.
    # Exclude num_layers: _param_num_dynamic counts all physical layers regardless of the depth
    # hparam; actual depth pruning (drop_mcore_language_model_layers) is not called here.
    local_config = {
        n.split(".")[-1]: hp.active
        for n, hp in named_hparams(model, configurable=True)
        if n.split(".")[-1] != "num_layers"
    }
    width_ss_config = dist.DistributedProcessGroup.get_dist_syncd_obj(
        local_config,
        dist.DistributedProcessGroup(get_pipeline_model_parallel_group()),
        op=lambda all_rank_configs: {k: v for d in all_rank_configs for k, v in d.items()},
    )
    formula_total, _ = mcore_param_count(
        model.config,
        model.vocab_size,
        model.share_embeddings_and_output_weights,
        hybrid_layer_pattern=hybrid_layer_pattern,
        **width_ss_config,
    )
    dynamic_count = int(_param_num_dynamic(model))

    assert formula_total == dynamic_count, (
        f"Formula ({formula_total:,}) != _param_num_dynamic ({dynamic_count:,}) "
        f"for min-width subnet {width_ss_config} (PP={size})"
    )


def test_param_num_dynamic_matches_formula(dist_workers):
    dist_workers.run(_test_param_num_dynamic_matches_formula)


def test_mamba_num_heads_hp():
    num_heads = MambaNumHeadsHp(8, ngroups=2)  # 4 heads per group
    assert num_heads.choices == [2, 4, 6, 8]
    assert num_heads.active_slice == slice(8)

    num_heads.active = 4  # 2 heads per group
    assert num_heads.active_slice.tolist() == [0, 1, 4, 5]

    num_heads_ranking = torch.tensor([1, 0, 3, 2, 4, 7, 6, 5])
    num_heads_ranking.argsort = lambda *args, **kwargs: num_heads_ranking
    num_heads._get_importance = lambda: num_heads_ranking
    num_heads.enforce_order(num_heads.importance.argsort(descending=True))
    assert num_heads.active_slice.tolist() == [1, 0, 4, 7]


def test_mamba_d_inner_hp():
    num_heads = TracedHp([2, 4, 6, 8])
    head_dim = TracedHp([1, 2, 3])
    d_inner = MambaDInnerHp(num_heads, head_dim)

    assert d_inner.choices == [2, 4, 6, 8, 12, 16, 18, 24]
    assert d_inner.active_slice == slice(24)

    # Set importance and slice order
    num_heads._get_importance = lambda: torch.tensor([2.2, 0.1, 1.1, 2.1, 3.0, 2.0, 0.0, 1.0])
    head_dim._get_importance = lambda: torch.tensor([2.0, 3.0, 1.0])
    num_heads.enforce_order(torch.argsort(num_heads.importance, descending=True))
    head_dim.enforce_order(torch.argsort(head_dim.importance, descending=True))
    assert num_heads.active_slice.tolist() == [4, 0, 3, 5, 2, 7, 1, 6]
    assert head_dim.active_slice.tolist() == [1, 0, 2]

    # check if we get correct selection of sorted + pruned heads after setting active values
    num_heads.active = 6  # top 6 heads
    head_dim.active = 2  # top 2 dims per head
    assert d_inner.active == 12  # (6 * 2)
    assert d_inner.active_slice.tolist() == [13, 12, 1, 0, 10, 9, 16, 15, 7, 6, 22, 21]
