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


import pytest
import torch
from _test_utils.import_helper import skip_if_no_mamba

skip_if_no_mamba()

from _test_utils.torch.megatron.models import get_mcore_hybrid_model
from _test_utils.torch.megatron.utils import run_mcore_inference
from megatron.core.parallel_state import is_pipeline_first_stage, is_pipeline_last_stage

import modelopt.torch.nas as mtn
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
from modelopt.torch.nas.traced_hp import TracedHp
from modelopt.torch.opt.utils import named_dynamic_modules, search_space_size
from modelopt.torch.prune.plugins.mcore_minitron import get_mcore_minitron_config
from modelopt.torch.utils.random import centroid

SEED = 1234
CHANNEL_DIVISOR = 4
MAMBA_HEAD_DIM_DIVISOR = 4


def _get_mamba_search_space(size):
    model = get_mcore_hybrid_model(
        tensor_model_parallel_size=1,
        pipeline_model_parallel_size=size,
        initialize_megatron=True,
        num_layers=size,
        hybrid_layer_pattern="M" * size,
        hidden_size=CHANNEL_DIVISOR * 4,
        mamba_state_dim=CHANNEL_DIVISOR,
        mamba_head_dim=MAMBA_HEAD_DIM_DIVISOR * 2,
        mamba_num_groups=2,
        max_sequence_length=8,
        vocab_size=32,
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
                    hidden_size_divisor=CHANNEL_DIVISOR,
                    ffn_hidden_size_divisor=CHANNEL_DIVISOR,
                    mamba_head_dim_divisor=MAMBA_HEAD_DIM_DIVISOR,
                    num_layers_divisor=1,
                ),
            )
        ],
    )
    return model, mamba_num_heads


def _mamba_subnet_outputs(model):
    prompt_tokens = torch.randint(0, model.vocab_size, (2, model.max_sequence_length)).cuda()
    for sample_func in [min, max, centroid]:
        mtn.sample(model, sample_func)
        yield run_mcore_inference(model, prompt_tokens, model.hidden_size)
    mtn.export(model)
    yield run_mcore_inference(model, prompt_tokens, model.hidden_size)


def _compile_mamba_kernels(rank, size):
    model, _ = _get_mamba_search_space(size)
    for _ in _mamba_subnet_outputs(model):
        pass
    torch.cuda.synchronize()


# Keep compilation first so the functional test reuses the module's warmed workers.
@pytest.mark.timeout(240)
def test_mamba_kernel_compilation(dist_workers):
    dist_workers.run(_compile_mamba_kernels)


def _test_mamba_search_space(rank, size):
    model, mamba_num_heads = _get_mamba_search_space(size)

    assert isinstance(model, _DynamicMCoreLanguageModel)
    if is_pipeline_first_stage():
        assert isinstance(model.embedding.word_embeddings, _DynamicEmbedding)
    for layer in model.decoder.layers:
        assert isinstance(layer, _DynamicMambaLayer)
        assert isinstance(layer.mixer, _DynamicMambaMixer)
        assert isinstance(layer.mixer.in_proj, _DynamicTELayerNormColumnParallelLinear)
        assert isinstance(layer.mixer.out_proj, _DynamicTERowParallelLinear)
        # The mixer's conv is raw parameters (dynamically sliced), not a module
        assert {"conv1d_weight", "conv1d_bias"} <= layer.mixer._dm_attribute_manager.da_keys()
        if layer.mixer.rmsnorm:
            assert isinstance(layer.mixer.norm, _DynamicExtendedRMSNorm)
    if is_pipeline_last_stage():
        assert isinstance(model.decoder.final_norm, _DynamicTENorm)
        assert isinstance(model.output_layer, _DynamicColumnParallelLinear)

    # NOTE: `search_space_size` does not reduce across TP/PP groups
    ss_size_per_pp = search_space_size(model)
    num_heads_choices = mamba_num_heads // model.config.mamba_num_groups
    head_dim_choices = model.config.mamba_head_dim // MAMBA_HEAD_DIM_DIVISOR
    hidden_size_choices = model.config.hidden_size // CHANNEL_DIVISOR
    num_layers_per_pp = model.config.num_layers // size
    assert (
        ss_size_per_pp
        == (num_heads_choices * head_dim_choices) ** num_layers_per_pp
        * model.config.num_layers
        * hidden_size_choices
    )

    for output in _mamba_subnet_outputs(model):
        assert output.shape == (2, model.max_sequence_length, model.vocab_size)
    assert not any(named_dynamic_modules(model))


def test_mamba_search_space(dist_workers):
    dist_workers.run(_test_mamba_search_space)


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
