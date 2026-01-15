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

from dataclasses import asdict

import torch
from modelopt.torch.puzzletron.export.MCore.puzzletron_layer_specs import (
    PuzzletronAttentionConfig,
    PuzzletronHeterogeneousTransformerConfig,
    PuzzletronMLPConfig,
    get_gpt_heterogeneous_layer_spec_puzzletron,
)
from megatron.core.transformer.spec_utils import ModuleSpec
from nemo.collections.common.tokenizers.huggingface.auto_tokenizer import (
    AutoTokenizer as NemoAutoTokenizer,
)
from nemo.collections.llm.gpt.model.base import GPTModel
from nemo.collections.llm.gpt.model.llama_nemotron import HFLlamaNemotronImporter
from nemo.lightning import io, teardown
from nemo.lightning.io.state import TransformFns
from nemo.utils.import_utils import safe_import
from transformers import AutoModelForCausalLM, AutoTokenizer


def convert_attention_config_from_cfg_object(attention_config, num_attention_heads, head_dim):
    for unsupported_key in [
        "llama4",
        "num_sink_tokens",
        "sparsify",
        "unshifted_sink",
        "use_prefill_window_in_sink_attention",
    ]:
        if hasattr(attention_config, unsupported_key) and getattr(attention_config, unsupported_key) not in [
            None,
            False,
        ]:
            #
            #        if attention_config.get(unsupported_key, None) not in [None, False]:
            raise NotImplementedError(f"{unsupported_key} is not supported")
    window_size = attention_config.window_size if hasattr(attention_config, "window_size") else None
    if window_size is not None:
        window_size = (window_size, 0)
    is_mamba = attention_config.mamba if hasattr(attention_config, "mamba") else None
    n_heads_in_group = attention_config.n_heads_in_group if hasattr(attention_config, "n_heads_in_group") else 1
    if n_heads_in_group is None:
        n_heads_in_group = 1
    return asdict(
        PuzzletronAttentionConfig(
            no_op=attention_config.no_op if hasattr(attention_config, "no_op") else False,
            replace_with_linear=(
                attention_config.replace_with_linear if hasattr(attention_config, "replace_with_linear") else False
            ),
            num_attention_heads=num_attention_heads,
            num_query_groups=num_attention_heads // n_heads_in_group,
            kv_channels=head_dim,
            window_size=window_size,
            multi_latent_attention=False,
            is_mamba=is_mamba,
            mamba_state_dim=(
                attention_config.mamba.state_dim if is_mamba and hasattr(attention_config.mamba, "state_dim") else 128
            ),
            mamba_head_dim=(
                attention_config.mamba.head_dim if is_mamba and hasattr(attention_config.mamba, "head_dim") else 64
            ),
            mamba_num_groups=(
                attention_config.mamba.num_groups if is_mamba and hasattr(attention_config.mamba, "num_groups") else 8
            ),
            mamba_num_heads=(
                attention_config.mamba.num_heads if is_mamba and hasattr(attention_config.mamba, "num_heads") else None
            ),
        )
    )


def convert_mlp_config_from_cfg_object(mlp_config, parallel_blocks, default_hidden_act):
    if parallel_blocks is not None:
        raise NotImplementedError("parallel_blocks is not supported")
    if not hasattr(mlp_config, "gated") or mlp_config.gated is False:
        raise NotImplementedError("non-gated MLP is not supported")
    if not hasattr(mlp_config, "hidden_act") or mlp_config.hidden_act not in [default_hidden_act]:
        raise NotImplementedError(f"all mlps must have the same activation ({default_hidden_act})")
    if hasattr(mlp_config, "sparsify") and mlp_config.sparsify is not None:
        raise NotImplementedError("sparsify is not supported")
    is_moe = hasattr(mlp_config, "moe") and mlp_config.moe is not None
    return asdict(
        PuzzletronMLPConfig(
            no_op=mlp_config.no_op if hasattr(mlp_config, "no_op") else False,
            replace_with_linear=mlp_config.replace_with_linear if hasattr(mlp_config, "replace_with_linear") else False,
            ffn_hidden_size=mlp_config.intermediate_size if hasattr(mlp_config, "intermediate_size") else None,
            num_moe_experts=(
                mlp_config.moe.num_local_experts if is_moe and hasattr(mlp_config.moe, "num_local_experts") else None
            ),
            moe_shared_expert_intermediate_size=(
                mlp_config.moe.shared_expert_intermediate_dim
                if is_moe and hasattr(mlp_config.moe, "shared_expert_intermediate_dim")
                else None
            ),
            moe_ffn_hidden_size=(
                mlp_config.moe.expert_intermediate_dim
                if is_moe and hasattr(mlp_config.moe, "expert_intermediate_dim")
                else None
            ),
            moe_router_topk=(
                mlp_config.moe.num_experts_per_tok if is_moe and hasattr(mlp_config.moe, "num_experts_per_tok") else 2
            ),
        )
    )
