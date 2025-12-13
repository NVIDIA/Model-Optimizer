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
from functools import partial

import pytest
import torch
from _test_utils.import_helper import skip_if_no_megatron

skip_if_no_megatron(apex_or_te_required=True)

from _test_utils.torch.distributed.utils import spawn_multiprocess_job
from _test_utils.torch.megatron.models import get_mcore_gpt_model

import modelopt.torch.speculative as mtsp
from modelopt.torch.speculative.plugins.megatron_eagle import _DynamicEagleGPTModel
from modelopt.torch.speculative.plugins.megatron_medusa import _DynamicMedusaGPTModel

ALGO_TO_CONFIG = {
    "eagle1": mtsp.config.EAGLE1_DEFAULT_CFG,
    "eagle3": mtsp.config.EAGLE3_DEFAULT_CFG,
    "eagle-mtp": mtsp.config.EAGLE_MTP_DEFAULT_CFG,
}


def _test_speculative_gpt_model(
    algo, num_medusa_heads_or_eagle_layers, activation_func, normalization, rank, size
):
    num_attention_heads = 8
    num_query_groups = size
    max_sequence_length = 32
    vocab_size = 64
    batch_size = 2

    model = get_mcore_gpt_model(
        tensor_model_parallel_size=size,
        pipeline_model_parallel_size=1,
        initialize_megatron=True,
        num_attention_heads=num_attention_heads,
        num_query_groups=num_query_groups,
        max_sequence_length=max_sequence_length,
        vocab_size=vocab_size,
        activation_func=activation_func,
        normalization=normalization,
    ).cuda()

    if algo == "medusa":
        config = {
            "medusa_num_heads": num_medusa_heads_or_eagle_layers,
            "medusa_num_layers": 1,
        }

        model = mtsp.convert(model, [("medusa", config)])

        # Type checking
        assert isinstance(model, _DynamicMedusaGPTModel)
    elif algo in {"eagle1", "eagle3"}:
        mtsp_config = ALGO_TO_CONFIG[algo]

        mtsp_config["config"]["eagle_architecture_config"]["num_hidden_layers"] = (
            num_medusa_heads_or_eagle_layers
        )
        mtsp_config["config"]["eagle_architecture_config"]["hidden_size"] = model.config.hidden_size
        mtsp_config["config"]["eagle_architecture_config"]["vocab_size"] = model.vocab_size
        mtsp_config["config"]["eagle_architecture_config"]["draft_vocab_size"] = model.vocab_size

        model = mtsp.convert(model, mtsp_config)

        # Type checking
        assert isinstance(model, _DynamicEagleGPTModel)
    else:
        raise ValueError("Only algo={eagle1, eagle3, medusa} are supported!")

    if algo == "eagle3":
        first_layer = model.eagle_module.decoder.layers[0]
        last_layer = model.eagle_module.decoder.layers[-1]
        # Eagle3 QKV input_dim is 2x of hidden_size
        assert (
            first_layer.self_attention.linear_qkv.weight.shape[-1] == model.config.hidden_size * 2
        )
        # Eagle3 attention has a forward_pre_hook to handle additional features to be concatenated
        assert len(first_layer.self_attention._forward_pre_hooks) > 0
        # Eagle3 last layer has a forward hook to extrat the pre_norm hidden_state
        assert len(last_layer._forward_hooks) > 0
    elif algo == "eagle1":
        first_layer = model.eagle_module.decoder.layers[0]
        last_layer = model.eagle_module.decoder.layers[-1]
        # Eagle1 QKV input_dim the same as hidden_size
        assert first_layer.self_attention.linear_qkv.weight.shape[-1] == model.config.hidden_size
        # No forward_hook or forward_pre_hook are needed
        assert len(first_layer.self_attention._forward_pre_hooks) == 0
        assert len(last_layer._forward_hooks) == 0

    # Bfloat16
    model = model.to(torch.bfloat16)

    # Prepare inputs for forward.
    prompt_tokens = torch.randint(0, vocab_size, (batch_size, max_sequence_length)).cuda()
    attention_mask = torch.tril(torch.ones((1, 1, max_sequence_length, max_sequence_length))).cuda()
    position_ids = torch.arange(max_sequence_length, dtype=torch.long).unsqueeze(0).cuda()
    attention_mask = attention_mask < 0.5

    # When no labels provided, model.forward should return logits[b, s, vocab / tp]
    logits = model(prompt_tokens, position_ids, attention_mask, labels=None)
    assert logits.shape[0] == batch_size
    assert logits.shape[1] == max_sequence_length
    assert logits.shape[2] == vocab_size / size

    if algo == "medusa":
        # When label provided, model.forward should return
        # medusa_loss[b, s * (num_medusa_heads + 1), b]
        labels = torch.randint(
            0,
            vocab_size,
            (batch_size, max_sequence_length),
        ).cuda()
        medusa_loss = model(prompt_tokens, position_ids, attention_mask, labels=labels)

        assert medusa_loss.shape[0] == batch_size
        assert medusa_loss.shape[1] == max_sequence_length
    elif algo in {"eagle1", "eagle3"}:
        labels = torch.randint(0, vocab_size, (batch_size, max_sequence_length)).cuda()
        eagle_loss = model(prompt_tokens, position_ids, attention_mask, labels=labels)

        assert eagle_loss.shape[0] == batch_size
        assert eagle_loss.shape[1] == max_sequence_length


@pytest.mark.parametrize(
    ("algo", "num_medusa_heads_or_eagle_layers", "activation_func", "normalization"),
    [
        ("eagle1", 1, "squared_relu", "LayerNorm"),  # MHA
        ("eagle1", 2, "swiglu", "RMSNorm"),  # GQA
        ("eagle3", 1, "swiglu", "RMSNorm"),  # GQA
        ("eagle3", 2, "swiglu", "RMSNorm"),  # GQA
        ("medusa", 1, "squared_relu", "LayerNorm"),  # MHA
        ("medusa", 2, "swiglu", "RMSNorm"),  # GQA
    ],
)
def test_speculative_gpt_model(
    algo, num_medusa_heads_or_eagle_layers, activation_func, normalization
):
    if algo == "eagle":
        try:
            import megatron.core.post_training  # noqa: F401
        except ImportError:
            pytest.skip("megatron.core.post_training not found")

    spawn_multiprocess_job(
        size=torch.cuda.device_count(),
        job=partial(
            _test_speculative_gpt_model,
            algo,
            num_medusa_heads_or_eagle_layers,
            activation_func,
            normalization,
        ),
        backend="nccl",
    )
