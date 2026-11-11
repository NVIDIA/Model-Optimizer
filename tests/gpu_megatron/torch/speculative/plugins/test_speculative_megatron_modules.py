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
from copy import deepcopy
from functools import partial

import pytest
import torch
from _test_utils.torch.megatron.models import get_mcore_gpt_model
from megatron.core.parallel_state import get_tensor_model_parallel_group

import modelopt.torch.speculative as mtsp
from modelopt.torch.speculative.plugins.megatron_eagle import _DynamicEagleGPTModel

ALGO_TO_CONFIG = {
    "eagle3": mtsp.config.EAGLE3_DEFAULT_CFG,
    "eagle-mtp": mtsp.config.EAGLE_MTP_DEFAULT_CFG,
}


def _test_speculative_gpt_model(
    algo,
    num_layers,
    activation_func,
    normalization,
    mix_hidden_states_mode,
    sequence_parallel,
    rank,
    size,
):
    num_attention_heads = 8
    num_query_groups = size
    max_sequence_length = 32
    vocab_size = 64
    batch_size = 2

    create_model = partial(
        get_mcore_gpt_model,
        tensor_model_parallel_size=size,
        pipeline_model_parallel_size=1,
        num_attention_heads=num_attention_heads,
        num_query_groups=num_query_groups,
        max_sequence_length=max_sequence_length,
        vocab_size=vocab_size,
        activation_func=activation_func,
        normalization=normalization,
        hidden_dropout=0.0,
        attention_dropout=0.0,
    )
    model = create_model(initialize_megatron=True, sequence_parallel=sequence_parallel).cuda()

    if algo in ("eagle1", "eagle3"):
        mtsp_config = deepcopy(ALGO_TO_CONFIG["eagle3"])
        mtsp_config["config"]["eagle_architecture_config"]["use_aux_hidden_state"] = (
            algo == "eagle3"
        )

        mtsp_config["config"]["eagle_architecture_config"]["num_hidden_layers"] = num_layers
        mtsp_config["config"]["eagle_architecture_config"]["hidden_size"] = model.config.hidden_size
        mtsp_config["config"]["eagle_architecture_config"]["vocab_size"] = model.vocab_size
        mtsp_config["config"]["eagle_architecture_config"]["draft_vocab_size"] = model.vocab_size

        if mix_hidden_states_mode is not None:
            mtsp_config["config"].update(
                eagle_mix_hidden_states=True,
                eagle_mix_hidden_states_mode=(
                    "sparse_replace"
                    if mix_hidden_states_mode == "unknown"
                    else mix_hidden_states_mode
                ),
                eagle_ttt_steps=4,
            )

        model = mtsp.convert(model, mtsp_config)
        if algo == "eagle3":
            # The duplicated FC leaves initialization to the training/checkpoint loader.
            torch.nn.init.normal_(
                model.eagle_module.fc.weight,
                std=0.02,
                generator=torch.Generator(device="cuda").manual_seed(1234),
            )

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

    # Bfloat16
    model = model.to(torch.bfloat16)

    # Prepare inputs for forward.
    prompt_tokens = torch.randint(0, vocab_size, (batch_size, max_sequence_length)).cuda()
    attention_mask = torch.tril(torch.ones((1, 1, max_sequence_length, max_sequence_length))).cuda()
    position_ids = torch.arange(max_sequence_length, dtype=torch.long).unsqueeze(0).cuda()
    attention_mask = attention_mask < 0.5

    if mix_hidden_states_mode == "unknown":
        model.eagle_mix_hidden_states_mode = "unknown"
        with pytest.raises(ValueError):
            model(prompt_tokens, position_ids, attention_mask, labels=prompt_tokens)
        return

    # When no labels provided, model.forward should return logits[b, s, vocab / tp]
    logits = model(prompt_tokens, position_ids, attention_mask, labels=None)
    assert logits.shape[0] == batch_size
    assert logits.shape[1] == max_sequence_length
    assert logits.shape[2] == vocab_size / size

    if algo in ("eagle1", "eagle3"):
        labels = torch.randint(0, vocab_size, (batch_size, max_sequence_length)).cuda()

        def check_tp_inputs(module, inputs):
            hidden_states = inputs[1].detach().contiguous()
            gathered = [torch.empty_like(hidden_states) for _ in range(size)]
            torch.distributed.all_gather(
                gathered, hidden_states, group=get_tensor_model_parallel_group()
            )
            for other in gathered[1:]:
                torch.testing.assert_close(other, gathered[0], rtol=0, atol=0)

        # Replicated draft inputs must agree even when ranks consume different random numbers.
        handle = None
        if mix_hidden_states_mode == "uniform_layer_sample" and not sequence_parallel:
            handle = model.eagle_module.register_forward_pre_hook(check_tp_inputs)
            torch.cuda.manual_seed(1234 + rank)
        mixing_rng_state = torch.cuda.get_rng_state()
        try:
            eagle_loss = model(prompt_tokens, position_ids, attention_mask, labels=labels)
        finally:
            if handle is not None:
                handle.remove()

        assert eagle_loss.shape[0] == batch_size
        assert eagle_loss.shape[1] == max_sequence_length
        if mix_hidden_states_mode is not None:
            assert torch.isfinite(eagle_loss).all()
            eagle_loss.mean().backward()
            # EAGLE1 carries gradients through TTT outputs; EAGLE3 detaches them.
            parameter_name = (
                "fc.weight"
                if algo == "eagle3"
                else "decoder.layers.0.self_attention.linear_qkv.weight"
            )
            parameter = model.eagle_module.get_parameter(parameter_name)
            grad = parameter.grad
            assert grad is not None and torch.isfinite(grad).all()
            assert torch.count_nonzero(grad) > 0

            if sequence_parallel:
                reference = create_model(initialize_megatron=False, sequence_parallel=False).cuda()
                reference = mtsp.convert(reference, deepcopy(mtsp_config)).to(torch.bfloat16)
                reference.load_state_dict(model.state_dict())
                torch.cuda.set_rng_state(mixing_rng_state)
                reference_loss = reference(
                    prompt_tokens, position_ids, attention_mask, labels=labels
                )
                reference_loss.mean().backward()
                torch.testing.assert_close(eagle_loss, reference_loss, rtol=1e-2, atol=1e-3)

                # Match Megatron's sum for replicated SP parameters; QKV weights are TP-sharded.
                sp_grad = grad.detach().float().clone()
                if getattr(parameter, "sequence_parallel", False):
                    torch.distributed.all_reduce(sp_grad, group=get_tensor_model_parallel_group())
                reference_grad = reference.eagle_module.get_parameter(parameter_name).grad.float()
                # Allow BF16 accumulation differences, but reject TP-scaled gradients.
                assert (sp_grad - reference_grad).norm() <= 0.01 * reference_grad.norm()


@pytest.mark.parametrize(
    (
        "algo",
        "num_layers",
        "activation_func",
        "normalization",
        "mix_hidden_states_mode",
        "sequence_parallel",
    ),
    [
        ("eagle3", 1, "swiglu", "RMSNorm", None, False),  # GQA
        ("eagle3", 2, "swiglu", "RMSNorm", None, False),  # GQA
        ("eagle3", 1, "swiglu", "RMSNorm", "sparse_replace", False),
        ("eagle3", 1, "swiglu", "RMSNorm", "unknown", False),
        ("eagle3", 1, "swiglu", "RMSNorm", "sparse_replace", True),
        ("eagle3", 1, "swiglu", "RMSNorm", None, True),
        ("eagle3", 1, "swiglu", "RMSNorm", "uniform_layer_sample", False),
        ("eagle3", 1, "swiglu", "RMSNorm", "uniform_layer_sample", True),
        ("eagle1", 1, "swiglu", "RMSNorm", "uniform_layer_sample", True),
        ("eagle1", 1, "swiglu", "RMSNorm", "sparse_replace", True),
    ],
)
def test_speculative_gpt_model(
    dist_workers,
    algo,
    num_layers,
    activation_func,
    normalization,
    mix_hidden_states_mode,
    sequence_parallel,
):
    if sequence_parallel and torch.cuda.device_count() < 2:
        pytest.skip("Sequence parallelism requires at least two GPUs.")
    dist_workers.run(
        partial(
            _test_speculative_gpt_model,
            algo,
            num_layers,
            activation_func,
            normalization,
            mix_hidden_states_mode,
            sequence_parallel,
        ),
    )
