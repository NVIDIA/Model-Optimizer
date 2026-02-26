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

import copy
from functools import partial

import pytest
import torch
import torch.nn.init as init
from _test_utils.torch.distributed.utils import spawn_multiprocess_job
from _test_utils.torch.megatron.models import get_mcore_gpt_model
from _test_utils.torch.megatron.utils import initialize_for_megatron
from megatron.core import dist_checkpointing

import modelopt.torch.peft as mtpeft
import modelopt.torch.quantization as mtq
from modelopt.torch.opt.plugins.mcore_dist_checkpointing import (
    restore_sharded_modelopt_state,
    save_sharded_modelopt_state,
)
from modelopt.torch.peft.lora.layer import LoRAModule
from modelopt.torch.utils.plugins import megatron_prefill

NVFP4_DEFAULT_CONFIG = {
    "quant_cfg": {
        "*weight_quantizer": {
            "num_bits": (2, 1),
            "block_sizes": {-1: 16, "type": "dynamic", "scale_bits": (4, 3)},
            "axis": None,
            "enable": True,
        },
        "*input_quantizer": {
            "num_bits": (2, 1),
            "block_sizes": {-1: 16, "type": "dynamic", "scale_bits": (4, 3)},
            "axis": None,
            "enable": True,
        },
        "*output_quantizer": {"enable": False},
        "*output_layer*": {"enable": False},  # Note: only output_layer is disabled.
        "default": {"enable": False},
    },
    "algorithm": "max",
}

DEFAULT_LORA_CFG_TEST = {
    "adapter_type": "lora",
    "adapter_name": "default",
    "adapter_cfg": {
        "*": {
            "rank": 32,
            "scale": 1,
            "enable": True,
        },
        "*output_layer*": {"enable": False},
    },
}

LARGE_LORA_CFG_TEST = {
    "adapter_type": "lora",
    "adapter_name": "default",
    "adapter_cfg": {
        "*": {
            "rank": 128,
            "scale": 1,
            "enable": True,
        },
        "*output_layer*": {"enable": False},
    },
}

DEFAULT_LORA_CFG_RANDOM_INIT_TEST = {
    "adapter_type": "lora",
    "adapter_name": "random",
    "adapter_cfg": {
        "*": {
            "rank": 32,
            "scale": 1,
            "lora_a_init": init.kaiming_uniform_,
            "lora_b_init": init.kaiming_uniform_,
            "enable": True,
        },
        "*output_layer*": {"enable": False},
    },
}

LARGE_LORA_CFG_RANDOM_INIT_TEST = {
    "adapter_type": "lora",
    "adapter_name": "random",
    "adapter_cfg": {
        "*": {
            "rank": 128,
            "scale": 1,
            "lora_a_init": init.kaiming_uniform_,
            "lora_b_init": init.kaiming_uniform_,
            "enable": True,
        },
        "*output_layer*": {"enable": False},
    },
}

DEFAULT_LORA_CFG_RANDOM_INIT_SMALL_RANK_TEST = {
    "adapter_type": "lora",
    "adapter_name": "small",
    "adapter_cfg": {
        "*": {
            "rank": 8,
            "scale": 1,
            "lora_a_init": init.kaiming_uniform_,
            "lora_b_init": init.kaiming_uniform_,
            "enable": True,
        },
        "*output_layer*": {"enable": False},
    },
}

SELECTIVE_LAYER_LORA_CFG = {
    "adapter_type": "lora",
    "adapter_name": "selective",
    "adapter_cfg": {
        "*": {"enable": False},
        "*self_attention*": {
            "rank": 16,
            "scale": 1,
            "enable": True,
        },
        "*output_layer*": {"enable": False},
    },
}


def save_distributed_checkpoint(checkpoint_path, gpt_model):
    sharded_state_dict = gpt_model.sharded_state_dict(prefix="")
    dist_checkpointing.save(sharded_state_dict=sharded_state_dict, checkpoint_dir=checkpoint_path)


def load_distributed_checkpoint(checkpoint_path, gpt_model):
    sharded_state_dict = gpt_model.sharded_state_dict(prefix="")
    checkpoint = dist_checkpointing.load(
        sharded_state_dict=sharded_state_dict, checkpoint_dir=checkpoint_path
    )
    gpt_model.load_state_dict(checkpoint)
    return gpt_model


def _gpt_model_provider(tp_size: int, hidden_size=256, vocab_size=64, meta_device=False):
    """Build the model."""

    if meta_device:
        with torch.device("meta"):
            gpt_model = get_mcore_gpt_model(
                tensor_model_parallel_size=tp_size,
                num_layers=2,
                ffn_hidden_size=None,
                num_attention_heads=4,
                activation_func="squared_relu",
                transformer_impl="local",
                hidden_size=hidden_size,
                vocab_size=vocab_size,
                use_cpu_initialization=meta_device,
            )
    else:
        gpt_model = get_mcore_gpt_model(
            tensor_model_parallel_size=tp_size,
            num_layers=2,
            ffn_hidden_size=None,
            num_attention_heads=4,
            activation_func="squared_relu",
            transformer_impl="local",
            hidden_size=hidden_size,
            vocab_size=vocab_size,
        ).cuda()
    return gpt_model.eval()


def _test_forward_with_one_lora(lora_config, rank, size):
    """Test forward pass with a single LoRA adapter with various configurations."""
    hidden_size = 320
    initialize_for_megatron(tensor_model_parallel_size=size, pipeline_model_parallel_size=1)
    model = _gpt_model_provider(tp_size=size, hidden_size=hidden_size)
    prompt_tokens = torch.randint(0, model.vocab_size, (2, model.max_sequence_length)).cuda()

    original_output = megatron_prefill(model, prompt_tokens)
    mtpeft.update_model(model, lora_config)
    lora_output = megatron_prefill(model, prompt_tokens)
    assert lora_output.shape == original_output.shape
    if lora_config == DEFAULT_LORA_CFG_RANDOM_INIT_TEST:
        # Task: To verify that the LoRA output differs from the original
        # output since two LoRA layers are initialized randomly.
        assert not torch.allclose(lora_output, original_output, rtol=1e-5)
    else:
        # Task: The LoRA output should match the original output if two
        # LoRA layers are initialized in the standard way
        # (one with random values and one with zeros).
        assert torch.allclose(lora_output, original_output, rtol=1e-5), (
            f"{lora_output}, {original_output}"
        )
    mtpeft.disable_adapters(model)
    lora_disabled_output = megatron_prefill(model, prompt_tokens)
    # Task: Since all LoRA layers are disabled, the output should
    # be identical to the original output.
    assert torch.allclose(lora_disabled_output, original_output, rtol=1e-5)
    mtpeft.enable_adapters(model)
    lora_reenabled_output = megatron_prefill(model, prompt_tokens)
    # Task: To verify that toggling LoRA layers from disabled
    # to enabled does not alter the output, the output should remain unchanged.
    assert torch.allclose(lora_reenabled_output, lora_output, rtol=1e-5)
    lora_module_count = 0
    lora_with_adapter_count = 0
    for name, module in model.named_modules():
        if isinstance(module, LoRAModule):
            lora_module_count += 1

            if lora_config == SELECTIVE_LAYER_LORA_CFG:
                if "self_attention" in name:
                    # Task: Only self_attention modules should have the adapter
                    assert hasattr(module, f"lora_a_{lora_config['adapter_name']}")
                    assert hasattr(module, f"lora_b_{lora_config['adapter_name']}")
                    assert lora_config["adapter_name"] in module._lora_adapters
                    assert module._lora_adapters[lora_config["adapter_name"]]["enable"]
                    lora_with_adapter_count += 1
                else:
                    # Task: Other modules should NOT have the adapter at all
                    assert not hasattr(module, f"lora_a_{lora_config['adapter_name']}")
                    assert not hasattr(module, f"lora_b_{lora_config['adapter_name']}")
                    assert lora_config["adapter_name"] not in module._lora_adapters
            else:
                # Task: For non-selective configs, all LoRA modules should have the adapter
                for adapter_name in module._lora_adapters:
                    assert hasattr(module, f"lora_a_{adapter_name}")
                    assert hasattr(module, f"lora_b_{adapter_name}")
                lora_with_adapter_count += 1

    assert lora_module_count > 0
    assert lora_with_adapter_count > 0


@pytest.mark.parametrize(
    "lora_config",
    [
        DEFAULT_LORA_CFG_TEST,
        DEFAULT_LORA_CFG_RANDOM_INIT_TEST,
        SELECTIVE_LAYER_LORA_CFG,
    ],
)
def test_forward_with_one_lora(lora_config):
    spawn_multiprocess_job(
        size=torch.cuda.device_count(),
        job=partial(_test_forward_with_one_lora, lora_config),
        backend="nccl",
    )


def _test_forward_with_two_loras(lora_config_1, lora_config_2, rank, size):
    """Test forward pass with two LoRA adapters and adapter switching."""
    hidden_size = 320
    initialize_for_megatron(tensor_model_parallel_size=size, pipeline_model_parallel_size=1)
    model = _gpt_model_provider(tp_size=size, hidden_size=hidden_size)
    prompt_tokens = torch.randint(0, model.vocab_size, (2, model.max_sequence_length)).cuda()

    original_output = megatron_prefill(model, prompt_tokens)
    mtpeft.update_model(model, lora_config_1)
    # output from the first lora only
    lora_1_output = megatron_prefill(model, prompt_tokens)

    mtpeft.update_model(model, lora_config_2)

    mtpeft.disable_adapters(model, adapters_to_disable=[lora_config_1["adapter_name"]])
    mtpeft.enable_adapters(model, adapters_to_enable=[lora_config_2["adapter_name"]])

    # output from the 2nd lora only
    lora_2_output = megatron_prefill(model, prompt_tokens)

    assert lora_1_output.shape == lora_2_output.shape
    # Should not be the same
    assert not torch.allclose(lora_1_output, lora_2_output)

    mtpeft.enable_adapters(model, adapters_to_enable=[lora_config_1["adapter_name"]])
    mtpeft.enable_adapters(model, adapters_to_enable=[lora_config_2["adapter_name"]])
    lora_all_output = megatron_prefill(model, prompt_tokens)

    assert not torch.allclose(lora_all_output, lora_1_output)
    assert not torch.allclose(lora_all_output, lora_2_output)

    mtpeft.disable_adapters(model)
    both_disabled_output = megatron_prefill(model, prompt_tokens)
    assert torch.allclose(both_disabled_output, original_output)

    for _, module in model.named_modules():
        if isinstance(module, LoRAModule):
            for adapter_name in module._lora_adapters:
                assert hasattr(module, f"lora_a_{adapter_name}")
                assert hasattr(module, f"lora_b_{adapter_name}")


@pytest.mark.parametrize(
    ("lora_config_1", "lora_config_2"),
    [
        (DEFAULT_LORA_CFG_RANDOM_INIT_TEST, DEFAULT_LORA_CFG_RANDOM_INIT_SMALL_RANK_TEST),
    ],
)
def test_forward_with_two_loras(lora_config_1, lora_config_2):
    spawn_multiprocess_job(
        size=torch.cuda.device_count(),
        job=partial(_test_forward_with_two_loras, lora_config_1, lora_config_2),
        backend="nccl",
    )


# TODO: Rank check


def _test_attr_changes_with_one_lora(lora_config, rank, size):
    """Test forward pass with a single LoRA adapter with various configurations."""
    hidden_size = 320
    initialize_for_megatron(tensor_model_parallel_size=size, pipeline_model_parallel_size=1)
    model = _gpt_model_provider(tp_size=size, hidden_size=hidden_size)
    prompt_tokens = torch.randint(0, model.vocab_size, (2, model.max_sequence_length)).cuda()

    mtpeft.update_model(model, lora_config)
    lora_1_output = megatron_prefill(model, prompt_tokens)

    for _, module in model.named_modules():
        if isinstance(module, LoRAModule):
            for adapter_name in module._lora_adapters:
                adapter = module._lora_adapters[adapter_name]
                adapter["scale"] = 10.0

    lora_2_output = megatron_prefill(model, prompt_tokens)
    assert not torch.allclose(lora_1_output, lora_2_output)

    for _, module in model.named_modules():
        if isinstance(module, LoRAModule):
            for adapter_name in module._lora_adapters:
                adapter = module._lora_adapters[adapter_name]
                adapter["scale"] = 1.0
    lora_back_output = megatron_prefill(model, prompt_tokens)

    assert torch.allclose(lora_1_output, lora_back_output)


@pytest.mark.parametrize(
    "lora_config",
    [
        DEFAULT_LORA_CFG_RANDOM_INIT_TEST,
    ],
)
def test_attr_changes_with_one_lora(lora_config):
    spawn_multiprocess_job(
        size=torch.cuda.device_count(),
        job=partial(_test_attr_changes_with_one_lora, lora_config),
        backend="nccl",
    )


def _test_mcore_save_restore(lora_config, tmp_path, rank, size):
    hidden_size = 512
    initialize_for_megatron(tensor_model_parallel_size=size, pipeline_model_parallel_size=1)
    model_ref = _gpt_model_provider(tp_size=size, hidden_size=hidden_size)
    model_test = _gpt_model_provider(tp_size=size, hidden_size=hidden_size)
    prompt_tokens = torch.randint(
        0, model_ref.vocab_size, (2, model_ref.max_sequence_length)
    ).cuda()
    original_output_test = megatron_prefill(model_test, prompt_tokens)

    mtpeft.update_model(model_ref, lora_config)

    lora_output_ref = megatron_prefill(model_ref, prompt_tokens)

    save_distributed_checkpoint(tmp_path, model_ref)
    save_sharded_modelopt_state([model_ref], tmp_path)

    restore_sharded_modelopt_state([model_test], tmp_path)
    model_test = load_distributed_checkpoint(tmp_path, model_test)

    lora_output_test = megatron_prefill(model_test, prompt_tokens)

    # Task: If the save and restore functions work correctly, they should produce the same output.
    assert torch.allclose(lora_output_test, lora_output_ref)

    assert not torch.allclose(original_output_test, lora_output_test)


@pytest.mark.parametrize(
    "lora_config",
    [
        DEFAULT_LORA_CFG_RANDOM_INIT_TEST,
    ],
)
def test_mcore_save_restore(lora_config, tmp_path):
    spawn_multiprocess_job(
        size=torch.cuda.device_count(),
        job=partial(_test_mcore_save_restore, lora_config, str(tmp_path)),
        backend="nccl",
    )


def _test_adapter_gradient_flow_freeze_base_model(lora_config, tmp_path, rank, size):
    hidden_size = 512
    initialize_for_megatron(tensor_model_parallel_size=size, pipeline_model_parallel_size=1)
    model = _gpt_model_provider(tp_size=size, hidden_size=hidden_size)
    prompt_tokens = torch.randint(0, model.vocab_size, (2, model.max_sequence_length)).cuda()

    mtpeft.update_model(model, lora_config)
    model.train()

    # Use a simple forward pass instead for grad check
    batch_size = prompt_tokens.shape[0]
    seq_len = prompt_tokens.shape[-1]
    device = prompt_tokens.device

    attention_mask = (
        torch.triu(torch.ones((batch_size, seq_len, seq_len), device=device), diagonal=1)
        .bool()
        .view(batch_size, 1, seq_len, seq_len)
    )

    output = model(prompt_tokens, position_ids=None, attention_mask=attention_mask)

    loss = output.sum()
    loss.backward()

    for name, param in model.named_parameters():
        if "lora" in name:
            assert param.grad is not None
            assert torch.any(param.grad != 0)
        else:
            assert param.grad is None


@pytest.mark.parametrize(
    "lora_config",
    [
        LARGE_LORA_CFG_RANDOM_INIT_TEST,  # Use random init so gradients flow to both lora_a and lora_b
    ],
)
def test_adapter_gradient_flow_freeze_base_model(lora_config, tmp_path):
    spawn_multiprocess_job(
        size=torch.cuda.device_count(),
        job=partial(_test_adapter_gradient_flow_freeze_base_model, lora_config, str(tmp_path)),
        backend="nccl",
    )


def _test_adapter_gradient_flow_freeze_lora_model(lora_config, tmp_path, rank, size):
    hidden_size = 512
    local_cfg = copy.deepcopy(lora_config)
    local_cfg["freeze_lora_weights"] = True
    local_cfg["freeze_base_model"] = False

    initialize_for_megatron(tensor_model_parallel_size=size, pipeline_model_parallel_size=1)
    model = _gpt_model_provider(tp_size=size, hidden_size=hidden_size)
    prompt_tokens = torch.randint(0, model.vocab_size, (2, model.max_sequence_length)).cuda()

    mtpeft.update_model(model, local_cfg)
    model.train()

    # Use a simple forward pass instead for grad check
    batch_size = prompt_tokens.shape[0]
    seq_len = prompt_tokens.shape[-1]
    device = prompt_tokens.device

    attention_mask = (
        torch.triu(torch.ones((batch_size, seq_len, seq_len), device=device), diagonal=1)
        .bool()
        .view(batch_size, 1, seq_len, seq_len)
    )

    output = model(prompt_tokens, position_ids=None, attention_mask=attention_mask)

    loss = output.sum()
    loss.backward()

    for name, param in model.named_parameters():
        if "lora" in name:
            assert param.grad is None
        else:
            assert param.grad is not None
            assert torch.any(param.grad != 0)


@pytest.mark.parametrize(
    "lora_config",
    [
        LARGE_LORA_CFG_RANDOM_INIT_TEST,  # Use random init so gradients flow to both lora_a and lora_b
    ],
)
def test_adapter_gradient_flow_freeze_lora_model(lora_config, tmp_path):
    spawn_multiprocess_job(
        size=torch.cuda.device_count(),
        job=partial(_test_adapter_gradient_flow_freeze_lora_model, lora_config, str(tmp_path)),
        backend="nccl",
    )


def _test_adapter_gradient_flow(lora_config, tmp_path, rank, size):
    hidden_size = 512
    lora_config["freeze_lora_weights"] = False
    lora_config["freeze_base_model"] = False

    initialize_for_megatron(tensor_model_parallel_size=size, pipeline_model_parallel_size=1)
    model = _gpt_model_provider(tp_size=size, hidden_size=hidden_size)
    prompt_tokens = torch.randint(0, model.vocab_size, (2, model.max_sequence_length)).cuda()

    mtpeft.update_model(model, lora_config)
    model.train()

    # Use a simple forward pass instead for grad check
    batch_size = prompt_tokens.shape[0]
    seq_len = prompt_tokens.shape[-1]
    device = prompt_tokens.device

    attention_mask = (
        torch.triu(torch.ones((batch_size, seq_len, seq_len), device=device), diagonal=1)
        .bool()
        .view(batch_size, 1, seq_len, seq_len)
    )

    output = model(prompt_tokens, position_ids=None, attention_mask=attention_mask)

    loss = output.sum()
    loss.backward()

    for name, param in model.named_parameters():
        assert param.grad is not None
        assert torch.any(param.grad != 0), "weight gradient is all zeros"


@pytest.mark.parametrize(
    "lora_config",
    [
        LARGE_LORA_CFG_RANDOM_INIT_TEST,  # Use random init so gradients flow to both lora_a and lora_b
    ],
)
def test_adapter_gradient_flow(lora_config, tmp_path):
    spawn_multiprocess_job(
        size=torch.cuda.device_count(),
        job=partial(_test_adapter_gradient_flow, lora_config, str(tmp_path)),
        backend="nccl",
    )


def _test_adapter_gradient_flow_freeze_lora_with_api(lora_config, tmp_path, rank, size):
    hidden_size = 256

    initialize_for_megatron(tensor_model_parallel_size=size, pipeline_model_parallel_size=1)
    model = _gpt_model_provider(tp_size=size, hidden_size=hidden_size)
    prompt_tokens = torch.randint(0, model.vocab_size, (2, model.max_sequence_length)).cuda()
    lora_config["freeze_lora_weights"] = False
    lora_config["freeze_base_model"] = False

    mtpeft.update_model(model, lora_config)
    # Freeze the self_attention layers only
    mtpeft.freeze_lora_weights(model, layer_patterns="*self_attention*")
    model.train()

    # Use a simple forward pass instead for grad check
    batch_size = prompt_tokens.shape[0]
    seq_len = prompt_tokens.shape[-1]
    device = prompt_tokens.device

    attention_mask = (
        torch.triu(torch.ones((batch_size, seq_len, seq_len), device=device), diagonal=1)
        .bool()
        .view(batch_size, 1, seq_len, seq_len)
    )

    output = model(prompt_tokens, position_ids=None, attention_mask=attention_mask)

    loss = output.sum()
    loss.backward()

    for name, param in model.named_parameters():
        if "lora" in name and "self_attention" in name:
            assert param.grad is None
        else:
            assert param.grad is not None
            assert torch.any(param.grad != 0), "weight gradient is all zeros"

    for p in model.parameters():
        p.grad = None

    mtpeft.freeze_lora_weights(model)
    model.train()

    # Use a simple forward pass instead for grad check
    batch_size = prompt_tokens.shape[0]
    seq_len = prompt_tokens.shape[-1]
    device = prompt_tokens.device

    attention_mask = (
        torch.triu(torch.ones((batch_size, seq_len, seq_len), device=device), diagonal=1)
        .bool()
        .view(batch_size, 1, seq_len, seq_len)
    )

    output = model(prompt_tokens, position_ids=None, attention_mask=attention_mask)

    loss = output.sum()
    loss.backward()

    for name, param in model.named_parameters():
        if "lora" in name:
            assert param.grad is None
        else:
            assert param.grad is not None
            assert torch.any(param.grad != 0), "weight gradient is all zeros"


@pytest.mark.parametrize(
    "lora_config",
    [
        LARGE_LORA_CFG_RANDOM_INIT_TEST,
    ],
)
def test_adapter_gradient_flow_freeze_lora_with_api(lora_config, tmp_path):
    spawn_multiprocess_job(
        size=torch.cuda.device_count(),
        job=partial(_test_adapter_gradient_flow_freeze_lora_with_api, lora_config, str(tmp_path)),
        backend="nccl",
    )


def _test_quantize_then_lora(lora_config, tmp_path, rank, size):
    hidden_size = 512
    initialize_for_megatron(tensor_model_parallel_size=size, pipeline_model_parallel_size=1)
    model = _gpt_model_provider(tp_size=size, hidden_size=hidden_size)
    prompt_tokens = torch.randint(0, model.vocab_size, (2, model.max_sequence_length)).cuda()

    def forward_func(mod):
        _ = megatron_prefill(model, prompt_tokens)

    mtq.quantize(model, NVFP4_DEFAULT_CONFIG, forward_func)

    # Then add the lora
    mtpeft.update_model(model, lora_config)

    # Bypass the output layer
    for name, module in model.named_modules():
        if isinstance(module, LoRAModule) and "output_layer" not in name:
            assert hasattr(module, "input_quantizer")
            assert hasattr(module, "weight_quantizer")
            assert hasattr(module.input_quantizer, "amax")
            assert hasattr(module.weight_quantizer, "amax")
            assert getattr(module.input_quantizer, "amax") is not None
            assert getattr(module.weight_quantizer, "amax") is not None
            # Check if the lora have the quantizer, they should not have them.
            for adapter_name in module._lora_adapters:
                lora_a = module._lora_adapters[adapter_name]["lora_a"]
                lora_b = module._lora_adapters[adapter_name]["lora_b"]
                assert not hasattr(lora_a, "input_quantizer")
                assert not hasattr(lora_b, "weight_quantizer")

    quantized_lora_output = megatron_prefill(model, prompt_tokens)
    mtq.disable_quantizer(model, "*")
    unquantized_lora_output = megatron_prefill(model, prompt_tokens)
    # Task: Quantize and unquantize should produce different tensor values
    assert not torch.allclose(quantized_lora_output, unquantized_lora_output)


@pytest.mark.parametrize(
    "lora_config",
    [
        LARGE_LORA_CFG_RANDOM_INIT_TEST,  # Use random init so gradients flow to both lora_a and lora_b
    ],
)
def test_quantize_then_lora(lora_config, tmp_path):
    spawn_multiprocess_job(
        size=torch.cuda.device_count(),
        job=partial(_test_quantize_then_lora, lora_config, str(tmp_path)),
        backend="nccl",
    )


def _test_lora_then_quantize(lora_config, tmp_path, rank, size):
    hidden_size = 512
    initialize_for_megatron(tensor_model_parallel_size=size, pipeline_model_parallel_size=1)
    model = _gpt_model_provider(tp_size=size, hidden_size=hidden_size)
    prompt_tokens = torch.randint(0, model.vocab_size, (2, model.max_sequence_length)).cuda()

    mtpeft.update_model(model, lora_config)
    lora_output = megatron_prefill(model, prompt_tokens)

    def forward_func(mod):
        _ = megatron_prefill(model, prompt_tokens)

    mtq.quantize(model, NVFP4_DEFAULT_CONFIG, forward_func)
    quantized_output = megatron_prefill(model, prompt_tokens)
    # Bypass the output layer
    for name, module in model.named_modules():
        if isinstance(module, LoRAModule) and "output_layer" not in name:
            assert hasattr(module, "input_quantizer")
            assert hasattr(module, "weight_quantizer")
            assert hasattr(module.input_quantizer, "amax")
            assert hasattr(module.weight_quantizer, "amax")
            assert getattr(module.input_quantizer, "amax") is not None
            assert getattr(module.weight_quantizer, "amax") is not None
            # Check if the lora have the quantizer, they should not have them.
            for adapter_name in module._lora_adapters:
                lora_a = module._lora_adapters[adapter_name]["lora_a"]
                lora_b = module._lora_adapters[adapter_name]["lora_b"]
                assert hasattr(lora_a, "input_quantizer")
                assert hasattr(lora_b, "weight_quantizer")
                assert hasattr(lora_a.input_quantizer, "amax")
                assert hasattr(lora_b.weight_quantizer, "amax")
                assert getattr(lora_a.input_quantizer, "amax") is not None
                assert getattr(lora_b.weight_quantizer, "amax") is not None

    assert not torch.allclose(lora_output, quantized_output)

    mtq.disable_quantizer(model, "*lora_a*")
    disabled_lora_a_quantized_output = megatron_prefill(model, prompt_tokens)
    # Should not be the same since we disable the lora_a quantizers
    assert not torch.allclose(disabled_lora_a_quantized_output, quantized_output)

    mtq.disable_quantizer(model, "*lora_b*")
    disabled_lora_ab_quantized_output = megatron_prefill(model, prompt_tokens)
    assert not torch.allclose(disabled_lora_a_quantized_output, disabled_lora_ab_quantized_output)
    assert not torch.allclose(quantized_output, disabled_lora_ab_quantized_output)


@pytest.mark.parametrize(
    "lora_config",
    [
        LARGE_LORA_CFG_RANDOM_INIT_TEST,  # Use random init so gradients flow to both lora_a and lora_b
    ],
)
def test_lora_then_quantize(lora_config, tmp_path):
    spawn_multiprocess_job(
        size=torch.cuda.device_count(),
        job=partial(_test_lora_then_quantize, lora_config, str(tmp_path)),
        backend="nccl",
    )


def _test_mcore_quantize_then_lora_save_restore(lora_config, tmp_path, rank, size):
    hidden_size = 512
    initialize_for_megatron(tensor_model_parallel_size=size, pipeline_model_parallel_size=1)
    model_ref = _gpt_model_provider(tp_size=size, hidden_size=hidden_size)
    model_test = _gpt_model_provider(tp_size=size, hidden_size=hidden_size)
    prompt_tokens = torch.randint(
        0, model_ref.vocab_size, (2, model_ref.max_sequence_length)
    ).cuda()
    original_output_test = megatron_prefill(model_test, prompt_tokens)

    def forward_func(mod):
        _ = megatron_prefill(model_ref, prompt_tokens)

    mtq.quantize(model_ref, NVFP4_DEFAULT_CONFIG, forward_func)
    mtpeft.update_model(model_ref, lora_config)

    quantize_lora_output_ref = megatron_prefill(model_ref, prompt_tokens)

    save_distributed_checkpoint(tmp_path, model_ref)
    save_sharded_modelopt_state([model_ref], tmp_path)

    restore_sharded_modelopt_state([model_test], tmp_path)
    model_test = load_distributed_checkpoint(tmp_path, model_test)

    quantize_lora_output_test = megatron_prefill(model_test, prompt_tokens)

    # Task: If the save and restore functions work correctly, they should produce the same output.
    assert torch.allclose(quantize_lora_output_test, quantize_lora_output_ref)

    assert not torch.allclose(original_output_test, quantize_lora_output_test)

    # Check the quantizer and lora layers after restore
    for name, module in model_test.named_modules():
        if isinstance(module, LoRAModule) and "output_layer" not in name:
            # print(f"{name} {module}")
            assert hasattr(module, "input_quantizer")
            assert hasattr(module, "weight_quantizer")
            assert hasattr(module.input_quantizer, "amax")
            assert hasattr(module.weight_quantizer, "amax")
            assert getattr(module.input_quantizer, "amax") is not None
            assert getattr(module.weight_quantizer, "amax") is not None
            # Check if the lora have the quantizer, they should not have them.
            for adapter_name in module._lora_adapters:
                lora_a = module._lora_adapters[adapter_name]["lora_a"]
                lora_b = module._lora_adapters[adapter_name]["lora_b"]
                assert not hasattr(lora_a, "input_quantizer")
                assert not hasattr(lora_b, "weight_quantizer")


@pytest.mark.parametrize(
    "lora_config",
    [
        DEFAULT_LORA_CFG_RANDOM_INIT_TEST,
    ],
)
def test_mcore_quantize_then_lora_save_restore(lora_config, tmp_path):
    spawn_multiprocess_job(
        size=torch.cuda.device_count(),
        job=partial(_test_mcore_quantize_then_lora_save_restore, lora_config, str(tmp_path)),
        backend="nccl",
    )


def _test_mcore_lora_then_quantize_save_restore(lora_config, tmp_path, rank, size):
    hidden_size = 512
    initialize_for_megatron(tensor_model_parallel_size=size, pipeline_model_parallel_size=1)
    model_ref = _gpt_model_provider(tp_size=size, hidden_size=hidden_size)
    model_test = _gpt_model_provider(tp_size=size, hidden_size=hidden_size)
    prompt_tokens = torch.randint(
        0, model_ref.vocab_size, (2, model_ref.max_sequence_length)
    ).cuda()
    original_output_test = megatron_prefill(model_test, prompt_tokens)

    mtpeft.update_model(model_ref, lora_config)

    def forward_func(mod):
        _ = megatron_prefill(model_ref, prompt_tokens)

    mtq.quantize(model_ref, NVFP4_DEFAULT_CONFIG, forward_func)

    lora_quantize_output_ref = megatron_prefill(model_ref, prompt_tokens)

    save_distributed_checkpoint(tmp_path, model_ref)
    save_sharded_modelopt_state([model_ref], tmp_path)

    restore_sharded_modelopt_state([model_test], tmp_path)
    model_test = load_distributed_checkpoint(tmp_path, model_test)

    lora_quantize_output_test = megatron_prefill(model_test, prompt_tokens)

    # Task: If the save and restore functions work correctly, they should produce the same output.
    assert torch.allclose(lora_quantize_output_test, lora_quantize_output_ref)

    assert not torch.allclose(original_output_test, lora_quantize_output_test)

    # Check the lora and quantize layers after restore
    for name, module in model_test.named_modules():
        if isinstance(module, LoRAModule) and "output_layer" not in name:
            assert hasattr(module, "input_quantizer")
            assert hasattr(module, "weight_quantizer")
            assert hasattr(module.input_quantizer, "amax")
            assert hasattr(module.weight_quantizer, "amax")
            assert getattr(module.input_quantizer, "amax") is not None
            assert getattr(module.weight_quantizer, "amax") is not None
            # Check if the lora have the quantizer, they should not have them.
            for adapter_name in module._lora_adapters:
                lora_a = module._lora_adapters[adapter_name]["lora_a"]
                lora_b = module._lora_adapters[adapter_name]["lora_b"]
                assert hasattr(lora_a, "input_quantizer")
                assert hasattr(lora_b, "weight_quantizer")
                assert hasattr(lora_a.input_quantizer, "amax")
                assert hasattr(lora_b.weight_quantizer, "amax")
                assert getattr(lora_a.input_quantizer, "amax") is not None
                assert getattr(lora_b.weight_quantizer, "amax") is not None


@pytest.mark.parametrize(
    "lora_config",
    [
        DEFAULT_LORA_CFG_RANDOM_INIT_TEST,
    ],
)
def test_mcore_lora_then_quantize_save_restore(lora_config, tmp_path):
    spawn_multiprocess_job(
        size=torch.cuda.device_count(),
        job=partial(_test_mcore_lora_then_quantize_save_restore, lora_config, str(tmp_path)),
        backend="nccl",
    )
