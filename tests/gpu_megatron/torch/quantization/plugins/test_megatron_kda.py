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

from contextlib import nullcontext

import pytest
import torch
from _test_utils.torch.megatron.utils import (
    initialize_for_megatron,
    load_distributed_checkpoint,
    save_distributed_checkpoint,
)
from megatron.core import parallel_state
from megatron.core.models.hybrid.hybrid_layer_specs import hybrid_stack_spec
from megatron.core.process_groups_config import ProcessGroupCollection
from megatron.core.transformer import TransformerConfig

import modelopt.torch.quantization as mtq
from modelopt.recipe import load_recipe
from modelopt.torch.opt.plugins.mcore_dist_checkpointing import (
    restore_sharded_modelopt_state,
    save_sharded_modelopt_state,
)
from modelopt.torch.quantization.config import QuantizerAttributeConfig
from modelopt.torch.quantization.linear_attention import (
    LinearAttentionConfig,
    linear_attention_training_phase,
)
from modelopt.torch.quantization.nn import TensorQuantizer

KimiDeltaAttention = pytest.importorskip("megatron.core.ssm.gated_delta_net.kda").KimiDeltaAttention


def _layer():
    config = TransformerConfig(
        num_layers=1,
        hidden_size=64,
        num_attention_heads=2,
        linear_num_key_heads=2,
        linear_num_value_heads=2,
        linear_key_head_dim=32,
        linear_value_head_dim=32,
        linear_conv_kernel_dim=4,
        experimental_attention_variant="kda",
        is_hybrid_model=True,
        normalization="RMSNorm",
        activation_func=torch.nn.functional.silu,
        params_dtype=torch.float32,
        gradient_accumulation_fusion=False,
        recompute_granularity="selective",
        recompute_modules=["gdn"],
    )
    spec = hybrid_stack_spec.submodules.kda_layer.submodules.self_attention
    model = (
        KimiDeltaAttention(
            config,
            submodules=spec.submodules,
            layer_number=1,
            pg_collection=ProcessGroupCollection(
                tp=parallel_state.get_tensor_model_parallel_group(),
                cp=parallel_state.get_context_parallel_group(),
            ),
        )
        .cuda()
        .train()
    )
    with torch.no_grad():
        model.A_log.fill_(-4)
        model.dt_bias.zero_()
    return model


def _case(cfg):
    initialize_for_megatron(tensor_model_parallel_size=1, pipeline_model_parallel_size=1, seed=73)
    model = _layer()
    hidden = torch.randn(73, 1, 64, device="cuda", dtype=torch.bfloat16, requires_grad=True)

    def forward(layer):
        policy = getattr(layer, "linear_attention_config", None)
        phase = (
            linear_attention_training_phase(layer, [31])
            if policy is not None and policy.decode is not None
            else nullcontext()
        )
        with phase, torch.autocast("cuda", dtype=torch.bfloat16):
            return layer(hidden, attention_mask=None)[0]

    with torch.no_grad():
        baseline = forward(model)
    mtq.quantize(model, cfg)
    return model, hidden, forward, baseline


def _compile_kda(rank, size, cfg):
    model, hidden, forward, _ = _case(cfg)
    with linear_attention_training_phase(model, [31]):
        forward(model).float().square().mean().backward()
    torch.cuda.synchronize()


def _test_kda(rank, size, cfg, checkpoint_path):
    model, hidden, forward, baseline = _case(cfg)
    assert model.kda_state_quantizer.is_enabled
    assert model.kda_state_quantizer.num_bits == 8
    assert model.linear_attention_config.decode.state_codec == "int8_hadamard32"
    kernel = model.gated_delta_rule
    with torch.no_grad():
        assert not torch.equal(forward(model), baseline)
    assert model.gated_delta_rule is kernel

    policy = model.linear_attention_config
    model.kda_state_quantizer.disable()
    model.linear_attention_config = LinearAttentionConfig()
    with torch.no_grad():
        torch.testing.assert_close(forward(model), baseline, rtol=0, atol=0)
    model.kda_state_quantizer.enable()
    model.linear_attention_config = policy

    with torch.no_grad():
        expected = forward(model)
    # Old checkpoints carried a disabled WY placeholder; it must not reappear on restore.
    model.kda_w_quantizer = TensorQuantizer(QuantizerAttributeConfig(enable=False))
    save_distributed_checkpoint(checkpoint_path, model)
    save_sharded_modelopt_state([model], checkpoint_path)
    del model.kda_w_quantizer
    restored = _layer()
    restore_sharded_modelopt_state([restored], checkpoint_path)
    load_distributed_checkpoint(checkpoint_path, restored)
    # Megatron recomputes the core during backward, after forward's kernel wrapper exits.
    with linear_attention_training_phase(restored, [31]):
        actual = forward(restored)
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        actual.float().square().mean().backward()
    assert not hasattr(restored, "kda_w_quantizer")
    assert restored._linear_attention_prefill_lengths is None
    assert restored.linear_attention_config == policy
    assert restored.gated_delta_rule is kernel
    assert torch.isfinite(hidden.grad).all()
    for parameter in restored.parameters():
        assert parameter.grad is not None
        assert torch.isfinite(parameter.grad).all()
    before = restored.in_proj.weight.detach().clone()
    torch.optim.SGD(restored.parameters(), lr=0.1).step()
    assert not torch.equal(before, restored.in_proj.weight)


@pytest.fixture(scope="module")
def compiled_kda_workers(dist_workers_size_1):
    """Warm one KDA shape outside the functional test timer."""
    cfg = load_recipe("general/ptq/linear_attention_state_int8_dynamic").quantize.model_dump()
    cfg["linear_attention"][0]["cfg"]["decode"].update(
        mode="replay", replay={"window": 5}, decay_log_step=1 / 256
    )
    dist_workers_size_1.run(_compile_kda, cfg)
    return dist_workers_size_1, cfg


def test_kda_qat_and_sharded_restore(compiled_kda_workers, tmp_path):
    workers, cfg = compiled_kda_workers
    workers.run(_test_kda, cfg, tmp_path)
