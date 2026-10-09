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
from _test_utils.torch.megatron.utils import initialize_for_megatron
from megatron.core import parallel_state
from megatron.core.models.hybrid.hybrid_layer_specs import hybrid_stack_spec
from megatron.core.process_groups_config import ProcessGroupCollection
from megatron.core.transformer import TransformerConfig

import modelopt.torch.quantization as mtq
from modelopt.recipe import load_recipe
from modelopt.torch.quantization.linear_attention import linear_attention_training_phase

KimiDeltaAttention = pytest.importorskip("megatron.core.ssm.gated_delta_net.kda").KimiDeltaAttention
pytest.importorskip("vllm")


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

    def forward(layer, *, decode_aware=True, prefix=31, length=None):
        policy = getattr(layer, "linear_attention_config", None)
        phase = (
            linear_attention_training_phase(
                layer, [prefix], sequence_lengths=None if length is None else [length]
            )
            if policy is not None and policy.backend == "serving" and decode_aware
            else nullcontext()
        )
        with phase, torch.autocast("cuda", dtype=torch.bfloat16):
            return layer(hidden, attention_mask=None)[0]

    with torch.no_grad():
        baseline = forward(model)

    def calibrate(m):
        with linear_attention_training_phase(m, [73]):
            return forward(m, decode_aware=False)

    mtq.quantize(model, {**cfg, "algorithm": "max"}, calibrate)
    return model, hidden, forward, baseline


def _compile_kda(rank, size, cfg):
    model, hidden, forward, _ = _case(cfg)
    for prefix in (31, 32):
        forward(model, prefix=prefix, length=prefix + 4).float().square().mean().backward()
    torch.cuda.synchronize()


def _test_kda(rank, size, cfg):
    model, hidden, forward, baseline = _case(cfg)
    for name, value in (("kda_safe_gate", True), ("kda_lower_bound", -5.0)):
        original = getattr(model.config, name)
        try:
            setattr(model.config, name, value)
            with pytest.raises(NotImplementedError, match="unbounded softplus gates"):
                mtq.quantize(model, cfg)
        finally:
            setattr(model.config, name, original)
    assert model.kda_state_quantizer.is_enabled
    assert model.kda_state_quantizer.num_bits == 8
    assert model.linear_attention_config.precision == "vllm"
    assert model.kda_state_quantizer.block_sizes == {-1: 32}
    kernel = model.gated_delta_rule
    with torch.no_grad():
        assert not torch.equal(forward(model), baseline)
    assert model.gated_delta_rule is kernel

    with pytest.raises(ValueError, match="explicit prefill lengths"):
        forward(model, decode_aware=False)
    # Two in-flight microbatches must keep their own phases after both contexts exit.
    expected_grads = []
    model.recompute_gdn = False
    for prefix in (31, 32):
        model.zero_grad(set_to_none=True)
        hidden.grad = None
        forward(model, prefix=prefix, length=prefix + 4).float().square().mean().backward()
        expected_grads.append((hidden.grad.clone(), model.in_proj.weight.grad.clone()))
    model.recompute_gdn = True
    pending = [forward(model, prefix=prefix, length=prefix + 4) for prefix in (31, 32)]
    for output, (hidden_grad, weight_grad) in zip(pending, expected_grads):
        model.zero_grad(set_to_none=True)
        hidden.grad = None
        output.float().square().mean().backward()
        torch.testing.assert_close(hidden.grad, hidden_grad, rtol=0, atol=0)
        torch.testing.assert_close(model.in_proj.weight.grad, weight_grad, rtol=0, atol=0)
    model.eval()
    forward(model).float().sum().backward()  # No checkpoint hook may reject eval gradients.
    model.train()

    mtq.disable_quantizer(model, "*")
    with torch.no_grad():
        torch.testing.assert_close(forward(model, decode_aware=False), baseline, rtol=0, atol=0)
    mtq.enable_quantizer(model, "*kda_state_quantizer")
    before = model.in_proj.weight.detach().clone()
    torch.optim.SGD(model.parameters(), lr=0.1).step()
    assert not torch.equal(before, model.in_proj.weight)


@pytest.fixture(scope="module")
def compiled_kda_workers(dist_workers_size_1):
    """Warm one KDA shape outside the functional test timer."""
    cfg = load_recipe(
        "general/ptq/linear_attention_state_int8_block32_dynamic"
    ).quantize.model_dump()
    dist_workers_size_1.run(_compile_kda, cfg)
    return dist_workers_size_1, cfg


def test_kda_qat_recompute(compiled_kda_workers):
    workers, cfg = compiled_kda_workers
    workers.run(_test_kda, cfg)
