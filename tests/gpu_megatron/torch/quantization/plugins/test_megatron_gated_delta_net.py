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
from _test_utils.torch.megatron.models import get_mcore_gpt_model
from _test_utils.torch.megatron.utils import (
    get_forward,
    initialize_for_megatron,
    sharded_state_dict_test_helper,
)

import modelopt.torch.quantization as mtq
from modelopt.torch.quantization.linear_attention import (
    LinearAttentionConfig,
    linear_attention_training_phase,
)
from modelopt.torch.quantization.nn import TensorQuantizer

pytest.importorskip("fla")  # Megatron-Core GatedDeltaNet needs FLA for its baseline kernels.
GatedDeltaNet = pytest.importorskip("megatron.core.ssm.gated_delta_net").GatedDeltaNet
pytest.importorskip("vllm.model_executor.layers.fla.ops.kda", exc_type=ModuleNotFoundError)

from modelopt.torch.quantization.plugins.megatron import _QuantGatedDeltaNet

SEED = 1234


def _make_model(tp_size):
    model = (
        get_mcore_gpt_model(
            tensor_model_parallel_size=tp_size,
            num_layers=1,
            hidden_size=64,
            num_attention_heads=4,
            vocab_size=32,
            max_sequence_length=128,
            experimental_attention_variant="gated_delta_net",
            linear_num_key_heads=1,
            linear_num_value_heads=1,
            linear_key_head_dim=32,
            linear_value_head_dim=32,
        )
        .cuda()
        .eval()
    )
    # Retain enough history for recurrent-state rounding to affect later tokens.
    with torch.no_grad():
        for module in model.modules():
            if isinstance(module, GatedDeltaNet):
                module.A_log.fill_(-4)
                module.dt_bias.zero_()
    return model


def _gdn_config():
    return {
        "quant_cfg": [
            {"quantizer_name": "*", "enable": False},
            {
                "quantizer_name": "*gdn_state_quantizer",
                "cfg": {
                    "num_bits": 8,
                    "block_sizes": {-1: 32},
                    "type": "dynamic",
                    "unsigned": False,
                    "narrow_range": True,
                },
            },
        ],
        "algorithm": None,
        "linear_attention": [
            {
                "module_name": "decoder.layers.*.self_attention",
                "cfg": {
                    "backend": "serving",
                    "state_block_v": 16,
                    "precision": "vllm_0_15",
                },
            }
        ],
    }


def _gdn_forward(model):
    original_forward = get_forward(model)

    def forward(m):
        enabled = any(
            getattr(getattr(layer, "linear_attention_config", None), "backend", None) == "serving"
            for layer in m.modules()
        )
        with linear_attention_training_phase(m, [64, 64]) if enabled else nullcontext():
            return original_forward(m)

    return forward


def _test_gdn_qat_helper(rank, size, checkpoint_path):
    initialize_for_megatron(
        tensor_model_parallel_size=size, pipeline_model_parallel_size=1, seed=SEED
    )
    model = _make_model(size)
    forward = _gdn_forward(model)

    outputs = []
    handles = [
        module.register_forward_hook(
            lambda module, args, result: outputs.append(result[0].detach().clone())
        )
        for module in model.modules()
        if isinstance(module, GatedDeltaNet)
    ]
    with torch.no_grad():
        loss_ref = forward(model)
    gdn_ref = outputs.copy()
    outputs.clear()

    mtq.quantize(model, _gdn_config())
    gdn_modules = [m for m in model.modules() if isinstance(m, _QuantGatedDeltaNet)]
    assert gdn_modules, "no GatedDeltaNet layer was wrapped"
    for module in gdn_modules:
        assert module.gdn_state_qdq_block_v == 16
        assert module.gdn_state_quantizer.is_enabled
        assert not module.gdn_w_quantizer.is_enabled
        # Checkpointing must save the resolved policy, including edits after conversion.
        module.linear_attention_config = LinearAttentionConfig(
            **{**module.linear_attention_config.model_dump(), "state_block_v": 32}
        )

    with torch.no_grad():
        loss_quant = forward(model)
    assert torch.isfinite(loss_quant).all()
    # BF16 residual additions can hide a small GDN change at the final loss.
    assert len(outputs) == len(gdn_ref) == len(gdn_modules)
    assert any(not torch.equal(actual, ref) for actual, ref in zip(outputs, gdn_ref)), (
        "enabled QDQ must change the GDN branch output"
    )
    for handle in handles:
        handle.remove()

    enabled = [
        q
        for m in gdn_modules
        for q in m.modules()
        if isinstance(q, TensorQuantizer) and q.is_enabled
    ]
    policies = [m.linear_attention_config for m in gdn_modules]
    for q in enabled:
        q.disable()
    for module in gdn_modules:
        module.linear_attention_config = LinearAttentionConfig()
    with torch.no_grad():
        torch.testing.assert_close(forward(model), loss_ref, rtol=1e-4, atol=1e-4)
    for q in enabled:
        q.enable()
    for module, policy in zip(gdn_modules, policies):
        module.linear_attention_config = policy

    restored = _make_model(size)
    sharded_state_dict_test_helper(checkpoint_path, model, restored, forward)
    restored_gdn = [m for m in restored.modules() if isinstance(m, _QuantGatedDeltaNet)]
    assert len(restored_gdn) == len(gdn_modules)
    for module, original in zip(restored_gdn, gdn_modules):
        assert module.linear_attention_config == original.linear_attention_config
        assert module.gdn_state_quantizer.is_enabled
        assert not module.gdn_w_quantizer.is_enabled
        assert module.gdn_state_quantizer.num_bits == original.gdn_state_quantizer.num_bits
        assert module.gdn_state_qdq_block_v == 32
        assert module.in_proj.weight.grad is not None
        assert torch.isfinite(module.in_proj.weight.grad).all()

    # The checkpoint helper already ran backward through the restored model.
    optimizer = torch.optim.SGD(restored.parameters(), lr=1e-3)
    before = restored_gdn[0].in_proj.weight.detach().clone()
    optimizer.step()
    assert not torch.equal(restored_gdn[0].in_proj.weight, before)

    fused_model = _make_model(size)
    for module in fused_model.modules():
        if isinstance(module, GatedDeltaNet):
            module.gdn_pre_gated_delta_rule_fusion = True
    with pytest.raises(NotImplementedError, match="unfused input-preparation hook"):
        mtq.quantize(fused_model, _gdn_config())


def _compile_gdn_qat_kernels(rank, size):
    initialize_for_megatron(
        tensor_model_parallel_size=size, pipeline_model_parallel_size=1, seed=SEED
    )
    model = _make_model(size)
    forward = _gdn_forward(model)
    with torch.no_grad():
        forward(model)
    cfg = _gdn_config()
    # The functional case switches state tiles before running inference.
    cfg["linear_attention"][0]["cfg"]["state_block_v"] = 32
    mtq.quantize(model, cfg)
    with torch.no_grad():
        forward(model)
    model.train()
    forward(model).sum().backward()
    torch.cuda.synchronize()


@pytest.fixture
def compiled_gdn_workers(dist_workers_size_1):
    """Warm one small QAT model in the same worker, outside the test-call budget."""
    dist_workers_size_1.run(_compile_gdn_qat_kernels)
    return dist_workers_size_1


def test_gdn_qat_and_sharded_restore(compiled_gdn_workers, tmp_path):
    """Train through QDQ after a Megatron distributed-checkpoint round trip."""
    compiled_gdn_workers.run(_test_gdn_qat_helper, tmp_path)
