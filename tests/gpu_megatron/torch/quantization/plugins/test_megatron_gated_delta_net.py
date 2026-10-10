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
from megatron.core.packed_seq_params import PackedSeqParams

import modelopt.torch.opt as mto
import modelopt.torch.quantization as mtq
from modelopt.recipe import load_recipe
from modelopt.torch.quantization.linear_attention import linear_attention_training_phase

pytest.importorskip("fla")  # Megatron-Core GatedDeltaNet needs FLA for its baseline kernels.
GatedDeltaNet = pytest.importorskip("megatron.core.ssm.gated_delta_net").GatedDeltaNet
pytest.importorskip("vllm")

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
                # Shared hybrid-model KDA settings must not reject a GDN layer.
                module.config.kda_safe_gate = True
                module.config.kda_lower_bound = -5.0
    return model


def _gdn_config():
    return load_recipe(
        "general/ptq/linear_attention_state_int8_block32_dynamic"
    ).quantize.model_dump()


def _gdn_forward(model):
    original_forward = get_forward(model)

    def forward(m, *, decode_aware=True):
        with linear_attention_training_phase(m, [64, 64]) if decode_aware else nullcontext():
            return original_forward(m)

    return forward


def _test_gdn_qat_helper(rank, size, checkpoint_path):
    initialize_for_megatron(
        tensor_model_parallel_size=size, pipeline_model_parallel_size=1, seed=SEED
    )
    model = _make_model(size)
    forward = _gdn_forward(model)
    with torch.no_grad():
        baseline = forward(model, decode_aware=False)
    mtq.quantize(model, _gdn_config())
    module = next(m for m in model.modules() if isinstance(m, _QuantGatedDeltaNet))
    calls = []
    hook = module.gdn_state_quantizer.register_forward_hook(lambda *_: calls.append(True))
    try:
        assert torch.isfinite(forward(model)).all()
        assert calls
    finally:
        hook.remove()
    mtq.disable_quantizer(model, "*")
    with torch.no_grad():
        torch.testing.assert_close(forward(model, decode_aware=False), baseline, rtol=0, atol=0)
    mtq.enable_quantizer(model, "*gdn_state_quantizer")
    module.linear_attention_config.state_block_v = 32

    restored = _make_model(size)
    sharded_state_dict_test_helper(checkpoint_path, model, restored, forward)
    restored_gdn = next(m for m in restored.modules() if isinstance(m, _QuantGatedDeltaNet))
    assert restored_gdn.linear_attention_config == module.linear_attention_config
    assert restored_gdn.gdn_state_quantizer.is_enabled
    assert torch.isfinite(restored_gdn.in_proj.weight.grad).all()
    before = restored_gdn.in_proj.weight.detach().clone()
    torch.optim.SGD(restored.parameters(), lr=1e-3).step()
    assert not torch.equal(restored_gdn.in_proj.weight, before)
    _check_packed_recompute(restored_gdn)

    for name, value, restore in (
        ("context_parallel_size", 2, True),
        ("recompute_granularity", "full", True),
        ("recompute_granularity", "full", False),
    ):
        receiver = _make_model(size)
        layer = next(m for m in receiver.modules() if isinstance(m, GatedDeltaNet))
        original = getattr(layer.config, name)
        try:
            setattr(layer.config, name, value)
            receiver.train()
            if restore:
                mto.restore_from_modelopt_state(receiver, mto.modelopt_state(model))
            else:
                mtq.quantize(receiver, _gdn_config())
            assert layer.training
            for enabled in (False, True):
                if enabled:
                    layer.gdn_state_quantizer.enable()
                else:
                    layer.gdn_state_quantizer.disable()
                layer.modelopt_post_restore()
            with (
                pytest.raises(NotImplementedError, match=r"context parallelism|full-layer"),
                linear_attention_training_phase(layer, [0]),
            ):
                layer(torch.zeros(1, 1, 64, device="cuda"), attention_mask=None)
        finally:
            setattr(layer.config, name, original)


def _check_packed_recompute(module):
    # Older Megatron releases lack selective GDN recompute and its native core hook.
    if not hasattr(module, "recompute_gdn") or not hasattr(GatedDeltaNet, "_forward_compute"):
        return
    hidden = torch.randn(128, 1, 64, device="cuda", dtype=torch.bfloat16, requires_grad=True)
    packed = PackedSeqParams(
        qkv_format="thd",
        cu_seqlens_q=torch.tensor([0, 60, 118], device="cuda", dtype=torch.int32),
        cu_seqlens_kv=torch.tensor([0, 60, 118], device="cuda", dtype=torch.int32),
        cu_seqlens_q_padded=torch.tensor([0, 64, 128], device="cuda", dtype=torch.int32),
        cu_seqlens_kv_padded=torch.tensor([0, 64, 128], device="cuda", dtype=torch.int32),
        max_seqlen_q=64,
        max_seqlen_kv=64,
        total_tokens=128,
    )
    original = module.recompute_gdn
    was_training = module.training
    module.train()
    try:
        results = []
        for recompute, lengths in ((False, [60, 58]), (False, None), (True, None)):
            module.recompute_gdn = recompute
            module.zero_grad(set_to_none=True)
            hidden.grad = None
            with (
                linear_attention_training_phase(
                    module,
                    [56, 54],
                    sequence_lengths=lengths,
                    cu_seqlens=packed.cu_seqlens_q_padded,
                ),
                torch.autocast("cuda", dtype=torch.bfloat16),
            ):
                output = module(hidden, attention_mask=None, packed_seq_params=packed)[0]
            # Backward runs after the phase exits, as in pipeline schedules.
            output.float().square().sum().backward()
            results.append(
                (output.detach(), hidden.grad.clone(), module.in_proj.weight.grad.clone())
            )
        for actual in results[1:]:
            for value, expected in zip(actual, results[0]):
                torch.testing.assert_close(value, expected, rtol=0, atol=0)
    finally:
        module.recompute_gdn = original
        module.train(was_training)


def _compile_gdn_qat_kernels(rank, size):
    initialize_for_megatron(
        tensor_model_parallel_size=size, pipeline_model_parallel_size=1, seed=SEED
    )
    model = _make_model(size)
    forward = _gdn_forward(model)
    with torch.no_grad():
        forward(model)
    cfg = _gdn_config()
    cfg["linear_attention"][0]["cfg"]["state_block_v"] = 32
    mtq.quantize(model, cfg)
    with torch.no_grad():
        forward(model)
    model.train()
    forward(model).sum().backward()
    _check_packed_recompute(next(m for m in model.modules() if isinstance(m, _QuantGatedDeltaNet)))
    torch.cuda.synchronize()


@pytest.fixture
def compiled_gdn_workers(dist_workers_size_1):
    """Warm one small QAT model in the same worker, outside the test-call budget."""
    dist_workers_size_1.run(_compile_gdn_qat_kernels)
    return dist_workers_size_1


def test_gdn_qat_and_sharded_restore(compiled_gdn_workers, tmp_path):
    """Train through QDQ after a Megatron distributed-checkpoint round trip."""
    compiled_gdn_workers.run(_test_gdn_qat_helper, tmp_path)
