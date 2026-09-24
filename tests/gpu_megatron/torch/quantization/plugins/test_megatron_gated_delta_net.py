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

pytest.importorskip("fla")  # Megatron-Core GatedDeltaNet and the state QDQ kernel need fla
GatedDeltaNet = pytest.importorskip("megatron.core.ssm.gated_delta_net").GatedDeltaNet

from modelopt.torch.quantization.plugins.gated_delta_net import _state_qdq_chunk_gated_delta_rule
from modelopt.torch.quantization.plugins.megatron import _QuantGatedDeltaNet

try:
    _state_qdq_chunk_gated_delta_rule()
except RuntimeError as e:
    pytest.skip(str(e), allow_module_level=True)

SEED = 1234


def _make_model(tp_size, **config_kwargs):
    model = (
        get_mcore_gpt_model(
            tensor_model_parallel_size=tp_size,
            num_layers=2,
            hidden_size=64,
            num_attention_heads=4,
            vocab_size=32,
            max_sequence_length=128,
            experimental_attention_variant="gated_delta_net",
            linear_value_head_dim=64,
            **config_kwargs,
        )
        .cuda()
        .eval()
    )
    # Retain enough history for chunk-boundary state rounding to affect the next chunk.
    with torch.no_grad():
        for module in model.modules():
            if isinstance(module, GatedDeltaNet):
                module.A_log.fill_(-4)
                module.dt_bias.zero_()
    return model


def _gdn_config(sites):
    return {
        "quant_cfg": [{"quantizer_name": "*", "enable": False}]
        + [
            {
                "quantizer_name": f"*gdn_{site}_quantizer",
                "cfg": {
                    "num_bits": (4, 3),
                    "axis": (0, 1) if site == "state" else (0, 1, 2),
                    "type": "dynamic",
                },
            }
            for site in sites
        ],
        "algorithm": None,
        "linear_attention": [
            {"module_name": "decoder.layers.*.self_attention", "cfg": {"state": {"block_v": 16}}}
        ],
    }


def _config_for_mode(mode):
    sites = ("state", "w") if mode == "both" else ((mode,) if mode in ("state", "w") else ())
    if mode in ("prefill_fp8", "prefill_nvfp4"):
        sites = ("w",)
    if mode.startswith("decode"):
        sites = ("state",)
    if mode == "state_int8":
        sites = ("state",)
    cfg = _gdn_config(sites)
    if mode.endswith("int8"):
        cfg["quant_cfg"][1]["cfg"].update(num_bits=8, unsigned=False, narrow_range=True)
    if mode.startswith("decode"):
        policy = cfg["linear_attention"][0]["cfg"]
        policy.update(backend="matmul", decode={"implementation": "triton"})
        if mode.startswith("decode_replay"):
            policy["decode"].update(mode="replay", replay={"window": 8})
    if mode.startswith("prefill"):
        policy = cfg["linear_attention"][0]["cfg"]
        policy["backend"] = "matmul"
        if mode == "prefill_solve":
            policy["solve"] = {"method": "neumann", "degree": 1, "implementation": "triton"}
        elif mode == "prefill_arithmetic":
            policy["matmul"] = {
                "output_value": {"accumulator_dtype": "float16", "reduction_block": 16}
            }
            policy["elementwise"] = {"value_residual": "bfloat16"}
        else:
            attributes = {"num_bits": (4, 3), "axis": (0, 1, 2), "type": "dynamic"}
            if mode == "prefill_nvfp4":
                attributes = {
                    "num_bits": (2, 1),
                    "type": "dynamic",
                    "block_sizes": {-1: 16, "type": "dynamic", "scale_bits": (4, 3)},
                }
            cfg["quant_cfg"].append({"quantizer_name": "*linear_attn_sites.*", "cfg": attributes})
            cfg["quant_cfg"].append({"quantizer_name": "*gdn_w_quantizer", "cfg": attributes})
    return cfg, sites


def _test_gdn_qat_helper(rank, size, mode, checkpoint_path):
    initialize_for_megatron(
        tensor_model_parallel_size=size, pipeline_model_parallel_size=1, seed=SEED
    )
    model = _make_model(size)
    original_forward = get_forward(model)

    def forward(m):
        enabled = any(
            getattr(getattr(layer, "linear_attention_config", None), "decode", None) is not None
            for layer in m.modules()
        )
        with linear_attention_training_phase(m, [64, 64]) if enabled else nullcontext():
            return original_forward(m)

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

    cfg, sites = _config_for_mode(mode)
    mtq.quantize(model, cfg)
    gdn_modules = [m for m in model.modules() if isinstance(m, _QuantGatedDeltaNet)]
    assert gdn_modules, "no GatedDeltaNet layer was wrapped"
    for module in gdn_modules:
        assert module.gdn_state_qdq_block_v == 16
        assert module.gdn_state_quantizer.is_enabled == ("state" in sites)
        assert module.gdn_w_quantizer.is_enabled == ("w" in sites)
        # Checkpointing must save the resolved policy, including edits after conversion.
        module.linear_attention_config = LinearAttentionConfig(
            **{**module.linear_attention_config.model_dump(), "state": {"block_v": 32}}
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
        assert module.linear_attn_sites.is_enabled == original.linear_attn_sites.is_enabled
        assert module.gdn_state_quantizer.is_enabled == ("state" in sites)
        assert module.gdn_w_quantizer.is_enabled == ("w" in sites)
        assert module._linear_attn_state_format == original._linear_attn_state_format
        assert module.gdn_state_qdq_block_v == 32
        assert module.in_proj.weight.grad is not None
        assert torch.isfinite(module.in_proj.weight.grad).all()

    # The checkpoint helper already ran backward through the restored model.
    optimizer = torch.optim.SGD(restored.parameters(), lr=1e-3)
    before = restored_gdn[0].in_proj.weight.detach().clone()
    optimizer.step()
    assert not torch.equal(restored_gdn[0].in_proj.weight, before)


# Cold FLA/TileLang compilation plus a two-rank checkpoint exceeds the lane's 120s default.
@pytest.mark.timeout(300)
@pytest.mark.parametrize("tp_size", [1, 2])
@pytest.mark.parametrize(
    "mode",
    [
        "state",
        "w",
        "both",
        "prefill_fp8",
        "prefill_nvfp4",
        "prefill_arithmetic",
        "prefill_solve",
        "decode_token",
        "decode_replay",
        "state_int8",
        "decode_token_int8",
        "decode_replay_int8",
    ],
)
def test_gdn_qat_and_sharded_restore(request, tmp_path, tp_size, mode):
    """Train through state/W QDQ after a Megatron distributed-checkpoint round trip."""
    if mode in ("state", "both") and torch.cuda.get_device_capability() < (8, 9):
        pytest.skip("State QDQ needs native E4M3 conversion (SM89+)")
    workers = request.getfixturevalue(f"dist_workers_size_{tp_size}")
    workers.run(_test_gdn_qat_helper, mode, tmp_path)


def _test_gdn_context_parallel_helper(rank, size, mode):
    initialize_for_megatron(context_parallel_size=size, seed=SEED)
    model = _make_model(1, context_parallel_size=size)
    with pytest.raises(NotImplementedError, match="context parallelism"):
        mtq.quantize(model, _config_for_mode(mode)[0])


@pytest.mark.parametrize(
    "mode",
    [
        "state",
        "w",
        "prefill_fp8",
        "prefill_arithmetic",
        "prefill_solve",
        "decode_token",
        "decode_replay",
        "state_int8",
        "decode_token_int8",
        "decode_replay_int8",
    ],
)
def test_gdn_context_parallel_rejected(dist_workers_size_2, mode):
    """Reject unqualified Megatron CP even when it does not pass an FLA CP context."""
    dist_workers_size_2.run(_test_gdn_context_parallel_helper, mode)
