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

import pytest
import torch
from _test_utils.torch.megatron.models import get_mcore_gpt_model
from _test_utils.torch.megatron.utils import (
    get_forward,
    initialize_for_megatron,
    sharded_state_dict_test_helper,
)

import modelopt.torch.quantization as mtq

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
    }


def _test_gdn_qat_helper(rank, size, mode, checkpoint_path):
    initialize_for_megatron(
        tensor_model_parallel_size=size, pipeline_model_parallel_size=1, seed=SEED
    )
    model = _make_model(size)
    forward = get_forward(model)
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

    sites = ("state", "w") if mode == "both" else (mode,)
    mtq.quantize(model, _gdn_config(sites))
    gdn_modules = [m for m in model.modules() if isinstance(m, _QuantGatedDeltaNet)]
    assert gdn_modules, "no GatedDeltaNet layer was wrapped"
    for module in gdn_modules:
        assert module.gdn_state_quantizer.is_enabled == ("state" in sites)
        assert module.gdn_w_quantizer.is_enabled == ("w" in sites)

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

    for site in sites:
        mtq.disable_quantizer(model, f"*gdn_{site}_quantizer")
    with torch.no_grad():
        torch.testing.assert_close(forward(model), loss_ref, rtol=1e-4, atol=1e-4)
    for site in sites:
        mtq.enable_quantizer(model, f"*gdn_{site}_quantizer")

    restored = _make_model(size)
    sharded_state_dict_test_helper(checkpoint_path, model, restored, forward)
    restored_gdn = [m for m in restored.modules() if isinstance(m, _QuantGatedDeltaNet)]
    assert len(restored_gdn) == len(gdn_modules)
    for module in restored_gdn:
        assert module.gdn_state_quantizer.is_enabled == ("state" in sites)
        assert module.gdn_w_quantizer.is_enabled == ("w" in sites)
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
@pytest.mark.parametrize("mode", ["state", "w", "both"])
def test_gdn_qat_and_sharded_restore(request, tmp_path, tp_size, mode):
    """Train through state/W QDQ after a Megatron distributed-checkpoint round trip."""
    if mode != "w" and torch.cuda.get_device_capability() < (8, 9):
        pytest.skip("State QDQ needs native E4M3 conversion (SM89+)")
    workers = request.getfixturevalue(f"dist_workers_size_{tp_size}")
    workers.run(_test_gdn_qat_helper, mode, tmp_path)


def _test_gdn_context_parallel_helper(rank, size, mode):
    initialize_for_megatron(context_parallel_size=size, seed=SEED)
    model = _make_model(1, context_parallel_size=size)
    with pytest.raises(NotImplementedError, match="context parallelism"):
        mtq.quantize(model, _gdn_config((mode,)))


@pytest.mark.parametrize("mode", ["state", "w"])
def test_gdn_context_parallel_rejected(dist_workers_size_2, mode):
    """Reject unqualified Megatron CP even when it does not pass an FLA CP context."""
    dist_workers_size_2.run(_test_gdn_context_parallel_helper, mode)
