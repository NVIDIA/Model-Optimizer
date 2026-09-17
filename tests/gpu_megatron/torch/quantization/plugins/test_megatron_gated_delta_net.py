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
from _test_utils.torch.megatron.utils import get_forward, initialize_for_megatron

import modelopt.torch.quantization as mtq

pytest.importorskip("fla")  # Megatron-Core GatedDeltaNet and the state QDQ kernel need fla
pytest.importorskip("megatron.core.ssm.gated_delta_net")

from modelopt.torch.quantization.plugins.gated_delta_net import _state_qdq_chunk_gated_delta_rule
from modelopt.torch.quantization.plugins.megatron import _QuantGatedDeltaNet

try:
    _state_qdq_chunk_gated_delta_rule()
except RuntimeError as e:
    pytest.skip(str(e), allow_module_level=True)

SEED = 1234
GDN_STATE_QUANT_CFG = {
    "quant_cfg": [
        {"quantizer_name": "*", "enable": False},
        {
            "quantizer_name": "*gdn_state_quantizer",
            "cfg": {"num_bits": (4, 3), "axis": (0, 1), "type": "dynamic"},
        },
    ],
    "algorithm": "max",
}


def _test_gdn_state_quant_helper(rank, size):
    initialize_for_megatron(
        tensor_model_parallel_size=size, pipeline_model_parallel_size=1, seed=SEED
    )
    model = get_mcore_gpt_model(
        tensor_model_parallel_size=size,
        num_layers=2,
        hidden_size=64,
        num_attention_heads=4,
        vocab_size=32,
        max_sequence_length=128,
        experimental_attention_variant="gated_delta_net",
    ).cuda()
    model.eval()  # no dropout, so the forwards differ only through the state quantizer
    forward = get_forward(model)
    with torch.no_grad():
        loss_ref = forward(model)

    model = mtq.quantize(model, GDN_STATE_QUANT_CFG, forward)

    gdn_modules = [m for m in model.modules() if isinstance(m, _QuantGatedDeltaNet)]
    assert gdn_modules, "no GatedDeltaNet layer was wrapped"
    assert all(m.gdn_state_quantizer.is_enabled for m in gdn_modules)
    assert all(m.gdn_state_qdq_block_v is None for m in gdn_modules)  # kernel's 64-column tile

    with torch.no_grad():
        loss_quant = forward(model)
    assert torch.isfinite(loss_quant).all()
    assert not torch.allclose(loss_quant, loss_ref, rtol=1e-4, atol=1e-4), (
        "FP8 state quantization must change the output"
    )

    mtq.disable_quantizer(model, "*gdn_state_quantizer")
    with torch.no_grad():
        assert torch.allclose(forward(model), loss_ref, rtol=1e-4, atol=1e-4)
    mtq.enable_quantizer(model, "*gdn_state_quantizer")

    # QAD trains through the segmented kernel: gradients must reach the GDN projections.
    forward(model).sum().backward()
    assert all(m.in_proj.weight.grad is not None for m in gdn_modules)
    assert all(torch.isfinite(m.in_proj.weight.grad).all() for m in gdn_modules)


def test_gdn_state_quant(dist_workers_size_1):
    """GatedDeltaNet layers get a ``gdn_state_quantizer`` whose FP8 quant-dequant of the recurrent
    state runs inside the chunked kernel during forward and backward."""
    dist_workers_size_1.run(_test_gdn_state_quant_helper)
