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

"""End-to-end scalar selection using ModelOpt weight quantization."""

import copy

import pytest
import torch
from torch import nn

import modelopt.torch.opt as mto
import modelopt.torch.quantization as mtq
from experimental.softmax_reparameterization import head_kl, search_head


@pytest.mark.parametrize("bias", [False, True])
def test_selection_matches_independent_candidates_and_preserves_source(bias, tmp_path):
    torch.manual_seed(12)
    head = nn.Linear(16, 32, bias=bias)
    embedding = nn.Embedding(32, 16)
    embedding.weight = head.weight
    original = head.weight.detach().clone()
    states = torch.randn(23, 16)
    fit, val, test = states[:8], states[8:17], states[17:]
    config = {
        "quant_cfg": [
            {"quantizer_name": "*", "enable": False},
            {"quantizer_name": "*weight_quantizer", "cfg": {"num_bits": 4, "axis": 0}},
        ],
        "algorithm": "max",
    }
    original_config = copy.deepcopy(config)
    grid = (0, 1, -1, 4)
    result = search_head(head, fit, val, config, grid, batch_size=4)
    independent = []
    for t in grid:
        shifted = copy.deepcopy(head).eval()
        with torch.no_grad():
            shifted.weight.copy_(original - t * original.mean(0))
        torch.testing.assert_close(
            shifted(test).softmax(-1), head(test).softmax(-1), atol=1e-6, rtol=1e-5
        )
        quantized = mtq.quantize(nn.Sequential(shifted), copy.deepcopy(config))
        independent.append(head_kl(head, quantized, val, batch_size=9))
    assert [kl for _, kl in result.validation_kl] == pytest.approx(independent, abs=1e-7)
    assert result.coefficient == grid[independent.index(min(independent))]
    assert head_kl(head, result.model, val) <= independent[0] + 1e-7
    assert config == original_config
    assert embedding.weight is head.weight
    assert head.training
    assert not hasattr(head, "weight_quantizer")
    torch.testing.assert_close(head.weight, original, rtol=0, atol=0)
    assert result.model[0].weight.data_ptr() != head.weight.data_ptr()
    path = tmp_path / "head.pt"
    mto.save(result.model, path)
    restored = mto.restore(nn.Sequential(copy.deepcopy(head)), path)
    torch.testing.assert_close(restored(test), result.model(test), rtol=0, atol=0)


def test_ties_keep_first_and_invalid_candidates_fail():
    head = nn.Linear(16, 4, bias=False)
    with torch.no_grad():
        head.weight.zero_()
    states = torch.ones(3, 16)
    config = {
        "quant_cfg": [
            {"quantizer_name": "*", "enable": False},
            {"quantizer_name": "*weight_quantizer", "cfg": {"num_bits": 4, "axis": 0}},
        ],
        "algorithm": "max",
    }
    result = search_head(head, states, states.clone(), config, (1, 0, -1))
    assert result.coefficient == 1
    assert all(kl == 0 for _, kl in result.validation_kl)
    for grid in ((1,), (0, 0), (0, float("nan"))):
        with pytest.raises(ValueError, match="coefficients"):
            search_head(head, states, states, config, grid)
    with pytest.raises(ValueError, match="nonempty"):
        search_head(head, states, states[:0], config)
    config["quant_cfg"].append(
        {"quantizer_name": "*input_quantizer", "cfg": {"num_bits": 8, "axis": None}}
    )
    with pytest.raises(ValueError, match="weight-only"):
        search_head(head, states, states, config, (0,))
