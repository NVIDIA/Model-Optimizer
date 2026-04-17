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

"""GPU end-to-end tests for WaterSIC KV-cache quantization."""

import pytest
import torch
from _test_utils.torch.transformers_models import get_tiny_llama

import modelopt.torch.quantization as mtq
from modelopt.torch.quantization.plugins.huggingface import _QuantAttention


@pytest.fixture
def tiny_llama():
    return get_tiny_llama()


@pytest.fixture
def calib_loop():
    def forward_loop(m):
        # Use vocab_size=32 matching the tiny_llama fixture
        input_ids = torch.randint(0, 32, (2, 32), device=next(m.parameters()).device)
        m(input_ids)

    return forward_loop


class TestWaterSICKVEndToEnd:
    """End-to-end GPU tests for WaterSIC KV-cache quantization."""

    def test_standalone_watersic_kv(self, tiny_llama, calib_loop):
        """Test WaterSIC KV-cache quantization as a standalone algorithm."""
        model = tiny_llama.to("cuda")
        model.eval()

        config = {
            "quant_cfg": [
                {"quantizer_name": "*", "enable": False},
                {"quantizer_name": "*[kv]_bmm_quantizer", "enable": True},
            ],
            "algorithm": {
                "method": "watersic_kv",
                "target_rate": 4.0,
                "use_sequential": False,
            },
        }

        model = mtq.quantize(model, config, forward_loop=calib_loop)

        # Verify _watersic_kv_state exists on _QuantAttention modules
        attn_modules = [m for m in model.modules() if isinstance(m, _QuantAttention)]
        assert len(attn_modules) > 0, "No _QuantAttention modules found"

        for m in attn_modules:
            assert hasattr(m, "_watersic_kv_state"), f"Module {m} missing _watersic_kv_state"
            state = m._watersic_kv_state
            assert state.rate > 0, f"Rate should be positive, got {state.rate}"
            assert state.rate < 10, f"Rate should be < 10, got {state.rate}"

    def test_composable_with_fp8_weights(self, tiny_llama, calib_loop):
        """Test composition with FP8 weight quantization."""
        model = tiny_llama.to("cuda")
        model.eval()

        # Step 1: FP8 weight quantization
        model = mtq.quantize(model, mtq.FP8_DEFAULT_CFG, forward_loop=calib_loop)

        # Step 2: WaterSIC KV-cache quantization
        watersic_config = {
            "quant_cfg": [
                {"quantizer_name": "*", "enable": False},
                {"quantizer_name": "*[kv]_bmm_quantizer", "enable": True},
            ],
            "algorithm": {
                "method": "watersic_kv",
                "target_rate": 3.0,
                "use_sequential": False,
            },
        }

        model = mtq.quantize(model, watersic_config, forward_loop=calib_loop)

        # Verify model produces valid output (no NaN)
        input_ids = torch.randint(0, 32, (1, 16), device="cuda")
        with torch.no_grad():
            output = model(input_ids)
        assert not torch.isnan(output.logits).any(), "Output contains NaN values"

    def test_kl_aware_mode(self, tiny_llama, calib_loop):
        """Test KL-aware importance weighting."""
        model = tiny_llama.to("cuda")
        model.eval()

        config = {
            "quant_cfg": [
                {"quantizer_name": "*", "enable": False},
                {"quantizer_name": "*[kv]_bmm_quantizer", "enable": True},
            ],
            "algorithm": {
                "method": "watersic_kv",
                "target_rate": 4.0,
                "use_sequential": False,
                "kl_aware": True,
                "importance_clip": 20.0,
            },
        }

        model = mtq.quantize(model, config, forward_loop=calib_loop)

        # Verify _watersic_kv_state exists on _QuantAttention modules
        attn_modules = [m for m in model.modules() if isinstance(m, _QuantAttention)]
        assert len(attn_modules) > 0, "No _QuantAttention modules found"

        for m in attn_modules:
            assert hasattr(m, "_watersic_kv_state"), f"Module {m} missing _watersic_kv_state"
            state = m._watersic_kv_state
            assert state.rate > 0, f"Rate should be positive, got {state.rate}"
            assert state.rate < 10, f"Rate should be < 10, got {state.rate}"
