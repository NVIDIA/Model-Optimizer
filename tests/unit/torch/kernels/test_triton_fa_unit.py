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

"""Unit tests for Triton flash attention pure-Python helpers (no GPU required).

The ``@triton.jit`` kernels themselves require a GPU and are covered by
``tests/gpu/torch/sparsity/attention_sparsity/test_triton_fa*.py``.
These unit tests only exercise argument validation and module-level constants.
"""

import pytest
import torch


def _require_triton_module():
    """Import ``modelopt.torch.kernels.triton_fa`` directly, bypassing CUDA gates."""
    try:
        import triton  # noqa: F401
    except ImportError:
        pytest.skip("triton is not installed")

    from modelopt.torch.kernels import triton_fa

    return triton_fa


class TestPublicAPI:
    """Public surface: imports, exported names, and trivial attributes."""

    def test_module_exports(self):
        mod = _require_triton_module()
        assert "attention" in mod.__all__
        assert "attention_calibrate" in mod.__all__

    def test_log2e_constant(self):
        mod = _require_triton_module()
        import math

        assert pytest.approx(math.log2(math.e), rel=1e-5) == mod.LOG2E


class TestAttentionCalibrateArgValidation:
    """``attention_calibrate`` argument validation (runs before kernel launch)."""

    def test_empty_threshold_trials_raises(self):
        mod = _require_triton_module()
        # Tiny CPU tensors — the validation path raises before reaching the kernel.
        q = torch.empty(1, 1, 8)
        k = torch.empty(1, 1, 8)
        v = torch.empty(1, 1, 8)
        b_start_loc = torch.zeros(1, dtype=torch.int32)
        b_seq_len = torch.ones(1, dtype=torch.int32)

        with pytest.raises(ValueError, match="threshold_trials"):
            mod.attention_calibrate(
                q,
                k,
                v,
                b_start_loc,
                b_seq_len,
                max_input_len=1,
                threshold_trials=[],
            )

    def test_none_threshold_trials_raises(self):
        mod = _require_triton_module()
        q = torch.empty(1, 1, 8)
        k = torch.empty(1, 1, 8)
        v = torch.empty(1, 1, 8)
        b_start_loc = torch.zeros(1, dtype=torch.int32)
        b_seq_len = torch.ones(1, dtype=torch.int32)
        with pytest.raises(ValueError, match="threshold_trials"):
            mod.attention_calibrate(
                q,
                k,
                v,
                b_start_loc,
                b_seq_len,
                max_input_len=1,
                threshold_trials=None,
            )
