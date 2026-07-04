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

"""CPU tests for the Triton flash attention module.

The ``@triton.jit`` kernels and the ``attention`` / ``attention_calibrate``
Python wrappers require a GPU and are fully exercised in
``tests/gpu/torch/sparsity/attention_sparsity/test_triton_fa*.py``.

These tests verify CPU-safe wrapper behavior without executing a Triton kernel.
"""

from contextlib import nullcontext

import pytest
import torch


def test_triton_fa_importable_on_cpu():
    """Module imports cleanly without CUDA; exports the public API names."""
    try:
        import triton  # noqa: F401
    except ImportError:
        pytest.skip("triton is not installed")

    from modelopt.torch.kernels.common.attention import triton_fa
    from modelopt.torch.kernels.sparsity.attention import calibrate

    assert "attention" in triton_fa.__all__
    assert callable(calibrate.attention_calibrate)


def test_forward_buckets_autotune_key_without_bucketing_grid(monkeypatch):
    """Reuse autotune results by length regime without launching extra query tiles."""
    pytest.importorskip("triton")

    from modelopt.torch.kernels.common.attention import triton_fa

    class CapturingKernel:
        def __getitem__(self, grid):
            self.grid = grid

            def launch(*args, **kwargs):
                self.kwargs = kwargs

            return launch

    kernel = CapturingKernel()
    monkeypatch.setattr(triton_fa, "_attn_fwd", kernel)
    monkeypatch.setattr(triton_fa.torch.cuda, "device", lambda _device: nullcontext())
    monkeypatch.setattr(triton_fa, "_load_sparsity_helpers", lambda: None)
    monkeypatch.setattr(triton_fa, "_load_qdq_helpers", lambda: None)

    seq_len = 129
    q = torch.empty(seq_len, 2, 16)
    k = torch.empty(seq_len, 1, 16)
    v = torch.empty_like(k)
    starts = torch.tensor([0], dtype=torch.int32)
    lengths = torch.tensor([seq_len], dtype=torch.int32)

    triton_fa.attention(q, k, v, starts, lengths, seq_len)

    assert kernel.kwargs["N_CTX"] == 256
    assert kernel.grid({"BLOCK_M": 64}) == (1, 2, 3)
