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


class _CapturingKernel:
    def __init__(self):
        self.launch_count = 0

    def __getitem__(self, grid):
        self.grid = grid

        def launch(*args, **kwargs):
            self.launch_count += 1
            self.kwargs = kwargs

        return launch


class _ForbiddenKernel:
    def __getitem__(self, grid):
        raise AssertionError("unexpected kernel launch")


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

    kernel = _CapturingKernel()
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


def test_forward_uses_minimal_shared_autotune_configs():
    pytest.importorskip("triton")

    from modelopt.torch.kernels.common.attention import triton_fa

    configs = triton_fa._FWD_CONFIGS
    assert [(config.kwargs["BLOCK_M"], config.kwargs["BLOCK_N"]) for config in configs] == [
        (16, 32),
        (64, 32),
        (128, 32),
    ]

    assert triton_fa._attn_fwd.keys == ["N_CTX", "HEAD_DIM", "Q_IS_FP32", "P_QDQ", "V_QDQ"]
    assert {(config.num_stages, config.num_warps) for config in triton_fa._SKIP_SERVE_CONFIGS} == {
        (stages, warps) for stages in (1, 2, 3) for warps in (4, 8)
    }
    assert triton_fa._attn_fwd_skip_serve.keys == ["N_CTX", "HEAD_DIM", "Q_IS_FP32", "IS_PAGED"]


@pytest.mark.parametrize(
    ("attention_kwargs", "expected_p_qdq", "expected_v_qdq"),
    [
        ({}, 0, 0),
        ({"p_qdq": "fp8"}, 1, 0),
        ({"p_qdq": "nvfp4", "v_qdq": "nvfp4"}, 2, 2),
        ({"p_qdq": "nvfp4", "sparsity_n": 2, "sparsity_m": 4}, 2, 0),
    ],
)
def test_forward_routes_every_mode_to_single_autotuner(
    monkeypatch, attention_kwargs, expected_p_qdq, expected_v_qdq
):
    """Every non-measurement launch uses the unified autotuner."""
    pytest.importorskip("triton")

    from modelopt.torch.kernels.common.attention import triton_fa

    kernel = _CapturingKernel()
    kernel.fn = _ForbiddenKernel()
    monkeypatch.setattr(triton_fa, "_attn_fwd", kernel)
    monkeypatch.setattr(triton_fa, "_attn_fwd_p_qdq", _ForbiddenKernel(), raising=False)
    monkeypatch.setattr(triton_fa.torch.cuda, "device", lambda _device: nullcontext())
    monkeypatch.setattr(triton_fa, "_load_sparsity_helpers", lambda: None)
    monkeypatch.setattr(triton_fa, "_load_qdq_helpers", lambda: None)

    seq_len = 129
    q = torch.empty(seq_len, 2, 16)
    k = torch.empty(seq_len, 1, 16)
    v = torch.empty_like(k)
    starts = torch.tensor([0], dtype=torch.int32)
    lengths = torch.tensor([seq_len], dtype=torch.int32)

    triton_fa.attention(q, k, v, starts, lengths, seq_len, **attention_kwargs)

    assert kernel.kwargs["P_QDQ"] == expected_p_qdq
    assert kernel.kwargs["V_QDQ"] == expected_v_qdq


def test_forward_measurement_uses_one_fixed_launch(monkeypatch):
    """Counter measurement bypasses autotuning to avoid repeated atomic updates."""
    pytest.importorskip("triton")

    from modelopt.torch.kernels.common.attention import triton_fa

    kernel = _ForbiddenKernel()
    kernel.fn = _CapturingKernel()
    monkeypatch.setattr(triton_fa, "_attn_fwd", kernel)
    monkeypatch.setattr(triton_fa, "_attn_fwd_skip_serve", _ForbiddenKernel())
    monkeypatch.setattr(triton_fa.torch.cuda, "device", lambda _device: nullcontext())
    monkeypatch.setattr(triton_fa, "_load_sparsity_helpers", lambda: None)
    monkeypatch.setattr(triton_fa, "_load_qdq_helpers", lambda: None)

    seq_len = 129
    q = torch.empty(seq_len, 2, 16)
    k = torch.empty(seq_len, 1, 16)
    v = torch.empty_like(k)
    starts = torch.tensor([0], dtype=torch.int32)
    lengths = torch.tensor([seq_len], dtype=torch.int32)

    triton_fa.attention(
        q, k, v, starts, lengths, seq_len, skip_softmax_threshold=0.1, measure_sparsity=True
    )

    assert kernel.fn.launch_count == 1
    assert kernel.fn.kwargs["BLOCK_M"] == 128
    assert kernel.fn.kwargs["BLOCK_N"] == 128
    assert kernel.fn.kwargs["num_stages"] == 1
    assert kernel.fn.kwargs["num_warps"] == 4


@pytest.mark.parametrize(
    ("q_len", "kv_len", "expected_block_m"),
    [(129, 129, 128), (1, 256, 16)],
)
def test_forward_skip_serving_keeps_kv_tile_and_uses_phase_q_tile(
    monkeypatch, q_len, kv_len, expected_block_m
):
    """Serving keeps BLOCK_N=128 while decode avoids padded 128-row Q work."""
    pytest.importorskip("triton")

    from modelopt.torch.kernels.common.attention import triton_fa

    kernel = _ForbiddenKernel()
    kernel.fn = _ForbiddenKernel()
    serving_kernel = _CapturingKernel()
    monkeypatch.setattr(triton_fa, "_attn_fwd", kernel)
    monkeypatch.setattr(triton_fa, "_attn_fwd_skip_serve", serving_kernel)
    monkeypatch.setattr(triton_fa.torch.cuda, "device", lambda _device: nullcontext())
    monkeypatch.setattr(triton_fa, "_load_sparsity_helpers", lambda: None)
    monkeypatch.setattr(triton_fa, "_load_qdq_helpers", lambda: None)

    q = torch.empty(q_len, 2, 16)
    k = torch.empty(kv_len, 1, 16)
    v = torch.empty_like(k)
    starts = torch.tensor([0], dtype=torch.int32)
    lengths = torch.tensor([q_len], dtype=torch.int32)
    kv_lengths = torch.tensor([kv_len], dtype=torch.int32)

    triton_fa.attention(
        q,
        k,
        v,
        starts,
        lengths,
        q_len,
        b_start_loc_k=starts,
        b_seq_len_k=kv_lengths,
        max_input_len_k=kv_len,
        skip_softmax_threshold=0.1,
    )

    assert serving_kernel.kwargs["BLOCK_M"] == expected_block_m
    assert serving_kernel.kwargs["BLOCK_N"] == 128


@pytest.mark.parametrize(
    "qdq_kwargs",
    [{"p_qdq": "nvfp4"}, {"p_qdq": "fp8"}, {"v_qdq": "nvfp4", "v_qdq_amax": 1.0}],
)
def test_forward_rejects_skip_softmax_with_qdq(monkeypatch, qdq_kwargs):
    """Active skip-softmax rejects P/V QDQ before any kernel launch."""
    pytest.importorskip("triton")

    from modelopt.torch.kernels.common.attention import triton_fa

    kernel = _ForbiddenKernel()
    kernel.fn = _ForbiddenKernel()
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

    with pytest.raises(ValueError, match="cannot be combined with attention quantization"):
        triton_fa.attention(
            q, k, v, starts, lengths, seq_len, skip_softmax_threshold=0.1, **qdq_kwargs
        )
