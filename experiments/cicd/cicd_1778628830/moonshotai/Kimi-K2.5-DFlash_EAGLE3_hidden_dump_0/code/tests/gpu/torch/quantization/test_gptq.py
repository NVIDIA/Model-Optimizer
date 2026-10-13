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

import copy
import time

import pytest
import torch
from _test_utils.torch.transformers_models import get_tiny_llama, get_tiny_tokenizer
from conftest import requires_triton

import modelopt.torch.quantization as mtq
from modelopt.torch.export.unified_export_hf import _export_quantized_weight
from modelopt.torch.quantization.model_calib import gptq
from modelopt.torch.quantization.qtensor.nvfp4_tensor import NVFP4QTensor
from modelopt.torch.quantization.utils import promote_nvfp4_static_quantizers
from modelopt.torch.quantization.utils.calib_utils import (
    compute_hessian_inverse,
    gptq_blockwise_update,
    gptq_blockwise_update_fused_scalar,
    update_hessian,
)
from modelopt.torch.utils.dataset_utils import create_forward_loop, get_dataset_dataloader

RAND_SEED = 42
torch.manual_seed(RAND_SEED)


def test_update_hessian():
    """Test for update_hessian function with both random and known inputs."""
    # Test 1: Random input - general functionality test
    torch.manual_seed(42)
    batch_size = 2
    seq_len = 3
    features = 4
    input_tensor = torch.randn(batch_size, seq_len, features, dtype=torch.float32)

    hessian = torch.zeros(features, features, dtype=torch.float32)
    n_samples = 0

    updated_hessian, new_n_samples = update_hessian(input_tensor, hessian, n_samples)

    # Verify output shape
    assert updated_hessian.shape == (features, features), (
        f"Expected hessian shape ({features}, {features}), got {updated_hessian.shape}"
    )

    # Verify sample count is updated correctly (incremented by total tokens = batch * seq_len)
    expected_n_samples = batch_size * seq_len
    assert new_n_samples == expected_n_samples, (
        f"Expected n_samples={expected_n_samples}, got {new_n_samples}"
    )

    # Verify hessian is not all zeros after update
    assert not torch.allclose(updated_hessian, torch.zeros_like(updated_hessian)), (
        "Hessian should not be all zeros after update"
    )

    # Verify hessian is symmetric (should be for outer product X @ X.T)
    assert torch.allclose(updated_hessian, updated_hessian.t()), "Hessian should be symmetric"

    # Test 2: Known input - verify correct hessian calculation
    batch_size = 6
    seq_len = 2
    features = 2
    input_tensor = torch.ones(batch_size, seq_len, features, dtype=torch.float32)

    hessian = torch.zeros(features, features, dtype=torch.float32)
    n_samples = 0

    updated_hessian, new_n_samples = update_hessian(input_tensor, hessian, n_samples)

    # Manual calculation:
    # input_flat shape: (features, batch*seq) = (2, 12), all ones
    # n_samples = batch * seq = 12 (token count after flattening)
    # scaled_input = sqrt(2/12) * ones(2, 12)
    # outer_product = (2/12) * ones(2,12) @ ones(12,2) = [[2,2], [2,2]]
    expected_n_samples = batch_size * seq_len  # 12 tokens
    expected_hessian = torch.ones(features, features, dtype=torch.float32) * 2.0

    assert torch.allclose(updated_hessian, expected_hessian, atol=1e-5), (
        f"Expected hessian {expected_hessian}, got {updated_hessian}"
    )
    assert new_n_samples == expected_n_samples

    # Test 3: Accumulated hessians - verify equivalence
    # Processing [6,2,2] in one step should equal processing [2,2,2] three times
    seq_len = 2
    features = 2

    # Process in 3 steps of batch_size=2 (4 tokens each, 12 total)
    hessian_accumulated = torch.zeros(features, features, dtype=torch.float32)
    n_samples_accumulated = 0

    for i in range(3):
        input_batch = torch.ones(2, seq_len, features, dtype=torch.float32)
        hessian_accumulated, n_samples_accumulated = update_hessian(
            input_batch, hessian_accumulated, n_samples_accumulated
        )

    # Verify that accumulated result matches single-step result from Test 2
    assert torch.allclose(hessian_accumulated, updated_hessian, atol=1e-5), (
        f"Accumulated hessian should match single-step: expected {updated_hessian}, got {hessian_accumulated}"
    )
    assert torch.allclose(hessian_accumulated, expected_hessian, atol=1e-5), (
        f"Accumulated hessian should match expected: expected {expected_hessian}, got {hessian_accumulated}"
    )
    # 3 batches * 2 batch_size * 2 seq_len = 12 tokens
    assert n_samples_accumulated == 12, f"Expected n_samples=12, got {n_samples_accumulated}"


@pytest.mark.parametrize(
    ("block_size", "dim", "model_weight", "expect_weight_change"),
    [
        (16, 128, torch.randn(128, 128).to("cuda"), True),  # random weight
        (
            16,
            128,
            torch.ones(128, 128).to("cuda"),
            False,
        ),  # all same weight -> no quantization error -> no GPTQ update
    ],
)
def test_gptq_updates(block_size, dim, model_weight, expect_weight_change):
    model = torch.nn.Linear(dim, dim).to("cuda")
    model.weight.data = model_weight
    original_weight = model_weight.clone()
    input_tensor = torch.randn(2, 16, dim).to("cuda")
    quant_cfg = mtq.NVFP4_DEFAULT_CFG

    mtq.quantize(model, quant_cfg, forward_loop=lambda model: model(input_tensor))

    # Get qdq weight
    q_dq_weight = model.weight_quantizer(model.weight.data)

    # Restore original weight before GPTQ
    model.weight.data = original_weight.clone()

    # Run GPTQ through the public API
    gptq(model, forward_loop=lambda m: m(input_tensor), perc_damp=0.1, block_size=block_size)
    if expect_weight_change:
        # Weight must change as GPTQ updates weights to adjust for quantization error
        assert not torch.allclose(model.weight.data, q_dq_weight), "Weight should not be equal"
    else:
        assert torch.allclose(model.weight.data, q_dq_weight), "Weight should be equal"


def test_gptq_export_roundtrip():
    """Test that GPTQ export + dequantize produces weights matching in-memory QDQ."""
    torch.manual_seed(RAND_SEED)
    dim = 128
    block_size = 16

    # Step 1: Create a simple linear model and quantize to install NVFP4 quantizers
    model = torch.nn.Linear(dim, dim, dtype=torch.bfloat16).to("cuda")
    original_weight = model.weight.data.clone()
    input_tensor = torch.randn(2, 16, dim, dtype=torch.bfloat16).to("cuda")
    quant_cfg = mtq.NVFP4_DEFAULT_CFG

    mtq.quantize(model, quant_cfg, forward_loop=lambda m: m(input_tensor))

    # Restore original weight before GPTQ
    model.weight.data = original_weight.clone()

    # Step 2: Perform GPTQ — compute Hessian and update weights
    gptq(model, forward_loop=lambda m: m(input_tensor), perc_damp=0.1, block_size=block_size)

    # Save the QDQ reference from the quantizer applied to GPTQ'd weights
    gptq_weight_shape = model.weight.data.shape
    gptq_weight_dtype = model.weight.data.dtype
    qdq_ref = model.weight.data.clone()

    # Step 3: Export — converts weight to packed NVFP4 and registers scale buffers
    _export_quantized_weight(model, torch.bfloat16)

    # Verify export produced the expected buffers
    assert hasattr(model, "weight_scale"), "Export should register weight_scale buffer"
    assert hasattr(model, "weight_scale_2"), "Export should register weight_scale_2 buffer"

    # Step 4: Dequantize the exported packed weight and compare with QDQ reference
    packed_weight = model.weight.data
    weight_scale = model.weight_scale
    weight_scale_2 = model.weight_scale_2

    nvfp4_qtensor = NVFP4QTensor(gptq_weight_shape, gptq_weight_dtype, packed_weight)
    deq_weight = nvfp4_qtensor.dequantize(
        dtype=torch.bfloat16,
        scale=weight_scale,
        double_scale=weight_scale_2,
        block_sizes={-1: 16},
    )

    assert deq_weight.shape == qdq_ref.shape, (
        f"Shape mismatch: dequantized {deq_weight.shape} vs QDQ ref {qdq_ref.shape}"
    )
    assert torch.allclose(deq_weight, qdq_ref, atol=1e-2), (
        f"Dequantized weight does not match QDQ reference. "
        f"Max diff: {(deq_weight - qdq_ref).abs().max().item()}"
    )


@pytest.mark.parametrize(
    "quant_cfg", [mtq.NVFP4_DEFAULT_CFG, mtq.FP8_DEFAULT_CFG, mtq.INT4_BLOCKWISE_WEIGHT_ONLY_CFG]
)
def test_gptq_e2e_flow(quant_cfg):
    tokenizer = get_tiny_tokenizer()
    model = get_tiny_llama(vocab_size=tokenizer.vocab_size).to("cuda")

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    tokenizer.padding_side = "left"

    assert tokenizer.pad_token is not None, "Pad token cannot be set!"
    model.eval()

    quant_cfg = copy.deepcopy(quant_cfg)
    quant_cfg["algorithm"] = {"method": "gptq", "layerwise": True}
    calib_dataloader = get_dataset_dataloader(
        dataset_name="cnn_dailymail",
        tokenizer=tokenizer,
        batch_size=2,
        num_samples=8,
        device="cuda",
        include_labels=False,
    )

    calibrate_loop = create_forward_loop(dataloader=calib_dataloader)
    model = mtq.quantize(model, quant_cfg, forward_loop=calibrate_loop)


# ---------------------------------------------------------------------------
# Fused Triton GPTQ kernel tests for NVFP4 scalar quantization
# ---------------------------------------------------------------------------


def _make_nvfp4_test_data(quant_block_size, out_features, dim):
    """Create weight, weight_quantizer, block_amax, global_scale, and h_inv for NVFP4 GPTQ tests."""
    # Build a quantized Linear with NVFP4 static config at the desired block size
    model = torch.nn.Linear(dim, out_features, bias=False, device="cuda")
    weight = model.weight.data.clone()

    nvfp4_static_cfg = {
        "num_bits": (2, 1),
        "block_sizes": {-1: quant_block_size, "type": "static", "scale_bits": (4, 3)},
    }
    quant_cfg = {
        "quant_cfg": [
            {"quantizer_name": "*", "enable": False},
            {"quantizer_name": "*weight_quantizer", "cfg": nvfp4_static_cfg},
        ],
        "algorithm": "max",
    }
    inp = torch.randn(4, 32, dim, device="cuda")
    mtq.quantize(model, quant_cfg, forward_loop=lambda m: m(inp))
    promote_nvfp4_static_quantizers(model)

    # Restore original weight (GPTQ operates on original weights)
    model.weight.data = weight.clone()

    weight_quantizer = model.weight_quantizer
    block_amax = weight_quantizer.amax.reshape(out_features, -1).float()
    global_scale = weight_quantizer.global_amax.float().item() / (6.0 * 448.0)

    # Compute Hessian
    hessian = torch.zeros(dim, dim, dtype=torch.float32)
    hessian, _ = update_hessian(inp, hessian, 0)
    hessian = hessian.to("cuda")
    h_inv = compute_hessian_inverse(hessian, weight, perc_damp=0.01)

    return weight, weight_quantizer, block_amax, global_scale, h_inv


def _run_unfused_gptq_nvfp4(weight, weight_quantizer, h_inv, gptq_block_size):
    """Unfused NVFP4 GPTQ using the production blockwise update with weight_quantizer."""
    w = weight.float().clone()
    gptq_blockwise_update(w, h_inv, gptq_block_size, weight_quantizer)
    return w


def _run_fused_gptq_nvfp4(
    weight, block_amax, global_scale, h_inv, gptq_block_size, quant_block_size
):
    """Fused Triton GPTQ for NVFP4 using the production fused update."""
    w = weight.float().clone()
    gptq_blockwise_update_fused_scalar(
        w, block_amax, global_scale, h_inv, gptq_block_size, quant_block_size
    )
    return w


_NVFP4_QUANT_BLOCK_SIZES = [16, 128]
_NVFP4_GPTQ_BLOCK_SIZES = [16, 128]


@requires_triton
@pytest.mark.parametrize("quant_block_size", _NVFP4_QUANT_BLOCK_SIZES)
@pytest.mark.parametrize("gptq_block_size", _NVFP4_GPTQ_BLOCK_SIZES)
def test_fused_vs_unfused_nvfp4(quant_block_size, gptq_block_size):
    """Fused Triton NVFP4 GPTQ must match unfused production reference."""
    torch.manual_seed(42)
    dim = max(256, quant_block_size * 4)
    out_features = 64

    weight, weight_quantizer, block_amax, global_scale, h_inv = _make_nvfp4_test_data(
        quant_block_size,
        out_features,
        dim,
    )

    weight_fused = _run_fused_gptq_nvfp4(
        weight,
        block_amax,
        global_scale,
        h_inv,
        gptq_block_size,
        quant_block_size,
    )
    weight_unfused = _run_unfused_gptq_nvfp4(
        weight,
        weight_quantizer,
        h_inv,
        gptq_block_size,
    )

    assert not torch.equal(weight_fused, weight.float()), "Fused did not update weights"
    assert not torch.equal(weight_unfused, weight.float()), "Unfused did not update weights"

    diff = (weight_fused - weight_unfused).abs()
    max_abs = diff.max().item()
    mean_abs = diff.mean().item()
    denom = weight_unfused.abs().max().item()
    rel_max = max_abs / denom if denom > 0 else 0.0

    print(
        f"\n[nvfp4] gptq_bs={gptq_block_size} quant_bs={quant_block_size}: "
        f"max_abs={max_abs:.2e}  mean_abs={mean_abs:.2e}  rel_max={rel_max:.2e}"
    )

    torch.testing.assert_close(weight_fused, weight_unfused, atol=1e-4, rtol=1e-4)


_NVFP4_BENCH_CONFIGS = [
    (16, 128, 256, 512),
    (16, 128, 256, 2048),
    (16, 128, 256, 4096),
    (128, 128, 256, 512),
    (128, 128, 256, 2048),
    (128, 128, 256, 4096),
]


def bench_fused_nvfp4():
    """Benchmark fused Triton NVFP4 GPTQ vs unfused production loop (informational-only).

    Not collected by pytest. Run directly: ``python tests/gpu/torch/quantization/test_gptq.py``
    """

    def _bench(fn, n_warmup=2, n_iters=5):
        for _ in range(n_warmup):
            fn()
        torch.cuda.synchronize()
        total = 0.0
        for _ in range(n_iters):
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            fn()
            torch.cuda.synchronize()
            total += time.perf_counter() - t0
        return total / n_iters

    for quant_block_size, gptq_block_size, out_features, dim in _NVFP4_BENCH_CONFIGS:
        torch.manual_seed(42)
        weight, weight_quantizer, block_amax, global_scale, h_inv = _make_nvfp4_test_data(
            quant_block_size, out_features, dim
        )

        def run_fused():
            return _run_fused_gptq_nvfp4(
                weight, block_amax, global_scale, h_inv, gptq_block_size, quant_block_size
            )

        def run_unfused():
            return _run_unfused_gptq_nvfp4(weight, weight_quantizer, h_inv, gptq_block_size)

        t_fused = _bench(run_fused)
        t_unfused = _bench(run_unfused)
        speedup = t_unfused / t_fused if t_fused > 0 else float("inf")

        tag = f"qbs{quant_block_size}_gbs{gptq_block_size}_{out_features}x{dim}"
        print(
            f"[{tag}] Fused: {t_fused * 1e3:8.2f} ms | "
            f"Unfused: {t_unfused * 1e3:8.2f} ms | Speedup: {speedup:.1f}x"
        )


if __name__ == "__main__":
    bench_fused_nvfp4()
