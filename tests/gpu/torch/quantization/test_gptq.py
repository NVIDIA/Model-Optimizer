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
from _test_utils.torch.quantization.models import ToyFusedExperts, ToyMoE
from _test_utils.torch.transformers_models import get_tiny_llama
from conftest import requires_triton

import modelopt.torch.quantization as mtq
from modelopt.torch.export.unified_export_hf import _export_quantized_weight
from modelopt.torch.quantization.extensions import get_cuda_ext_ggml
from modelopt.torch.quantization.ggml.common import pinned_packed_weight
from modelopt.torch.quantization.ggml.gptq import GGMLGPTQHelper
from modelopt.torch.quantization.model_calib import gptq
from modelopt.torch.quantization.nn import QuantModuleRegistry
from modelopt.torch.quantization.qtensor.nvfp4_tensor import NVFP4QTensor
from modelopt.torch.quantization.utils import calib_utils, promote_nvfp4_static_quantizers
from modelopt.torch.quantization.utils.calib_utils import (
    GPTQHelper,
    compute_hessian_inverse,
    gptq_blockwise_update,
    gptq_blockwise_update_fused_scalar,
    update_hessian,
)
from modelopt.torch.utils import get_available_gpu_memory
from modelopt.torch.utils.dataset_utils import create_forward_loop, get_dataset_dataloader

RAND_SEED = 42
torch.manual_seed(RAND_SEED)


def _report_available(monkeypatch, available):
    """Make each GPU report ``available[gpu]`` bytes for a new allocation, with no forward measured."""
    monkeypatch.setattr(calib_utils, "get_available_gpu_memory", lambda gpu: available[gpu])
    monkeypatch.setattr(calib_utils, "_calib_forward_peak", {})


def test_hessian_of_a_full_gpu_goes_to_an_idle_gpu(need_2_gpus, monkeypatch):
    """A Hessian whose GPU is full lands on an idle GPU rather than the CPU, with the same values."""
    total = torch.cuda.get_device_properties(0).total_memory
    _report_available(monkeypatch, {0: total // 10, 1: total * 9 // 10})
    linear = torch.nn.Linear(64, 8, bias=False, device="cuda:0")
    shard = torch.zeros(1, device="cuda:1")  # the model also lives on cuda:1
    helper = GPTQHelper(linear, "linear", offload_to_cpu=True)
    assert helper.hessian.device == shard.device

    inputs = torch.randn(32, 64, device="cuda:0")
    helper.accumulate(inputs)
    reference, _ = update_hessian(inputs, torch.zeros(64, 64, device="cuda:0"), 0)
    torch.testing.assert_close(helper.hessian.to("cuda:0"), reference)


@pytest.mark.parametrize("inference_mode", [False, True])
def test_expert_updates_on_another_gpu_match_one_gpu(need_2_gpus, monkeypatch, inference_mode):
    """GGML expert updates spread over two GPUs give the one-GPU weights and codes, pinned at home,
    also on inference tensors under the caller's inference mode, which worker threads do not inherit.
    """
    pytest.importorskip("transformers")  # the fused-experts quant wrapper is the HF plugin's
    get_cuda_ext_ggml(raise_if_failed=True)  # updates move only with the CUDA encoders
    # Two GGML groups per expert input, so the Hessian carries the first group's error to the next.
    width = 512
    shard = torch.zeros(1, device="cuda:1")  # noqa: F841  # the model also lives on cuda:1
    config = {
        "quant_cfg": [
            {"quantizer_name": "*", "enable": False},
            {
                "quantizer_name": "*weight_quantizer",
                "cfg": {"num_bits": "iq2_xxs", "block_sizes": {-1: 256}, "backend": "ggml"},
            },
        ],
        "algorithm": {"method": "gptq", "block_size": 256, "perc_damp": 0.3},
    }
    devices = []
    update_weights = GGMLGPTQHelper.update_weights

    def recording_update_weights(self, *args, device=None, **kwargs):
        devices.append(device)
        return update_weights(self, *args, device=device, **kwargs)

    monkeypatch.setattr(GGMLGPTQHelper, "update_weights", recording_update_weights)
    experts = []
    try:
        with torch.inference_mode(inference_mode):
            inputs = torch.randn(96, width, device="cuda")
            for gpus in ([0], [0, 1]):
                monkeypatch.setattr(calib_utils, "get_process_gpus", lambda gpus=gpus: gpus)
                torch.manual_seed(RAND_SEED)
                model = ToyMoE(width, width).cuda()
                mtq.quantize(model, config, forward_loop=lambda m: m(inputs))
                experts.append(model.mlp.experts)
    finally:
        QuantModuleRegistry.unregister(ToyFusedExperts)

    assert torch.device("cuda", 1) in devices
    one_gpu, two_gpus = experts
    for name in ("gate_up_proj", "down_proj"):
        assert torch.equal(getattr(one_gpu, name), getattr(two_gpus, name))
        for q1, q2 in zip(
            getattr(one_gpu, f"{name}_weight_quantizers"),
            getattr(two_gpus, f"{name}_weight_quantizers"),
        ):
            assert pinned_packed_weight(q2, "iq2_xxs").device == getattr(two_gpus, name).device
            assert torch.equal(
                pinned_packed_weight(q1, "iq2_xxs"), pinned_packed_weight(q2, "iq2_xxs")
            )


@pytest.mark.parametrize(
    ("num_bits", "shape", "cuda_extension"),
    [
        ("iq2_xxs", (16384, 2048), True),  # a projection 8x wider than its input
        ("iq2_xxs", (65536, 256), True),  # one GPTQ block spans the input
        ("iq1_s", (1024, 512), True),
        # The PyTorch encoder's scratch grows with block_chunk_size, past the room check.
        ("iq1_s", (1024, 512), False),
    ],
)
def test_an_update_that_may_move_fits_its_room(
    need_2_gpus, monkeypatch, num_bits, shape, cuda_extension
):
    """An expert update that may run on another GPU peaks there within the room it is admitted with."""
    if cuda_extension:
        get_cuda_ext_ggml(raise_if_failed=True)  # without it, no update may move to check
    else:
        monkeypatch.setattr(get_cuda_ext_ggml, "extension", None, raising=False)
    out_features, in_features = shape
    torch.manual_seed(RAND_SEED)
    linear = torch.nn.Linear(
        in_features, out_features, bias=False, dtype=torch.bfloat16, device="cuda:0"
    )
    inputs = torch.randn(512, in_features, dtype=torch.bfloat16, device="cuda:0")
    config = {
        "quant_cfg": [
            {"quantizer_name": "*", "enable": False},
            {
                "quantizer_name": "*weight_quantizer",
                "cfg": {"num_bits": num_bits, "block_sizes": {-1: 256}, "backend": "ggml"},
            },
        ],
        "algorithm": "max",
    }
    mtq.quantize(linear, config, forward_loop=lambda m: m(inputs))
    helper = GGMLGPTQHelper(linear, "linear", log_mse=False)
    helper.accumulate(inputs)
    block_size = 256
    worker = torch.device("cuda", 1)
    torch.cuda.synchronize(worker)
    torch.cuda.reset_peak_memory_stats(worker)
    allocated = torch.cuda.memory_allocated(worker)
    with torch.cuda.device(worker):
        helper.update_weights(block_size, 0.3, linear.weight, device=worker)
    peak = torch.cuda.max_memory_allocated(worker) - allocated

    if helper.updates_on_any_device():
        assert peak <= calib_utils._update_room(helper, linear.weight, block_size)


@pytest.mark.parametrize("blocker", ["idle_gpu_full", "idle_gpu_too_small", "multi_rank"])
def test_hessian_of_a_full_gpu_falls_back_to_the_cpu(need_2_gpus, monkeypatch, blocker):
    """With no other GPU that can take it, or on any rank of a multi-rank run, it goes to the CPU."""
    total = torch.cuda.get_device_properties(0).total_memory
    reserve = int(calib_utils._HESSIAN_UNMEASURED_RESERVE * total)
    idle = {
        "idle_gpu_full": reserve // 2,
        "idle_gpu_too_small": reserve + 64 * 64 * 4 - 1,  # one byte short of reserve + Hessian
        "multi_rank": total,
    }[blocker]
    _report_available(monkeypatch, {0: total // 10, 1: idle})
    if blocker == "multi_rank":
        monkeypatch.setattr(calib_utils.dist, "is_initialized", lambda: True)
        monkeypatch.setattr(calib_utils.dist, "get_world_size", lambda group=None: 2)
    linear = torch.nn.Linear(64, 8, bias=False, device="cuda:0")
    shard = torch.zeros(1, device="cuda:1")  # noqa: F841  # the model also lives on cuda:1
    helper = GPTQHelper(linear, "linear", offload_to_cpu=True)
    assert helper.hessian.device == torch.device("cpu")


def test_bf16_hessian_input_gives_the_float32_hessian():
    """``input_dtype=torch.bfloat16`` gives the float32 Hessian, on the inputs' GPU or the CPU,
    and with float32-output GEMMs makes no float32 copy of the inputs."""
    torch.manual_seed(0)
    batches = [torch.randn(4096, 1024, device="cuda", dtype=torch.bfloat16) for _ in range(3)]
    inputs = torch.cat(batches).double()
    expected = 2 * inputs.t() @ inputs / inputs.shape[0]
    fast = calib_utils._has_float32_out_gemm(batches[0].device)

    for device in ("cuda", "cpu"):
        hessian, n_samples = torch.zeros(1024, 1024, device=device), 0
        for batch in batches:
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats()
            base = torch.cuda.memory_allocated()
            hessian, n_samples = update_hessian(batch, hessian, n_samples, torch.bfloat16)
            growth = torch.cuda.max_memory_allocated() - base
        # Tensor cores round float32 sums their own way: ~1e-5 of the diagonal (2), not BF16's 1e-2.
        torch.testing.assert_close(hessian.cuda().double(), expected, rtol=0, atol=1e-4)
        if device == "cuda" and fast:
            assert growth < batch.numel() * 4


def test_gptq_hessian_input_dtype_reaches_the_hessian(monkeypatch):
    """A recipe's ``hessian_input_dtype: bfloat16`` drops the float32 copy of each batch that the
    default float32 Hessian GEMM makes."""
    if not calib_utils._has_float32_out_gemm(torch.device("cuda", torch.cuda.current_device())):
        pytest.skip("needs CUDA GEMMs that take out_dtype")
    inputs = torch.randn(4, 1024, 2048, device="cuda", dtype=torch.bfloat16)
    peaks = {}
    for hessian_input_dtype in ("float32", "bfloat16"):
        monkeypatch.setattr(calib_utils, "_calib_forward_peak", {})
        model = torch.nn.Linear(2048, 16, bias=False, device="cuda", dtype=torch.bfloat16)
        quant_cfg = {
            "quant_cfg": [
                {"quantizer_name": "*", "enable": False},
                {"quantizer_name": "*weight_quantizer", "cfg": {"num_bits": 8, "axis": 0}},
            ],
            "algorithm": {"method": "gptq", "hessian_input_dtype": hessian_input_dtype},
        }
        mtq.quantize(model, quant_cfg, forward_loop=lambda m: m(inputs))
        peaks[hessian_input_dtype] = calib_utils._calib_forward_peak[torch.cuda.current_device()]
    # The float32 GEMM reads a float32 copy of the batch: 4 bytes per element.
    assert peaks["float32"] - peaks["bfloat16"] >= inputs.numel() * 2


@pytest.mark.parametrize(
    ("measured_fraction", "stays"),
    [(None, False), (0.05, True), (0.25, False)],
    ids=["unmeasured", "small_forward", "large_forward"],
)
def test_a_measured_calibration_forward_sets_the_hessian_reserve(
    monkeypatch, measured_fraction, stays
):
    """At 30% available, a GPU keeps a Hessian only once a measured forward leaves it room."""
    total = torch.cuda.get_device_properties(0).total_memory
    _report_available(monkeypatch, {0: total * 3 // 10})
    monkeypatch.setattr(calib_utils, "get_process_gpus", lambda: [0])
    if measured_fraction is not None:
        calib_utils._calib_forward_peak[0] = int(measured_fraction * total)
    linear = torch.nn.Linear(64, 8, bias=False, device="cuda:0")
    helper = GPTQHelper(linear, "linear", offload_to_cpu=True)
    assert helper.hessian.device == (torch.device("cuda", 0) if stays else torch.device("cpu"))


def test_available_gpu_memory_is_what_one_allocation_can_get(monkeypatch):
    """Free pieces of partly used cached blocks do not count: one allocation of all the memory
    reported beyond the driver's free memory comes from the cache."""
    torch.cuda.empty_cache()
    whole = torch.empty(64 * 2**20, dtype=torch.uint8, device="cuda")
    pieces = [torch.empty(3 * 2**20, dtype=torch.uint8, device="cuda") for _ in range(24)]
    del whole, pieces[::2]  # a whole free block, and free pieces of partly used ones
    monkeypatch.setattr(torch.cuda, "mem_get_info", lambda device=None: (0, 2**40))
    available = get_available_gpu_memory(torch.cuda.current_device())
    reserved = torch.cuda.memory_reserved()
    block = torch.empty(available, dtype=torch.uint8, device="cuda")  # noqa: F841
    assert torch.cuda.memory_reserved() == reserved


def test_gptq_records_its_calibration_forward_memory(monkeypatch):
    """The next layer's Hessian reserve covers what this layer's calibration forward added, and
    not an earlier peak such as the previous layer's Hessians."""
    monkeypatch.setattr(calib_utils, "_calib_forward_peak", {})
    model = torch.nn.Linear(256, 256, device="cuda")
    inputs = torch.randn(2, 16, 256, device="cuda")
    mtq.quantize(model, mtq.NVFP4_DEFAULT_CFG, forward_loop=lambda m: m(inputs))
    transient = 64 * 2**20

    def forward_loop(m):
        scratch = torch.empty(transient, dtype=torch.uint8, device="cuda")
        m(inputs)
        del scratch

    earlier = 16 * transient
    torch.empty(earlier, dtype=torch.uint8, device="cuda")  # freed at once; only the peak stays
    gptq(model, forward_loop=forward_loop, perc_damp=0.1, block_size=16)
    assert transient <= calib_utils._calib_forward_peak[torch.cuda.current_device()] < earlier // 2


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
def test_gptq_e2e_flow(quant_cfg, tiny_tokenizer, calib_jsonl):
    model = get_tiny_llama(vocab_size=tiny_tokenizer.vocab_size).to("cuda")
    model.eval()

    quant_cfg = copy.deepcopy(quant_cfg)
    quant_cfg["algorithm"] = {"method": "gptq", "layerwise": {"enable": True}}
    calib_dataloader = get_dataset_dataloader(
        dataset_name=calib_jsonl,
        tokenizer=tiny_tokenizer,
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


@requires_triton
def test_fused_nvfp4_gptq_runs_on_the_weights_gpu(need_2_gpus):
    """A layer device_map placed on another GPU is updated there, not from the current GPU."""
    torch.manual_seed(42)
    weight, _, block_amax, global_scale, h_inv = _make_nvfp4_test_data(16, 64, 256)
    expected = _run_fused_gptq_nvfp4(weight, block_amax, global_scale, h_inv, 128, 16)

    weight, block_amax, h_inv = (t.to("cuda:1") for t in (weight, block_amax, h_inv))
    got = _run_fused_gptq_nvfp4(weight, block_amax, global_scale, h_inv, 128, 16)

    assert torch.cuda.current_device() == 0
    torch.testing.assert_close(got.cpu(), expected.cpu(), rtol=0, atol=0)


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


@requires_triton
@pytest.mark.parametrize(
    ("in_features", "block_size"),
    [
        (96, 48),  # non-power-of-two GPTQ block
        (208, 128),  # non-power-of-two final block
        (70, 128),  # width the NVFP4 group size does not divide
    ],
)
def test_gptq_fused_matches_unfused_on_irregular_shapes(in_features, block_size):
    torch.manual_seed(0)
    model = torch.nn.Linear(in_features, 32, bias=False, device="cuda")
    inp = torch.randn(4, 16, in_features, device="cuda")
    weights = []
    for fused in (True, False):
        quant_cfg = copy.deepcopy(mtq.NVFP4_DEFAULT_CFG)
        quant_cfg["algorithm"] = {"method": "gptq", "block_size": block_size, "fused": fused}
        qmodel = mtq.quantize(copy.deepcopy(model), quant_cfg, forward_loop=lambda m: m(inp))
        weights.append(qmodel.weight.data)
    torch.testing.assert_close(weights[0], weights[1], atol=1e-4, rtol=1e-4)


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


@requires_triton
@pytest.mark.parametrize(("four_over_six", "expected"), [(False, True), (True, False)])
def test_fused_supported_skips_four_over_six(four_over_six, expected):
    model = torch.nn.Linear(64, 32, bias=False, device="cuda")
    cfg = {
        "num_bits": (2, 1),
        "block_sizes": {
            -1: 16,
            "type": "static",
            "scale_bits": (4, 3),
            "four_over_six": four_over_six,
        },
    }
    quant_cfg = {
        "quant_cfg": [
            {"quantizer_name": "*", "enable": False},
            {"quantizer_name": "*weight_quantizer", "cfg": cfg},
        ],
        "algorithm": "max",
    }
    mtq.quantize(model, quant_cfg, forward_loop=lambda m: m(torch.randn(2, 64, device="cuda")))
    promote_nvfp4_static_quantizers(model)
    helper = GPTQHelper(model, "linear", fused=True)
    expected = expected and torch.cuda.get_device_capability() >= (8, 9)
    assert helper._fused_supported(model.weight_quantizer, model.weight) == expected
