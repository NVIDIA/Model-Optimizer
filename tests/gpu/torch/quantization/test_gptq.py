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

import pytest
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

import modelopt.torch.quantization as mtq
from modelopt.torch.quantization.model_calib import blockwise_weight_update, update_hessian
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

    # Verify sample count is updated correctly (incremented by batch_size)
    assert new_n_samples == batch_size, f"Expected n_samples={batch_size}, got {new_n_samples}"

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
    # scaled_input = sqrt(2/6) * input_flat = sqrt(1/3) * ones(2, 12)
    # outer_product = scaled_input @ scaled_input.t() = (2/6) * ones(2,12) @ ones(12,2) = [[4,4], [4,4]]
    # Note: The scaling factor is (2/n_samples), so with n_samples=6 and 12 tokens: (2/6)*12 = 4
    expected_hessian = torch.ones(features, features, dtype=torch.float32) * 4.0

    assert torch.allclose(updated_hessian, expected_hessian, atol=1e-5), (
        f"Expected hessian {expected_hessian}, got {updated_hessian}"
    )
    assert new_n_samples == batch_size

    # Test 3: Accumulated hessians - verify equivalence
    # Processing [6,2,2] in one step should equal processing [2,2,2] three times
    seq_len = 2
    features = 2

    # Process in 3 steps of batch_size=2
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
    assert n_samples_accumulated == 6, f"Expected n_samples=6, got {n_samples_accumulated}"


@pytest.mark.parametrize(
    ("block_size", "dim", "model_weight", "expect_weight_change"),
    [
        (4, 16, torch.randn(16, 16).to("cuda"), True),  # random weight
        (
            4,
            16,
            torch.ones(16, 16).to("cuda"),
            False,
        ),  # all same weight -> no quantization error -> no GPTQ update
    ],
)
def test_gptq_updates(block_size, dim, model_weight, expect_weight_change):
    model = torch.nn.Linear(dim, dim).to("cuda")
    model.weight.data = model_weight
    model.name = "linear"
    original_weight = model_weight.clone()
    input = torch.randn(2, 16, dim).to("cuda")
    hessian = torch.zeros(dim, dim).to("cpu")
    n_samples = 0
    quant_cfg = mtq.NVFP4_DEFAULT_CFG

    mtq.quantize(model, quant_cfg, forward_loop=lambda model: model(input))

    # Get qdq weight
    q_dq_weight = model.weight_quantizer(model.weight.data)

    # Restore original weight
    model.weight.data = original_weight.clone()

    hessian, n_samples = update_hessian(input, hessian, n_samples)

    # Verify n_samples is update using hessian matrix
    assert n_samples == input.shape[0], "n_samples should be equal to input.shape[0]"

    # Perform another forward pass to update hessian matrix
    input_2 = torch.randn(3, 16, dim).to("cuda")
    hessian, n_samples = update_hessian(input_2, hessian, n_samples)
    assert n_samples == input.shape[0] + input_2.shape[0], (
        "n_samples should be equal to input.shape[0] + input_2.shape[0]"
    )

    hessian = hessian.to(input.device)
    blockwise_weight_update(model, hessian, block_size, 0.1)
    if expect_weight_change:
        # Weight must change as GPTQ updates weights to adjust for quantization error
        assert not torch.allclose(model.weight.data, q_dq_weight), "Weight should not be equal"
    else:
        assert torch.allclose(model.weight.data, q_dq_weight), "Weight should be equal"


@pytest.mark.parametrize(
    "quant_cfg", [mtq.NVFP4_DEFAULT_CFG, mtq.FP8_DEFAULT_CFG, mtq.INT4_BLOCKWISE_WEIGHT_ONLY_CFG]
)
def test_gptq_e2e_flow(quant_cfg):
    model = AutoModelForCausalLM.from_pretrained(
        "TinyLlama/TinyLlama-1.1B-Chat-v1.0", device_map="auto"
    )
    tokenizer = AutoTokenizer.from_pretrained(
        "TinyLlama/TinyLlama-1.1B-Chat-v1.0", trust_remote_code=True
    )

    # can't set attribute 'pad_token' for "<unk>"
    # We skip this step for Nemo models
    if tokenizer.pad_token != "<unk>" or tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Left padding usually provides better calibration result.
    tokenizer.padding_side = "left"

    assert tokenizer.pad_token is not None, "Pad token cannot be set!"
    model.eval()

    quant_cfg = copy.deepcopy(quant_cfg)
    quant_cfg["algorithm"] = "gptq_lite"
    # Define quantizer/dataloader
    calib_dataloader = get_dataset_dataloader(
        dataset_name="cnn_dailymail",
        tokenizer=tokenizer,
        batch_size=32,
        num_samples=512,
        device="cuda",
        include_labels=False,
    )
    # Only run single sample for preview
    prompt = "Where is New York city?"
    input_ids = tokenizer(prompt, return_tensors="pt")
    print(f"Input ids: {input_ids}")
    generated_ids_before_ptq = model.generate(
        input_ids["input_ids"].to("cuda"), max_new_tokens=100, do_sample=False, temperature=0.0
    )

    print(
        f"Generated ids before quantization: {tokenizer.decode(generated_ids_before_ptq[0], skip_special_tokens=True)}"
    )
    calibrate_loop = create_forward_loop(dataloader=calib_dataloader)
    model = mtq.quantize(model, quant_cfg, forward_loop=calibrate_loop)
    generated_ids_after_ptq = model.generate(
        input_ids["input_ids"].to("cuda"), max_new_tokens=100, do_sample=False, temperature=0.0
    )
    print(
        f"Generated ids after quantization: {tokenizer.decode(generated_ids_after_ptq[0], skip_special_tokens=True)}"
    )
