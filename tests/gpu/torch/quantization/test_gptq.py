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

import pytest
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

import modelopt.torch.quantization as mtq
from modelopt.torch.quantization.model_calib import blockwise_weight_update, update_hessian
from modelopt.torch.utils.dataset_utils import create_forward_loop, get_dataset_dataloader

RAND_SEED = 42
torch.manual_seed(RAND_SEED)


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
        (
            4,
            32,
            torch.tensor(
                [
                    0,
                    0.5,
                    1,
                    -0.5,
                    0,
                    0.5,
                    1,
                    -0.5,
                    0,
                    0.5,
                    1,
                    -0.5,
                    0,
                    0.5,
                    1,
                    -0.5,
                    -4,
                    -2,
                    0,
                    6,
                    -6,
                    -4,
                    -2,
                    0,
                    -4,
                    -2,
                    0,
                    6,
                    -6,
                    -4,
                    -2,
                    0,
                ]
            )
            .to("cuda")
            .expand(32, -1),
            False,
        ),  # weights with nvfp4 values -> no GPTQ update
    ],
)
def test_gptq_updates(block_size, dim, model_weight, expect_weight_change):
    model = torch.nn.Linear(dim, 1).to("cuda")
    model.weight.data = model_weight
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


@pytest.mark.parametrize(
    "quant_cfg", [mtq.NVFP4_DEFAULT_CFG, mtq.FP8_DEFAULT_CFG, mtq.INT4_BLOCKWISE_WEIGHT_ONLY_CFG]
)
def test_gptq_weight_folding(quant_cfg):
    """Test that folded weights match the quantized-dequantized weights from GPTQ."""
    # Setup model
    model = AutoModelForCausalLM.from_pretrained(
        "TinyLlama/TinyLlama-1.1B-Chat-v1.0", device_map="auto"
    )
    tokenizer = AutoTokenizer.from_pretrained(
        "TinyLlama/TinyLlama-1.1B-Chat-v1.0", trust_remote_code=True
    )

    if tokenizer.pad_token != "<unk>" or tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    model.eval()

    # Configure GPTQ quantization
    quant_cfg["algorithm"] = "gptq_lite"

    # Prepare calibration data
    calib_dataloader = get_dataset_dataloader(
        dataset_name="cnn_dailymail",
        tokenizer=tokenizer,
        batch_size=32,
        num_samples=128,  # Reduced for faster testing
        device="cuda",
        include_labels=False,
    )

    # Step 1: Quantize model using GPTQ
    calibrate_loop = create_forward_loop(dataloader=calib_dataloader)
    model = mtq.quantize(model, quant_cfg, forward_loop=calibrate_loop)

    # Collect quantized-dequantized weights before folding
    qdq_weights_before = {}
    for name, module in model.named_modules():
        if hasattr(module, "weight_quantizer") and hasattr(module, "weight"):
            # Get the quantized-dequantized weight using the quantizer
            with torch.no_grad():
                qdq_weight = module.weight_quantizer(module.weight.data)
                qdq_weights_before[name] = qdq_weight.clone()

    assert len(qdq_weights_before) > 0, "No quantized weights found in the model"

    # Step 2: Fold weights
    mtq.fold_weight(model)

    # Step 3: Compare weights between step 1 and 2
    for name, module in model.named_modules():
        if name in qdq_weights_before:
            # After folding, the weight should equal the QDQ weight from before
            folded_weight = module.weight.data
            qdq_weight_before = qdq_weights_before[name]

            # Check if weights are equal (use allclose for floating point comparison)
            assert torch.allclose(folded_weight, qdq_weight_before), (
                f"Folded weight for {name} doesn't match QDQ weight before folding"
            )

    print(f"Successfully verified weight folding for {len(qdq_weights_before)} layers")
