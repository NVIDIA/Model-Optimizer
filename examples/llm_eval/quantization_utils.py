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


import torch
from transformers import AutoTokenizer

import modelopt.torch.quantization as mtq
from modelopt.torch.quantization.config import need_calibration
from modelopt.torch.quantization.plugins import register_hf_attentions_on_the_fly
from modelopt.torch.utils.dataset_utils import (
    create_forward_loop,
    get_dataset_dataloader,
    get_max_batch_size,
)

MAX_SEQ_LEN = 4096
MAX_OUTPUT_LEN = 512

# This is an example to customize the quantization config.
# Modify your custom config for debugging or research purposes.
CUSTOM_CONFIG = {
    "MY_QUANT_CONFIG": {
        "quant_cfg": {
            "*weight_quantizer": {"num_bits": 4, "block_sizes": {-1: 128}, "enable": True},
            "*input_quantizer": {"num_bits": 8, "type": "dynamic", "block_sizes": {-1: None}},
            # Disable sensitive layers such as `lm_head`, gate layers in MoE etc.
            **mtq.config._default_disabled_quantizer_cfg,
        },
        "algorithm": "max",
    },
}


def get_tokenizer(ckpt_path, max_seq_len=MAX_SEQ_LEN, trust_remote_code=False):
    """Returns the tokenizer from the model ckpt_path."""
    print(f"Initializing tokenizer from {ckpt_path}")
    tokenizer = AutoTokenizer.from_pretrained(
        ckpt_path,
        model_max_length=max_seq_len,
        padding_side="left",
        trust_remote_code=trust_remote_code,
    )

    # can't set attribute 'pad_token' for "<unk>"
    if tokenizer.pad_token != "<unk>":
        tokenizer.pad_token = tokenizer.eos_token

    return tokenizer


def _quantize_model_with_dataset(
    lm,
    quant_cfg: str | list[str],
    calib_dataset,
    auto_quantize_bits=None,
    auto_quantize_method="gradient",
    auto_quantize_score_size=128,
    batch_size=1,
    compress=False,
    auto_quantize_checkpoint=None,
):
    if hasattr(lm, "gpt2"):
        net = lm.gpt2
    elif hasattr(lm, "model"):
        net = lm.model
    else:
        net = lm

    if auto_quantize_bits is not None:
        quant_cfg_for_search = [
            getattr(mtq, quant_fmt) for quant_fmt in quant_cfg if quant_fmt != "NONE"
        ]

        # Configure forward_step and loss_func based on method
        if auto_quantize_method == "gradient":
            # For gradient-based method, return full output with loss
            def forward_step(model, batch):
                return model(**batch)

            def loss_func(output, data):
                # For transformers AutoModelForCausalLM models, the outputs are wrapped in `CausalLMOutputWithPast`
                # which contains the loss attribute.
                return output.loss
        elif auto_quantize_method == "kl_div":
            # For KL divergence method, return only logits
            def forward_step(model, batch):
                return model(**batch).logits

            loss_func = None  # KL divergence doesn't need a custom loss function
        else:
            raise ValueError(
                f"Invalid auto_quantize_method: {auto_quantize_method}. "
                "Must be 'gradient' or 'kl_div'"
            )

        net, _ = mtq.auto_quantize(
            net,
            constraints={"effective_bits": auto_quantize_bits},
            quantization_formats=quant_cfg_for_search,
            data_loader=calib_dataset,
            forward_step=forward_step,
            loss_func=loss_func,
            num_calib_steps=len(calib_dataset),
            # Most time is spent on score estimation; fewer samples speed it up with little accuracy impact.
            num_score_steps=min(len(calib_dataset), max(auto_quantize_score_size // batch_size, 1)),
            verbose=True,
            method=auto_quantize_method,
            # disabled_layers=["*lm_head*", "*mlp.gate.*"],
            checkpoint=auto_quantize_checkpoint,
        )
    else:
        mtq_cfg = CUSTOM_CONFIG.get(quant_cfg)  # type: ignore [arg-type]
        if mtq_cfg is None:
            mtq_cfg = getattr(mtq, quant_cfg)  # type: ignore [arg-type]

        calibrate_loop = None
        use_calibration = need_calibration(mtq_cfg)
        if not use_calibration:
            print("Dynamic quantization. Calibration skipped.")
        else:
            # The calibrate_loop is a custom defined method to run the model with the input data.
            # The basic version looks like:
            #
            # def calibrate_loop(model, dataloader):
            #     for data in dataloader:
            #         model(**data)
            #
            # We also provided a util method to generate the forward_loop with additional error handlings.
            calibrate_loop = create_forward_loop(dataloader=calib_dataset)

        quantize_bmm_attention = False
        for key in mtq_cfg["quant_cfg"]:
            if "bmm_quantizer" in key:
                quantize_bmm_attention = True
        if quantize_bmm_attention:
            register_hf_attentions_on_the_fly(net)

        net = mtq.quantize(net, mtq_cfg, calibrate_loop)
    mtq.print_quant_summary(net)
    # Compress or fold weights for faster evaluation.
    if compress:
        mtq.compress(net)
    else:
        mtq.fold_weight(net)


def quantize_model(
    model,
    quant_cfg: str | list[str],
    tokenizer,
    batch_size,
    calib_size,
    data="cnn_dailymail",
    test_generated=True,
    compress=False,
    auto_quantize_bits=None,
    auto_quantize_method="gradient",
    auto_quantize_score_size=128,
    auto_quantize_checkpoint=None,
):
    """Quantizes the model with the provided calibration dataset.

    Args:
        model: the model to be quantized.
        quant_cfg: the quantization algorithm config name if simple quantization is used.
                   the list of quantization algorithm config names if auto quantization is used.
        tokenizer: the tokenizer.
        batch_size: the calibration batch size for each calibration inference run.
        calib_size: the total calibration dataset size.
        data: the name of the calibration dataset.
        test_generated:  If ``True``, test the generated text before and after quantization.
        compress: If ``True``, compress the model after quantization.
        auto_quantize_bits: The effective bits constraint for auto_quantize.
        auto_quantize_method: The method for auto_quantize ('gradient' or 'kl_div').
        auto_quantize_score_size: Number of samples used for auto_quantize scoring.
        auto_quantize_checkpoint: Path to checkpoint file for saving/restoring auto_quantize search state
            (sensitivity scores, costs, etc.). Only used when auto_quantize_bits is specified.
    """
    if "AWQ" in quant_cfg:
        print(
            "\n####\nAWQ calibration could take longer than other calibration methods. "
            "Consider reducing calib_size to reduce calibration time.\n####\n"
        )

    device = model.device
    if hasattr(model, "model"):
        device = model.model.device

    is_gradient_based = auto_quantize_bits is not None and auto_quantize_method == "gradient"

    if batch_size == 0:
        if is_gradient_based or torch.distributed.is_initialized():
            raise ValueError("We dont support automatic batch size inference for this case.")

        net = model.gpt2 if hasattr(model, "gpt2") else model.model

        # We let the system to determine the max data batch for each forward.
        batch_size = get_max_batch_size(net)
        print(f"Update calib batch {batch_size}")

    calib_dataloader = get_dataset_dataloader(
        dataset_name=data,
        tokenizer=tokenizer,
        batch_size=batch_size,
        num_samples=calib_size,
        device=device,
        include_labels=is_gradient_based,
    )

    if test_generated:
        input_str = tokenizer.decode(next(iter(calib_dataloader))["input_ids"][0])
        generated_str_before_ptq = model.run(input_str)

    _quantize_model_with_dataset(
        model,
        quant_cfg,
        calib_dataloader,
        auto_quantize_bits,
        auto_quantize_method,
        auto_quantize_score_size,
        batch_size,
        compress,
        auto_quantize_checkpoint,
    )

    if test_generated:
        generated_str_after_ptq = model.run(input_str)

        print("--------")
        print(f"example test input: {input_str}")
        print("--------")
        print(f"example outputs before ptq: {generated_str_before_ptq}")
        print("--------")
        print(f"example outputs after ptq: {generated_str_after_ptq}")
