# SPDX-FileCopyrightText: Copyright (c) 2023-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

import argparse
from collections import defaultdict

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

import modelopt.torch.opt as mto
import modelopt.torch.quantization as mtq
from modelopt.torch.utils import create_forward_loop
from modelopt.torch.utils.dataset_utils import get_dataset_dataloader

SUPPORT_QUANT_FORMAT = {
    "fp8": mtq.FP8_DEFAULT_CFG,
    "nvfp4": mtq.NVFP4_DEFAULT_CFG,
}


def update_weight_quantizer_amax_for_fusion(model: torch.nn.Module):
    """Group modules that take the same input and set amax to enable gemm fusion."""
    input_to_linear = defaultdict(list)

    def _input_hook(module, input, output):
        input_to_linear[input[0]].append(module)

    handles = []

    for name, module in model.named_modules():
        if "QuantLinear" in type(module).__name__:
            module.name = name
            handle = module.register_forward_hook(_input_hook)
            handles.append(handle)

    with torch.no_grad():
        fake_input = torch.ones([1, 2], dtype=torch.long).to(model.device)
        # Run forward pass so that all modules sharing the same input are collected using forward hook.
        model(fake_input)
        for handle in handles:
            handle.remove()

    for modules in input_to_linear.values():
        # make sure they have the same input amax
        if modules[0].input_quantizer.is_enabled:
            amax = modules[0].input_quantizer.amax
            for m in modules:
                assert m.input_quantizer.is_enabled
                assert m.input_quantizer.amax == amax

        # set amax of weight_quantizer
        if modules[0].weight_quantizer.is_enabled:
            max_weight_amax = max([m.weight_quantizer.amax for m in modules])
            for m in modules:
                m.weight_quantizer.amax = max_weight_amax


def auto_quantize(
    model, qformat, auto_quantize_bits, calib_dataloader, calibrate_loop, batch_size=1
):
    qformat_list = qformat.split(",")
    # Check if all provided quantization formats are supported
    assert all(qformat in SUPPORT_QUANT_FORMAT for qformat in qformat_list), (
        "One or more quantization formats provided are not supported for unified checkpoint export"
    )

    def loss_func(output, data):
        # For transformers AutoModelForCausalLM models, the outputs are wrapped in `CausalLMOutputWithPast`
        # which contains the loss attribute.
        return output.loss

    model, _ = mtq.auto_quantize(
        model,
        constraints={"effective_bits": auto_quantize_bits},
        data_loader=calib_dataloader,
        forward_step=lambda model, batch: model(**batch),
        loss_func=loss_func,
        quantization_formats=[SUPPORT_QUANT_FORMAT[format] for format in qformat_list],
        num_calib_steps=len(calib_dataloader),
        num_score_steps=min(
            len(calib_dataloader), 128 // batch_size
        ),  # Limit the number of score steps to avoid long calibration time
        verbose=True,
    )

    # We need to explicitly calibrate for kv cache quantization
    enable_kv_cache_quantization = "int8" not in qformat
    if enable_kv_cache_quantization:
        mtq.set_quantizer_by_cfg(
            model,
            quant_cfg=[
                {
                    "quantizer_name": "*output_quantizer",
                    "cfg": {"num_bits": (4, 3), "axis": None},
                    "enable": True,
                }
            ],
        )
        # Lets calibrate only the output quantizer this time. Let's disable all other quantizers.
        with mtq.set_quantizer_by_cfg_context(
            model,
            [
                {"quantizer_name": "*", "enable": False},
                {"quantizer_name": "*output_quantizer", "enable": True},
            ],
        ):
            mtq.calibrate(model, algorithm="max", forward_loop=calibrate_loop)
    return model


def modelopt_ptq(
    model_path: str,
    output_dir: str,
    qformat: str | None = None,
    num_samples: int = 512,
    auto_quantize_bits: float | None = None,
    calib_dataset: str = "cnn_dailymail",
    calib_batch_size: int = 8,
    trust_remote_code: bool = False,
) -> torch.nn.Module:
    """Quantize the model with modelopt."""
    model = AutoModelForCausalLM.from_pretrained(
        model_path, trust_remote_code=trust_remote_code, dtype="auto", device_map="auto"
    )
    model.eval()

    tokenizer = AutoTokenizer.from_pretrained(
        model_path,
        model_max_length=2048,
        padding_side="left",
        trust_remote_code=trust_remote_code,
    )
    # sanitize tokenizer
    if tokenizer.pad_token != "<unk>":
        tokenizer.pad_token = tokenizer.eos_token
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # create the forward loop for calibration
    calib_dataloader = get_dataset_dataloader(
        dataset_name=calib_dataset,
        tokenizer=tokenizer,
        batch_size=calib_batch_size,
        num_samples=num_samples,
        device=model.device,
        include_labels=auto_quantize_bits is not None,
    )
    calibrate_loop = create_forward_loop(dataloader=calib_dataloader)

    # quantize the model
    model = auto_quantize(
        model,
        qformat,
        auto_quantize_bits,
        calib_dataloader,
        calibrate_loop,
        calib_batch_size,
    )

    # Post processing for weight scaling factors
    # 1. detect modules which will be fused and shared the same input
    # 2. update the weight amax, as the modules will be fused during deployment
    # This is required for nvfp4, fp8 for gemm fusion
    update_weight_quantizer_amax_for_fusion(model)

    # enable huggingface checkpointing for ModelOpt
    mto.enable_huggingface_checkpointing()

    print(f"Saving the quantized model to {output_dir}.")
    tokenizer.save_pretrained(output_dir)
    model.save_pretrained(output_dir)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--hf_ckpt",
        help="Specify where the unqunatized HF checkpoint path is.",
        required=True,
    )
    parser.add_argument(
        "--quant",
        help=(f"Quantization format. Available options: {list(SUPPORT_QUANT_FORMAT.keys())}."),
        default="fp8",
    )
    parser.add_argument(
        "--output_dir",
        help=("Output directory of the quantized checkpoint with tokenizer."),
    )
    parser.add_argument(
        "--num_samples", help="Number of samples for calibration.", type=int, default=512
    )
    parser.add_argument(
        "--calib_batch_size", help="Batch size for calibration.", type=int, default=8
    )
    parser.add_argument(
        "--effective_bits",
        default=8.0,
        type=float,
        help=(
            "Effective bits constraint for auto_quantize. If not set, "
            "regular quantization without auto_quantize search will be applied."
        ),
    )
    parser.add_argument(
        "--trust_remote_code",
        action="store_true",
        help="Set trust_remote_code for Huggingface models and tokenizers",
    )

    args = parser.parse_args()

    modelopt_ptq(
        args.hf_ckpt,
        args.output_dir,
        args.quant,
        args.num_samples,
        auto_quantize_bits=args.effective_bits,
        calib_batch_size=args.calib_batch_size,
        trust_remote_code=args.trust_remote_code,
    )
