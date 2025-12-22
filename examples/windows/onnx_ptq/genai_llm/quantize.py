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

import argparse
import logging
import os
import time

import numpy as np
import onnx
import torch
from datasets import load_dataset
from onnx import helper
from torch.utils.data import DataLoader
from transformers import AutoConfig, AutoTokenizer

from modelopt.onnx.quantization.int4 import quantize as quantize_int4

logging.getLogger().setLevel(logging.INFO)

pt_to_np = {"torch.int64": np.int64, "torch.float32": np.float32, "torch.float16": np.float16}


def prepare_input_shapes_string(
    batch_size, seq_len, past_seq_len, num_layers, num_kv_heads, head_dim
):
    shapes = ""

    shapes += f"input_ids:{batch_size}x{seq_len}"
    shapes += f",attention_mask:{batch_size}x{seq_len}"

    for i in range(num_layers):
        key_name = f"past_key_values.{i}.key"
        value_name = f"past_key_values.{i}.value"
        shapes += f",{key_name}:{batch_size}x{num_kv_heads}x{past_seq_len}x{head_dim}"
        shapes += f",{value_name}:{batch_size}x{num_kv_heads}x{past_seq_len}x{head_dim}"

    return shapes


def get_input_shapes_profile(model_name_or_path):
    config = AutoConfig.from_pretrained(model_name_or_path)

    head_dim = config.hidden_size // config.num_attention_heads
    if hasattr(config, "head_dim") and config.head_dim is not None:
        head_dim = config.head_dim
    num_kv_heads = config.num_key_value_heads
    num_layers = config.num_hidden_layers

    min_shapes = prepare_input_shapes_string(1, 1, 0, num_layers, num_kv_heads, head_dim)
    max_shapes = prepare_input_shapes_string(1, 1024, 1024, num_layers, num_kv_heads, head_dim)
    opt_shapes = prepare_input_shapes_string(1, 512, 512, num_layers, num_kv_heads, head_dim)

    return min_shapes, max_shapes, opt_shapes


def make_input_shapes_profile_for_ep_list(ep_list, model_name_or_path):
    # Input-shapes-profile will be used in provider-options for ORT session creation.
    # Provider options (even if {}) are needed for all EPs when we provide for any one of them.
    # Using empty shapes_profile for non-NvTensorRtRtx EPs.
    input_shapes_profile_sequence = []
    for ep in ep_list:
        if ep == "NvTensorRtRtx":
            min_shapes, max_shapes, opt_shapes = get_input_shapes_profile(model_name_or_path)
            input_shapes_profile = {
                "nv_profile_min_shapes": min_shapes,
                "nv_profile_max_shapes": max_shapes,
                "nv_profile_opt_shapes": opt_shapes,
            }
            input_shapes_profile_sequence.append(input_shapes_profile)
        else:
            input_shapes_profile_sequence.append({})

    return input_shapes_profile_sequence


def make_model_input(
    config,
    input_ids_arg,
    attention_mask_arg,
    add_past_kv_inputs,
    device,
    use_fp32,
    use_buffer_share,
    add_position_ids,
):
    input_ids = input_ids_arg
    attention_mask = attention_mask_arg

    if isinstance(input_ids_arg, list):
        input_ids = torch.tensor(input_ids_arg, device=device, dtype=torch.int64)
        attention_mask = torch.tensor(attention_mask_arg, device=device, dtype=torch.int64)

    inputs = {
        "input_ids": input_ids.contiguous(),
        "attention_mask": attention_mask.contiguous(),
    }

    if add_position_ids:
        # print(f"\n--Quantize-Script-- adding position ids\n")
        position_ids = attention_mask.long().cumsum(-1) - 1
        position_ids.masked_fill_(attention_mask == 0, 1)
        inputs["position_ids"] = position_ids.contiguous()

    if add_past_kv_inputs:
        # print(f"\n--Quantize-Script-- adding past KV cache value\n")
        torch_dtype = torch.float32 if use_fp32 else torch.float16
        batch_size, sequence_length = input_ids.shape
        max_sequence_length = config.max_position_embeddings
        num_heads, head_size = (
            config.num_key_value_heads,
            config.hidden_size // config.num_attention_heads,
        )

        if hasattr(config, "head_dim") and config.head_dim is not None:
            head_size = config.head_dim

        for i in range(config.num_hidden_layers):
            past_key = torch.zeros(
                batch_size,
                num_heads,
                max_sequence_length if use_buffer_share else 0,
                head_size,
                device=device,
                dtype=torch_dtype,
            )
            past_value = torch.zeros(
                batch_size,
                num_heads,
                max_sequence_length if use_buffer_share else 0,
                head_size,
                device=device,
                dtype=torch_dtype,
            )
            inputs.update(
                {
                    f"past_key_values.{i}.key": past_key.contiguous(),
                    f"past_key_values.{i}.value": past_value.contiguous(),
                }
            )

    return inputs


def get_initial_inputs(
    config,
    tokenizer,
    prompt,
    device,
    use_fp32,
    use_buffer_share,
    add_past_kv_inputs,
    add_position_ids,
):
    """
    Get initial inputs for the model for inference.
    These values will include the input_ids, attention_mask, and position_ids and past_key_values.

    Args:
        config: Huggingface config of the model.
        tokenizer: Tokenizer to encode and decode text.
        prompt: List of prompts to be supplied for inference.
        device: Device used to run the inference.
        use_fp32: Flag to select the float32 dtype in torch.
        use_buffer_share: True when --use_gqa is passed during the onnx export process
        add_past_kv_inputs: True when we want to also pass past_key_values input to model
        add_position_ids: True when we want to also pass position_ids input to model
    """
    # tokenizer.pad_token = "[PAD]"
    tokenizer.pad_token = tokenizer.eos_token
    encodings_dict = tokenizer.batch_encode_plus(prompt, padding=True)

    # max_length = model.config.max_position_embeddings
    # input_ids = tokenizer.encode(text, truncation=True, padding='max_length', max_length=max_length)

    return make_model_input(
        config,
        encodings_dict["input_ids"],
        encodings_dict["attention_mask"],
        add_past_kv_inputs,
        device,
        use_fp32,
        use_buffer_share,
        add_position_ids,
    )


def get_calib_inputs(
    dataset_name,
    model_name,
    cache_dir,
    calib_size,
    batch_size,
    block_size,
    device,
    use_fp32,
    use_buffer_share,
    add_past_kv_inputs,
    max_calib_rows_to_load,
    add_position_ids,
    trust_remote_code,
):
    # from transformers import LlamaConfig
    # config = LlamaConfig.from_pretrained(
    #     model_name, use_auth_token=True, cache_dir=cache_dir, trust_remote_code=trust_remote_code
    # )
    config = AutoConfig.from_pretrained(
        model_name, use_auth_token=True, cache_dir=cache_dir, trust_remote_code=trust_remote_code
    )
    tokenizer = AutoTokenizer.from_pretrained(
        model_name, use_auth_token=True, cache_dir=cache_dir, trust_remote_code=trust_remote_code
    )
    tokenizer.add_special_tokens({"pad_token": "[PAD]"})
    tokenizer.pad_token = tokenizer.eos_token

    assert calib_size <= max_calib_rows_to_load, (
        "calib size should be no more than max_calib_rows_to_load"
    )

    if "cnn" in dataset_name:
        dataset2 = load_dataset("cnn_dailymail", name="3.0.0", split="train").select(
            range(max_calib_rows_to_load)
        )
        column = "article"
    elif "pilevel" in dataset_name:
        dataset2 = load_dataset("mit-han-lab/pile-val-backup", split="validation")
        column = "text"
    else:
        raise ValueError(f'dataset "{dataset_name}" not supported')

    # dataset2 = dataset2.shuffle(seed=42)
    dataset2 = dataset2[column][:calib_size]
    batch_encoded = tokenizer.batch_encode_plus(
        dataset2, return_tensors="pt", padding=True, truncation=True, max_length=block_size
    )  # return_tensors="pt",
    batch_encoded = batch_encoded.to(device)
    batch_encoded_input_ids = batch_encoded["input_ids"]
    batch_encoded_attention_mask = batch_encoded["attention_mask"]
    calib_dataloader_input_ids = DataLoader(
        batch_encoded_input_ids, batch_size=batch_size, shuffle=False
    )
    calib_dataloader_attenton_mask = DataLoader(
        batch_encoded_attention_mask, batch_size=batch_size, shuffle=False
    )

    assert len(calib_dataloader_input_ids.dataset) == len(calib_dataloader_attenton_mask.dataset)
    assert len(calib_dataloader_input_ids) == len(calib_dataloader_attenton_mask)

    number_of_batched_samples = calib_size // batch_size

    batched_input_ids = []
    for idx, data in enumerate(calib_dataloader_input_ids):
        batched_input_ids.append(data)
        if idx == (number_of_batched_samples - 1):
            break

    batched_attention_mask = []
    for idx, data in enumerate(calib_dataloader_attenton_mask):
        batched_attention_mask.append(data)
        if idx == (number_of_batched_samples - 1):
            break

    print(
        f"\n--Quantize-Script-- number_of_batched_samples={number_of_batched_samples}, "
        f"batch-input-ids-list-len={len(batched_input_ids)}, "
        f"batched_attention_mask={len(batched_attention_mask)}\n"
    )

    batched_inputs_list = []
    for i in range(number_of_batched_samples):
        input_ids = batched_input_ids[i]
        attention_mask = batched_attention_mask[i]

        inputs = make_model_input(
            config,
            input_ids,
            attention_mask,
            add_past_kv_inputs,
            device,
            use_fp32,
            use_buffer_share,
            add_position_ids,
        )
        inputs = {
            input_name: torch_tensor.cpu().numpy() for input_name, torch_tensor in inputs.items()
        }
        batched_inputs_list.append(inputs)

    print(f"\n--Quantize-Script-- number of batched inputs = {len(batched_inputs_list)}\n")
    return batched_inputs_list


def parse_calibration_eps(value):
    """Parse and validate the calibration_eps input."""
    valid_choices = {"cuda", "cpu", "dml", "NvTensorRtRtx"}
    # Split the input by commas and remove any surrounding whitespace
    eps = [item.strip() for item in value.split(",")]
    # Validate each calibration endpoint
    for ep in eps:
        if ep not in valid_choices:
            raise argparse.ArgumentTypeError(
                f"Invalid calibration endpoint: '{ep}'. Choose from 'cuda', 'cpu', 'dml', 'NvTensorRtRtx'."
            )
    return eps


def convert_opset_to_21_proto(model_proto):
    """Modify the model's opset to 21 if it's not already, operating on a ModelProto.

    Args:
        model_proto (ModelProto): The ONNX model proto to modify.

    Returns:
        ModelProto: The updated ONNX model proto with opset version 21.

    """
    current_opset = {opset.domain: opset.version for opset in model_proto.opset_import}

    default_domain_version = current_opset.get("", 0)
    if default_domain_version >= 21:
        logging.info(
            "Model already uses opset version %s for the default domain. Skip conversion.",
            default_domain_version,
        )
        return model_proto  # No conversion needed

    new_opset_imports = [
        helper.make_opsetid("", 21),  # Default domain with opset version 21
        helper.make_opsetid("com.microsoft", 1),  # Microsoft domain with version 1
    ]

    for domain, version in current_opset.items():
        if domain not in ["", "com.microsoft"]:
            new_opset_imports.append(helper.make_opsetid(domain, version))

    # Update the model's opset imports
    model_proto.ClearField("opset_import")
    model_proto.opset_import.extend(new_opset_imports)

    logging.info("Model opset successfully converted to 21.")

    return model_proto


def main(args):
    cache_dir = "C:\\tmp"

    # device_id = 0
    # device = torch.device(f"cuda:{device_id}")
    device = torch.device(args.device)

    if args.layers_8bit:
        args.enable_mixed_quant = True

    print(
        f"\n--Quantize-Script-- algo={args.algo}, dataset={args.dataset}, calib_size={args.calib_size}, "
        f"batch_size={args.batch_size}, block_size={args.block_size}, add-position-ids={args.add_position_ids}, "
        f"past-kv={args.add_past_kv_inputs}, rcalib={args.use_random_calib}, device={args.device}, "
        f"use_zero_point={args.use_zero_point}, use_fp32={args.use_fp32} enable_mixed_quant={args.enable_mixed_quant}, "
        f"layers_8bit={args.layers_8bit}\n"
    )

    print(
        f"\n\n--Quantize-Script-- awqlite_alpha_step={args.awqlite_alpha_step}, "
        f"awqlite_fuse_nodes={args.awqlite_fuse_nodes}, "
        f"awqlite_run_per_subgraph={args.awqlite_run_per_subgraph}, "
        f"awqclip_alpha_step={args.awqclip_alpha_step}, "
        f"awqclip_alpha_min={args.awqclip_alpha_min}, "
        f"awqclip_bsz_col={args.awqclip_bsz_col}, "
        f"calibration_eps={args.calibration_eps}\n"
    )

    if os.path.isabs(args.output_path):
        output_dir_path = os.path.dirname(args.output_path)
    else:
        output_dir_path = os.path.dirname(os.path.join(os.getcwd(), args.output_path))

    print(
        f"\n\n--Quantize-Script-- input onnx path = {args.onnx_path},"
        f"\n    given output path = {args.output_path}"
        f"\n    absolute output directory path = {output_dir_path}\n\n"
    )

    if not os.path.exists(output_dir_path):
        os.makedirs(output_dir_path)
        os.chmod(output_dir_path, 0o777)

    calib_inputs = get_calib_inputs(
        args.dataset,
        args.model_name,
        cache_dir,
        args.calib_size,
        args.batch_size,
        512,
        device,
        args.use_fp32,
        args.use_buffer_share,
        args.add_past_kv_inputs,
        128,
        args.add_position_ids,
        args.trust_remote_code,
    )

    input_shapes_profile_data = None
    if "NvTensorRtRtx" in args.calibration_eps and (args.algo not in ["rtn", "rtn_dq"]):
        # NvTensorRtRtx EP uses (min, max, opt) profile for dynamic shapes in the model's inputs.
        input_shapes_profile_data = make_input_shapes_profile_for_ep_list(
            args.calibration_eps, args.model_name
        )
    print(
        f"\n--Quantize-Script-- input_shapes_profile is None? - {input_shapes_profile_data is None}\n"
    )

    t = time.time()
    logging.info("\nQuantizing the model....\n")
    quantized_onnx_model = quantize_int4(
        args.onnx_path,
        calibration_method=args.algo,
        calibration_data_reader=None if args.use_random_calib else calib_inputs,
        calibration_eps=args.calibration_eps,
        use_zero_point=args.use_zero_point,
        block_size=args.block_size,
        input_shapes_profile=input_shapes_profile_data,
        awqlite_alpha_step=args.awqlite_alpha_step,
        awqlite_run_per_subgraph=args.awqlite_run_per_subgraph,
        awqlite_fuse_nodes=args.awqlite_fuse_nodes,
        awqclip_alpha_step=args.awqclip_alpha_step,
        awqclip_alpha_min=args.awqclip_alpha_min,
        awqclip_bsz_col=args.awqclip_bsz_col,
        enable_mixed_quant=args.enable_mixed_quant,
        layers_8bit=args.layers_8bit,
        gather_block_size=args.gather_block_size,
        gather_quantize_axis=args.gather_quantize_axis,
    )
    logging.info(f"\nQuantization process took {time.time() - t} seconds")

    quantized_onnx_model = convert_opset_to_21_proto(quantized_onnx_model)

    t = time.time()
    onnx.save_model(
        quantized_onnx_model,
        args.output_path,
        save_as_external_data=True,
        location=os.path.basename(args.output_path) + "_data",
        size_threshold=0,
    )
    logging.info(f"Saving to {args.output_path} took {time.time() - t} seconds")
    print("\nDone\n")


if __name__ == "__main__":
    st = time.time()
    parser = argparse.ArgumentParser(description="Quantize ONNX model.")
    parser.add_argument(
        "--model_name", type=str, required=True, help="Name of the llama model variant."
    )
    parser.add_argument(
        "--onnx_path",
        type=str,
        required=True,
        help="Input ONNX model path.",
    )
    parser.add_argument(
        "--output_path",
        type=str,
        required=True,
        help="Output quantized model path.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        help="Device for calibration data",
    )
    parser.add_argument(
        "--algo",
        type=str,
        default="awq_lite",
        help="Algorithm or calibration-method to use. Choose from [awq_lite, awq_clip, rtn, rtn_dq]",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="cnn",
        help="Dataset for calibration",
    )
    parser.add_argument(
        "--calib_size",
        type=int,
        default=32,
        help="Number of input calibration samples, should be no more than 128",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=1,
        help="Batch size for calibration samples",
    )
    parser.add_argument(
        "--use_fp32",
        default=False,
        action="store_true",
        help="True when KV cache inputs/outputs are in float32.",
    )
    parser.add_argument(
        "--awqlite_alpha_step",
        type=float,
        default=0.1,
        help="Alpha step-size for AWQ scale search",
    )
    parser.add_argument(
        "--awqlite_run_per_subgraph",
        default=False,
        action="store_true",
        help="If true, then AWQ scale search will iterate over each subgraphs with quantizable matmuls",
    )
    parser.add_argument(
        "--awqlite_disable_fuse_nodes",
        dest="awqlite_fuse_nodes",
        default=True,
        action="store_false",
        help="If fuse_nodes is true, then input scaling part will be fused in the parent node as and when possible",
    )
    parser.add_argument(
        "--awqclip_alpha_step",
        type=float,
        default=0.05,
        help="Step-size for AWQ weight clipping",
    )
    parser.add_argument(
        "--awqclip_alpha_min",
        type=float,
        default=0.5,
        help="Minimum AWQ weight-clipping threshold",
    )
    parser.add_argument(
        "--awqclip_bsz_col",
        type=int,
        default=1024,
        help="Number of Columns taken in one chunk during AWQ weight clipping loss computation",
    )
    parser.add_argument(
        "--block_size",
        type=int,
        default=128,
        help="Block size for INT4 quantization of MatMul/Gemm nodes",
    )
    parser.add_argument(
        "--gather_block_size",
        type=int,
        default=32,
        help="Block size for INT4 quantization of Gather nodes",
    )
    parser.add_argument(
        "--gather_quantize_axis",
        type=int,
        default=None,
        help="Quantization axis for INT4 quantization of Gather nodes",
    )
    parser.add_argument(
        "--use_zero_point",
        default=False,
        action="store_true",
        help="True when we want to perform zero-point based quantization",
    )
    parser.add_argument(
        "--no_past_kv_inputs",
        dest="add_past_kv_inputs",
        default=True,
        action="store_false",
        help="Add past KV cache values in inputs",
    )
    parser.add_argument(
        "--use_buffer_share",
        default=False,
        action="store_true",
        help="True when --use_gqa was passed during export.",
    )
    parser.add_argument(
        "--no_position_ids",
        dest="add_position_ids",
        action="store_false",
        default=True,
        help="True when we want to also pass position_ids input to model",
    )
    parser.add_argument(
        "--use_random_calib",
        default=False,
        action="store_true",
        help="True when we want to use a random calibration data",
    )

    parser.add_argument(
        "--calibration_eps",
        type=parse_calibration_eps,  # Use the custom parser
        default=["dml", "cpu"],  # Default as a list
        help="Comma-separated list of calibration endpoints. Choose from 'cuda', 'cpu', 'dml', 'NvTensorRtRtx'.",
    )
    parser.add_argument(
        "--trust_remote_code",
        help="Set trust_remote_code for Huggingface models and tokenizers",
        default=False,
        action="store_true",
    )
    parser.add_argument(
        "--enable_mixed_quant",
        default=False,
        action="store_true",
        help=(
            "Use default mixed quantization strategy: first 1/8, last 1/8, and every 3rd attn, "
            "mlp layers quantized to 8 bits; others to 4 bits."
        ),
    )
    parser.add_argument(
        "--layers_8bit",
        type=str,
        default="",
        help=("Overrides default mixed quant strategy. Example: 'layers.0,lm_head'"),
    )
    args = parser.parse_args()
    main(args)
