# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
import time

import numpy as np
import torch
from datasets import load_dataset
from optimum.onnxruntime import ORTModelForSpeechSeq2Seq
from transformers import WhisperProcessor

from modelopt.onnx.quantization.quantize import quantize as quantize_top_level_api

logging.getLogger().setLevel(logging.INFO)


USE_MERGED = False


def get_ep_for_decoder_calib_data_preparation(calibration_eps: list[str]):
    provider = None
    if "cuda" in calibration_eps:
        provider = "CUDAExecutionProvider"
    elif "dml" in calibration_eps:
        provider = "DmlExecutionProvider"
    elif "cpu" in calibration_eps:
        provider = "CPUExecutionProvider"
    else:
        raise ValueError("unknown ep")

    return provider


def parse_calibration_eps(value):
    """Parse and validate the calibration_eps input."""
    valid_choices = {"cuda", "cpu", "dml"}
    # Split the input by commas and remove any surrounding whitespace
    eps = [item.strip() for item in value.split(",")]
    # Validate each calibration endpoint
    for ep in eps:
        if ep not in valid_choices:
            raise argparse.ArgumentTypeError(
                f"Invalid calibration endpoint: '{ep}'. Choose from 'cuda', 'cpu', 'dml'."
            )
    return eps


def is_encoder_model(input_onnx_path):
    is_encoder = False

    if "encoder_model.onnx" in args.onnx_path:
        is_encoder = True

    return is_encoder


def is_decoder_model(input_onnx_path):
    is_decoder = False

    if "decoder_model.onnx" in args.onnx_path:
        is_decoder = True

    return is_decoder


def is_decoder_with_past_model(input_onnx_path):
    is_decoder_with_past = False

    if "decoder_with_past_model.onnx" in args.onnx_path:
        is_decoder_with_past = True

    return is_decoder_with_past


def get_calib_data_for_encoder(asr_dataset, processor, calib_size, data_type):
    np_dtype = np.float16 if data_type == "fp16" else np.float32

    calib_data = {}

    for idx, batch in enumerate(asr_dataset):
        audio = batch["audio"]
        # inp = processor(audio["array"], sampling_rate=audio["sampling_rate"], return_tensors="pt").input_features
        inp = (
            processor(audio["array"], sampling_rate=audio["sampling_rate"], return_tensors="pt")
            .input_features.cpu()
            .numpy()
            .astype(np_dtype)
        )
        x = calib_data.get("input_features")
        if x is None:
            calib_data["input_features"] = inp
        else:
            calib_data["input_features"] = np.concatenate((x, inp[np.newaxis :,]), axis=0)
        if idx == calib_size:
            break

    print(f"\nCalibration data for ENCODER is created. calib_size={calib_size}\n")
    return calib_data


def get_calib_data_for_decoder(
    asr_dataset, processor, calib_size, base_model_dir, ep_list, data_type
):
    torch_dtype = torch.float16 if data_type == "fp16" else torch.float32

    calib_data = {}

    provider = get_ep_for_decoder_calib_data_preparation(ep_list)

    model = ORTModelForSpeechSeq2Seq.from_pretrained(
        base_model_dir, provider=provider, cache_dir=args.cache_dir, use_merged=USE_MERGED
    )

    for idx, batch in enumerate(asr_dataset):
        audio = batch["audio"]
        inp = processor(
            audio["array"], sampling_rate=audio["sampling_rate"], return_tensors="pt"
        ).to(dtype=torch_dtype)
        encoder_outputs = model.encoder(inp.input_features, attention_mask=None)
        last_hidden_state = encoder_outputs.last_hidden_state.cpu().numpy()
        decoder_input_ids = (
            torch.tensor([[model.config.decoder_start_token_id]]).to(torch.int64).cpu().numpy()
        )  # to("cuda")
        # decoder_input_ids = model.decoder.embed_tokens(decoder_input_ids)
        # decoder_input_ids = torch.ones((batch_size, 2), dtype=torch.int64, device="cuda")
        #                      * model.config.decoder_start_token_id
        x = calib_data.get("input_ids")
        if x is None:
            assert calib_data.get("encoder_hidden_states") is None, (
                "encoder-hidden-states is not None but input-ids is"
            )
            calib_data["input_ids"] = decoder_input_ids
            calib_data["encoder_hidden_states"] = last_hidden_state
        else:
            calib_data["input_ids"] = np.concatenate((x, decoder_input_ids[np.newaxis :,]), axis=0)
            x = calib_data.get("encoder_hidden_states")
            assert x is not None, "encoder-hidden-states is None but not input-ids"
            calib_data["encoder_hidden_states"] = np.concatenate(
                (x, last_hidden_state[np.newaxis :,]), axis=0
            )

        if idx == calib_size:
            break

    print(f"\nCalibration data for DECODER is created. calib_size={calib_size}\n")
    return calib_data


def get_calib_data_for_decoder_with_past(
    asr_dataset, processor, calib_size, base_model_dir, ep_list, data_type
):
    torch_dtype = torch.float16 if data_type == "fp16" else torch.float32

    calib_data = {}

    provider = get_ep_for_decoder_calib_data_preparation(ep_list)

    model = ORTModelForSpeechSeq2Seq.from_pretrained(
        base_model_dir, provider=provider, cache_dir=args.cache_dir, use_merged=USE_MERGED
    )

    for idx, batch in enumerate(asr_dataset):
        audio = batch["audio"]
        inp = processor(
            audio["array"], sampling_rate=audio["sampling_rate"], return_tensors="pt"
        ).to(dtype=torch_dtype)

        encoder_outputs = model.encoder(inp.input_features, attention_mask=None)

        last_hidden_state = encoder_outputs.last_hidden_state.cpu()

        # TODO Read suitable config instead of hard-coding token-ids for language (en),
        #      task (transcribe), and no_timestamp_ids. The task-transcribe-id might be same as some
        #      of forced-ids - check Optimum-ORT pipeline e.g. see _retrieve_init_tokens()
        decoder_input_ids = (
            torch.tensor([[model.config.decoder_start_token_id, 50259, 50359, 50363]])
            .to(torch.int64)
            .cpu()
        )

        decoder_outputs = model.decoder(decoder_input_ids, last_hidden_state)

        # TODO batch size = 1 -> assert here after passing argument
        next_token_id = (
            torch.argmax(decoder_outputs.logits[:, -1, :], dim=-1).view(-1, 1).cpu().numpy()
        )
        cache_position = torch.tensor([4]).to(torch.int64).cpu().numpy()

        assert decoder_outputs.past_key_values is not None, "missing past-KV values"
        assert len(decoder_outputs.past_key_values) == model.config.num_hidden_layers, (
            "different amount of KV-data"
        )
        assert len(decoder_outputs.past_key_values[0]) == 4, "different per-layer KV-data length"

        x = calib_data.get("input_ids")
        if x is None:
            assert calib_data.get("cache_position") is None, (
                "cache_position is not None but input-ids is"
            )
            calib_data["input_ids"] = next_token_id
            calib_data["cache_position"] = cache_position
            for i, kv_data in enumerate(decoder_outputs.past_key_values):
                # kv_data = kv_data.cpu().numpy()
                calib_data[f"past_key_values.{i}.decoder.key"] = kv_data[0].cpu().numpy()
                calib_data[f"past_key_values.{i}.decoder.value"] = kv_data[1].cpu().numpy()
                calib_data[f"past_key_values.{i}.encoder.key"] = kv_data[2].cpu().numpy()
                calib_data[f"past_key_values.{i}.encoder.value"] = kv_data[3].cpu().numpy()
        else:
            calib_data["input_ids"] = np.concatenate((x, next_token_id), axis=0)
            x = calib_data.get("cache_position")
            assert x is not None, "cache_position is None but not input-ids"
            calib_data["cache_position"] = np.concatenate((x, cache_position), axis=0)
            for i, kv_data in enumerate(decoder_outputs.past_key_values):
                # kv_data = kv_data.cpu().numpy()
                x = calib_data[f"past_key_values.{i}.decoder.key"]
                calib_data[f"past_key_values.{i}.decoder.key"] = np.concatenate(
                    (x, kv_data[0].cpu().numpy()), axis=0
                )
                x = calib_data[f"past_key_values.{i}.decoder.value"]
                calib_data[f"past_key_values.{i}.decoder.value"] = np.concatenate(
                    (x, kv_data[1].cpu().numpy()), axis=0
                )
                x = calib_data[f"past_key_values.{i}.encoder.key"]
                calib_data[f"past_key_values.{i}.encoder.key"] = np.concatenate(
                    (x, kv_data[2].cpu().numpy()), axis=0
                )
                x = calib_data[f"past_key_values.{i}.encoder.value"]
                calib_data[f"past_key_values.{i}.encoder.value"] = np.concatenate(
                    (x, kv_data[3].cpu().numpy()), axis=0
                )

        if idx == calib_size:
            break

    print(f"\nCalibration data for DECODER_WITH_PAST is created. calib_size={calib_size}\n")
    return calib_data


def main(args):
    start_time = time.time()

    assert args.batch_size == 1, "batch size is NOT 1"

    # args.qdq_for_weights = True

    print("\n\n######### Whisper's 8-bit Quantization:  Settings...\n\n")

    print(
        f"  quantization_mode={args.quant_mode},\n  calibrartion_method={args.calib_method},"
        f"\n  calib_size={args.calib_size},\n  batch_size={args.batch_size},"
        f"\n  use-random-calib-data={args.use_random_calib},\n  torch-is-cuda-available={torch.cuda.is_available()},"
        f"\n  calibration-EPs={args.calibration_eps},\n  dtype={args.dtype},\n  USE_MERGED={USE_MERGED},"
        f"\n  qdq_for_weights={args.qdq_for_weights}, dq-only-for-weights={not args.qdq_for_weights}\n"
    )
    print(
        f"\n  model-name (id)={args.model_name},\n  input-onnx-path ={args.onnx_path},"
        f"\n  output-path={args.output_path},\n  base_model_dir={args.base_model_dir},\n  cache_dir={args.cache_dir}\n"
    )

    print("\n=========================================================\n\n")

    processor = WhisperProcessor.from_pretrained(args.model_name, cache_dir=args.cache_dir)

    asr_dataset = load_dataset("librispeech_asr", "clean", split="test")
    # asr_dataset = load_dataset("librispeech_asr", "all", split="test.clean")

    calib_data = None
    if not args.use_random_calib and is_decoder_model(args.onnx_path):
        calib_data = get_calib_data_for_decoder(
            asr_dataset,
            processor,
            args.calib_size,
            args.base_model_dir,
            args.calibration_eps,
            args.dtype,
        )
    elif not args.use_random_calib and is_encoder_model(args.onnx_path):
        calib_data = get_calib_data_for_encoder(asr_dataset, processor, args.calib_size, args.dtype)
    elif not args.use_random_calib and is_decoder_with_past_model(args.onnx_path):
        calib_data = get_calib_data_for_decoder_with_past(
            asr_dataset,
            processor,
            args.calib_size,
            args.base_model_dir,
            args.calibration_eps,
            args.dtype,
        )

    assert args.use_random_calib or calib_data is not None, "calibration data not prepared"

    logging.info("\nQuantizing the model....\n")
    quantize_top_level_api(
        onnx_path=args.onnx_path,
        quantize_mode=args.quant_mode,
        calibration_method=args.calib_method,
        calibration_data=None if args.use_random_calib else calib_data,
        calibration_eps=args.calibration_eps,
        use_external_data_format=True,
        output_path=args.output_path,
        op_types_to_quantize=["MatMul"],
        nodes_to_quantize=[r"\S*MatMul[\S]*"],
        nodes_to_exclude=[r"/lm_head", r"/Shape"],
        dq_only=not args.qdq_for_weights,
        verbose=True,
        high_precision_dtype="fp16" if args.dtype == "fp16" else "fp32",
        mha_accumulation_dtype="fp16" if args.dtype == "fp16" else "fp32",
        enable_gemv_detection_for_trt=False,
        enable_shared_constants_duplication=False,
    )
    logging.info(
        f"\nQuantization process (along with saving) took {time.time() - start_time} seconds\n"
    )

    print("\n\nDone\n\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Quantize Whisper ONNX model with INT8/FP8.")
    parser.add_argument("--model_name", type=str, required=True, help="Name or HF id of the model")
    parser.add_argument(
        "--base_model_dir",
        type=str,
        required=True,
        help="Directory of the base ONNX model",
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
        "--calib_method",
        type=str,
        default="max",
        help="calibration method for quantization (max or entropy)",
    )
    parser.add_argument(
        "--quant_mode",
        type=str,
        default="int8",
        help="quantization mode to be used (int8 or fp8)",
    )
    parser.add_argument(
        "--dtype",
        type=str,
        default="fp32",
        help="precision of the model tensors. Choose from 'fp32', 'fp16'.",
    )
    parser.add_argument(
        "--calib_size",
        type=int,
        default=32,
        help="Number of input calibration samples, should be no more than 256",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=1,
        help="Batch size for calibration samples",
    )
    parser.add_argument(
        "--use_random_calib",
        type=bool,
        default=False,
        help="True when we want to use one randomly generated calibration sample",
    )
    parser.add_argument(
        "--qdq_for_weights",
        default=False,
        action="store_true",
        help="If True, Q->DQ nodes will be added for weights, otherwise only DQ nodes will be added.",
    )
    parser.add_argument(
        "--calibration_eps",
        type=parse_calibration_eps,  # Use the custom parser
        default=["cuda", "cpu"],  # Default as a list
        help="Comma-separated list of calibration endpoints. Choose from 'cuda', 'cpu', 'dml'.",
    )
    parser.add_argument(
        "--cache_dir",
        type=str,
        help="cache directory for HuggingFace files",
    )

    args = parser.parse_args()
    main(args)
