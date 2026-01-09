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
import time

import torch
import torchaudio
from datasets import load_dataset
from evaluate import load
from optimum.onnxruntime import ORTModelForSpeechSeq2Seq
from tqdm import tqdm
from transformers import WhisperProcessor

USE_MERGED = False


def get_ep(inference_ep: list[str]):
    provider = None
    if "cuda" in inference_ep:
        provider = "CUDAExecutionProvider"
    elif "dml" in inference_ep:
        provider = "DmlExecutionProvider"
    elif "cpu" in inference_ep:
        provider = "CPUExecutionProvider"
    else:
        raise ValueError("unknown ep")

    return provider


def main(args):
    data_type = torch.float16 if args.dtype == "fp16" else torch.float32

    start_time = time.time()

    print("\n########### ONNX Whisper Inference:    Settings...\n\n")
    print(
        f"\n  model-name (id)={args.model_name},\n  onnx_model_dir={args.onnx_model_dir},"
        f"\n  inference_ep={args.inference_ep},\n  test_samples_count={args.test_samples_count},"
        f"\n  log_model_outputs={args.log_model_output},\n  cache_dir={args.cache_dir},"
        f"\n  dtype={data_type},\n  USE_MERGED={USE_MERGED},"
        f"\n  audio_file_path={args.audio_file_path},\n  run_wer_test={args.run_wer_test}\n"
    )
    print("\n=========================================================\n\n")

    processor = WhisperProcessor.from_pretrained(args.model_name, cache_dir=args.cache_dir)
    model = ORTModelForSpeechSeq2Seq.from_pretrained(
        args.onnx_model_dir,
        provider=get_ep(args.inference_ep),
        cache_dir=args.cache_dir,
        use_merged=USE_MERGED,
    )

    # Fix for latest transformers: Disable the new Dynamic Cache Format
    # that optimum 1.23 doesn't support. This forces the legacy tuple format
    model._supports_default_dynamic_cache = lambda: False

    # print(model.encoder)
    # print(model.decoder)

    test_sample, sample_rate = torchaudio.load(args.audio_file_path)
    test_sample = test_sample.numpy()[0]
    inp = processor(test_sample, sampling_rate=sample_rate, return_tensors="pt").input_features.to(
        "cuda", dtype=data_type
    )
    # print(f"\n\n--demo-audio-- -input-features- type={type(inp)}, shape={inp.shape}, {inp}\n")

    predicted_ids = model.generate(inp)[0]
    transcription = processor.decode(predicted_ids)
    prediction = processor.tokenizer._normalize(transcription)

    print(f"\n\n-- Content of input audio-file = {prediction}\n\n")

    if args.run_wer_test:
        librispeech_test_clean = load_dataset(
            "librispeech_asr", "clean", split="test", trust_remote_code=True
        )

        references = []
        predictions = []

        for idx in tqdm(range(args.test_samples_count), desc="Evaluating..."):
            # audio = batch["audio"]
            audio = librispeech_test_clean[idx]["audio"]
            inp = processor(
                audio["array"], sampling_rate=audio["sampling_rate"], return_tensors="pt"
            )
            input_features = inp.input_features
            reference = processor.tokenizer._normalize(
                librispeech_test_clean[idx]["text"]
            )  # batch['text'], librispeech_test_clean[idx]['text']
            references.append(reference)
            if args.log_model_output:
                print(f"\n\n--Evaluate-- reference-{idx}={reference}\n\n")
            input_features = input_features.to("cuda", dtype=data_type)
            predicted_ids = model.generate(input_features)[0]
            transcription = processor.decode(predicted_ids)
            prediction = processor.tokenizer._normalize(transcription)
            predictions.append(prediction)
            if args.log_model_output:
                print(f"\n\n--Evaluate-- prediction-{idx}={prediction}\n\n")

        wer = load("wer")
        wer_result = wer.compute(references=references, predictions=predictions)

    print(
        f"\n## DONE ## - wer = {wer_result}, wer% = {wer_result * 100}, accuracy% = {(1 - wer_result) * 100},"
        f"\n  total-time = {time.time() - start_time} seconds,"
        f"\n  num-distinct-inputs={len(set(references))},"
        f"\n  len-reference={len(references)}, len-predictions={len(predictions)}\n\n"
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Inference of ONNX Whisper model.")
    parser.add_argument("--model_name", type=str, required=True, help="Name or HF id of the model")
    parser.add_argument(
        "--onnx_model_dir",
        type=str,
        required=True,
        help="Directory of the ONNX model files",
    )
    parser.add_argument(
        "--inference_ep",
        type=str,
        default="cuda",
        help="ORT-EP to be used by optimum-ORT for inference. Choose from 'cuda', 'cpu', 'dml'.",
    )
    parser.add_argument(
        "--dtype",
        type=str,
        default="fp32",
        help="Precision of the model tensors. Choose from 'fp32', 'fp16'.",
    )
    parser.add_argument(
        "--test_samples_count",
        type=int,
        default=100,
        help="Number of audio samples to evaluate",
    )
    parser.add_argument(
        "--log_model_output",
        default=False,
        action="store_true",
        help="If True, model's output are logged along with reference texts for the input audio sample",
    )
    parser.add_argument(
        "--cache_dir",
        type=str,
        help="cache directory for HuggingFace files",
    )
    parser.add_argument(
        "--audio_file_path",
        type=str,
        required=True,
        help="Path of the input audio file in .wav format",
    )
    parser.add_argument(
        "--run_wer_test",
        default=False,
        action="store_true",
        help="If True, runs WER accuracy benchmarking using samples from librispeech_asr dataset",
    )
    args = parser.parse_args()
    main(args)
