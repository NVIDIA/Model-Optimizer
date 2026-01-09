# Adapted from: https://github.com/lm-sys/FastChat/blob/322e9238dd586815ea405ddb70607958773a0464/fastchat/llm_judge/gen_model_answer.py

# Copyright 2024 Large Model Systems Organization (LMSYS)

#    Licensed under the Apache License, Version 2.0 (the "License");
#    you may not use this file except in compliance with the License.
#    You may obtain a copy of the License at

#        http://www.apache.org/licenses/LICENSE-2.0

#    Unless required by applicable law or agreed to in writing, software
#    distributed under the License is distributed on an "AS IS" BASIS,
#    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#    See the License for the specific language governing permissions and
#    limitations under the License.

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
"""Generate answers with local models.

Usage:
python3 gen_model_answer.py --model-path lmsys/fastchat-t5-3b-v1.0 --model-id fastchat-t5-3b-v1.0
"""

import argparse
import json
import os
import random
import time

import openai
import shortuuid
import torch
from fastchat.llm_judge.common import load_questions, temperature_config
from fastchat.model import get_conversation_template, load_model
from fastchat.utils import str_to_torch_dtype
from quantization_utils import get_tokenizer, quantize_model
from tqdm import tqdm

try:
    from modelopt.deploy.llm import LLM
except ImportError:
    LLM = None  # type: ignore[misc]

# API setting constants
API_MAX_RETRY = 16
API_RETRY_SLEEP = 10
API_ERROR_OUTPUT = "$ERROR$"


# Model Optimizer modification
# Reference:
# https://github.com/lm-sys/FastChat/blob/5c0443edb8545babb37d495ade86c75ebbf68009/fastchat/llm_judge/common.py#L407
def chat_completion_openai(model, conv, temperature, top_p, max_tokens, api_dict=None):
    """Chat completion with the OpenAI API.

    Args:
        model: The model handle.
        conv: The FastChat conversation template.
        temperature: The temperature for sampling.
        top_p: The nucleus sampling parameter.
        max_tokens: The maximum number of tokens to generate.
        api_dict: The API settings.

    Returns:
        The generated output.
    """

    if api_dict is not None:
        openai.api_base = api_dict["api_base"]
        openai.api_key = api_dict["api_key"]
    output = API_ERROR_OUTPUT
    for _ in range(API_MAX_RETRY):
        try:
            messages = conv.to_openai_api_messages()
            response = openai.ChatCompletion.create(
                model=model,
                messages=messages,
                n=1,
                temperature=temperature,
                top_p=top_p,
                max_tokens=max_tokens,
            )
            output = response["choices"][0]["message"]["content"]
            break
        except openai.error.OpenAIError as e:
            print(type(e), e)
            time.sleep(API_RETRY_SLEEP)

    return output


def run_eval(
    model_path,
    model_id,
    question_file,
    question_begin,
    question_end,
    answer_file,
    max_new_token,
    top_p,
    temperature,
    num_choices,
    num_gpus_per_model,
    num_gpus_total,
    max_gpu_memory,
    dtype,
    revision,
    checkpoint_dir,
    nim_model,
    args,
):
    questions = load_questions(question_file, question_begin, question_end)
    # random shuffle the questions to balance the loading
    random.shuffle(questions)

    # Split the question file into `num_gpus` files
    assert num_gpus_total % num_gpus_per_model == 0
    use_ray = num_gpus_total // num_gpus_per_model > 1

    if use_ray:
        get_answers_func = ray.remote(num_gpus=num_gpus_per_model)(get_model_answers).remote
    else:
        get_answers_func = get_model_answers

    chunk_size = len(questions) // (num_gpus_total // num_gpus_per_model)
    ans_handles = [
        get_answers_func(
            model_path,
            model_id,
            questions[i : i + chunk_size],
            answer_file,
            max_new_token,
            num_choices,
            num_gpus_per_model,
            max_gpu_memory,
            dtype=dtype,
            revision=revision,
            top_p=top_p,
            temperature=temperature,
            checkpoint_dir=checkpoint_dir,
            nim_model=nim_model,
        )
        for i in range(0, len(questions), chunk_size)
    ]

    if use_ray:
        ray.get(ans_handles)


@torch.inference_mode()
def get_model_answers(
    model_path,
    model_id,
    questions,
    answer_file,
    max_new_token,
    num_choices,
    num_gpus_per_model,
    max_gpu_memory,
    dtype,
    revision,
    top_p=None,
    temperature=None,
    checkpoint_dir=None,
    nim_model=None,
):
    # Model Optimizer modification
    tokenizer = get_tokenizer(model_path, trust_remote_code=args.trust_remote_code)
    if checkpoint_dir:
        assert LLM is not None, "tensorrt_llm APIs could not be imported."
        model = LLM(checkpoint_dir, tokenizer=tokenizer, max_batch_size=1)
    elif not nim_model:
        model, _ = load_model(
            model_path,
            revision=revision,
            device="cuda",
            num_gpus=num_gpus_per_model,
            max_gpu_memory=max_gpu_memory,
            dtype=dtype,
            load_8bit=False,
            cpu_offloading=False,
            debug=False,
        )
        if args.quant_cfg:
            quantize_model(
                model,
                args.quant_cfg,
                tokenizer,
                args.calib_batch_size,
                args.calib_size,
                test_generated=False,
                auto_quantize_bits=args.auto_quantize_bits,
                auto_quantize_method=args.auto_quantize_method,
                auto_quantize_score_size=args.auto_quantize_score_size,
                auto_quantize_checkpoint=args.auto_quantize_checkpoint,
            )

    for question in tqdm(questions):
        if not temperature:
            if question["category"] in temperature_config:
                temperature = temperature_config[question["category"]]
            else:
                temperature = 0.7
        top_p = top_p if top_p is not None else 1.0
        choices = []
        for i in range(num_choices):
            torch.manual_seed(i)
            conv = get_conversation_template(model_id)
            turns = []
            for j in range(len(question["turns"])):
                qs = question["turns"][j]
                conv.append_message(conv.roles[0], qs)
                conv.append_message(conv.roles[1], None)
                prompt = conv.get_prompt()
                # Model Optimizer modification
                if nim_model:
                    api_key = os.getenv("OPENAI_API_KEY")

                    if not api_key:
                        raise ValueError("OPENAI_API_KEY environment variable is not set")
                    api_dict = {}
                    api_dict["api_base"] = os.getenv(
                        "OPENAI_API_BASE", "https://integrate.api.nvidia.com/v1"
                    )
                    api_dict["api_key"] = api_key
                    output = chat_completion_openai(
                        nim_model, conv, temperature, top_p, max_new_token, api_dict
                    )
                    conv.update_last_message(output)
                    turns.append(output)
                    continue

                input_ids = tokenizer([prompt]).input_ids

                # Model Optimizer modification
                do_sample = not temperature <= 0.0001

                # some models may error out when generating long outputs
                try:
                    if not checkpoint_dir:
                        output_ids = model.generate(
                            torch.as_tensor(input_ids).cuda(),
                            do_sample=do_sample,
                            temperature=temperature,
                            max_new_tokens=max_new_token,
                            top_p=top_p,
                        )
                        if model.config.is_encoder_decoder:
                            output_ids = output_ids[0]
                        else:
                            output_ids = output_ids[0][len(input_ids[0]) :]
                    else:
                        # Model Optimizer modification
                        output_ids = model.generate_tokens(
                            [prompt],
                            max_new_tokens=max_new_token,
                            temperature=temperature,
                            top_p=top_p,
                        )[0]

                    # be consistent with the template's stop_token_ids
                    if conv.stop_token_ids:
                        stop_token_ids_index = [
                            i for i, id in enumerate(output_ids) if id in conv.stop_token_ids
                        ]
                        if len(stop_token_ids_index) > 0:
                            output_ids = output_ids[: stop_token_ids_index[0]]

                    output = tokenizer.decode(
                        output_ids,
                        spaces_between_special_tokens=False,
                    )
                    if conv.stop_str and isinstance(conv.stop_str, list):
                        stop_str_indices = sorted(
                            [
                                output.find(stop_str)
                                for stop_str in conv.stop_str
                                if output.find(stop_str) > 0
                            ]
                        )
                        if len(stop_str_indices) > 0:
                            output = output[: stop_str_indices[0]]
                    elif conv.stop_str and output.find(conv.stop_str) > 0:
                        output = output[: output.find(conv.stop_str)]

                    for special_token in tokenizer.special_tokens_map.values():
                        if isinstance(special_token, list):
                            for special_tok in special_token:
                                output = output.replace(special_tok, "")
                        else:
                            output = output.replace(special_token, "")
                    output = output.strip()

                    if conv.name == "xgen" and output.startswith("Assistant:"):
                        output = output.replace("Assistant:", "", 1).strip()
                except RuntimeError:
                    print("ERROR question ID: ", question["question_id"])
                    output = "ERROR"

                conv.update_last_message(output)
                turns.append(output)

            choices.append({"index": i, "turns": turns})

        # Dump answers
        os.makedirs(os.path.dirname(answer_file), exist_ok=True)
        with open(os.path.expanduser(answer_file), "a") as fout:
            ans_json = {
                "category": question["category"],
                "turns": turns,
                "question_id": question["question_id"],
                "answer_id": shortuuid.uuid(),
                "model_id": model_id,
                "choices": choices,
                "tstamp": time.time(),
            }
            fout.write(json.dumps(ans_json) + "\n")


def reorg_answer_file(answer_file):
    """Sort by question id and de-duplication"""
    answers = {}
    with open(answer_file) as fin:
        for line in fin:
            qid = json.loads(line)["question_id"]
            answers[qid] = line

    qids = sorted(answers.keys())
    with open(answer_file, "w") as fout:
        for qid in qids:
            fout.write(answers[qid])


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model-path",
        type=str,
        required=True,
        help="The path to the weights. This can be a local folder or a Hugging Face repo ID.",
    )
    parser.add_argument("--model-id", type=str, required=True, help="A custom name for the model.")
    parser.add_argument(
        "--bench-name",
        type=str,
        default="mt_bench",
        help="The name of the benchmark question set.",
    )
    parser.add_argument(
        "--question-begin",
        type=int,
        help="A debug option. The begin index of questions.",
    )
    parser.add_argument(
        "--question-end", type=int, help="A debug option. The end index of questions."
    )
    parser.add_argument("--answer-file", type=str, help="The output answer file.")
    parser.add_argument(
        "--max-new-token",
        type=int,
        default=1024,
        help="The maximum number of new generated tokens.",
    )
    parser.add_argument(
        "--top-p",
        type=float,
        default=None,
        help="The nucleus sampling parameter.",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=None,
        help="The temperature for sampling.",
    )
    parser.add_argument(
        "--num-choices",
        type=int,
        default=1,
        help="How many completion choices to generate.",
    )
    parser.add_argument(
        "--num-gpus-per-model",
        type=int,
        default=1,
        help="The number of GPUs per model.",
    )
    parser.add_argument("--num-gpus-total", type=int, default=1, help="The total number of GPUs.")
    parser.add_argument(
        "--max-gpu-memory",
        type=str,
        help="Maximum GPU memory used for model weights per GPU.",
    )
    parser.add_argument(
        "--dtype",
        type=str,
        choices=["float32", "float16", "bfloat16"],
        help="Override the default dtype. If not set, it will use float16 on GPU and float32 on CPU.",
        default=None,
    )
    parser.add_argument(
        "--revision",
        type=str,
        default="main",
        help="The model revision to load.",
    )
    parser.add_argument(
        "--checkpoint-dir",
        type=str,
        help="The path to the model checkpoint directory.",
    )
    parser.add_argument(
        "--nim-model",
        type=str,
        help="The NIM model handle to use",
    )
    parser.add_argument(
        "--quant_cfg",
        type=str,
        help="The quantization algorithm config name.",
    )
    parser.add_argument(
        "--calib_size",
        type=int,
        default=512,
        help="The number of quantization calibration samples.",
    )
    parser.add_argument(
        "--calib_batch_size",
        type=int,
        default=4,
        help="The calibration batch size for quantization calibration.",
    )
    parser.add_argument(
        "--auto_quantize_bits",
        type=float,
        default=None,
        help=(
            "Effective bits constraint for auto_quantize. If not set, "
            "regular quantization without auto_quantize search will be applied."
        ),
    )
    parser.add_argument(
        "--auto_quantize_method",
        type=str,
        default="gradient",
        choices=["gradient", "kl_div"],
        help=(
            "Method for auto_quantize sensitivity analysis. 'gradient' uses gradient-based method "
            "(requires labels in dataset). 'kl_div' uses KL divergence between original and "
            "quantized model outputs (no labels required). Default: 'gradient'"
        ),
    )
    parser.add_argument(
        "--auto_quantize_score_size",
        type=int,
        default=128,
        help=(
            "Number of samples to use for auto_quantize scoring. Most of auto_quantize time is spent on "
            "sensitivity score estimation, so reducing this speeds it up while only minimally affecting "
            "final model accuracy compared to lowering --calib_size (the number of samples used for calibration)."
        ),
    )
    parser.add_argument(
        "--auto_quantize_checkpoint",
        type=str,
        default=None,
        help=(
            "Path to checkpoint file for saving/restoring auto_quantize search state "
            "(sensitivity scores, costs, etc.). Only used when auto_quantize_bits is specified."
        ),
    )
    parser.add_argument(
        "--trust_remote_code",
        help="Set trust_remote_code for Huggingface models and tokenizers",
        default=False,
        action="store_true",
    )

    args = parser.parse_args()
    print(args)

    if args.num_gpus_total // args.num_gpus_per_model > 1:
        import ray

        ray.init()

    question_file = f"data/{args.bench_name}/question.jsonl"
    answer_file = (
        args.answer_file
        if args.answer_file
        else f"data/{args.bench_name}/model_answer/{args.model_id}.jsonl"
    )

    print(f"Output to {answer_file}")

    run_eval(
        model_path=args.model_path,
        model_id=args.model_id,
        question_file=question_file,
        question_begin=args.question_begin,
        question_end=args.question_end,
        answer_file=answer_file,
        max_new_token=args.max_new_token,
        top_p=args.top_p,
        temperature=args.temperature,
        num_choices=args.num_choices,
        num_gpus_per_model=args.num_gpus_per_model,
        num_gpus_total=args.num_gpus_total,
        max_gpu_memory=args.max_gpu_memory,
        dtype=str_to_torch_dtype(args.dtype),
        revision=args.revision,
        checkpoint_dir=args.checkpoint_dir,
        nim_model=args.nim_model,
        args=args,
    )

    reorg_answer_file(answer_file)
