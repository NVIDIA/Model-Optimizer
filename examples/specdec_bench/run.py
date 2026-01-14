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
import asyncio

import yaml
from specdec_bench import datasets, metrics, models, runners
from specdec_bench.utils import (
    decode_chat,
    encode_chat,
    get_tokenizer,
    postprocess_base,
    postprocess_gptoss,
)

engines_available = {
    "TRTLLM": models.TRTLLMPYTModel,
    "VLLM": models.VLLMModel,
    "SGLANG": models.SGLANGModel,
}


async def run_loop(runner, dataset, tokenizer, output_length, postprocess, concurrency=10):
    """
    Async version of run_loop with concurrency control using a semaphore.

    Args:
        runner: The model runner instance
        dataset: The dataset containing requests
        tokenizer: The tokenizer instance
        output_length: Maximum output length
        concurrency: Maximum number of concurrent requests (default: 10)
    """
    semaphore = asyncio.Semaphore(concurrency)
    max_length = output_length
    end_id = tokenizer.eos_token_id

    async def process_single_request(request, i):
        """Process a single request with all its conversation turns."""
        async with semaphore:
            messages = []
            if request.system_prompt is not None:
                messages.append({"role": "system", "content": request.system_prompt})

            for turn_id, question in enumerate(request.turns):
                messages.append({"role": "user", "content": question})
                entry_encoded = encode_chat(tokenizer, messages)

                # Run the async runner.run directly
                output_tokens = await runner.run(
                    entry_encoded, max_length, end_id, request_id=i, turn_id=turn_id
                )
                output_text = decode_chat(tokenizer, output_tokens["output_ids"][0])
                output_text = postprocess(output_text)
                messages.append({"role": "assistant", "content": output_text})

            return messages

    tasks = [process_single_request(request, i) for i, request in enumerate(dataset.data)]
    text_outputs = await asyncio.gather(*tasks, return_exceptions=True)

    # Check for any exceptions and handle them
    for i, result in enumerate(text_outputs):
        if isinstance(result, Exception):
            print(f"Error processing request {i}: {result}")
            raise result

    runner.process_metrics_final(text_outputs)
    return text_outputs


def run_simple(args):
    tokenizer = get_tokenizer(args.tokenizer)
    dataset_kwargs = args.runtime_params.get("dataset_kwargs", {})
    if args.mtbench is not None:
        dataset = datasets.MTBench(args.mtbench, args.num_requests, **dataset_kwargs)
    elif args.random_isl is not None:
        dataset = datasets.RandomToken(
            tokenizer, args.random_isl, args.num_requests, **dataset_kwargs
        )
    engine_args = args.runtime_params.get("engine_args", {})
    sampling_kwargs = args.runtime_params.get("sampling_kwargs", {"temperature": 0})
    model_class = engines_available[args.engine]
    model = model_class(
        args.model_dir,
        max_concurrent_requests=args.concurrency,
        sampling_kwargs=sampling_kwargs,
        speculative_algorithm=args.speculative_algorithm,
        draft_model_dir=args.draft_model_dir,
        speculative_num_steps=args.draft_length,
        tensor_parallel_size=args.tp_size,
        moe_expert_parallel_size=args.ep_size,
        **engine_args,
    )

    metrics_list = [metrics.Timing(args.tp_size)]
    if args.aa_timing:
        metrics_list.append(metrics.AATiming(tokenizer))
    if args.mtbench is not None:
        metrics_list.insert(0, metrics.MTBench())
    else:
        metrics_list.insert(0, metrics.AcceptanceRate())
    runner = runners.SimpleRunner(model, metrics=metrics_list)

    if args.postprocess == "base":
        postprocess = postprocess_base
    elif args.postprocess == "gptoss":
        postprocess = postprocess_gptoss
    else:
        raise ValueError(f"Invalid postprocess: {args.postprocess}")

    asyncio.run(
        run_loop(runner, dataset, tokenizer, args.output_length, postprocess, args.concurrency)
    )

    runner.clear_metrics()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--tokenizer", type=str, required=True, help="Path to the tokenizer directory"
    )
    parser.add_argument(
        "--mtbench", type=str, required=False, default=None, help="Path to the mtbench dataset"
    )
    parser.add_argument(
        "--random_isl",
        type=int,
        required=False,
        default=None,
        help="How many tokens random input should be.",
    )
    parser.add_argument("--num_requests", type=int, required=True, help="Number of requests to run")
    parser.add_argument(
        "--engine",
        type=str,
        required=False,
        default="TRTLLM",
        choices=list(engines_available.keys()),
        help="Engine to use",
    )
    parser.add_argument(
        "--speculative_algorithm",
        type=str,
        required=False,
        default="EAGLE3",
        choices=["EAGLE3", "EAGLE", "DRAFT_TARGET", "NGRAM", "MTP", "NONE"],
        help="Speculative algorithm to use",
    )
    parser.add_argument("--model_dir", type=str, required=True, help="Path to the model directory")
    parser.add_argument(
        "--draft_model_dir",
        type=str,
        required=False,
        default=None,
        help="Path to the draft model directory",
    )
    parser.add_argument(
        "--runtime_params",
        type=str,
        required=False,
        default=None,
        help="Path to the runtime params yaml file",
    )
    parser.add_argument(
        "--output_length", type=int, required=False, default=4096, help="Output length"
    )
    parser.add_argument("--draft_length", type=int, required=False, default=3, help="Draft length")
    parser.add_argument(
        "--tp_size", type=int, required=False, default=4, help="Tensor parallel size"
    )
    parser.add_argument(
        "--ep_size", type=int, required=False, default=2, help="Expert parallel size"
    )
    parser.add_argument(
        "--concurrency",
        type=int,
        required=False,
        default=1,
        help="Maximum number of concurrent requests",
    )
    parser.add_argument("--aa_timing", action="store_true", help="Enable AA timing metric")
    parser.add_argument(
        "--postprocess",
        type=str,
        required=False,
        default="base",
        choices=["base", "gptoss"],
        help="Postprocess to use",
    )

    args = parser.parse_args()

    if args.runtime_params is not None:
        with open(args.runtime_params) as f:
            args.runtime_params = yaml.safe_load(f)
    else:
        args.runtime_params = {}

    assert args.mtbench is not None or args.random_isl is not None, (
        "Either mtbench or random_isl must be provided"
    )

    run_simple(args)
