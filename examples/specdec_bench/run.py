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
from tqdm.asyncio import tqdm

engines_available = {
    "TRTLLM": models.TRTLLMPYTModel,
    "VLLM": models.VLLMModel,
    "SGLANG": models.SGLANGModel,
    "AUTO_DEPLOY": models.AutoDeployModel,
    "SPECBENCH_MEDUSA": models.SpecBenchMedusaModel,
}
datasets_available = {
    "mtbench": datasets.MTBench,
    "random": datasets.RandomToken,
    "specbench": datasets.SpecBench,
    "speed": datasets.SPEEDBench,
}


async def tqdm_gather(*fs, return_exceptions=False, **kwargs):
    if not return_exceptions:
        return await tqdm.gather(*fs, **kwargs)

    async def wrap(f):
        try:
            return await f
        except Exception as e:
            return e

    return await tqdm.gather(*map(wrap, fs), **kwargs)


async def run_loop(
    runner,
    dataset,
    tokenizer,
    output_length,
    postprocess,
    concurrency=10,
    end_id=-1,
    show_progress=False,
    completions=False,
    chat_template_args={},
):
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

    async def process_single_request(request, i):
        """Process a single request with all its conversation turns."""
        async with semaphore:
            messages = []
            if request.system_prompt is not None:
                messages.append({"role": "system", "content": request.system_prompt})

            for turn_id, question in enumerate(request.turns):
                messages.append({"role": "user", "content": question})
                entry_encoded = encode_chat(
                    tokenizer,
                    messages,
                    chat_template_args=chat_template_args,
                    completions=completions,
                )

                # Run the async runner.run directly
                output_tokens = await runner.run(
                    entry_encoded, max_length, end_id, request_id=i, turn_id=turn_id
                )
                output_text = decode_chat(tokenizer, output_tokens["output_ids"][0])
                output_text = postprocess(output_text)
                messages.append({"role": "assistant", "content": output_text})

            return messages

    tasks = [process_single_request(request, i) for i, request in enumerate(dataset.data)]
    if show_progress:
        text_outputs = await tqdm_gather(
            *tasks,
            return_exceptions=True,
            desc=f"Running requests (concurrency={concurrency})",
        )
    else:
        text_outputs = await asyncio.gather(*tasks, return_exceptions=True)

    # Check for any exceptions and handle them
    for i, result in enumerate(text_outputs):
        if isinstance(result, Exception):
            print(f"Error processing request {i}/{dataset.data[i].question_id}: {result}")
            raise result

    runner.process_metrics_final(text_outputs)
    return text_outputs


def run_simple(args):
    tokenizer = get_tokenizer(args.tokenizer, trust_remote_code=args.trust_remote_code)
    chat_template_args = args.runtime_params.get("chat_template_args", {})
    dataset_kwargs = args.runtime_params.get("dataset_kwargs", {})
    if args.num_requests is not None:
        dataset_kwargs["num_samples"] = args.num_requests
    if args.dataset is not None:
        if args.dataset == "random":
            assert args.random_isl is not None, "Random input length must be provided"
            dataset = datasets.RandomToken(tokenizer, args.random_isl, **dataset_kwargs)
        else:
            dataset = datasets_available[args.dataset](args.dataset_path, **dataset_kwargs)
    elif args.mtbench is not None:
        dataset = datasets.MTBench(args.mtbench, **dataset_kwargs)
    elif args.random_isl is not None:
        dataset = datasets.RandomToken(tokenizer, args.random_isl, **dataset_kwargs)
    elif args.specbench is not None:
        dataset = datasets.SpecBench(args.specbench, **dataset_kwargs)
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
        trust_remote_code=args.trust_remote_code,
        parallel_drafting=args.parallel_drafting,
        **engine_args,
    )

    metrics_list = [metrics.Timing(args.tp_size)]
    if args.aa_timing:
        metrics_list.append(metrics.AATiming(tokenizer))
    if args.mtbench is not None:
        metrics_list.insert(0, metrics.MTBench())
    elif args.specbench is not None or args.dataset == "speed":
        metrics_list.insert(0, metrics.SpecBench(requests=dataset.data))
    else:
        metrics_list.insert(0, metrics.AcceptanceRate())

    if args.save_dir is not None:
        for metric in metrics_list:
            metric.update_directory(args.save_dir)

    runner = runners.SimpleRunner(model, metrics=metrics_list)

    if args.postprocess == "base":
        postprocess = postprocess_base
    elif args.postprocess == "gptoss":
        postprocess = postprocess_gptoss
    else:
        raise ValueError(f"Invalid postprocess: {args.postprocess}")

    end_id = tokenizer.eos_token_id if not args.ignore_eos else -1

    asyncio.run(
        run_loop(
            runner,
            dataset,
            tokenizer,
            args.output_length,
            postprocess,
            args.concurrency,
            end_id,
            args.show_progress,
            args.completions,
            chat_template_args,
        )
    )

    runner.clear_metrics()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--tokenizer", type=str, required=True, help="Path to the tokenizer directory"
    )
    parser.add_argument(
        "--mtbench",
        type=str,
        required=False,
        default=None,
        help="Path to the mtbench dataset",
    )
    parser.add_argument(
        "--specbench",
        type=str,
        required=False,
        default=None,
        help="Path to the specbench dataset",
    )
    parser.add_argument(
        "--random_isl",
        type=int,
        required=False,
        default=None,
        help="How many tokens random input should be.",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        required=False,
        default=None,
        choices=list(datasets_available.keys()),
        help="Dataset to use",
    )
    parser.add_argument(
        "--dataset_path",
        type=str,
        required=False,
        default=None,
        help="Path to the dataset or config name for SPEEDBench",
    )
    parser.add_argument(
        "--num_requests",
        type=int,
        required=False,
        default=None,
        help="Number of requests to run. If not provided, all requests from the dataset will be run.",
    )
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
    parser.add_argument("--parallel_drafting", action="store_true", help="Enable parallel drafting")
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
    parser.add_argument(
        "--trust_remote_code", action="store_true", help="Trust remote code for tokenizer and model"
    )
    parser.add_argument("--aa_timing", action="store_true", help="Enable AA timing metric")
    parser.add_argument("--ignore_eos", action="store_true", help="Ignore EOS token")
    parser.add_argument("--show_progress", action="store_true", help="Show progress bar")
    parser.add_argument(
        "--completions",
        action="store_true",
        help="Skip chat template, tokenize the message directly",
    )
    parser.add_argument(
        "--postprocess",
        type=str,
        required=False,
        default="base",
        choices=["base", "gptoss"],
        help="Postprocess to use",
    )
    parser.add_argument(
        "--save_dir",
        type=str,
        required=False,
        default=None,
        help="Directory to save the results",
    )
    args = parser.parse_args()

    if args.runtime_params is not None:
        with open(args.runtime_params) as f:
            args.runtime_params = yaml.safe_load(f)
    else:
        args.runtime_params = {}
    if args.dataset is None:
        assert (
            args.mtbench is not None or args.random_isl is not None or args.specbench is not None
        ), "Either mtbench or random_isl or specbench must be provided"
    else:
        assert args.dataset_path is not None, "Dataset path must be provided"
        if args.dataset == "specbench":
            args.specbench = args.dataset_path
        elif args.dataset == "mtbench":
            args.mtbench = args.dataset_path

    if args.ignore_eos:
        print(
            "Warning: Ignore EOS should only be used in certain cases, do no activate unless necessary"
        )

    run_simple(args)
