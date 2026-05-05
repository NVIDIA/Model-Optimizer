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
import os

import yaml
from specdec_bench import datasets, metrics, models, runners
from specdec_bench.run_utils import (
    build_dataset,
    build_metrics,
    expand_sweep_run_specs,
    gather_limited,
    resolve_run_save_dir,
    resolve_single_run_spec,
    resolve_sweep_output_root,
)
from specdec_bench.utils import (
    decode_chat,
    dump_env,
    encode_chat,
    get_tokenizer,
    postprocess_base,
    postprocess_gptoss,
)

engines_available = {
    "TRTLLM": models.TRTLLMPYTModel,
    "VLLM": models.VLLMModel,
    "SGLANG": models.SGLANGModel,
    "AUTO_DEPLOY": models.AutoDeployModel,
    "SPECBENCH_MEDUSA": models.SpecBenchMedusaModel,
}
datasets_available = {
    "humaneval": datasets.HumanEval,
    "mtbench": datasets.MTBench,
    "random": datasets.RandomToken,
    "specbench": datasets.SpecBench,
    "speed": datasets.SPEEDBench,
}


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
    chat_template_args=None,
):
    """Run requests with bounded concurrency and preserve input order."""
    max_length = output_length
    if chat_template_args is None:
        chat_template_args = {}

    async def process_single_request(request, i):
        """Process a single request with all its conversation turns."""
        # Pre-built messages (e.g. from a trace JSON): single forward pass
        if request.messages is not None:
            entry_encoded = encode_chat(
                tokenizer,
                request.messages,
                chat_template_args=chat_template_args,
                completions=completions,
            )
            output_tokens = await runner.run(
                entry_encoded, max_length, end_id, request_id=i, turn_id=0
            )
            raw_output = decode_chat(tokenizer, output_tokens["output_ids"][0])
            output_text = postprocess(raw_output)
            messages = [
                *request.messages,
                {
                    "role": "assistant",
                    "content": output_text,
                    "raw_content": raw_output,
                    "output_token_ids": output_tokens["output_ids"][0],
                    "generated": True,
                },
            ]
            return messages

        messages = []
        raw_outputs = []
        if request.system_prompt is not None:
            messages.append({"role": "system", "content": request.system_prompt})

        for turn_id, question in enumerate(request.turns):
            messages.append({"role": "user", "content": question})
            entry_encoded = encode_chat(
                tokenizer, messages, chat_template_args=chat_template_args, completions=completions
            )
            output_tokens = await runner.run(
                entry_encoded, max_length, end_id, request_id=i, turn_id=turn_id
            )
            output_text = decode_chat(tokenizer, output_tokens["output_ids"][0])
            raw_outputs.append(output_text)
            output_text = postprocess(output_text)
            messages.append(
                {
                    "role": "assistant",
                    "content": output_text,
                    "output_token_ids": output_tokens["output_ids"][0],
                    "generated": True,
                }
            )

        assistant_idx = 0
        for msg in messages:
            if msg["role"] == "assistant":
                msg["raw_content"] = raw_outputs[assistant_idx]
                assistant_idx += 1

        return messages

    text_outputs = await gather_limited(
        dataset.data,
        process_single_request,
        concurrency=concurrency,
        show_progress=show_progress,
        progress_desc=f"Running requests (concurrency={concurrency})",
    )

    for i, result in enumerate(text_outputs):
        if isinstance(result, Exception):
            print(f"Error processing request {i}/{dataset.data[i].question_id}: {result}")
            raise result

    runner.process_metrics_final(text_outputs)
    return text_outputs


def run_simple(args):
    tokenizer = get_tokenizer(args.tokenizer, trust_remote_code=args.trust_remote_code)
    chat_template_args = args.runtime_params.get("chat_template_args", {})
    dataset_kwargs = dict(args.runtime_params.get("dataset_kwargs", {}))
    if args.num_requests is not None:
        dataset_kwargs["num_samples"] = args.num_requests
    if args.category is not None:
        dataset_kwargs["category"] = args.category

    if args.sweep_config is None:
        run_specs = resolve_single_run_spec(args)
        is_sweep = False
        sweep_output_root = None
    else:
        run_specs = expand_sweep_run_specs(args, datasets_available)
        is_sweep = True
        sweep_output_root = resolve_sweep_output_root(args)
        print(f"Sweep output root: {sweep_output_root}")

    max_engine_concurrency = max(run_spec["concurrency"] for run_spec in run_specs)
    engine_args = args.runtime_params.get("engine_args", {})
    sampling_kwargs = args.runtime_params.get("sampling_kwargs", {"temperature": args.temperature})
    model_class = engines_available[args.engine]
    model = model_class(
        args.model_dir,
        max_concurrent_requests=max_engine_concurrency,
        sampling_kwargs=sampling_kwargs,
        speculative_algorithm=args.speculative_algorithm,
        draft_model_dir=args.draft_model_dir,
        speculative_num_steps=args.draft_length,
        tensor_parallel_size=args.tp_size,
        moe_expert_parallel_size=args.ep_size,
        parallel_drafting=args.parallel_drafting,
        trust_remote_code=args.trust_remote_code,
        **engine_args,
    )

    if args.postprocess == "base":
        postprocess = postprocess_base
    elif args.postprocess == "gptoss":
        postprocess = postprocess_gptoss
    else:
        raise ValueError(f"Invalid postprocess: {args.postprocess}")

    end_id = tokenizer.eos_token_id if not args.ignore_eos else -1

    async def run_all():
        for run_index, run_spec in enumerate(run_specs):
            run_dataset_kwargs = dict(dataset_kwargs)
            run_num_requests = run_spec.get("num_requests")
            if run_num_requests is not None:
                run_dataset_kwargs["num_samples"] = run_num_requests
            run_category = run_spec.get("category")
            if run_category is not None:
                run_dataset_kwargs["category"] = run_category
            dataset = build_dataset(
                run_spec, tokenizer, run_dataset_kwargs, datasets_available, datasets
            )
            metrics_list = build_metrics(args, tokenizer, run_spec["dataset"], dataset, metrics)
            output_length = run_spec.get("output_length") or args.output_length
            run_temperature = run_spec.get("temperature")
            if run_temperature is None:
                run_temperature = args.temperature
            model.sampling_kwargs["temperature"] = run_temperature
            run_save_dir = resolve_run_save_dir(
                args, run_spec, run_index, is_sweep, sweep_output_root
            )
            if run_save_dir is not None:
                if is_sweep:
                    dump_env(
                        args,
                        run_save_dir,
                        overrides={
                            "dataset": run_spec["dataset"],
                            "dataset_path": run_spec.get("dataset_path"),
                            "random_isl": run_spec.get("random_isl"),
                            "category": run_spec.get("category") or args.category,
                            "concurrency": run_spec["concurrency"],
                            "num_requests": run_spec.get("num_requests") or args.num_requests,
                            "output_length": output_length,
                            "temperature": run_temperature,
                        },
                    )
                for metric in metrics_list:
                    metric.update_directory(run_save_dir)

            print(
                f"Run {run_index + 1}/{len(run_specs)} | "
                f"dataset={run_spec['dataset']} | concurrency={run_spec['concurrency']} | "
                f"temperature={run_temperature} | requests={len(dataset.data)} | "
                f"save_dir={run_save_dir if run_save_dir is not None else './'}"
            )

            runner = runners.SimpleRunner(model, metrics=metrics_list)
            await run_loop(
                runner,
                dataset,
                tokenizer,
                output_length,
                postprocess,
                run_spec["concurrency"],
                end_id,
                args.show_progress,
                args.completions,
                chat_template_args,
            )
            runner.clear_metrics()

    try:
        asyncio.run(run_all())
    finally:
        model.stop()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--tokenizer", type=str, required=True, help="Path to the tokenizer directory"
    )
    parser.add_argument(
        "--mtbench", type=str, required=False, default=None, help="Path to the mtbench dataset"
    )
    parser.add_argument(
        "--specbench", type=str, required=False, default=None, help="Path to the specbench dataset"
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
        "--category",
        type=str,
        required=False,
        default=None,
        help="For datasets that provide the category field, only run requests in this category",
    )
    parser.add_argument(
        "--engine",
        type=str,
        required=False,
        default="TRTLLM",
        choices=sorted(engines_available.keys()),
        help="Engine to use",
    )
    parser.add_argument(
        "--speculative_algorithm",
        type=str,
        required=False,
        default="EAGLE3",
        choices=["EAGLE3", "EAGLE", "DRAFT_TARGET", "NGRAM", "MTP", "PARD", "NONE"],
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
        "--sweep_config",
        type=str,
        required=False,
        default=None,
        help="Path to YAML defining dataset/concurrency sweep runs",
    )
    parser.add_argument(
        "--sweep_output_root",
        type=str,
        required=False,
        default=None,
        help="Root directory for sweep outputs (defaults to save_dir or ./sweep_outputs/<timestamp>)",
    )
    parser.add_argument(
        "--output_length", type=int, required=False, default=4096, help="Output length"
    )
    parser.add_argument("--draft_length", type=int, required=False, default=3, help="Draft length")
    parser.add_argument(
        "--parallel_drafting", action="store_true", help="Enable parallel drafting (for vLLM)"
    )
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
        "--stop_think_id",
        type=int,
        nargs="+",
        required=False,
        default=None,
        help="Token IDs that mark the stop-think boundary (can specify multiple sequential tokens)",
    )
    parser.add_argument("--ignore_eos", action="store_true", help="Ignore EOS token")
    parser.add_argument(
        "--trust_remote_code",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Trust remote code when loading tokenizer and model (default: False)",
    )
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
        "--temperature", type=float, required=False, default=0.0, help="Temperature to use"
    )
    parser.add_argument(
        "--save_dir", type=str, required=False, default=None, help="Directory to save the results"
    )
    args = parser.parse_args()

    if args.runtime_params is not None:
        with open(args.runtime_params) as f:
            args.runtime_params = yaml.safe_load(f) or {}
    else:
        args.runtime_params = {}

    if args.sweep_config is None:
        if args.dataset is None:
            assert (
                args.mtbench is not None
                or args.random_isl is not None
                or args.specbench is not None
            ), "Either mtbench or random_isl or specbench must be provided"
        elif args.dataset != "random":
            assert args.dataset_path is not None, "Dataset path must be provided"

    if args.save_dir is not None:
        if os.path.exists(args.save_dir) and os.listdir(args.save_dir):
            raise ValueError(f"Save directory {args.save_dir} already exists and is not empty")
        dump_env(args, args.save_dir)

    if args.ignore_eos:
        print(
            "Warning: Ignore EOS should only be used in certain cases, do no activate unless necessary"
        )

    run_simple(args)
