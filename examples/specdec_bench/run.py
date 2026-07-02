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
from specdec_bench.datasets.agentic_speed import AgenticSPEEDBench
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
    get_tokenizer,
    postprocess_base,
    postprocess_gptoss,
)

ANSI_RED = "\033[91m"
ANSI_RESET = "\033[0m"

engines_available = {
    "TRTLLM": models.TRTLLMPYTModel,
    "VLLM": models.VLLMModel,
    "SGLANG": models.SGLANGModel,
    "AUTO_DEPLOY": models.AutoDeployModel,
    "SPECBENCH_MEDUSA": models.SpecBenchMedusaModel,
}

# Translation table for --max_seq_len. Each engine spells the same
# concept (max input + output sequence the engine should reserve)
# differently:
#   VLLM   → max_model_len   (AsyncEngineArgs)
#   TRTLLM → max_seq_len     (LLM(...))
#   SGLANG → context_length  (sgl.Engine)
# Mapping applied in run_simple() so cell YAMLs use one CLI flag
# regardless of --engine. New engines: add an entry + a comment in
# the wrapper's __init__ pointing back here.
_MAX_SEQ_LEN_KEY = {
    "VLLM": "max_model_len",
    "TRTLLM": "max_seq_len",
    "SGLANG": "context_length",
}
datasets_available = {
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
    chat_template_args={},
    allowed_failure_rate=0,
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
    max_length = output_length

    async def process_single_request(request, i):
        """Process a single request with all its conversation turns."""
        # Pre-built messages (e.g. from a trace JSON): single forward pass
        if request.messages is not None:
            request_chat_template_args = dict(
                chat_template_args
            )  # make a copy of the chat template args, so we pass tools only for the requests that have tools
            if request.tools is not None:
                request_chat_template_args["tools"] = request.tools
            entry_encoded = dataset.encode_chat(
                tokenizer,
                request.messages,
                chat_template_args=request_chat_template_args,
                completions=completions,
                request=request,
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
            entry_encoded = dataset.encode_chat(
                tokenizer,
                messages,
                chat_template_args=chat_template_args,
                completions=completions,
                request=request,
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

    failed_request_ids = set()
    failed_question_ids = []
    max_failed_requests = int(allowed_failure_rate * len(dataset.data))

    def record_failed_request(i, result):
        if not isinstance(result, Exception):
            return

        question_id = getattr(dataset.data[i], "question_id", i)
        print(
            f"{ANSI_RED}Error processing request {i} (question_id={question_id}): "
            f"{type(result).__name__}: {result}{ANSI_RESET}"
        )
        failed_request_ids.add(i)
        failed_question_ids.append(question_id)

        if len(failed_request_ids) > max_failed_requests:
            raise ValueError(
                f"Too many requests failed: {len(failed_request_ids)}/{len(dataset.data)} "
                f"requests failed, allowed failure rate is {allowed_failure_rate}, aborting run."
            )

    text_outputs = await gather_limited(
        dataset.data,
        process_single_request,
        concurrency=concurrency,
        show_progress=show_progress,
        progress_desc=f"Running requests (concurrency={concurrency})",
        on_result=record_failed_request,
    )

    for i, result in enumerate(text_outputs):
        if isinstance(result, Exception):
            question_id = getattr(dataset.data[i], "question_id", i)
            text_outputs[i] = [
                {
                    "role": "error",
                    "content": f"{type(result).__name__}: {result}",
                    "error": True,
                    "question_id": question_id,
                }
            ]

    if len(failed_request_ids) > max_failed_requests:
        raise ValueError(
            f"Too many requests failed: {len(failed_request_ids)}/{len(text_outputs)} "
            f"requests failed, allowed failure rate is {allowed_failure_rate}, aborting run."
        )

    runner.process_metrics_final(text_outputs, failed_request_ids=failed_request_ids)

    if failed_request_ids:
        print(
            f"{ANSI_RED}[WARNING] {len(failed_request_ids)}/{len(text_outputs)} request(s) "
            f"failed while processing the dataset. Failed question_ids: "
            f"{failed_question_ids}{ANSI_RESET}"
        )

    return text_outputs


def run_simple(args):
    tokenizer = get_tokenizer(args.tokenizer, trust_remote_code=args.trust_remote_code)
    chat_template_args = args.runtime_params.get("chat_template_args", {})
    dataset_kwargs = dict(args.runtime_params.get("dataset_kwargs", {}))
    if args.num_requests is not None:
        dataset_kwargs["num_samples"] = args.num_requests
    if args.category is not None:
        dataset_kwargs["category"] = args.category

    # When --agentic is set, swap the "speed" dataset class to AgenticSPEEDBench
    active_datasets = dict(datasets_available)
    if args.agentic:
        active_datasets["speed"] = AgenticSPEEDBench
        dataset_kwargs["skip_turns_delta"] = args.agentic_skip_turns_delta

    if args.sweep_config is None:
        run_specs = resolve_single_run_spec(args)
        is_sweep = False
        sweep_output_root = None
    else:
        run_specs = expand_sweep_run_specs(args, active_datasets)
        is_sweep = True
        sweep_output_root = resolve_sweep_output_root(args)
        print(f"Sweep output root: {sweep_output_root}")

    def _dataset_kwargs_for(run_spec):
        rk = dict(dataset_kwargs)
        if run_spec.get("num_requests") is not None:
            rk["num_samples"] = run_spec["num_requests"]
        if run_spec.get("category") is not None:
            rk["category"] = run_spec["category"]
        return rk

    datasets_per_run = [
        build_dataset(
            run_spec,
            tokenizer,
            _dataset_kwargs_for(run_spec),
            active_datasets,
            datasets,
        )
        for run_spec in run_specs
    ]

    max_engine_concurrency = max(run_spec["concurrency"] for run_spec in run_specs)
    # CLI overrides take precedence over --runtime_params; supplying neither
    # leaves engine_args empty (engine auto-derives sequence length) and
    # sampling_kwargs defaulting to greedy (temperature=0).
    #
    # --max_seq_len is the generic sequence-length cap; _MAX_SEQ_LEN_KEY
    # (module scope) maps it to the engine-specific kwarg so cell / variant
    # YAMLs can use one flag regardless of --engine. Engines outside the
    # table fall back to --runtime_params (engine_args.<their-key>).
    engine_args = args.runtime_params.get("engine_args", {})
    if args.max_seq_len is not None:
        key = _MAX_SEQ_LEN_KEY.get(args.engine)
        if key is None:
            raise ValueError(
                f"--max_seq_len is not wired for --engine {args.engine}. "
                f"Use --runtime_params with engine_args.<key> for this engine, "
                f"or extend _MAX_SEQ_LEN_KEY in run.py."
            )
        engine_args[key] = args.max_seq_len
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
            dataset = datasets_per_run[run_index]
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
                args.allowed_failure_rate,
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
        "--num_samples",
        "--num_requests",
        type=int,
        required=False,
        default=None,
        dest="num_requests",
        help="Number of samples from the dataset to run. If not provided, all samples from the dataset will be run.",
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
        choices=[
            "EAGLE3",
            "EAGLE",
            "DRAFT_TARGET",
            "NGRAM",
            "MTP",
            "PARD",
            "DFLASH",
            "NONE",
        ],
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
        "--max_seq_len",
        type=int,
        required=False,
        default=None,
        help=(
            "Max sequence length the engine should reserve (input + output). "
            "Maps to the engine-specific kwarg at the model-wrapper seam: "
            "VLLM → max_model_len, TRTLLM → max_seq_len, SGLANG → context_length. "
            "Overrides the same key in --runtime_params engine_args if both "
            "are set. When neither is set, the engine auto-derives from the "
            "model config + memory budget, which can cap below the input "
            "length on tight GPUs. Set to 40960 for the SPEED-Bench "
            "throughput_32k split (32K input + 4K output + 4K headroom)."
        ),
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
        "--parallel_drafting",
        action="store_true",
        help="Enable parallel drafting (for vLLM)",
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
    parser.add_argument(
        "--stop_think_id",
        type=int,
        nargs="+",
        required=False,
        default=None,
        help="Token IDs that mark the stop-think boundary (can specify multiple sequential tokens)",
    )
    parser.add_argument("--ignore_eos", action="store_true", help="Ignore EOS token")
    parser.add_argument("--aa_timing", action="store_true", help="Enable AA timing metric")
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
        "--agentic",
        action="store_true",
        help="Enable agentic mode: load parquet with full trajectories, "
        "cut at assistant turns, and include tool definitions",
    )
    parser.add_argument(
        "--agentic_skip_turns_delta",
        type=int,
        required=False,
        default=1,
        help="Agentic mode only: Skip every Nth turn per trajectory (1=all, 2=every 2nd, etc.)",
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
        "--temperature",
        type=float,
        required=False,
        default=0.0,
        help="Temperature to use",
    )
    parser.add_argument(
        "--save_dir",
        type=str,
        required=False,
        default=None,
        help="Directory to save the results",
    )
    parser.add_argument(
        "--allowed_failure_rate",
        type=float,
        required=False,
        default=0,
        help="Allowed failed requests rate (default: 0.0, i.e. no failures allowed)",
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
            base = args.save_dir.rstrip("/")
            max_idx = 0
            parent = os.path.dirname(base) or "."
            prefix = os.path.basename(base) + "_"
            for name in os.listdir(parent):
                if name.startswith(prefix):
                    suffix = name[len(prefix) :]
                    if suffix.isdigit():
                        max_idx = max(max_idx, int(suffix))
            args.save_dir = f"{base}_{max_idx + 1}"
            print(f"Save directory already exists, using {args.save_dir}")
        dump_env(args, args.save_dir)

    if args.allowed_failure_rate < 0 or args.allowed_failure_rate > 1:
        raise ValueError(
            f"Allowed failure rate must be between 0 and 1, got {args.allowed_failure_rate}"
        )

    if args.ignore_eos:
        print(
            "Warning: Ignore EOS should only be used in certain cases, do no activate unless necessary"
        )

    run_simple(args)
