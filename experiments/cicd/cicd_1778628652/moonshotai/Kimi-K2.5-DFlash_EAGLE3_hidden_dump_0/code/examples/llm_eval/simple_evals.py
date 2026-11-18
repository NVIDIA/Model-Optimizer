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
import json

import pandas as pd

from . import common
from .drop_eval import DropEval
from .gpqa_eval import GPQAEval
from .humaneval_eval import HumanEval
from .math_eval import MathEval
from .mgsm_eval import MGSMEval
from .mmlu_eval import MMLUEval
from .sampler.chat_completion_sampler import OPENAI_SYSTEM_MESSAGE_API, ChatCompletionSampler
from .simpleqa_eval import SimpleQAEval


def main():
    parser = argparse.ArgumentParser(
        description="Run sampling and evaluations using different samplers and evaluations."
    )
    parser.add_argument("--list-models", action="store_true", help="List available models")
    parser.add_argument("--model", type=str, help="Select a model by name")
    parser.add_argument("--debug", action="store_true", help="Run in debug mode")
    parser.add_argument(
        "--examples", type=int, help="Number of examples to use (overrides default)"
    )

    parser.add_argument("--evals", type=str, help="Comma separate eval tasks", default="mmlu")
    parser.add_argument("--max_tokens", type=int, help="max tokens", default=2048)
    parser.add_argument("--temperature", type=float, help="temperature", default=0)

    args = parser.parse_args()

    # modelopt modification
    models = {
        args.model: ChatCompletionSampler(
            model=args.model,
            system_message=OPENAI_SYSTEM_MESSAGE_API,
            max_tokens=args.max_tokens,
            temperature=args.temperature,
        ),
    }

    if args.list_models:
        print("Available models:")
        for model_name in models:
            print(f" - {model_name}")
        return

    if args.model:
        if args.model not in models:
            print(f"Error: Model '{args.model}' not found.")
            return
        models = {args.model: models[args.model]}

    grading_sampler = ChatCompletionSampler(model="gpt-4o")
    equality_checker = ChatCompletionSampler(model="gpt-4-turbo-preview")
    # ^^^ used for fuzzy matching, just for math

    def get_evals(eval_name, debug_mode):
        num_examples = args.examples if args.examples is not None else (5 if debug_mode else None)
        # Set num_examples = None to reproduce full evals
        match eval_name:
            case "mmlu":
                return MMLUEval(num_examples=1 if debug_mode else num_examples)
            case "math":
                return MathEval(
                    equality_checker=equality_checker,
                    num_examples=num_examples,
                    n_repeats=1 if debug_mode else 10,
                )
            case "gpqa":
                return GPQAEval(n_repeats=1 if debug_mode else 10, num_examples=num_examples)
            case "mgsm":
                return MGSMEval(num_examples_per_lang=10 if debug_mode else 250)
            case "drop":
                return DropEval(
                    num_examples=10 if debug_mode else num_examples,
                    train_samples_per_prompt=3,
                )
            case "humaneval":
                return HumanEval(num_examples=10 if debug_mode else num_examples)
            case "simpleqa":
                return SimpleQAEval(
                    grader_model=grading_sampler,
                    num_examples=10 if debug_mode else num_examples,
                )
            case _:
                raise Exception(f"Unrecognized eval type: {eval_name}")

    # modelopt modification
    eval_names = args.evals.split(",")
    available_eval_names = {"simpleqa", "mmlu", "math", "gpqa", "mgsm", "drop", "humaneval"}
    for eval_name in eval_names:
        assert eval_name in available_eval_names, f"{eval_name} not in {available_eval_names}"

    evals = {eval_name: get_evals(eval_name, args.debug) for eval_name in eval_names}
    print(evals)
    debug_suffix = "_DEBUG" if args.debug else ""
    print(debug_suffix)
    mergekey2resultpath = {}
    for model_name, sampler in models.items():
        for eval_name, eval_obj in evals.items():
            result = eval_obj(sampler)
            # ^^^ how to use a sampler
            file_stem = f"{eval_name}_{model_name}"
            report_filename = f"/tmp/{file_stem}{debug_suffix}.html"
            print(f"Writing report to {report_filename}")
            with open(report_filename, "w") as fh:
                fh.write(common.make_report(result))
            metrics = result.metrics | {"score": result.score}
            print(metrics)
            result_filename = f"/tmp/{file_stem}{debug_suffix}.json"
            with open(result_filename, "w") as f:
                f.write(json.dumps(metrics, indent=2))
            print(f"Writing results to {result_filename}")
            mergekey2resultpath[f"{file_stem}"] = result_filename
    merge_metrics = []
    for eval_model_name, result_filename in mergekey2resultpath.items():
        try:
            result = json.load(open(result_filename, "r+"))
        except Exception as e:
            print(e, result_filename)
            continue
        result = result.get("f1_score", result.get("score", None))
        eval_name = eval_model_name[: eval_model_name.find("_")]
        model_name = eval_model_name[eval_model_name.find("_") + 1 :]
        merge_metrics.append({"eval_name": eval_name, "model_name": model_name, "metric": result})
    merge_metrics_df = pd.DataFrame(merge_metrics).pivot(index=["model_name"], columns="eval_name")
    print("\nAll results: ")
    print(merge_metrics_df.to_markdown())
    return merge_metrics


if __name__ == "__main__":
    main()
