# Adapted from https://github.com/declare-lab/instruct-eval/blob/720e66f627369266ed1cfd74426666ec37e524bc/mmlu.py

# MIT License
#
# Copyright (c) 2020 Dan Hendrycks
# Copyright (c) 2023 Deep Cognition and Language Research (DeCLaRe) Lab
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

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

"""A simple MMLU evaluation for Megatron LM models."""

import requests
import torch
import transformers
from datasets import load_dataset

from .megatron_generate import megatron_generate

__all__ = ["megatron_mmlu"]


def _get_all_subjects():
    """All subjects (anatomy, ...) can be acquired from querying all subsets and splits."""
    response = requests.get(
        "https://datasets-server.huggingface.co/splits?dataset=cais/mmlu", timeout=10
    )
    data = response.json()
    all_subjects = set()
    for split in data["splits"]:
        all_subjects.add(split["config"])
    for name in ["all", "auxiliary_train"]:
        all_subjects.discard(name)
    return sorted(all_subjects)


def megatron_mmlu(
    model,
    tokenizer: transformers.PreTrainedTokenizer,
    few_shots: int = 0,
    percentage: float = 0.05,
    enable_kv_cache: bool = False,
) -> float:
    """Evaluate the model on MMLU.

    Args:
        model: The model to evaluate.
        tokenizer: The tokenizer to use.
        few_shots: The number of few-shot examples to use.
        percentage: The percentage of the test set to evaluate on.
        enable_kv_cache: Whether to disable KV-cache.
    """
    all_correct = {}
    all_subjects = _get_all_subjects()

    def _format_example(example, include_answer: bool = True):
        """Format an example into a multi-choices problem."""
        prompt = example["question"]
        for choice, answer in zip(["A", "B", "C", "D"], example["choices"]):
            prompt += f"\n{choice}. {answer}"
        if include_answer:
            prompt += "Answer: {}\n\n".format(example["answer"])
        else:
            prompt += "\nAnswer:"
        return prompt

    def _generate_prompt(test_example, dev_examples, few_shots=0):
        """Generating few-shot prompts."""
        prompt = "The following are multiple choice questions (with answers) about {}.\n\n".format(
            " ".join(test_example["subject"].split("_"))
        )
        for i in range(few_shots):
            prompt += _format_example(dev_examples[i])
        prompt += _format_example(test_example, include_answer=False)
        return prompt

    if torch.distributed.get_rank() == 0:
        print(f"\nMMLU ({percentage * 100}%, {few_shots}-shot) evaluation started...\n", flush=True)
        print("{:48} | (ACC) | Count/Total".format("Subject"), flush=True)
        print("{:48} | {:5} | {:11}".format("-" * 48, "-" * 5, "-" * 11), flush=True)

    for subject in all_subjects:
        test_data = load_dataset("cais/mmlu", subject, split="test")
        dev_data = load_dataset("cais/mmlu", subject, split="dev")

        correct = []
        for idx, test_example in enumerate(test_data):
            if idx > percentage * len(test_data):
                break
            prompt = _generate_prompt(test_example, dev_data, few_shots=few_shots)
            label = ["A", "B", "C", "D"][test_example["answer"]]
            tokens = tokenizer(prompt, return_tensors="pt")
            generated_ids = megatron_generate(
                model,
                tokens.input_ids.cuda(),
                osl=2,
                disable_tqdm=True,
                enable_kv_cache=enable_kv_cache,
            )
            predict = tokenizer.batch_decode(generated_ids)[0].strip()
            correct += [True] if predict.startswith(label) else [False]
        all_correct[subject] = correct

        if torch.distributed.get_rank() == 0:
            print(
                f"{subject:48} | {sum(correct) / len(correct):.3f} | {sum(correct):5}/{len(correct):5}",
                flush=True,
            )

        avg_correct = []

    for subject, correct in all_correct.items():
        avg_correct += correct

    if torch.distributed.get_rank() == 0:
        print("{:48} | {:5} | {:11}".format("-" * 48, "-" * 5, "-" * 11), flush=True)
        print(
            "{:48} | {:.3f} | {:5}/{:5}".format(
                "average", sum(avg_correct) / len(avg_correct), sum(avg_correct), len(avg_correct)
            ),
            flush=True,
        )

    return sum(avg_correct) / len(avg_correct)
