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
# SPDX-License-Identifier: Apache-2.0 AND MIT
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

import torch
from datasets import load_dataset
from tqdm import tqdm
from transformers import PreTrainedTokenizer

from .. import distributed as dist
from .. import print_rank_0
from .megatron_generate import megatron_prefill

__all__ = ["megatron_mmlu"]

_CHOICES = ["A", "B", "C", "D"]


def megatron_mmlu(
    model,
    tokenizer: PreTrainedTokenizer,
    few_shots: int = 0,
    fraction: float = 0.05,
    batch_size: int = 1,
) -> float:
    """Evaluate the model on MMLU using log-likelihood scoring over batched prefill passes.

    Instead of autoregressively generating tokens, a single prefill forward pass is run per
    batch and the answer is selected as argmax over the four choice token logits at the last
    prompt position. This is the same approach used by lm-evaluation-harness.

    Args:
        model: The model to evaluate.
        tokenizer: The tokenizer to use.
        few_shots: The number of few-shot examples to use.
        fraction: The fraction of the test set to evaluate on.
        batch_size: Number of examples to process in one forward pass.
    """
    print_rank_0(
        f"\nMMLU ({fraction * 100}%, {few_shots}-shot, Batch Size: {batch_size}) evaluation started...\n"
        "First batch may take longer to evaluate for Pipeline Parallel models."
    )
    assert 0 < fraction <= 1, "Fraction must be between 0 and 1"

    # Token IDs for " A", " B", " C", " D" — the last subword handles edge cases.
    choice_ids = [tokenizer.encode(f" {c}", add_special_tokens=False)[-1] for c in _CHOICES]

    def _format_example(example, include_answer: bool = True):
        prompt = example["question"]
        for choice, answer in zip(_CHOICES, example["choices"]):
            prompt += f"\n{choice}. {answer}"
        if include_answer:
            prompt += "Answer: {}\n\n".format(_CHOICES[example["answer"]])
        else:
            prompt += "\nAnswer:"
        return prompt

    def _generate_prompt(test_example, dev_examples, few_shots=0):
        prompt = "The following are multiple choice questions (with answers) about {}.\n\n".format(
            " ".join(test_example["subject"].split("_"))
        )
        for i in range(few_shots):
            prompt += _format_example(dev_examples[i])
        prompt += _format_example(test_example, include_answer=False)
        return prompt

    # Load all subjects in two dataset calls instead of 2x num_subjects calls.
    # The "all" config includes a "subject" field for per-subject reporting.
    test_dataset = load_dataset("cais/mmlu", "all", split="test")
    dev_dataset = load_dataset("cais/mmlu", "all", split="dev") if few_shots > 0 else None

    # Group dev examples by subject for few-shot prompt construction.
    dev_by_subject: dict = {}
    if dev_dataset is not None:
        for ex in dev_dataset:
            dev_by_subject.setdefault(ex["subject"], []).append(ex)

    # Collect all examples, tracking subject membership for per-subject reporting.
    all_subjects_seen: list[str] = []
    all_prompts: list[str] = []
    all_labels: list[str] = []

    # Count test examples per subject to apply the fraction cutoff correctly.
    subject_counts: dict[str, int] = {}
    for ex in test_dataset:
        subject_counts[ex["subject"]] = subject_counts.get(ex["subject"], 0) + 1

    subject_idx: dict[str, int] = {}
    for ex in test_dataset:
        subj = ex["subject"]
        idx = subject_idx.get(subj, 0)
        if idx >= fraction * subject_counts[subj]:
            continue
        subject_idx[subj] = idx + 1
        prompt = _generate_prompt(ex, dev_by_subject.get(subj, []), few_shots=few_shots)
        all_prompts.append(prompt)
        all_labels.append(_CHOICES[ex["answer"]])
        all_subjects_seen.append(subj)

    # Tokenize all prompts and sort by length to minimise padding waste within batches.
    encoded = [tokenizer(p, return_tensors="pt").input_ids[0] for p in all_prompts]
    lengths = [e.shape[0] for e in encoded]
    order = sorted(range(len(encoded)), key=lambda i: lengths[i], reverse=True)

    sorted_encoded = [encoded[i] for i in order]
    sorted_lengths = [lengths[i] for i in order]

    # Run inference in global batches.
    predictions: list[str] = [""] * len(encoded)
    n_batches = (len(sorted_encoded) + batch_size - 1) // batch_size
    pbar = tqdm(
        range(0, len(sorted_encoded), batch_size),
        total=n_batches,
        desc="MMLU",
        unit="batch",
        disable=not dist.is_master(),
    )
    for batch_start in pbar:
        batch_enc = sorted_encoded[batch_start : batch_start + batch_size]
        batch_len = sorted_lengths[batch_start : batch_start + batch_size]
        max_len = max(batch_len)

        # Right-pad to max_len; causal mask means the last real token is unaffected by padding.
        padded = torch.zeros(len(batch_enc), max_len, dtype=torch.long)
        for i, (e, seq_len) in enumerate(zip(batch_enc, batch_len)):
            padded[i, :seq_len] = e

        logits = megatron_prefill(model, padded.cuda())  # [B, max_len, vocab]

        for i, seq_len in enumerate(batch_len):
            answer_logits = logits[i, seq_len - 1, choice_ids]
            predictions[order[batch_start + i]] = _CHOICES[answer_logits.argmax().item()]

        examples_done = min(batch_start + batch_size, len(sorted_encoded))
        pbar.set_postfix(examples=f"{examples_done}/{len(sorted_encoded)}")

    # Compute per-subject accuracy and overall average.
    subject_correct: dict[str, list[bool]] = {}
    for pred, label, subj in zip(predictions, all_labels, all_subjects_seen):
        subject_correct.setdefault(subj, []).append(pred == label)

    all_correct = [pred == label for pred, label in zip(predictions, all_labels)]
    n_total = len(all_correct)
    avg = sum(all_correct) / n_total

    print_rank_0("{:48} | (ACC) | Count/Total".format("Subject"))
    print_rank_0("{:48} | {:5} | {:11}".format("-" * 48, "-" * 5, "-" * 11))
    for subj in sorted(subject_correct):
        correct = subject_correct[subj]
        n = len(correct)
        print_rank_0(f"{subj:48} | {sum(correct) / n:.3f} | {sum(correct):5}/{n:5}")
    print_rank_0("{:48} | {:5} | {:11}".format("-" * 48, "-" * 5, "-" * 11))
    print_rank_0("{:48} | {:.3f} | {:5}/{:5}".format("average", avg, sum(all_correct), n_total))

    return avg
