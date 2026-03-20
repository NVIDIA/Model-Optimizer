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

# mypy: ignore-errors
# ruff: noqa: D103, PERF401

"""Perplexity evaluation for language models.

Ported from FP-Quant: https://github.com/IST-DASLab/FP-Quant
"""

import torch
import torch.nn.functional as F
from tqdm import trange


@torch.no_grad()
def compute_perplexity(model, data, batch_size: int = 1):
    num_samples = len(data)
    device = next(model.parameters()).device
    # Running estimate of negative log-likelihood
    nll_running = 0
    # Number of tokens processed to far
    tokens_processed = 0
    # Loop through each batch
    for i in trange(0, num_samples, batch_size, desc="Computing perplexity", leave=False):
        j = min(i + batch_size, num_samples)
        inputs = torch.cat(data[i:j]).to(device)
        # Forward pass through the model
        lm_logits = model(inputs).logits
        # Shift logits and labels for next token prediction
        shift_logits = lm_logits[:, :-1, :].contiguous()
        shift_labels = inputs[:, 1:]
        # Compute loss
        loss = F.cross_entropy(
            shift_logits.reshape(-1, shift_logits.size(-1)), shift_labels.reshape(-1)
        )
        # Calculate negative log likelihood
        a = shift_labels.numel() / (tokens_processed + shift_labels.numel())
        b = tokens_processed / (tokens_processed + shift_labels.numel())
        nll_running = a * loss + b * nll_running
        # Update number of processed tokens
        tokens_processed += shift_labels.numel()
    # Compute perplexity
    ppl = nll_running.exp().item()
    return ppl


def get_wikitext2(tokenizer, sequence_length: int):
    """Load WikiText-2 test set as a list of tokenized sequences for perplexity evaluation.

    Args:
        tokenizer: HuggingFace tokenizer.
        sequence_length: Length of each evaluation sequence.

    Returns:
        List of tensors, each of shape ``[1, sequence_length]``.
    """
    from datasets import load_dataset

    test_dataset_raw = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
    test_dataset_tok = tokenizer(
        "\n\n".join(test_dataset_raw["text"]), return_tensors="pt"
    ).input_ids
    num_test_sequences = test_dataset_tok.numel() // sequence_length
    test_loader = []
    for i in range(num_test_sequences):
        test_loader.append(test_dataset_tok[:, i * sequence_length : (i + 1) * sequence_length])
    return test_loader
