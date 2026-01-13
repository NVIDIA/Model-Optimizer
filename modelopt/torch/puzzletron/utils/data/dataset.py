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
import functools
from collections.abc import Sequence

import numpy as np
import torch
from torch.utils.data import IterableDataset

FIM_TOKEN_START = "<fim"  # nosec B105
FIM_TOKEN_CONNECTOR_STAR = "_"  # nosec B105
FIM_TOKEN_CONNECTOR_SANTA = "-"  # nosec B105
FIM_TOKEN_END_LIST = ["prefix>", "middle>", "suffix>", "pad>"]
CODEGEN_FIM_TOKENS = ["<mask_1>", "<|endoftext|>", "<sep>"]


class ConstantLengthDataset(IterableDataset):
    """Iterable dataset that returns constant length chunks of tokens from stream of text files.

    Args:
            tokenizer (Tokenizer): The processor used for proccessing the data.
            dataset (dataset.Dataset): Dataset with text files.
            infinite (bool): If True the iterator is reset after dataset reaches end else stops.
            seq_length (int): Length of token sequences to return.
            num_of_sequences (int): Number of token sequences to keep in buffer.
            chars_per_token (int): Number of characters per token used to estimate number of tokens in text buffer.
            fim_rate (float): Rate (0.0 to 1.0) that sample will be permuted with FIM.
            fim_spm_rate (float): Rate (0.0 to 1.0) of FIM permuations that will use SPM.
            seed (int): Seed for random number generator.
            label_shift (bool): Whether to shift labels by 1 or not.
    """

    def __init__(
        self,
        tokenizer,
        dataset,
        infinite=False,
        seq_length=1024,
        num_of_sequences=1024,
        chars_per_token=3.6,
        content_field="content",
        fim_rate=0.5,
        fim_spm_rate=0.5,
        seed=0,
        label_shift=True,
        max_sample_length=200_000,
        tokens_field="token_ids",
        source_datasets_to_discard: Sequence[str] | None = tuple(),
        bos_rate: float = 1.0,
        return_cu_seqlens: bool = False,
        seqlen_cap: int | None = None,
    ):
        self.tokenizer = tokenizer
        self.concat_token_id = tokenizer.eos_token_id
        # self.concat_token_id = tokenizer.eos_id # for lit-lamma tokenizer
        self.dataset = dataset
        self.is_dataset_already_tokenized = tokens_field in self.dataset.column_names
        self.seq_length = seq_length
        self.infinite = infinite
        self.current_size = 0
        if not self.is_dataset_already_tokenized:
            self.max_buffer_size = seq_length * chars_per_token * num_of_sequences
            self.max_sample_length = max_sample_length
        else:
            self.max_buffer_size = seq_length * num_of_sequences
            # self.max_sample_length = int(max_sample_length / chars_per_token)
            self.max_sample_length = max_sample_length  # we don't know the exact chars_per_token
        self.content_field = content_field
        self.tokens_field = tokens_field
        self.fim_rate = fim_rate
        self.fim_spm_rate = fim_spm_rate
        self.seed = seed
        self.max_sample_length = max_sample_length

        self.fim_token_ids = get_fim_token_ids(self.tokenizer)
        if None in self.fim_token_ids.values() and self.fim_rate > 0:
            self.fim_rate = 0
        self.label_shift = label_shift
        self.bos_rate = bos_rate
        self.source_datasets_to_discard = (
            source_datasets_to_discard if source_datasets_to_discard is not None else tuple()
        )
        self.return_cu_seqlens = return_cu_seqlens
        self.seqlen_cap = seqlen_cap
        self.np_rng = np.random.RandomState(seed=self.seed)

    def __iter__(self) -> dict[str, torch.Tensor]:
        iterator = iter(self.dataset)
        more_examples = True
        while more_examples:
            buffer, buffer_len = [], 0
            while True:
                if buffer_len >= self.max_buffer_size:
                    break
                try:
                    sample = next(iterator)
                    if (
                        len(self.source_datasets_to_discard) > 0
                        and sample["dataset_name"] in self.source_datasets_to_discard
                    ):
                        continue
                    if not self.is_dataset_already_tokenized:
                        sample = sample[self.content_field]
                        if (
                            isinstance(sample, list)
                            and isinstance(sample[0], dict)
                            and {"content", "role"}.issubset(sample[0])
                        ):
                            if len(sample) > 1:
                                sample = self.tokenizer.apply_chat_template(sample, tokenize=False)
                            else:
                                sample = sample[0]["content"]
                    else:
                        sample = sample[self.tokens_field]
                    sample = sample[: self.max_sample_length]
                    buffer.append(sample)
                    buffer_len += len(sample)
                except StopIteration:
                    if self.infinite:
                        iterator = iter(self.dataset)
                    else:
                        more_examples = False
                        break

            if not self.is_dataset_already_tokenized:
                tokenized_inputs = self.tokenizer(buffer, truncation=False)["input_ids"]
            else:
                tokenized_inputs = buffer

            all_token_ids = []

            for tokenized_input in tokenized_inputs:
                if (
                    self.bos_rate < 1.0
                    and not self.np_rng.binomial(1, self.bos_rate)
                    and self.tokenizer.bos_token_id is not None
                    and tokenized_input[0] == self.tokenizer.bos_token_id
                ):
                    tokenized_input = tokenized_input[1:]
                # optionally do FIM permutations
                if self.fim_rate > 0:
                    tokenized_input, np_rng = permute(
                        sample=tokenized_input,
                        np_rng=self.np_rng,
                        fim_token_ids=self.fim_token_ids,
                        fim_rate=self.fim_rate,
                        fim_spm_rate=self.fim_spm_rate,
                        truncate_or_pad=False,
                    )

                all_token_ids.extend(tokenized_input + [self.concat_token_id])

            examples = []
            # cuts code snippets in the middle to yield constant length instances
            for i in range(0, len(all_token_ids), self.seq_length):
                input_ids = all_token_ids[i : i + self.seq_length]
                labels = all_token_ids[
                    i + int(self.label_shift) : i + int(self.label_shift) + self.seq_length
                ]
                # ignores last short example in the buffer
                if len(labels) == self.seq_length:
                    examples.append((input_ids, labels))

            shuffling_indices = self.np_rng.permutation(len(examples))
            examples = [examples[i] for i in shuffling_indices]

            for input_ids, labels in examples:
                self.current_size += 1
                input_ids = torch.LongTensor(input_ids)
                if self.return_cu_seqlens:
                    cu_seqlens = self.prepare_cu_seqlens(input_ids)
                    yield {
                        "input_ids": input_ids,
                        "targets": torch.LongTensor(labels),
                        "cu_seqlens": cu_seqlens,
                    }
                else:
                    yield {
                        "input_ids": input_ids,
                        "targets": torch.LongTensor(labels),
                    }

    def prepare_cu_seqlens(self, input_ids):
        if not self.return_cu_seqlens:
            return None
        # seqlens is of shape (num_seqs+1,) and with the property that
        # the i-th sequnce is input_ids[seqlens[i-1]:seqlens[i]]
        cu_seqlens = (input_ids == self.concat_token_id).nonzero().squeeze(-1).int() + 1
        cu_seqlens = torch.cat(
            (
                torch.IntTensor([0]),
                cu_seqlens,
                torch.IntTensor([len(input_ids)]),
            )
        )
        if self.seqlen_cap is not None:
            i = 1
            while i < len(cu_seqlens):
                curr_seqlen = cu_seqlens[i] - cu_seqlens[i - 1]
                if curr_seqlen > self.seqlen_cap:
                    cu_seqlens = torch.cat(
                        (cu_seqlens[:i], cu_seqlens[[i - 1]] + self.seqlen_cap, cu_seqlens[i:])
                    )
                i += 1
        if cu_seqlens[-1] == cu_seqlens[-2]:
            cu_seqlens = cu_seqlens[:-1]
        return cu_seqlens


## Adapted from https://github.com/NVIDIA/Megatron-LM/blob/6c4bf908df8fd86b4977f54bf5b8bd4b521003d1/megatron/data/gpt_dataset.py
def permute(
    sample,
    np_rng,
    fim_token_ids,
    fim_rate=0.5,
    fim_spm_rate=0.5,
    truncate_or_pad=False,
):
    """Take in a sample (list of tokens) and perform a FIM transformation on it with a probability of fim_rate, using two FIM modes:
    PSM and SPM (with a probability of fim_spm_rate).
    """
    if np_rng.binomial(1, fim_rate):
        boundaries = list(np_rng.randint(low=0, high=len(sample) + 1, size=2))
        boundaries.sort()

        prefix = np.array(sample[: boundaries[0]], dtype=np.int64)
        middle = np.array(sample[boundaries[0] : boundaries[1]], dtype=np.int64)
        suffix = np.array(sample[boundaries[1] :], dtype=np.int64)

        if truncate_or_pad:
            raise NotImplementedError

        if "<mask_1>" in fim_token_ids:  # use codegen FIM pattern
            assert fim_spm_rate == 0
            new_sample = np.concatenate(
                [
                    prefix,
                    [fim_token_ids["<mask_1>"]],
                    suffix,
                    [fim_token_ids["<|endoftext|>"]],
                    [fim_token_ids["<sep>"]],
                    [fim_token_ids["<mask_1>"]],
                    middle,
                ]
            )
        elif np_rng.binomial(1, fim_spm_rate):
            # SPM (variant 2 from FIM paper)
            new_sample = np.concatenate(
                [
                    [fim_token_ids["prefix_tok_id"], fim_token_ids["suffix_tok_id"]],
                    suffix,
                    [fim_token_ids["middle_tok_id"]],
                    prefix,
                    middle,
                ]
            )
        else:
            # PSM
            new_sample = np.concatenate(
                [
                    [fim_token_ids["prefix_tok_id"]],
                    prefix,
                    [fim_token_ids["suffix_tok_id"]],
                    suffix,
                    [fim_token_ids["middle_tok_id"]],
                    middle,
                ]
            )
    else:
        # don't do FIM preproc
        new_sample = sample

    return list(new_sample), np_rng


# this is expensive so we cache it
@functools.cache
def get_fim_token_ids(tokenizer):
    # ugly fix for Salesforce/codegen25-7b-multi tokenizer
    if hasattr(tokenizer, "encoder"):
        search_vocab = tokenizer.encoder._special_tokens
        fim_token_ids = {tok: search_vocab.get(tok, None) for tok in CODEGEN_FIM_TOKENS}
    else:
        search_vocab = tokenizer.vocab
        if (FIM_TOKEN_START + FIM_TOKEN_CONNECTOR_STAR + FIM_TOKEN_END_LIST[0]) in search_vocab:
            prefix_tok_id, middle_tok_id, suffix_tok_id, pad_tok_id = (
                search_vocab.get(FIM_TOKEN_START + FIM_TOKEN_CONNECTOR_STAR + tok, None)
                for tok in FIM_TOKEN_END_LIST
            )
        else:
            prefix_tok_id, middle_tok_id, suffix_tok_id, pad_tok_id = (
                search_vocab.get(FIM_TOKEN_START + FIM_TOKEN_CONNECTOR_SANTA + tok, None)
                for tok in FIM_TOKEN_END_LIST
            )
        fim_token_ids = {
            "suffix_tok_id": suffix_tok_id,
            "prefix_tok_id": prefix_tok_id,
            "middle_tok_id": middle_tok_id,
            "pad_tok_id": pad_tok_id,
        }
    return fim_token_ids
