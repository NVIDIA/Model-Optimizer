# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
"""Unit tests for ``examples/hf_ptq/kl_divergence``."""

import math

import pytest
import torch
import torch.nn.functional as F
from _test_utils.examples.hf_ptq_example_utils import kl_divergence as kld
from _test_utils.torch.transformers_models import get_tiny_qwen3

SEQ_LEN = 16
FIRST = SEQ_LEN // 2


def _tokens(num_tokens):
    return torch.randint(0, 32, (num_tokens,), generator=torch.Generator().manual_seed(0))


def test_reference_scores_the_second_half_of_each_whole_chunk():
    reference = kld.collect_reference(
        get_tiny_qwen3(), _tokens(3 * SEQ_LEN + 5), num_chunks=10, seq_len=SEQ_LEN, bos_token_id=7
    )

    assert reference.chunks.shape == (3, SEQ_LEN)
    assert (reference.chunks[:, 0] == 7).all()
    assert reference.log_probs.shape == (3, SEQ_LEN - 1 - FIRST, 32)


def test_unchanged_model_has_no_divergence():
    model = get_tiny_qwen3()
    reference = kld.collect_reference(model, _tokens(2 * SEQ_LEN), seq_len=SEQ_LEN)

    result = kld.kl_divergence(model, reference)

    assert result["kld"] == pytest.approx(0, abs=1e-4)
    assert result["same_top"] == 1
    assert result["ppl"] == pytest.approx(result["ppl_base"], rel=1e-3)


@torch.no_grad()
def test_matches_a_direct_computation():
    model = get_tiny_qwen3()
    reference = kld.collect_reference(model, _tokens(2 * SEQ_LEN), seq_len=SEQ_LEN)
    base = [model(input_ids=chunk[None]).logits[0, FIRST:-1].float() for chunk in reference.chunks]
    torch.manual_seed(0)
    for param in model.parameters():
        param.add_(0.05 * torch.randn_like(param))
    quant = [model(input_ids=chunk[None]).logits[0, FIRST:-1].float() for chunk in reference.chunks]

    expected_kld = torch.cat(
        [
            (F.softmax(b, -1) * (F.log_softmax(b, -1) - F.log_softmax(q, -1))).sum(-1)
            for b, q in zip(base, quant)
        ]
    ).mean()
    expected_nll = torch.cat(
        [
            F.cross_entropy(q, chunk[FIRST + 1 :], reduction="none")
            for q, chunk in zip(quant, reference.chunks)
        ]
    ).mean()
    result = kld.kl_divergence(model, reference)

    assert result["num_tokens"] == 2 * (SEQ_LEN - 1 - FIRST)
    assert result["kld"] == pytest.approx(expected_kld.item(), rel=1e-2)
    assert result["ppl"] == pytest.approx(math.exp(expected_nll.item()), rel=1e-3)


def test_mismatched_vocabulary_is_rejected():
    reference = kld.collect_reference(get_tiny_qwen3(), _tokens(SEQ_LEN), seq_len=SEQ_LEN)

    with pytest.raises(ValueError, match="another vocabulary"):
        kld.kl_divergence(get_tiny_qwen3(vocab_size=64), reference)


def test_reference_round_trips_through_a_file(tmp_path):
    reference = kld.collect_reference(get_tiny_qwen3(), _tokens(2 * SEQ_LEN), seq_len=SEQ_LEN)
    reference.metadata = {"data": "wikitext2", "model": "tiny"}
    path = tmp_path / "ref.pt"

    kld.save_reference(reference, str(path))
    loaded = kld.load_reference(str(path))

    assert torch.equal(loaded.chunks, reference.chunks)
    assert torch.equal(loaded.log_probs, reference.log_probs)
    assert loaded.metadata == reference.metadata


def _char_tokenizer(text, add_special_tokens):
    assert not add_special_tokens
    return {"input_ids": [ord(c) for c in text]}


def test_load_eval_tokens_tokenizes_a_text_file_without_special_tokens(tmp_path):
    path = tmp_path / "eval.txt"
    path.write_text(" \n = Title = \n", encoding="utf-8")

    assert kld.load_eval_tokens(_char_tokenizer, str(path)).tolist() == [
        ord(c) for c in " \n = Title = \n"
    ]


def test_load_eval_tokens_draws_dataset_samples_until_long_enough(monkeypatch):
    requested = []

    def samples(name, num_samples, apply_chat_template, tokenizer):
        requested.append(num_samples)
        return ["abcd"] * num_samples

    monkeypatch.setattr(kld, "get_dataset_samples", samples)

    tokens = kld.load_eval_tokens(_char_tokenizer, "some-dataset", min_tokens=500)

    assert requested == [64, 128]
    assert tokens.numel() >= 500
