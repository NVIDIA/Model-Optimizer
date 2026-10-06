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

"""Offline correctness cases for the KL example; no checkpoints or datasets are downloaded."""

import copy
import importlib
import math
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from transformers import GPT2Config, GPT2LMHeadModel


@pytest.fixture
def kl_eval(monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[3] / "examples" / "llm_eval"))
    return importlib.import_module("kl_eval")


@pytest.mark.parametrize("chunk_size", [1, 2, 32])
def test_kl_direction_shared_support_and_chunk_reduction(kl_eval, chunk_size):
    p = [0.5, 0.3, 0.2]
    q = [0.25, 0.25, 0.5]  # Candidate's favorite token is outside the reference top-2.
    reference = torch.tensor([p, p, q]).log()
    quantized = torch.tensor([q, q, q]).log()
    actual = kl_eval.mean_kl(reference, quantized, top_k=2, chunk_size=chunk_size)
    expected_full = sum(a * math.log(a / b) for a, b in zip(p, q)) * 2 / 3
    expected_head = (0.625 * math.log(0.625 / 0.5) + 0.375 * math.log(0.375 / 0.5)) * 2 / 3
    assert actual["full_vocab_kl"] == pytest.approx(expected_full, abs=1e-6)
    assert actual["conditional_topk_kl"] == pytest.approx(expected_head, abs=1e-6)
    entire_vocab = kl_eval.mean_kl(reference, quantized, top_k=3)
    assert entire_vocab["conditional_topk_kl"] == pytest.approx(expected_full, abs=1e-6)


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_identical_logits_have_zero_kl(kl_eval, dtype):
    logits = torch.tensor([[1, 2, -3], [8, 4, 1]], dtype=dtype)
    metrics = kl_eval.mean_kl(logits, logits.clone(), top_k=2)
    assert metrics == pytest.approx({"full_vocab_kl": 0, "conditional_topk_kl": 0}, abs=1e-7)


def _tiny_model():
    return GPT2LMHeadModel(
        GPT2Config(
            vocab_size=16,
            n_positions=16,
            n_embd=8,
            n_layer=1,
            n_head=1,
            resid_pdrop=0,
            embd_pdrop=0,
            attn_pdrop=0,
            bos_token_id=1,
            eos_token_id=0,
            pad_token_id=0,
        )
    ).eval()


def test_scores_all_and_only_continuation_predictions(kl_eval):
    reference = _tiny_model()
    quantized = copy.deepcopy(reference)
    with torch.no_grad():
        quantized.lm_head.weight[3, 0].add_(3)
    sequence = torch.tensor([[4, 5, 6, 7, 8]])
    with torch.inference_mode():
        p = reference(sequence[:, :-1], use_cache=False).logits[0].float().softmax(-1)
        q = quantized(sequence[:, :-1], use_cache=False).logits[0].float().softmax(-1)
        expected = (p[2:] * (p[2:].log() - q[2:].log())).sum(-1).mean().item()
    assert expected > 1e-6
    actual = kl_eval.score_continuation(reference, quantized, sequence, prompt_tokens=3, top_k=4)
    assert actual["full_vocab_kl"] == pytest.approx(expected, abs=1e-6)


@pytest.mark.parametrize("detailed_results", [False, True])
def test_real_generation_stops_at_eos_and_scores_it(kl_eval, monkeypatch, detailed_results):
    argv = ["kl_eval.py", "--model", "local-test-model", "--recipe", "local-test-recipe"]
    if detailed_results:
        argv.append("--detailed_results")
    monkeypatch.setattr(sys, "argv", argv)
    args = kl_eval._parse_args()
    reference = _tiny_model()
    with torch.no_grad():
        for parameter in reference.parameters():
            parameter.zero_()
    # Equal logits greedily choose token 0, which is this model's EOS.
    # Checkpoint decoding settings must not override the evaluation's plain greedy policy.
    reference.generation_config.suppress_tokens = [0]
    reference.generation_config.return_dict_in_generate = True
    original_generation = copy.deepcopy(reference.generation_config)
    quantized = copy.deepcopy(reference)
    tokenizer = SimpleNamespace(eos_token_id=0, pad_token_id=0)
    prompts = [{"block_index": 3, "input_ids": [4, 5, 6]}]
    result = kl_eval.evaluate(
        reference,
        quantized,
        tokenizer,
        prompts,
        max_new_tokens=4,
        top_k=4,
        detailed_results=args.detailed_results,
    )
    assert reference.generation_config == original_generation
    expected = {"full_vocab_kl": 0, "conditional_topk_kl": 0}
    if detailed_results:
        assert result["examples"] == [
            {
                "example": 0,
                "block_index": 3,
                "prompt_ids": [4, 5, 6],
                "generated_ids": [0],
                "generated_tokens": 1,
                **expected,
            }
        ]
        assert result["summary"] == pytest.approx(expected, abs=1e-7)
    else:
        assert result == pytest.approx(expected, abs=1e-7)


def test_invalid_logits_fail_instead_of_dropping_positions(kl_eval):
    with pytest.raises(ValueError, match="Non-finite KL"):
        kl_eval.mean_kl(torch.zeros(2, 4), torch.full((2, 4), float("nan")), top_k=2)
