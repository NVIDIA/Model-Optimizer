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

import importlib
import math
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from datasets import Dataset
from transformers import GPT2Config


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


def test_identical_distributions_have_zero_kl(kl_eval):
    logprobs = torch.tensor([[1, 2, -3], [8, 4, 1]], dtype=torch.float32).log_softmax(-1)
    metrics = kl_eval.mean_kl(logprobs, logprobs.clone(), top_k=2)
    assert metrics == pytest.approx({"full_vocab_kl": 0, "conditional_topk_kl": 0}, abs=1e-7)


def _logprob_row(probabilities):
    # vLLM maps token IDs to Logprob records; dictionary order is not vocabulary order.
    return {
        token_id: SimpleNamespace(logprob=math.log(probabilities[token_id]))
        for token_id in reversed(range(len(probabilities)))
    }


def test_scores_all_and_only_continuation_predictions(kl_eval):
    probabilities = [[0.1, 0.2, 0.3, 0.4], [0.4, 0.3, 0.2, 0.1], [0.2, 0.4, 0.1, 0.3]]
    output = SimpleNamespace(
        prompt_token_ids=[2, 1, 3, 1, 2, 0],  # Three-token continuation ending in EOS=0.
        prompt_logprobs=[
            None,
            _logprob_row([0.25] * 4),
            _logprob_row([0.25] * 4),
            *[_logprob_row(row) for row in probabilities],
        ],
    )
    actual = kl_eval._continuation_logprobs(output, prompt_tokens=3, vocab_size=4)
    assert actual.dtype == torch.float32
    torch.testing.assert_close(actual, torch.tensor(probabilities).log())


@pytest.mark.parametrize("duplicate_actual_token", [False, True])
def test_flat_logprobs_preserve_token_ids(kl_eval, duplicate_actual_token):
    probabilities = [0.1, 0.2, 0.3, 0.4]
    token_ids = [3, 2, 1, 0]
    if duplicate_actual_token:
        token_ids.insert(0, 0)
    flat = SimpleNamespace(
        start_indices=[0, 0],
        end_indices=[0, len(token_ids)],
        token_ids=token_ids,
        logprobs=[math.log(probabilities[token_id]) for token_id in token_ids],
    )
    output = SimpleNamespace(prompt_token_ids=[1, 0], prompt_logprobs=flat)
    actual = kl_eval._continuation_logprobs(output, prompt_tokens=1, vocab_size=4)
    torch.testing.assert_close(actual, torch.tensor([probabilities]).log())


@pytest.mark.parametrize("bad_ids", [(0, 1, 2), (0, 1, 2, 4), (0, 1, 2, "0")])
def test_incomplete_or_mismatched_vocabulary_is_rejected(kl_eval, bad_ids):
    output = SimpleNamespace(
        prompt_token_ids=[1, 0],
        prompt_logprobs=[
            None,
            dict.fromkeys(bad_ids, SimpleNamespace(logprob=math.log(0.25))),
        ],
    )
    with pytest.raises(ValueError):
        kl_eval._continuation_logprobs(output, prompt_tokens=1, vocab_size=4)


@pytest.mark.parametrize("invalid", [float("nan"), float("inf"), float("-inf")])
def test_invalid_scores_fail_instead_of_dropping_positions(kl_eval, invalid):
    reference = torch.full((2, 4), math.log(0.25))
    quantized = reference.clone()
    quantized[1, 2] = invalid
    with pytest.raises(ValueError):
        kl_eval.mean_kl(reference, quantized, top_k=2)
    row = _logprob_row([0.25] * 4)
    row[2].logprob = invalid
    output = SimpleNamespace(prompt_token_ids=[1, 0], prompt_logprobs=[None, row])
    with pytest.raises(ValueError):
        kl_eval._continuation_logprobs(output, prompt_tokens=1, vocab_size=4)


@pytest.mark.parametrize("missing", [None, [None], [None, None]])
def test_missing_continuation_scores_are_rejected(kl_eval, missing):
    output = SimpleNamespace(prompt_token_ids=[1, 0], prompt_logprobs=missing)
    with pytest.raises(ValueError):
        kl_eval._continuation_logprobs(output, prompt_tokens=1, vocab_size=4)


@pytest.mark.parametrize("detailed_results", [False, True])
def test_cli_checkpoint_inputs_and_defaults(kl_eval, monkeypatch, detailed_results):
    argv = ["kl_eval.py", "--model", "local-bf16", "--quantized_model", "local-nvfp4"]
    if detailed_results:
        argv.append("--detailed_results")
    monkeypatch.setattr(sys, "argv", argv)
    args = kl_eval._parse_args()
    assert args.model == "local-bf16"
    assert args.quantized_model == "local-nvfp4"
    assert (args.num_examples, args.prompt_tokens, args.max_new_tokens, args.top_k, args.seed) == (
        100,
        128,
        512,
        128,
        0,
    )
    assert args.detailed_results is detailed_results
    assert not hasattr(args, "recipe")
    assert not hasattr(args, "calib_size")


@pytest.mark.parametrize("detailed_results", [False, True])
def test_report_weights_examples_equally_and_preserves_output_shape(kl_eval, detailed_results):
    examples = [
        {"generated_tokens": 1, "full_vocab_kl": 2.0, "conditional_topk_kl": 1.0},
        {"generated_tokens": 3, "full_vocab_kl": 0.0, "conditional_topk_kl": 0.0},
    ]
    expected = {"full_vocab_kl": 1.0, "conditional_topk_kl": 0.5}
    actual = kl_eval._build_report(examples, detailed_results=detailed_results)
    if detailed_results:
        assert actual == {"summary": expected, "examples": examples}
    else:
        assert actual == expected


def test_missing_generation_config_preserves_model_eos_ids(kl_eval, monkeypatch, tmp_path):
    GPT2Config(vocab_size=4, eos_token_id=[0, 2]).save_pretrained(tmp_path)
    tokenizer = SimpleNamespace(
        eos_token_id=0,
        get_vocab=lambda: {"a": 0, "b": 1, "c": 2, "d": 3},
        encode=lambda text, add_special_tokens: [1, 3, 1],
    )
    dataset = Dataset.from_dict({"text": ["Local evaluation text."]})
    monkeypatch.setattr(kl_eval.AutoTokenizer, "from_pretrained", lambda *a, **kw: tokenizer)
    monkeypatch.setattr(kl_eval, "load_dataset", lambda *a, **kw: dataset)
    args = SimpleNamespace(
        model=str(tmp_path),
        quantized_model=str(tmp_path),
        revision=None,
        quantized_revision=None,
        trust_remote_code=False,
        num_examples=1,
        prompt_tokens=3,
        seed=0,
    )
    manifest = kl_eval._prepare(args)
    assert manifest["eos_token_ids"] == [0, 2]
    assert manifest["prompts"] == [{"block_index": 0, "input_ids": [1, 3, 1]}]
