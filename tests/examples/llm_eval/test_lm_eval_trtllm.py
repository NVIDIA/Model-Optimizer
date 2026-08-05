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
"""Unit tests for ``examples/llm_eval/lm_eval_trtllm.py``.

The module lives next to the example script (not inside the ``modelopt`` package),
so we add ``examples/llm_eval/`` to ``sys.path`` before importing it. No GPU and no
TensorRT-LLM install is needed: ``_parse_logprobs`` is pure Python over a response
object, which these tests stub out.
"""

import sys
from dataclasses import dataclass
from pathlib import Path

import pytest

pytest.importorskip("lm_eval", reason="lm_eval is an examples/llm_eval requirement")

_LLM_EVAL_DIR = Path(__file__).resolve().parents[3] / "examples" / "llm_eval"
if str(_LLM_EVAL_DIR) not in sys.path:
    sys.path.insert(0, str(_LLM_EVAL_DIR))

import lm_eval_trtllm

# TensorRT-LLM aligns prompt_logprobs to the *next* token, so entry i holds the
# distribution that predicted tokens[i + 1]. These fixtures mirror that layout.
TOKENS = [10, 11, 12, 13, 14]
CTXLEN = 3  # continuation is tokens[3:] == [13, 14]


@dataclass
class _Logprob:
    """Stand-in for ``tensorrt_llm.executor.result.Logprob``."""

    logprob: float
    rank: int = 1


class _Outputs:
    """Stand-in for a TensorRT-LLM ``RequestOutput``."""

    def __init__(self, prompt_logprobs):
        self.outputs = [type("_Completion", (), {"prompt_logprobs": prompt_logprobs})()]


def _prompt_logprobs(continuation_ranks=(1, 1)):
    """One dict per prompt token; entry i contains tokens[i + 1], as TRT-LLM returns."""
    return [
        {TOKENS[1]: _Logprob(-2.0)},  # predicts tokens[1] -- context, not scored
        {TOKENS[2]: _Logprob(-3.0)},  # predicts tokens[2] -- context, not scored
        {TOKENS[3]: _Logprob(-0.5, continuation_ranks[0])},  # predicts tokens[3] -- scored
        {TOKENS[4]: _Logprob(-1.25, continuation_ranks[1])},  # predicts tokens[4] -- scored
        {999: _Logprob(-9.0)},  # predicts the first generated token -- unused
    ]


def test_sums_only_continuation_tokens():
    """Only tokens[ctxlen:] contribute, read from the entry one position earlier."""
    logprob, is_greedy = lm_eval_trtllm._parse_logprobs(
        tokens=TOKENS, outputs=_Outputs(_prompt_logprobs()), ctxlen=CTXLEN
    )
    assert logprob == pytest.approx(-0.5 + -1.25)
    assert is_greedy is True


def test_is_greedy_false_when_a_continuation_token_is_not_rank_one():
    logprob, is_greedy = lm_eval_trtllm._parse_logprobs(
        tokens=TOKENS,
        outputs=_Outputs(_prompt_logprobs(continuation_ranks=(1, 2))),
        ctxlen=CTXLEN,
    )
    assert logprob == pytest.approx(-0.5 + -1.25)
    assert is_greedy is False


def test_ctxlen_zero_skips_the_unscorable_first_token():
    """tokens[0] has no preceding distribution, so scoring starts at tokens[1]."""
    logprob, _ = lm_eval_trtllm._parse_logprobs(
        tokens=TOKENS, outputs=_Outputs(_prompt_logprobs()), ctxlen=0
    )
    assert logprob == pytest.approx(-2.0 + -3.0 + -0.5 + -1.25)


def test_raises_when_prompt_logprobs_is_too_short():
    """A short list means the engine scored a different prompt; every index would shift."""
    with pytest.raises(RuntimeError, match="entries for 5 tokens"):
        lm_eval_trtllm._parse_logprobs(
            tokens=TOKENS, outputs=_Outputs(_prompt_logprobs()[:3]), ctxlen=CTXLEN
        )


def test_raises_when_a_continuation_token_is_missing():
    """Dropping the term instead would silently inflate the reported accuracy."""
    entries = _prompt_logprobs()
    entries[2] = {777: _Logprob(-0.5)}  # tokens[3] absent from the entry that predicts it
    with pytest.raises(RuntimeError, match=r"tokens\[3\] is missing"):
        lm_eval_trtllm._parse_logprobs(tokens=TOKENS, outputs=_Outputs(entries), ctxlen=CTXLEN)


def test_upstream_is_still_misaligned():
    """Tripwire: when this fails, upstream fixed the bug and this file can be deleted.

    lm-eval's own ``_parse_logprobs`` reads ``prompt_logprobs[i][tokens[i]]``, but entry i
    holds ``tokens[i + 1]``, so it raises ``KeyError`` on the very first prompt token.
    """
    with pytest.raises(KeyError):
        lm_eval_trtllm._UPSTREAM_PARSE_LOGPROBS(
            tokens=TOKENS, outputs=_Outputs(_prompt_logprobs()), ctxlen=CTXLEN
        )
