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

"""Run lm-evaluation-harness against a TensorRT-LLM checkpoint.

Entry point around lm-eval's built-in ``trtllm`` backend
(``lm_eval.models.trtllm_causallms``, new in 0.4.12). It exists only to correct that
backend's ``prompt_logprobs`` handling -- everything else is upstream. Drop this file and
call ``lm_eval`` directly once the fix lands upstream.

    python lm_eval_trtllm.py --model trtllm \
        --model_args model=<quantized checkpoint dir>,tokenizer=<HF model folder>,\
tensor_parallel_size=<tp>,max_batch_size=<max batch size>,max_input_len=4096 \
        --tasks <comma separated tasks> --batch_size <max batch size>
"""

import sys
from importlib.metadata import version

from lm_eval.__main__ import cli_evaluate
from packaging.version import Version

if Version(version("lm_eval")) < Version("0.4.12"):
    # 0.4.12 is the first release shipping lm_eval.models.trtllm_causallms.
    raise ImportError(f"lm_eval_trtllm.py requires lm-eval >= 0.4.12; found {version('lm_eval')}.")

from lm_eval.models.trtllm_causallms import TRTLLM

# TensorRT-LLM only started passing the prompt token ids into `compute_logprobs` in
# 1.3.0rc11 (`executor/base_worker.py`), which is what makes the requested token always
# present in each `prompt_logprobs` entry. On 1.2.0 and earlier, `prompt_logprobs=1` keeps
# only the top-1 token, so a non-greedy continuation token is simply absent and no correct
# continuation logprob can be recovered -- by this file or by lm-eval's own version.
_MIN_TRTLLM_VERSION = "1.3.0rc11"
_trtllm_version_checked = False


def _check_trtllm_version() -> None:
    """Raise if TensorRT-LLM predates the `prompt_logprobs` layout scored below."""
    try:
        import tensorrt_llm
    except ImportError:
        # Nothing to check, and unreachable in a real run: the backend refuses to build a
        # model without tensorrt_llm long before any logprob is scored.
        return

    if Version(tensorrt_llm.__version__) < Version(_MIN_TRTLLM_VERSION):
        raise RuntimeError(
            f"Loglikelihood tasks need TensorRT-LLM >= {_MIN_TRTLLM_VERSION}; found "
            f"{tensorrt_llm.__version__}. Earlier releases return only the top-1 token per "
            "prompt position, so the continuation token's logprob is unavailable. Use a "
            "newer TensorRT-LLM container, or restrict the run to generative tasks."
        )


def _parse_logprobs(tokens: list[int], outputs, ctxlen: int) -> tuple[float, bool]:
    """Sum the continuation logprobs of one request, correcting upstream's alignment.

    TensorRT-LLM aligns ``prompt_logprobs`` to the *next* token: its worker computes them
    from ``prompt_token_ids[1:] + first_generated_token`` (``executor/base_worker.py``), so
    entry ``i`` is the distribution that predicted ``tokens[i + 1]`` and always contains
    that token's id -- either in the top-k or appended by ``_topk_logprobs``.

    lm-eval 0.4.12's ``TRTLLM._parse_logprobs`` instead reads
    ``prompt_logprobs[i][tokens[i]]`` and applies its own shift on top, which raises
    ``KeyError`` on the first request of every loglikelihood task (hellaswag, mmlu, arc).
    """
    global _trtllm_version_checked
    if not _trtllm_version_checked:
        # Checked here rather than at startup so generative-only runs, which never reach
        # this path, still work on older TensorRT-LLM releases.
        _check_trtllm_version()
        _trtllm_version_checked = True

    prompt_logprobs = outputs.outputs[0].prompt_logprobs
    # Scoring tokens[ctxlen:] reads entries ctxlen-1 .. len(tokens)-2; a shorter list means
    # the engine saw a different prompt than we asked about, which would shift every index.
    if len(prompt_logprobs) < len(tokens) - 1:
        raise RuntimeError(
            f"prompt_logprobs has {len(prompt_logprobs)} entries for {len(tokens)} tokens; "
            "the engine scored a different prompt than was requested."
        )

    continuation_logprobs = 0.0
    is_greedy = True
    # Token 0 has no preceding distribution, so it can never be scored.
    for i in range(max(ctxlen, 1), len(tokens)):
        logprob = prompt_logprobs[i - 1].get(tokens[i])
        if logprob is None:
            # Dropping the term instead would silently inflate the reported accuracy.
            raise RuntimeError(
                f"tokens[{i}] is missing from prompt_logprobs[{i - 1}]; the returned "
                "logprobs are misaligned with the requested tokens."
            )
        continuation_logprobs += logprob.logprob
        if logprob.rank != 1:
            is_greedy = False

    return continuation_logprobs, is_greedy


if not hasattr(TRTLLM, "_parse_logprobs"):
    raise RuntimeError(
        "lm_eval.models.trtllm_causallms.TRTLLM has no _parse_logprobs to override; the "
        f"backend changed shape in lm-eval {version('lm_eval')}. Recheck whether this file "
        "is still needed."
    )

# Kept so the unit tests can assert the upstream implementation is still the broken one.
# When that assertion starts failing, upstream has fixed the alignment and this whole file
# should be deleted in favour of calling `lm_eval` directly.
_UPSTREAM_PARSE_LOGPROBS = TRTLLM._parse_logprobs
TRTLLM._parse_logprobs = staticmethod(_parse_logprobs)


if __name__ == "__main__":
    # Warn up front so an unusable container is obvious before the model loads, but do not
    # abort: generative tasks are unaffected by the old prompt_logprobs layout.
    try:
        _check_trtllm_version()
    except RuntimeError as e:
        print(f"WARNING: {e}", file=sys.stderr)

    cli_evaluate()
