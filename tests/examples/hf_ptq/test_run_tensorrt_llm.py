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
"""Argument-forwarding tests for ``examples/hf_ptq/run_tensorrt_llm.py``.

``--trust_remote_code`` must reach the model load, not just the tokenizer: checkpoints
shipping custom modeling code (``auto_map``), e.g. Llama-3.3-Nemotron-Super-49B-v1
(DeciLM), otherwise fail executor init with "contains custom code which must be executed".
"""

import sys
from types import ModuleType, SimpleNamespace

import pytest
from _test_utils.examples.run_command import MODELOPT_ROOT

_HF_PTQ_DIR = MODELOPT_ROOT / "examples" / "hf_ptq"


class _RecordingLLM:
    """Stands in for the TRT-LLM wrapper, capturing how the script constructs it."""

    def __init__(self, *args, **kwargs):
        self.args = args
        self.kwargs = kwargs
        _RecordingLLM.last = self

    def generate_text(self, prompts, max_new_tokens):
        return ["" for _ in prompts]

    def generate_tokens(self, prompts, max_new_tokens):
        return [[0] for _ in prompts]

    def generate_context_logits(self, prompts):
        return [None for _ in prompts]


@pytest.fixture
def run_tensorrt_llm(monkeypatch):
    """Import the deploy script with its tensorrt_llm-dependent import stubbed out."""
    stub = ModuleType("modelopt.deploy.llm")
    stub.LLM = _RecordingLLM
    monkeypatch.setitem(sys.modules, "modelopt.deploy.llm", stub)
    monkeypatch.syspath_prepend(str(_HF_PTQ_DIR))
    monkeypatch.delitem(sys.modules, "run_tensorrt_llm", raising=False)

    import run_tensorrt_llm as module

    # run() reports GPU memory and toggles the profiler around generation; keep it CPU-only.
    monkeypatch.setattr(module.torch.cuda, "mem_get_info", lambda: (0, 0))
    monkeypatch.setattr(
        module.torch.cuda,
        "cudart",
        lambda: SimpleNamespace(cudaProfilerStart=lambda: None, cudaProfilerStop=lambda: None),
    )
    return module


@pytest.mark.parametrize("trust_remote_code", [True, False])
def test_run_forwards_trust_remote_code(run_tensorrt_llm, monkeypatch, trust_remote_code):
    tokenizer_kwargs = {}
    monkeypatch.setattr(
        run_tensorrt_llm,
        "get_tokenizer",
        lambda **kwargs: tokenizer_kwargs.update(kwargs) or object(),
    )

    run_tensorrt_llm.run(
        SimpleNamespace(
            tokenizer="",
            checkpoint_dir="/fake/checkpoint",
            max_output_len=8,
            input_texts="hello|world",
            trust_remote_code=trust_remote_code,
        )
    )

    # The model load, not just the tokenizer, must honor the flag.
    assert _RecordingLLM.last.kwargs["trust_remote_code"] is trust_remote_code
    assert tokenizer_kwargs["trust_remote_code"] is trust_remote_code
    # Context logits are requested below, so KV cache reuse must stay off.
    assert _RecordingLLM.last.kwargs["enable_kv_cache_reuse"] is False
