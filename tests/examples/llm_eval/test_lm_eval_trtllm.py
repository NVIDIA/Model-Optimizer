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

import importlib.machinery
import inspect
import sys
import types
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace

import pytest
import transformers

# Skip on the backend module, not just `lm_eval`: it currently guards its `tensorrt_llm`
# import, but if that ever becomes eager this should skip rather than error.
pytest.importorskip(
    "lm_eval.models.trtllm_causallms", reason="lm_eval is an examples/llm_eval requirement"
)

_LLM_EVAL_DIR = Path(__file__).resolve().parents[3] / "examples" / "llm_eval"
if str(_LLM_EVAL_DIR) not in sys.path:
    sys.path.insert(0, str(_LLM_EVAL_DIR))

import lm_eval_trtllm
from lm_eval.models.trtllm_causallms import TRTLLM
from lm_eval.utils import simple_parse_args_string

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


def test_trust_remote_code_reaches_the_backend(monkeypatch):
    """`--trust_remote_code` must land in model_args, since we call lm-eval's CLI directly.

    `huggingface_example.sh` passes the flag, and unlike `lm_eval_hf.py` this entry point
    does no rewriting of its own -- it relies on lm-eval doing it. On the live path that is
    `_cli/run.py:355` -> `EvaluatorConfig.from_cli` -> `_configure()` ->
    `_set_trust_remote_code()`.
    """
    import datasets
    from lm_eval.config.evaluate_config import EvaluatorConfig

    # raising=False: datasets 4.x dropped this attribute, and lm-eval creates it on
    # assignment. monkeypatch still removes it again on teardown.
    monkeypatch.setattr(datasets.config, "HF_DATASETS_TRUST_REMOTE_CODE", False, raising=False)

    cfg = EvaluatorConfig(
        model="trtllm", model_args={"model": "/ckpt", "tokenizer": "/tok"}, trust_remote_code=True
    )
    cfg._set_trust_remote_code()

    assert cfg.model_args["trust_remote_code"] is True
    assert datasets.config.HF_DATASETS_TRUST_REMOTE_CODE is True


def test_trust_remote_code_not_injected_when_unset(monkeypatch):
    import datasets
    from lm_eval.config.evaluate_config import EvaluatorConfig

    # raising=False: datasets 4.x dropped this attribute, and lm-eval creates it on
    # assignment. monkeypatch still removes it again on teardown.
    monkeypatch.setattr(datasets.config, "HF_DATASETS_TRUST_REMOTE_CODE", False, raising=False)

    cfg = EvaluatorConfig(model="trtllm", model_args={"model": "/ckpt"}, trust_remote_code=False)
    cfg._set_trust_remote_code()

    assert "trust_remote_code" not in cfg.model_args


def test_trtllm_backend_accepts_trust_remote_code():
    """The key lm-eval injects has to be a parameter the backend actually takes.

    Inspected on the upstream ``__init__``: ``TRTLLM.__init__`` is the wrapper below, which
    takes ``**kwargs`` and forwards them, so its own signature proves nothing.
    """
    assert "trust_remote_code" in inspect.signature(lm_eval_trtllm._UPSTREAM_INIT).parameters


def _fake_trtllm(monkeypatch, version):
    module = types.ModuleType("tensorrt_llm")
    module.__version__ = version
    monkeypatch.setitem(sys.modules, "tensorrt_llm", module)


@pytest.mark.parametrize("version", ["1.1.0rc2", "1.2.0", "1.3.0rc10"])
def test_rejects_trtllm_without_the_prompt_token_ids_fix(monkeypatch, version):
    """<1.3.0rc11 returns only the top-1 token per position, so continuations are absent."""
    _fake_trtllm(monkeypatch, version)
    with pytest.raises(RuntimeError, match=r"1\.3\.0rc11"):
        lm_eval_trtllm._check_trtllm_version()


@pytest.mark.parametrize("version", ["1.3.0rc11", "1.3.0rc23", "1.3.0", "1.4.0"])
def test_accepts_trtllm_with_the_prompt_token_ids_fix(monkeypatch, version):
    _fake_trtllm(monkeypatch, version)
    lm_eval_trtllm._check_trtllm_version()


def test_version_is_checked_before_scoring(monkeypatch):
    """The guard has to fire from _parse_logprobs, not only from __main__."""
    _fake_trtllm(monkeypatch, "1.2.0")
    monkeypatch.setattr(lm_eval_trtllm, "_trtllm_version_checked", False)
    with pytest.raises(RuntimeError, match=r"1\.3\.0rc11"):
        lm_eval_trtllm._parse_logprobs(
            tokens=TOKENS, outputs=_Outputs(_prompt_logprobs()), ctxlen=CTXLEN
        )


class _RecordingKvCacheConfig:
    """Stand-in for ``tensorrt_llm.llmapi.KvCacheConfig``, capturing its kwargs."""

    def __init__(self, **kwargs):
        self.kwargs = kwargs


@pytest.fixture
def stub_engine(monkeypatch):
    """Make the upstream ``__init__`` runnable without tensorrt_llm or a real tokenizer.

    Everything between the tokenizer load and the ``LLM(...)`` call is upstream code, so
    this exercises the real construction rather than a stand-in for it.
    """
    # The backend gates on find_spec(), which reads __spec__ off an already-imported module.
    trtllm = types.ModuleType("tensorrt_llm")
    trtllm.__spec__ = importlib.machinery.ModuleSpec("tensorrt_llm", loader=None)
    monkeypatch.setitem(sys.modules, "tensorrt_llm", trtllm)

    class _LLM:
        last_kwargs: dict = {}

        def __init__(self, **kwargs):
            _LLM.last_kwargs = kwargs

    monkeypatch.setattr(lm_eval_trtllm.trtllm_causallms, "LLM", _LLM, raising=False)
    monkeypatch.setattr(
        lm_eval_trtllm.trtllm_causallms, "KvCacheConfig", _RecordingKvCacheConfig, raising=False
    )

    class _Tokenizer:
        pad_token_id = 0
        eos_token_id = 0
        eos_token = "</s>"

    monkeypatch.setattr(
        transformers, "AutoTokenizer", SimpleNamespace(from_pretrained=lambda *a, **k: _Tokenizer())
    )
    return _LLM


def _build(**model_args):
    """Instantiate the backend the way lm-eval does from a parsed `--model_args` string."""
    # add_bos_token is pinned only to skip the tokenizer round-trip that auto-detects it.
    return TRTLLM.create_from_arg_obj({"model": "/ckpt", "add_bos_token": False, **model_args})


def test_kv_cache_fraction_reaches_the_engine(stub_engine):
    """`--model_args kv_cache_free_gpu_memory_fraction=...` must reach `KvCacheConfig`.

    Upstream drops unknown `--model_args` keys silently, so without the patch the KV cache
    takes TensorRT-LLM's default 90% of free memory and large-memory GPUs OOM (nvbug
    6701763).
    """
    _build(kv_cache_free_gpu_memory_fraction=0.5)

    kv_cache_config = stub_engine.last_kwargs["kv_cache_config"]
    assert kv_cache_config.kwargs == {"enable_block_reuse": False, "free_gpu_memory_fraction": 0.5}


def test_kv_cache_fraction_is_not_injected_when_unset(stub_engine):
    """Unset, TensorRT-LLM's own default must still apply -- we add no second default."""
    _build()

    assert "free_gpu_memory_fraction" not in stub_engine.last_kwargs["kv_cache_config"].kwargs


def test_kv_cache_fraction_is_parsed_from_a_model_args_string():
    """lm-eval coerces `--model_args` values itself; the key has to arrive as a float."""
    parsed = simple_parse_args_string("model=/ckpt,kv_cache_free_gpu_memory_fraction=0.5")

    assert parsed["kv_cache_free_gpu_memory_fraction"] == 0.5


def test_kv_cache_fraction_patch_is_reverted(monkeypatch):
    """Nothing may leak past the constructor; the patch is process-wide while held."""
    monkeypatch.setattr(
        lm_eval_trtllm.trtllm_causallms, "KvCacheConfig", _RecordingKvCacheConfig, raising=False
    )

    with pytest.raises(RuntimeError), lm_eval_trtllm._kv_cache_fraction_applied(0.5):
        raise RuntimeError("model load failed")

    assert lm_eval_trtllm.trtllm_causallms.KvCacheConfig is _RecordingKvCacheConfig


def test_kv_cache_fraction_without_tensorrt_llm(monkeypatch):
    """Without tensorrt_llm the backend raises its own error; don't pre-empt it here."""
    monkeypatch.delattr(lm_eval_trtllm.trtllm_causallms, "KvCacheConfig", raising=False)

    with lm_eval_trtllm._kv_cache_fraction_applied(0.5):
        pass


@pytest.mark.parametrize("fraction", [0.0, -0.1, 1.5])
def test_init_rejects_an_out_of_range_fraction(fraction):
    """Out of range, TensorRT-LLM either allocates nothing or fails deep in the engine."""
    with pytest.raises(ValueError, match="must be in"):
        lm_eval_trtllm._init(object(), model="/ckpt", kv_cache_free_gpu_memory_fraction=fraction)


def test_init_patch_is_installed():
    assert TRTLLM.__init__ is lm_eval_trtllm._init


def test_upstream_still_drops_the_kv_cache_fraction():
    """Tripwire: when upstream takes the argument itself, drop the `__init__` patch."""
    assert lm_eval_trtllm._UPSTREAM_TAKES_KV_CACHE_FRACTION is False

    source = inspect.getsource(lm_eval_trtllm._UPSTREAM_INIT)
    # It accepts **kwargs, and then builds the LLM kwargs from a fixed set of names.
    assert "free_gpu_memory_fraction" not in source


def test_upstream_is_still_misaligned():
    """Tripwire: when this fails, upstream fixed the bug and this file can be deleted.

    lm-eval's own ``_parse_logprobs`` reads ``prompt_logprobs[i][tokens[i]]``, but entry i
    holds ``tokens[i + 1]``, so it raises ``KeyError`` on the very first prompt token.
    """
    with pytest.raises(KeyError):
        lm_eval_trtllm._UPSTREAM_PARSE_LOGPROBS(
            tokens=TOKENS, outputs=_Outputs(_prompt_logprobs()), ctxlen=CTXLEN
        )
