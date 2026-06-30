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

"""GPU end-to-end tests for ``calc_runtime_for_subblocks`` via vLLM latency benchmarking."""

from __future__ import annotations

import math
from pathlib import Path  # noqa: TC003

import pytest
from _test_utils.torch.transformers_models import get_tiny_tokenizer
from omegaconf import OmegaConf

from modelopt.torch.puzzletron.block_config import AttentionConfig, FFNConfig, MambaConfig
from modelopt.torch.puzzletron.subblock_stats import calc_runtime_stats
from modelopt.torch.puzzletron.subblock_stats.calc_runtime_stats import calc_runtime_for_subblocks

_VOCAB_SIZE = 10016
_HIDDEN_SIZE = 256
_NUM_ATTENTION_HEADS = 4
_NUM_KV_HEADS = 2
_PREFILL_SEQ_LEN = 8
_GENERATION_SEQ_LEN = 4

_MAMBA_ATTN = AttentionConfig(
    mamba=MambaConfig(state_dim=16, num_heads=4, head_dim=16, num_groups=2)
)


@pytest.fixture(autouse=True)
def _clear_runtime_caches():
    """``calc_subblock_runtime`` and friends are ``@cache``d; clear between tests."""
    calc_runtime_stats.calc_subblock_runtime.cache_clear()
    calc_runtime_stats.calc_base_runtime.cache_clear()
    calc_runtime_stats.calc_no_block_runtime.cache_clear()
    yield None
    calc_runtime_stats.calc_subblock_runtime.cache_clear()
    calc_runtime_stats.calc_base_runtime.cache_clear()
    calc_runtime_stats.calc_no_block_runtime.cache_clear()


@pytest.mark.timeout(600)
@pytest.mark.skip(reason="AnyModel is not supported in vLLM yet")
def test_calc_runtime_for_subblocks(tmp_path: Path):
    """End-to-end: a tiny subblock set yields a runtime dict + positive no-block overhead."""
    tokenizer = get_tiny_tokenizer()
    tokenizer_dir = tmp_path / "tokenizer"
    tokenizer.save_pretrained(str(tokenizer_dir))

    attn = AttentionConfig(no_op=False, num_key_value_heads=2)
    ffn = FFNConfig(no_op=False, intermediate_size=256, moe=None)
    attn_noop = AttentionConfig(no_op=True)
    subblock_set = {attn, ffn, attn_noop}

    # vLLM's bench latency samples input ids in [0, 10000) (see
    # vllm/benchmarks/latency.py), and its input validator accepts an id when
    # it fits in max(tokenizer.max_token_id, model_vocab_size - 1). The tiny
    # tokenizer's vocab is ~200, so we size the model vocab past 10000 to
    # cover the sampled range.
    runtime_by_subblock, no_block_runtime_ms = calc_runtime_for_subblocks(
        subblock_config_set=subblock_set,
        runtime_stats_config=OmegaConf.create({"num_iters": 1, "num_warmup_iters": 1}),
        vocab_size=_VOCAB_SIZE,
        hidden_size=_HIDDEN_SIZE,
        num_attention_heads=_NUM_ATTENTION_HEADS,
        num_key_value_heads=_NUM_KV_HEADS,
        tokenizer_path=str(tokenizer_dir),
        prefill_seq_len=_PREFILL_SEQ_LEN,
        generation_seq_len=_GENERATION_SEQ_LEN,
        batch_size=1,
    )

    assert set(runtime_by_subblock) == subblock_set
    assert runtime_by_subblock[attn_noop] == 0.0
    assert math.isfinite(runtime_by_subblock[attn])
    assert math.isfinite(runtime_by_subblock[ffn])
    # The 1-block model is always slower than the per-block extrapolation from
    # the 10-block model, so the (embedding + LM-head) overhead is positive.
    assert no_block_runtime_ms > 0


@pytest.mark.timeout(600)
@pytest.mark.skip(reason="AnyModel is not supported in vLLM yet")
def test_calc_runtime_for_subblocks_mamba_end_to_end(tmp_path: Path):
    """End-to-end: Mamba subblock yields a finite per-layer runtime via vLLM."""
    try:
        from transformers import NemotronHConfig  # noqa: F401
    except ImportError:
        pytest.skip("NemotronHConfig requires a newer transformers build")

    pytest.importorskip("mamba_ssm")

    tokenizer = get_tiny_tokenizer()
    tokenizer_dir = tmp_path / "tokenizer"
    tokenizer.save_pretrained(str(tokenizer_dir))

    runtime_by_subblock, no_block_runtime_ms = calc_runtime_for_subblocks(
        subblock_config_set={_MAMBA_ATTN},
        runtime_stats_config=OmegaConf.create(
            {
                "num_iters": 1,
                "num_warmup_iters": 1,
                # mamba_block_size defaults to 16 for Nemotron-H benchmarks; override if needed.
            }
        ),
        vocab_size=_VOCAB_SIZE,
        hidden_size=_HIDDEN_SIZE,
        num_attention_heads=_NUM_ATTENTION_HEADS,
        num_key_value_heads=_NUM_KV_HEADS,
        tokenizer_path=str(tokenizer_dir),
        prefill_seq_len=_PREFILL_SEQ_LEN,
        generation_seq_len=_GENERATION_SEQ_LEN,
        batch_size=1,
    )

    assert set(runtime_by_subblock) == {_MAMBA_ATTN}
    assert math.isfinite(runtime_by_subblock[_MAMBA_ATTN])
    assert runtime_by_subblock[_MAMBA_ATTN] > 0
    assert no_block_runtime_ms > 0
