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

"""GPU test for ``calc_runtime_for_subblocks``.

Exercises the end-to-end vLLM latency benchmarking pipeline on a tiny model:
constructs a small subblock set, runs the benchmark for each candidate, and
checks the returned per-subblock runtime dict and no-block overhead.
"""

import math
from pathlib import Path

import pytest
from _test_utils.torch.transformers_models import get_tiny_tokenizer
from omegaconf import OmegaConf

from modelopt.torch.puzzletron.block_config import AttentionConfig, FFNConfig, MoEConfig
from modelopt.torch.puzzletron.subblock_stats.calc_runtime_stats import (
    _hybrid_override_pattern_for_block_configs,
    _make_moe_ffn_baseline_block_config,
    _make_moe_ffn_block_config,
    _uses_moe,
    calc_runtime_for_subblocks,
)


def test_moe_ffn_runtime_path_preserves_explicit_moe_block():
    """MoE expert-removal candidates should benchmark as real MoE layers."""
    pruned_ffn = FFNConfig(
        moe=MoEConfig(
            num_local_experts=16,
            num_experts_per_tok=4,
            expert_intermediate_dim=128,
            shared_expert_intermediate_dim=256,
        ),
    )

    target_block = _make_moe_ffn_block_config(pruned_ffn)
    baseline_block = _make_moe_ffn_baseline_block_config()

    assert _uses_moe(pruned_ffn)
    assert target_block.attention.no_op
    assert target_block.ffn.moe.num_local_experts == 16
    assert baseline_block.attention.no_op and baseline_block.ffn.no_op
    assert _hybrid_override_pattern_for_block_configs([target_block]) == "E"


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
        vocab_size=10016,
        hidden_size=256,
        num_attention_heads=4,
        num_key_value_heads=2,
        tokenizer_path=str(tokenizer_dir),
        prefill_seq_len=8,
        generation_seq_len=4,
        batch_size=1,
    )

    assert set(runtime_by_subblock) == subblock_set
    assert runtime_by_subblock[attn_noop] == 0.0
    assert math.isfinite(runtime_by_subblock[attn])
    assert math.isfinite(runtime_by_subblock[ffn])
    # The 1-block model is always slower than the per-block extrapolation from
    # the 10-block model, so the (embedding + LM-head) overhead is positive.
    assert no_block_runtime_ms > 0
