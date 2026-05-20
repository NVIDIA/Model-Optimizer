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

from functools import partial
from pathlib import Path

import pytest
from _test_utils.torch.distributed.utils import spawn_multiprocess_job
from _test_utils.torch.transformers_models import get_tiny_tokenizer

pytest.importorskip("vllm", reason="vLLM is required for calc_runtime_for_subblocks")


def test_calc_runtime_for_subblocks(tmp_path: Path):
    """End-to-end: a tiny subblock set yields a runtime dict + positive no-block overhead."""
    spawn_multiprocess_job(size=1, job=partial(_run, tmp_path), backend="nccl")


def _run(tmp_path: Path, rank: int, size: int):
    import math

    from omegaconf import OmegaConf

    from modelopt.torch.nas.subblock_stats import calc_runtime_for_subblocks
    from modelopt.torch.puzzletron.block_config import AttentionConfig, FFNConfig

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
    )

    assert set(runtime_by_subblock) == subblock_set
    assert runtime_by_subblock[attn_noop] == 0.0
    assert math.isfinite(runtime_by_subblock[attn])
    assert math.isfinite(runtime_by_subblock[ffn])
    # The 1-block model is always slower than the per-block extrapolation from
    # the 10-block model, so the (embedding + LM-head) overhead is positive.
    assert no_block_runtime_ms > 0
