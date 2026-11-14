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

import logging
import sys
import types

import pytest

try:
    import vllm.logger  # noqa: F401
except ModuleNotFoundError as exc:
    if exc.name != "vllm":
        raise
    vllm_module = types.ModuleType("vllm")
    vllm_module.__path__ = []
    vllm_logger_module = types.ModuleType("vllm.logger")
    vllm_logger_module.init_logger = logging.getLogger
    sys.modules["vllm"] = vllm_module
    sys.modules["vllm.logger"] = vllm_logger_module

from modelopt.torch.puzzletron.block_config import AttentionConfig
from modelopt.torch.puzzletron.subblock_stats.calc_runtime_stats import calc_subblock_runtime
from modelopt.torch.puzzletron.subblock_stats.runtime_utils import RuntimeConfig


@pytest.mark.parametrize(
    ("head_dim", "expected_head_dim"),
    [(24, 24), (None, 32)],
)
def test_calc_subblock_runtime_preserves_head_dim(monkeypatch, head_dim, expected_head_dim):
    """Use the model's explicit head dimension, falling back only when it is absent."""
    runtime_config = RuntimeConfig(
        vocab_size=32,
        hidden_size=128,
        num_attention_heads=4,
        num_key_value_heads=2,
        tokenizer_path="unused",
        repeat_block_n_times=1,
        prefill_seq_len=4,
        generation_seq_len=2,
        batch_size=1,
        num_iters=1,
        num_warmup_iters=0,
        head_dim=head_dim,
    )
    monkeypatch.setattr(
        "modelopt.torch.puzzletron.subblock_stats.calc_runtime_stats.calc_model_runtime",
        lambda model, _: model.model.layers[0].self_attn.q_proj.out_features,
    )
    calc_subblock_runtime.cache_clear()

    try:
        result = calc_subblock_runtime(
            runtime_config,
            AttentionConfig(no_op=False, num_key_value_heads=2),
        )
    finally:
        calc_subblock_runtime.cache_clear()

    assert result == 4 * expected_head_dim
