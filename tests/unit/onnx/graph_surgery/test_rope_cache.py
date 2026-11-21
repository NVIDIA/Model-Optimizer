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

"""RoPE cos/sin caches for GQA graph surgery."""

import json

import numpy as np
import pytest
import torch
from packaging.version import Version

transformers = pytest.importorskip(
    "transformers", reason="transformers required for RoPE cache tests"
)

from transformers import AutoConfig
from transformers.models.llama.modeling_llama import LlamaRotaryEmbedding

from modelopt.onnx.graph_surgery.utils.rope_cache import get_rope_caches

HEAD_DIM = 64
MAX_SEQ_LEN = 512
LLAMA3_SCALING = {
    "rope_type": "llama3",
    "factor": 32.0,
    "low_freq_factor": 1.0,
    "high_freq_factor": 4.0,
    "original_max_position_embeddings": 64,
}
DYNAMIC_SCALING = {"rope_type": "dynamic", "factor": 2.0}
YARN_SCALING = {"rope_type": "yarn", "factor": 4.0, "original_max_position_embeddings": 1024}
# Phi-3 and Phi-4-mini keep original_max_position_embeddings at the top level of config.json.
LONGROPE_SCALING = {
    "rope_type": "longrope",
    "short_factor": [1.0] * (HEAD_DIM // 2),
    "long_factor": [2.0] * (HEAD_DIM // 2),
}
TRANSFORMERS_4 = Version(transformers.__version__) < Version("5")


def _llama_config(rope_scaling, max_position_embeddings=4096, **top_level):
    """A checkpoint-style config, with rope_theta at the top level."""
    return {
        "model_type": "llama",
        "hidden_size": 4 * HEAD_DIM,
        "num_attention_heads": 4,
        "num_key_value_heads": 2,
        "num_hidden_layers": 1,
        "intermediate_size": 512,
        "vocab_size": 64,
        "max_position_embeddings": max_position_embeddings,
        "rope_theta": 500000.0,
        "rope_scaling": rope_scaling,
        **top_level,
    }


def _write_config(path, config):
    (path / "config.json").write_text(json.dumps(config))
    return str(path)


@pytest.mark.parametrize(
    "config",
    [
        pytest.param(_llama_config(None), id="default"),
        pytest.param(_llama_config(LLAMA3_SCALING), id="llama3"),
        pytest.param(_llama_config(DYNAMIC_SCALING), id="dynamic"),
        pytest.param(_llama_config(YARN_SCALING), id="yarn"),
        pytest.param(
            _llama_config(LONGROPE_SCALING, original_max_position_embeddings=MAX_SEQ_LEN),
            id="longrope-at-limit",
        ),
        pytest.param(
            _llama_config(YARN_SCALING, partial_rotary_factor=0.75), id="yarn-partial-rotary"
        ),
    ],
)
def test_rope_caches_match_hf_rotary_embedding(tmp_path, config):
    """Caches equal HF's rotary embedding, attention scaling included (dynamic, longrope: up to their limit)."""
    model_dir = _write_config(tmp_path, config)
    half_rotary_dim = int(HEAD_DIM * config.get("partial_rotary_factor", 1.0)) // 2

    cos, sin, _ = get_rope_caches(model_dir, MAX_SEQ_LEN, io_dtype="float32")

    rotary = LlamaRotaryEmbedding(config=AutoConfig.from_pretrained(model_dir))
    ref_cos, ref_sin = rotary(
        torch.zeros(1, MAX_SEQ_LEN, HEAD_DIM), torch.arange(MAX_SEQ_LEN)[None]
    )
    np.testing.assert_allclose(cos, ref_cos[0, :, :half_rotary_dim].numpy(), atol=1e-5)
    np.testing.assert_allclose(sin, ref_sin[0, :, :half_rotary_dim].numpy(), atol=1e-5)


@pytest.mark.parametrize(
    ("config", "match"),
    [
        pytest.param(
            _llama_config(DYNAMIC_SCALING, max_position_embeddings=256),
            "max_seq_len",
            id="dynamic-past-limit",
        ),
        pytest.param(
            _llama_config(LONGROPE_SCALING, original_max_position_embeddings=256),
            "max_seq_len",
            id="longrope-past-limit",
        ),
        pytest.param(
            # transformers 4 ignores this nested value and switches at max_position_embeddings.
            _llama_config(
                {**LONGROPE_SCALING, "factor": 16.0, "original_max_position_embeddings": 1024},
                max_position_embeddings=256,
            ),
            "max_seq_len",
            id="longrope-nested-limit-past-max-positions",
        ),
        pytest.param(
            _llama_config({"rope_type": "made_up", "factor": 2.0}),
            "rope_type",
            id="unknown-rope-type",
        ),
        pytest.param(
            {"model_type": "gemma3_text"},
            "Per-layer-type",
            id="per-layer-type",
            marks=pytest.mark.skipif(
                TRANSFORMERS_4, reason="per-layer RoPE parameters are a transformers 5 format"
            ),
        ),
    ],
)
def test_rope_caches_refuse_rope_one_static_cache_cannot_represent(tmp_path, config, match):
    """Such RoPE configs raise instead of getting wrong caches."""
    model_dir = _write_config(tmp_path, config)

    with pytest.raises(ValueError, match=match):
        get_rope_caches(model_dir, MAX_SEQ_LEN, io_dtype="float32")
