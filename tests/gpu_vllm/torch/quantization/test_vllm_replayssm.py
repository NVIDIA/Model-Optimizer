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

"""Tiny offline vLLM engine qualification of ReplaySSM cache allocation and reuse."""

import math
import os
import shutil
from pathlib import Path

import pytest
import torch
from packaging.version import Version
from transformers import Qwen3NextConfig
from vllm import LLM, SamplingParams
from vllm import __version__ as vllm_version

native = pytest.importorskip("emulation", reason="Requires pinned upstream ReplaySSM emulation")
ROOT = Path(__file__).parents[4]


def _audit(worker, install=False):
    # This optional adapter is only imported inside the vLLM 0.30 worker.
    from modelopt.torch.quantization.plugins.vllm_replayssm import _ReplaySSMGDN

    layers = [m for m in worker.model_runner.model.modules() if isinstance(m, _ReplaySSMGDN)]
    assert layers
    for layer in layers:
        cache = layer._replay_cache
        assert len(layer.kv_cache) == 9
        if install:
            layer._replay_counts = {"steps": 0, "resets": 0, "flushes": 0}
            original = cache.step
            references = {}

            def checked(
                qkv,
                a,
                b,
                rate,
                bias,
                indices,
                output,
                _cache=cache,
                _layer=layer,
                _original=original,
                _refs=references,
            ):
                expected = []
                for row, slot in enumerate(indices.tolist()):
                    if _cache.steps[slot] == 0:
                        _refs[slot] = native.GDNReplay(
                            _cache.anchor[slot : slot + 1].contiguous(), 2, "tp8_hi32"
                        )
                        _layer._replay_counts["resets"] += 1
                    _layer._replay_counts["flushes"] += int(_cache.steps[slot] % 16 == 15)
                    expected.append(
                        _refs[slot]
                        .step(
                            qkv[row : row + 1],
                            a[row : row + 1].float(),
                            b[row : row + 1].float(),
                            rate,
                            bias,
                        )[0]
                        .clone()
                    )
                _original(qkv, a, b, rate, bias, indices, output)
                torch.testing.assert_close(
                    output, torch.stack(expected).to(output.dtype), rtol=0, atol=0
                )
                for slot in indices.tolist():
                    torch.testing.assert_close(
                        _cache.anchor[slot], _refs[slot].state_hi[0], rtol=0, atol=0
                    )
                _layer._replay_counts["steps"] += len(indices)

            cache.step = checked
    return [layer._replay_counts for layer in layers]


@pytest.fixture
def replay_engine(tmp_path, monkeypatch):
    if Version(vllm_version).release[:2] != (0, 30):
        pytest.skip("ReplaySSM engine adapter requires vLLM 0.30.x")
    monkeypatch.setenv("VLLM_WORKER_MULTIPROC_METHOD", "spawn")
    monkeypatch.setenv("VLLM_GDN_DECODE_KERNEL", "triton")
    monkeypatch.setenv("RECIPE_PATH", "general/ptq/gdn_replayssm_fp8_read")
    paths = [str(ROOT), str(ROOT / "examples/vllm_serve"), str(Path(__file__).parent)]
    monkeypatch.setenv("PYTHONPATH", os.pathsep.join([*paths, os.environ.get("PYTHONPATH", "")]))
    for path in paths:
        monkeypatch.syspath_prepend(path)
    model = tmp_path / "gdn"
    Qwen3NextConfig(
        hidden_size=128,
        intermediate_size=256,
        num_attention_heads=4,
        head_dim=32,
        vocab_size=512,
        max_position_embeddings=128,
        num_hidden_layers=2,
        num_key_value_heads=2,
        linear_num_value_heads=4,
        linear_num_key_heads=2,
        linear_key_head_dim=128,
        linear_value_head_dim=128,
        num_experts=4,
        num_experts_per_tok=2,
        moe_intermediate_size=64,
        shared_expert_intermediate_size=64,
        mlp_only_layers=[],
        full_attention_interval=2,
        architectures=["Qwen3NextForCausalLM"],
    ).save_pretrained(model)
    shutil.copytree(ROOT / "tests/_test_utils/torch/tokenizer", model, dirs_exist_ok=True)
    llm = LLM(
        model=str(model),
        load_format="dummy",
        dtype="bfloat16",
        max_model_len=128,
        max_num_seqs=1,
        max_num_batched_tokens=64,
        enforce_eager=True,
        enable_prefix_caching=False,
        enable_chunked_prefill=True,
        async_scheduling=False,
        additional_config={"gdn_prefill_backend": "triton"},
        mamba_cache_dtype="float32",
        mamba_cache_mode="none",
        kv_cache_memory_bytes=128 * 1024**2,
        gpu_memory_utilization=0.1,
        worker_cls="replayssm_worker.ReplaySSMWorker",
        seed=17,
    )
    try:
        llm.collective_rpc(_audit, kwargs={"install": True})
        yield llm
    finally:
        llm.llm_engine.engine_core.shutdown()


def test_replayssm_engine(replay_engine):
    llm = replay_engine
    samples = [{"prompt_token_ids": [3] * length} for length in (73, 5, 65, 1)]
    params = SamplingParams(temperature=0, max_tokens=34, ignore_eos=True, logprobs=1)
    first = llm.generate(samples, params, use_tqdm=False)
    repeated = llm.generate(samples, params, use_tqdm=False)

    def values(outputs):
        return [(o.outputs[0].token_ids, o.outputs[0].cumulative_logprob) for o in outputs]

    assert values(first) == values(repeated)
    assert all(len(tokens) == 34 and math.isfinite(score) for tokens, score in values(first))
    counts = llm.collective_rpc(_audit)
    assert all(
        layer == {"steps": 264, "resets": 8, "flushes": 16} for rank in counts for layer in rank
    )
