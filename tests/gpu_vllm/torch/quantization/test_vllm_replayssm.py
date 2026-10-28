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
from types import SimpleNamespace

import pytest
import torch
import yaml
from packaging.version import Version
from transformers import Qwen3NextConfig
from vllm import LLM, SamplingParams
from vllm import __version__ as vllm_version

ROOT = Path(__file__).parents[4]


@pytest.mark.parametrize(
    ("num_prefills", "progress", "expected"),
    [
        (1, (96, 64), True),  # Recompute includes generated tokens past the original prompt.
        (0, (63, 64), True),  # One-token prompt tail is labeled decode by GDN metadata.
        (0, (64, 64), False),
        (0, None, False),  # CUDA-graph capture warmup.
    ],
)
def test_prompt_phase_routing(num_prefills, progress, expected):
    from modelopt.torch.quantization.plugins.vllm_replayssm import _is_prompt_phase

    metadata = SimpleNamespace(num_prefills=num_prefills)
    assert _is_prompt_phase(metadata, progress) is expected


def _fp8_read_copy(state):
    """Independent E4M3 block-K codec for canonical [H,K,V] states."""
    value_first = state.transpose(-1, -2)
    scale = value_first.abs().amax(-1, keepdim=True) / 448
    scale = torch.where(scale > 0, scale, 1)
    read = (value_first / scale).to(torch.float8_e4m3fn).float() * scale
    return read.transpose(-1, -2)


def _torch_fp8_replay(q, k, v, g, beta, initial):
    """Dense Torch oracle independent of the ModelOpt Triton replay backend."""

    def rounded(value, dtype=torch.bfloat16):
        return value.to(dtype).float()

    q = q / (q.square().sum(-1, keepdim=True) + 1e-6).sqrt() * q.shape[-1] ** -0.5
    k = k / (k.square().sum(-1, keepdim=True) + 1e-6).sqrt()
    beta = rounded(beta)
    state = initial
    outputs = []
    for start in range(0, len(q), 16):
        stop = min(start + 16, len(q))
        local_g = g[start:stop].cumsum(0)
        keys = k[start:stop].transpose(0, 1)
        read = _fp8_read_copy(state)
        updates = []
        for i in range(stop - start):
            t = start + i
            decay = local_g[i].exp()[:, None, None]
            weights = (local_g[i] - local_g[:i]).exp().T
            base = rounded(state if t == 0 or i == 15 else read)
            working = decay * base
            if i == 15:
                all_keys = rounded(keys)
                gram = all_keys @ all_keys.transpose(-1, -2)
                gates = local_g.T
                retentions = (gates[:, :, None] - gates[:, None, :]).exp()
                lower = torch.tril(gram * retentions * beta[start:stop].T[:, :, None], diagonal=-1)
                lower = lower + torch.eye(16, device=q.device)
                residual = rounded(
                    v[start:stop].transpose(0, 1)
                    - gates.exp()[..., None] * (all_keys @ rounded(state))
                )
                solved = torch.linalg.solve_triangular(
                    lower, beta[start:stop].T[..., None] * residual, upper=False
                )
                weighted = solved * (gates[:, -1:] - gates).exp()[..., None]
                output = torch.einsum("hk,hkv->hv", q[t], working)
                output += torch.einsum("hlk,hk,hlv->hv", keys, q[t], weighted)
                state = decay * state + all_keys.transpose(-1, -2) @ rounded(weighted)
            else:
                if i:
                    working += keys[:, :i].transpose(-1, -2) @ (
                        torch.stack(updates, 1) * weights[..., None]
                    )
                delta = beta[t, :, None] * (v[t] - torch.einsum("hk,hkv->hv", k[t], working))
                working += k[t, :, :, None] * delta[:, None, :]
                output = torch.einsum("hk,hkv->hv", q[t], working)
                updates.append(rounded(delta, torch.float16))
            outputs.append(output)
    return torch.stack(outputs)


def _audit(worker, install=False, compiled=False):
    # This optional adapter is only imported inside the vLLM 0.30 worker.
    from modelopt.torch.quantization.plugins.vllm_replayssm import _ReplaySSMGDN

    layers = [m for m in worker.model_runner.model.modules() if isinstance(m, _ReplaySSMGDN)]
    assert layers
    for layer in layers:
        cache = layer._replay_cache
        assert len(layer.kv_cache) == (10 if cache.read_mode == "tp4_hi32" else 9)
        if compiled:
            if install:
                original_reset = cache.reset

                def reset(indices, _layer=layer, _reset=original_reset):
                    _layer._replay_active_indices = indices.clone()
                    _reset(indices)

                cache.reset = reset
            continue
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
                        if _cache.read_mode != "tp8_hi32":
                            raise AssertionError(
                                "The independent eager oracle currently covers FP8"
                            )
                        _refs[slot] = {
                            "initial": _cache.anchor[slot].transpose(-1, -2).clone(),
                            "q": [],
                            "k": [],
                            "v": [],
                            "g": [],
                            "beta": [],
                        }
                        _layer._replay_counts["resets"] += 1
                    _layer._replay_counts["flushes"] += int(_cache.steps[slot] % 16 == 15)
                    reference = _refs[slot]
                    packed = qkv[row].float()
                    key_heads = _cache.key_heads
                    heads, values = _cache.anchor.shape[1:3]
                    q = packed[: key_heads * 128].view(key_heads, 128)
                    k = packed[key_heads * 128 : 2 * key_heads * 128].view(key_heads, 128)
                    repeat = heads // key_heads
                    reference["q"].append(q.repeat_interleave(repeat, 0))
                    reference["k"].append(k.repeat_interleave(repeat, 0))
                    reference["v"].append(packed[2 * key_heads * 128 :].view(heads, values))
                    reference["g"].append(
                        -rate.exp() * torch.nn.functional.softplus(a[row].float() + bias)
                    )
                    reference["beta"].append(b[row].float().sigmoid().bfloat16().float())
                    expected.append(
                        _torch_fp8_replay(
                            *(
                                torch.stack(reference[name])
                                for name in ("q", "k", "v", "g", "beta")
                            ),
                            reference["initial"],
                        )[-1]
                    )
                _original(qkv, a, b, rate, bias, indices, output)
                torch.testing.assert_close(
                    output.float(), torch.stack(expected), rtol=3e-3, atol=3e-3
                )
                for slot in indices.tolist():
                    assert torch.isfinite(_cache.anchor[slot]).all()
                _layer._replay_counts["steps"] += len(indices)

            cache.step = checked
    if compiled:
        return (
            []
            if install
            else [
                layer._replay_cache.steps.index_select(0, layer._replay_active_indices).tolist()
                for layer in layers
            ]
        )
    return [layer._replay_counts for layer in layers]


def _graph_replays(worker, start=False):
    if start:
        worker._replay_profiler = torch.profiler.profile(
            activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA]
        )
        worker._replay_profiler.start()
        return
    profiler = worker._replay_profiler
    profiler.stop()
    del worker._replay_profiler
    return sum(event.count for event in profiler.key_averages() if event.key == "cudaGraphLaunch")


@pytest.fixture(
    params=[("fp8", True), ("fp8", False), ("nvfp4", False), ("nvfp4_stochastic", False)],
    ids=["fp8_eager", "fp8_full_decode_graph", "nvfp4_full_decode_graph", "nvfp4_stochastic_graph"],
)
def replay_engine(tmp_path, monkeypatch, request):
    if Version(vllm_version).release[:2] != (0, 30):
        pytest.skip("ReplaySSM engine adapter requires vLLM 0.30.x")
    read_format, eager = request.param
    monkeypatch.setenv("VLLM_WORKER_MULTIPROC_METHOD", "spawn")
    monkeypatch.setenv("VLLM_GDN_DECODE_KERNEL", "triton")
    recipe_path = f"general/ptq/gdn_replayssm_{read_format.removesuffix('_stochastic')}_read"
    if read_format.endswith("_stochastic"):
        from modelopt.recipe import load_recipe

        recipe = {
            "metadata": {"recipe_type": "ptq"},
            "quantize": load_recipe(recipe_path).quantize.model_dump(),
        }
        for rule in recipe["quantize"]["quant_cfg"]:
            if rule.get("quantizer_name") == "*gdn_read_quantizer":
                rule["cfg"]["rounding"] = "stochastic"
        recipe_path = str(tmp_path / "stochastic.yaml")
        Path(recipe_path).write_text(yaml.safe_dump(recipe))
    monkeypatch.setenv("RECIPE_PATH", recipe_path)
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
        layer_types=["linear_attention", "full_attention"],
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
        enforce_eager=eager,
        compilation_config={"cudagraph_mode": "FULL_AND_PIECEWISE"} if not eager else None,
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
        llm.collective_rpc(_audit, kwargs={"install": True, "compiled": not eager})
        yield llm, not eager
    finally:
        llm.llm_engine.engine_core.shutdown()


def _seed_reads(worker):
    torch.cuda.manual_seed_all(101)


def test_replayssm_engine(replay_engine):
    llm, compiled = replay_engine
    samples = [{"prompt_token_ids": [3] * length} for length in (73, 5, 65, 1)]
    params = SamplingParams(temperature=0, max_tokens=34, ignore_eos=True, logprobs=1)
    llm.collective_rpc(_seed_reads)
    first = llm.generate(samples, params, use_tqdm=False)
    if compiled:
        llm.collective_rpc(_graph_replays, kwargs={"start": True})
    try:
        llm.collective_rpc(_seed_reads)
        repeated = llm.generate(samples, params, use_tqdm=False)
    finally:
        if compiled:
            assert all(count > 0 for count in llm.collective_rpc(_graph_replays))

    def values(outputs):
        return [(o.outputs[0].token_ids, o.outputs[0].cumulative_logprob) for o in outputs]

    assert values(first) == values(repeated)
    assert all(len(tokens) == 34 and math.isfinite(score) for tokens, score in values(first))
    counts = llm.collective_rpc(_audit, kwargs={"compiled": compiled})
    expected = [33] if compiled else {"steps": 264, "resets": 8, "flushes": 16}
    assert all(layer == expected for rank in counts for layer in rank)
