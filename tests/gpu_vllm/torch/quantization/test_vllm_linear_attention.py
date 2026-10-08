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

"""Native vLLM state-boundary QDQ and policy restoration on tiny offline models."""

import copy
import gc
import importlib.util
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
from vllm.transformers_utils.configs.kimi_linear import KimiLinearConfig

import modelopt.torch.opt as mto
from modelopt.torch.opt.conversion import ModeloptStateManager
from modelopt.torch.quantization.conversion import restore_quantizer_state
from modelopt.torch.quantization.linear_attention import LinearAttentionConfig
from modelopt.torch.quantization.plugins.vllm_linear_attention import _QuantVllmLinearAttention

ROOT = Path(__file__).parents[4]


def _example_module(name):
    spec = importlib.util.spec_from_file_location(
        name + "_test", ROOT / "examples/vllm_serve" / (name + ".py")
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _state_worker(worker, *, action="audit"):
    model = worker.model_runner.model
    if hasattr(model, "unwrap"):
        model = model.unwrap()
    layers = [m for m in model.modules() if isinstance(m, _QuantVllmLinearAttention)]
    assert layers
    if action == "restore":
        state = worker._state_checkpoint
        manager = ModeloptStateManager()
        manager.load_state_dict(state["modelopt_state_dict"], state["modelopt_version"])
        for _, config, metadata in manager.modes_with_states():
            restore_quantizer_state(model, config, metadata)
    for layer in layers:
        if action == "disable":
            layer._linear_attn_state.disable()
            layer.linear_attention_config = LinearAttentionConfig()
        elif action == "setup":
            layer.linear_attention_config.state_block_v = 16
            layer._state_calls = {"prefill": 0, "decode": 0, "changed": 0}
            original = layer._quantized_state_call

            def checked_call(native, *args, _layer=layer, _original=original, **kwargs):
                state = kwargs["initial_state"]
                indices = kwargs.get("ssm_state_indices")
                if indices is not None:
                    indices = indices[: kwargs["cu_seqlens"].numel() - 1].long()
                selected = state if indices is None else state.index_select(0, indices)
                quantizer = copy.deepcopy(_layer._linear_attn_state)
                rounded = torch.cat([quantizer(x) for x in selected.split(16, -1)], -1)
                expected = state.clone()
                if indices is None:
                    expected = rounded
                else:
                    expected.index_copy_(0, indices, rounded)

                def checked_native(*a, **kw):
                    # Check the full cache, including inactive slots, before delegating.
                    torch.testing.assert_close(kw["initial_state"], expected, atol=0, rtol=0)
                    return native(*a, **kw)

                _layer._state_calls["prefill" if indices is None else "decode"] += 1
                _layer._state_calls["changed"] += int(not torch.equal(selected, rounded))
                return _original(checked_native, *args, **kwargs)

            layer._quantized_state_call = checked_call
    if action == "setup":
        worker._state_checkpoint = copy.deepcopy(mto.modelopt_state(model))
    return [dict(layer._state_calls) for layer in layers]


def _tiny_model(path, kind):
    common = {
        "torch_dtype": "bfloat16",
        "hidden_size": 128,
        "intermediate_size": 256,
        "num_attention_heads": 4,
        "head_dim": 32,
        "vocab_size": 512,
        "max_position_embeddings": 128,
    }
    if kind == "gdn":
        config = Qwen3NextConfig(
            **common,
            num_hidden_layers=2,
            num_key_value_heads=2,
            linear_num_value_heads=4,
            linear_num_key_heads=2,
            linear_key_head_dim=32,
            linear_value_head_dim=64,
            num_experts=4,
            num_experts_per_tok=2,
            moe_intermediate_size=64,
            shared_expert_intermediate_size=64,
            mlp_only_layers=[],
            full_attention_interval=2,
            architectures=["Qwen3NextForCausalLM"],
        )
    else:
        config = KimiLinearConfig(
            **common,
            num_hidden_layers=1,
            architectures=["KimiLinearForCausalLM"],
            linear_attn_config={
                "head_dim": 32,
                "num_heads": 4,
                "short_conv_kernel_size": 4,
                "kda_layers": [1],
                "full_attn_layers": [],
            },
        )
    config.save_pretrained(path)
    shutil.copytree(ROOT / "tests/_test_utils/torch/tokenizer", path, dirs_exist_ok=True)


def _generate(llm):
    outputs = llm.generate(
        [{"prompt_token_ids": [3] * 73}, {"prompt_token_ids": [4]}],
        SamplingParams(temperature=0, max_tokens=3, ignore_eos=True, logprobs=1),
        use_tqdm=False,
    )
    values = [(o.outputs[0].token_ids, o.outputs[0].cumulative_logprob) for o in outputs]
    assert all(len(tokens) == 3 and math.isfinite(score) for tokens, score in values)
    return values


@pytest.fixture(params=["gdn", "kda"])
def compiled_worker(request, tmp_path, monkeypatch):
    """Compile one tiny native model outside the functional test's timeout."""
    if Version(vllm_version).release[:2] != (0, 15):
        pytest.skip("The state adapter requires vLLM 0.15.x")
    if not torch.cuda.is_available():
        pytest.skip("Requires a CUDA GPU")
    monkeypatch.setenv("VLLM_WORKER_MULTIPROC_METHOD", "spawn")
    paths = [str(ROOT), str(ROOT / "examples/vllm_serve"), str(Path(__file__).parent)]
    for path in reversed(paths):
        monkeypatch.syspath_prepend(path)
    monkeypatch.setenv("PYTHONPATH", os.pathsep.join([*paths, os.environ.get("PYTHONPATH", "")]))
    monkeypatch.setenv(
        "RECIPE_PATH", str(ROOT / "examples/vllm_serve/linear_attention_state_int8.yaml")
    )
    monkeypatch.delenv("MODELOPT_STATE_PATH", raising=False)
    model_path = tmp_path / "model"
    _tiny_model(model_path, request.param)
    llm = LLM(
        model=str(model_path),
        load_format="dummy",
        dtype="bfloat16",
        max_model_len=128,
        max_num_seqs=4,
        max_num_batched_tokens=64,
        enforce_eager=True,
        enable_prefix_caching=False,
        enable_chunked_prefill=True,
        async_scheduling=False,
        mamba_cache_dtype="float32",
        kv_cache_memory_bytes=128 * 1024**2,
        gpu_memory_utilization=0.1,
        worker_cls="fakequant_worker.FakeQuantWorker",
        seed=17,
    )
    try:
        llm.collective_rpc(_state_worker, kwargs={"action": "setup"})
        _generate(llm)
        yield llm
    finally:
        llm.llm_engine.engine_core.shutdown()
        del llm
        gc.collect()


def test_state_qdq_and_restore(compiled_worker):
    llm = compiled_worker
    expected = _generate(llm)
    reports = llm.collective_rpc(_state_worker)
    assert all(all(count > 0 for count in layer.values()) for rank in reports for layer in rank)
    llm.collective_rpc(_state_worker, kwargs={"action": "disable"})
    _generate(llm)
    assert llm.collective_rpc(_state_worker) == reports
    llm.collective_rpc(_state_worker, kwargs={"action": "restore"})
    assert _generate(llm) == expected


def test_saved_linear_attention_policy_and_quantizer_names_follow_mapper():
    module = _example_module("vllm_reload_utils")

    metadata = {
        "linear_attention": {"backbone.layers.0.attn": {"backend": "serving", "state_block_v": 16}},
        "quantizer_state": {"backbone.layers.0.attn.kda_state_quantizer": {"_disabled": False}},
    }
    state = {
        "modelopt_state_dict": [("quantize", {"config": {"quant_cfg": []}, "metadata": metadata})]
    }

    def mapper(values):
        return {
            k.replace("backbone.", "model.").replace(".attn.", ".self_attn."): v
            for k, v in values.items()
        }

    state["modelopt_state_dict"].append(
        ("quantize_algo", {"config": {}, "metadata": copy.deepcopy(metadata)})
    )
    result = module.convert_modelopt_state_to_vllm(state, mapper)
    converted = result["modelopt_state_dict"][0][1]
    assert list(converted["metadata"]["linear_attention"]) == ["model.layers.0.self_attn"]
    assert list(converted["metadata"]["quantizer_state"]) == [
        "model.layers.0.self_attn.kda_state_quantizer"
    ]
    assert converted["config"]["linear_attention"][0] == {
        "module_name": "model.layers.0.self_attn",
        "cfg": {"backend": "serving", "state_block_v": 16},
    }
    assert "linear_attention" not in result["modelopt_state_dict"][1][1]["config"]
