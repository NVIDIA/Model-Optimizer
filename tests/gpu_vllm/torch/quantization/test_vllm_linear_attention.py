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

"""Native vLLM state-boundary QDQ, worker reload, and TP on tiny offline models."""

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
import modelopt.torch.quantization as mtq
from modelopt.torch.quantization.linear_attention import LinearAttentionConfig
from modelopt.torch.quantization.plugins.vllm_linear_attention import _QuantVllmLinearAttention

pytestmark = pytest.mark.skipif(
    Version(vllm_version).release[:2] != (0, 15), reason="Requires the pinned vLLM 0.15.x ABI"
)
ROOT = Path(__file__).parents[4]


def _example_module(name):
    spec = importlib.util.spec_from_file_location(
        name + "_test", ROOT / "examples/vllm_serve" / (name + ".py")
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _calibrate_worker(worker):
    model = worker.model_runner.model
    if hasattr(model, "unwrap"):
        model = model.unwrap()
    layer = next(m for m in model.modules() if isinstance(m, _QuantVllmLinearAttention))
    projection = layer.q_proj if hasattr(layer, "q_proj") else layer.in_proj_qkvz
    quantizer = projection.input_quantizer
    quantizer.enable()
    try:
        loop = _example_module("vllm_ptq_utils").calibrate_fun(
            [{"input_ids": torch.tensor([[3] * 5, [4] * 5])}], worker
        )
        mtq.calibrate(model, algorithm="max", forward_loop=loop)
        assert torch.isfinite(quantizer.amax).all() and quantizer.amax.max() > 0
    finally:
        quantizer.disable()
    return True


def _set_state_format_worker(worker, *, fp8):
    for layer in worker._linear_attention_layers:
        layer._linear_attn_state.num_bits = (4, 3) if fp8 else 8
        layer.validate_linear_attention()


def _disable_worker(worker):
    for layer in worker._linear_attention_layers:
        layer._linear_attn_state.disable()
        layer._linear_attn_w.disable()
        layer.linear_attention_config = LinearAttentionConfig()


def _audit_worker(worker, *, checkpoint=None):
    model = worker.model_runner.model
    if hasattr(model, "unwrap"):
        model = model.unwrap()
    layers = [m for m in model.modules() if isinstance(m, _QuantVllmLinearAttention)]
    assert layers
    if checkpoint is not None and torch.distributed.get_rank() == 0:
        torch.save(mto.modelopt_state(model), checkpoint)
    reports = []
    for layer in layers:
        assert not hasattr(layer, "_linear_attention_cache")
        if not hasattr(layer, "_state_audit"):
            layer._state_audit = {"prefill": 0, "decode": 0, "changed": 0, "heads": 0}
            original = layer._quantized_state_call

            def checked_call(native, *args, _layer=layer, _original=original, **kwargs):
                state = kwargs["initial_state"]
                indices = kwargs.get("ssm_state_indices")
                expected_state = state.clone()
                selected = state if indices is None else state.index_select(0, indices.long())
                reference_quantizer = copy.deepcopy(_layer._linear_attn_state)
                rounded = reference_quantizer(selected.clone())
                if indices is None:
                    expected_state = rounded
                else:
                    expected_state.index_copy_(0, indices.long(), rounded)
                # Native KDA prefill writes its output into the value-input buffer.
                control_args = tuple(x.clone() if isinstance(x, torch.Tensor) else x for x in args)
                control_kwargs = {
                    name: x.clone() if isinstance(x, torch.Tensor) else x
                    for name, x in kwargs.items()
                }
                control_kwargs["initial_state"] = expected_state
                expected = native(*control_args, **control_kwargs)
                native_calls = []

                def checked_native(*native_args, **native_kwargs):
                    incoming = native_kwargs["initial_state"]
                    incoming = (
                        incoming if indices is None else incoming.index_select(0, indices.long())
                    )
                    torch.testing.assert_close(incoming, rounded, atol=0, rtol=0)
                    native_calls.append(True)
                    return native(*native_args, **native_kwargs)

                actual = _original(checked_native, *args, **kwargs)
                assert native_calls == [True]
                for result, control in zip(actual, expected):
                    if control is not None:
                        torch.testing.assert_close(result, control, atol=0, rtol=0)
                _layer._state_audit["prefill" if indices is None else "decode"] += 1
                _layer._state_audit["changed"] += int(not torch.equal(selected, rounded))
                _layer._state_audit["heads"] = state.shape[1]
                return actual

            layer._quantized_state_call = checked_call
        reports.append(dict(layer._state_audit))
    return reports


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
        SamplingParams(temperature=0, max_tokens=12, ignore_eos=True, logprobs=1),
        use_tqdm=False,
    )
    values = [(o.outputs[0].token_ids, o.outputs[0].cumulative_logprob) for o in outputs]
    assert all(len(tokens) == 12 and math.isfinite(score) for tokens, score in values)
    return values


@pytest.mark.parametrize("kind", ["gdn", "kda"])
@pytest.mark.parametrize("tp", [1, 2])
@pytest.mark.timeout(360)
def test_fakequant_worker_generation_reference_reload_and_tp(tmp_path, monkeypatch, kind, tp):
    if torch.cuda.device_count() < tp:
        pytest.skip(f"Requires {tp} GPUs")
    monkeypatch.setenv("VLLM_ENABLE_V1_MULTIPROCESSING", "1")
    monkeypatch.setenv("VLLM_WORKER_MULTIPROC_METHOD", "spawn")
    paths = [str(ROOT), str(ROOT / "examples/vllm_serve"), str(Path(__file__).parent)]
    monkeypatch.setenv("PYTHONPATH", os.pathsep.join([*paths, os.environ.get("PYTHONPATH", "")]))
    monkeypatch.setenv(
        "RECIPE_PATH", str(ROOT / "examples/vllm_serve/linear_attention_state_int8.yaml")
    )
    monkeypatch.delenv("MODELOPT_STATE_PATH", raising=False)
    model_path, checkpoint = tmp_path / "model", tmp_path / "state.pt"
    _tiny_model(model_path, kind)
    kwargs = {
        "model": str(model_path),
        "load_format": "dummy",
        "dtype": "bfloat16",
        "max_model_len": 128,
        "max_num_seqs": 4,
        "max_num_batched_tokens": 64,
        "enforce_eager": True,
        "enable_prefix_caching": False,
        "enable_chunked_prefill": True,
        "async_scheduling": False,
        "mamba_cache_dtype": "float32",
        "kv_cache_memory_bytes": 128 * 1024**2,
        "worker_cls": "fakequant_worker.FakeQuantWorker",
        "disable_custom_all_reduce": True,
        "tensor_parallel_size": tp,
        "seed": 17,
    }
    llm = LLM(**kwargs)
    try:
        llm.collective_rpc(_audit_worker, kwargs={"checkpoint": str(checkpoint)})
        actual = _generate(llm)
        assert _generate(llm) == actual  # Fresh requests can reuse the same native state slots.
        reports = llm.collective_rpc(_audit_worker)
        assert len(reports) == tp
        for rank in reports:
            assert all(
                r["prefill"] > 0 and r["decode"] > 0 and r["changed"] > 0 and r["heads"] == 4 // tp
                for r in rank
            )
        assert _generate(llm) == actual
        llm.collective_rpc(_set_state_format_worker, kwargs={"fp8": True})
        fp8 = _generate(llm)
        assert _generate(llm) == fp8
        llm.collective_rpc(_set_state_format_worker, kwargs={"fp8": False})
        assert _generate(llm) == actual
        assert all(llm.collective_rpc(_calibrate_worker))
        llm.collective_rpc(_disable_worker)
        disabled = _generate(llm)
    finally:
        llm.llm_engine.engine_core.shutdown()
        del llm
        gc.collect()

    monkeypatch.delenv("RECIPE_PATH")
    monkeypatch.setenv("MODELOPT_STATE_PATH", str(checkpoint))
    llm = LLM(**kwargs)
    try:
        assert _generate(llm) == actual
    finally:
        llm.llm_engine.engine_core.shutdown()
        del llm
        gc.collect()

    monkeypatch.delenv("MODELOPT_STATE_PATH")
    llm = LLM(**kwargs)
    try:
        assert _generate(llm) == disabled
    finally:
        llm.llm_engine.engine_core.shutdown()
        del llm
        gc.collect()


def test_saved_linear_attention_policy_and_quantizer_names_follow_mapper():
    module = _example_module("vllm_reload_utils")

    metadata = {
        "linear_attention": {"backbone.layers.0.attn": {"backend": "fla"}},
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
    assert converted["config"]["linear_attention"][0]["module_name"] == "model.layers.0.self_attn"
    assert "linear_attention" not in result["modelopt_state_dict"][1][1]["config"]
