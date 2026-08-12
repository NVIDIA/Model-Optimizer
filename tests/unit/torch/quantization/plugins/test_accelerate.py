# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

import pickle

import pytest
import torch
import torch.nn as nn

import modelopt.torch.quantization as mtq
from modelopt.torch.quantization.nn import QuantLinearConvBase, TensorQuantizer

try:
    from accelerate.hooks import ModelHook, add_hook_to_module
except ImportError:
    pytest.skip("accelerate not available", allow_module_level=True)

transformers = pytest.importorskip("transformers")


def test_linear_with_accelerate_monkey_patched_forward():
    module_test = nn.Linear(16, 16)
    add_hook_to_module(module_test, ModelHook())

    mtq.replace_quant_module(module_test)
    assert module_test._old_forward.__func__ == QuantLinearConvBase.forward

    module_test.input_quantizer.enable_calib()
    module_test.weight_quantizer.enable_calib()

    module_ref = nn.Linear(16, 16)
    mtq.replace_quant_module(module_ref)

    module_ref.load_state_dict(module_test.state_dict())

    x = torch.randn(1, 16)
    out1 = module_test(x)
    out2 = module_ref(x)
    assert torch.allclose(out1, out2)

    module_test.input_quantizer.load_calib_amax()
    module_test.weight_quantizer.load_calib_amax()

    assert module_test.input_quantizer.amax is not None
    assert module_test.weight_quantizer.amax is not None


def test_tensor_quantizer_modelopt_state_with_accelerate_hook():
    """Verify accelerate hook attributes are excluded from modelopt state.

    When accelerate's add_hook_to_module patches a TensorQuantizer, it adds
    _hf_hook, _old_forward, and an instance-level forward (a functools.partial
    wrapping a local function). These must be excluded from the modelopt state
    dict, otherwise torch.save / pickle will fail with:
        AttributeError: Can't get local object 'add_hook_to_module.<locals>.new_forward'
    """
    tq = TensorQuantizer()
    add_hook_to_module(tq, ModelHook())

    # The hook should have injected these instance attributes
    assert hasattr(tq, "_hf_hook")
    assert hasattr(tq, "_old_forward")
    assert "forward" in tq.__dict__

    # None of the accelerate attributes should appear in the modelopt state
    state = tq.get_modelopt_state()
    accelerate_attrs = {"_hf_hook", "_old_forward", "forward"}
    leaked = accelerate_attrs & state.keys()
    assert not leaked, f"Accelerate attributes leaked into modelopt state: {leaked}"

    # The state dict must be picklable (torch.save uses pickle internally)
    pickle.dumps(state)


def test_init_quantized_weights_dtype_and_kwargs_routing(monkeypatch):
    """Model-construction kwargs must reach from_config, not dispatch.

    `load_checkpoint_and_dispatch()` accepts `dtype` but not `torch_dtype` or
    `attn_implementation`, so forwarding **kwargs verbatim raised TypeError.
    The dtype fallback also has to survive `config.torch_dtype` being a
    deprecated alias that returns None instead of being absent, which defeats
    a `getattr(config, "torch_dtype", default)` fallback.

    Drives the real patched `from_pretrained` with the checkpoint I/O stubbed.
    """
    from modelopt.torch.quantization.plugins import accelerate as accel_plugin

    seen: dict[str, dict] = {}

    class _StubModel(torch.nn.Module):
        def tie_weights(self):
            seen["tie_weights_called"] = {"yes": True}

    def fake_from_config(cls, config, **kwargs):
        seen["from_config"] = kwargs
        return _StubModel()

    def fake_dispatch(model, **kwargs):
        seen["dispatch"] = kwargs
        return model

    # get_model_device_map is nested inside init_quantized_weights, so it is
    # neutralised through the accelerate helpers it calls rather than directly.
    monkeypatch.setattr(accel_plugin.mtq, "quantize", lambda model, cfg: model)
    monkeypatch.setattr(accel_plugin.mtq, "compress", lambda model, config=None: model)
    monkeypatch.setattr(accel_plugin, "get_max_memory", lambda: {"cpu": 1 << 30})
    monkeypatch.setattr(accel_plugin, "infer_auto_device_map", lambda *a, **k: {"": "cpu"})
    monkeypatch.setattr(accel_plugin, "load_checkpoint_and_dispatch", fake_dispatch)
    monkeypatch.setattr(
        transformers.AutoModelForCausalLM,
        "from_config",
        classmethod(fake_from_config),
    )

    config = transformers.PretrainedConfig()  # carries no dtype
    with accel_plugin.init_quantized_weights(mtq.INT8_DEFAULT_CFG):
        transformers.AutoModelForCausalLM.from_pretrained(
            "stub-checkpoint",
            config=config,
            torch_dtype=torch.bfloat16,  # legacy alias
            attn_implementation="sdpa",
        )

    # Construction kwargs went to from_config...
    assert seen["from_config"]["dtype"] is torch.bfloat16
    assert seen["from_config"]["attn_implementation"] == "sdpa"
    # ...and neither alias leaked into dispatch, which cannot accept them.
    assert "torch_dtype" not in seen["dispatch"]
    assert "attn_implementation" not in seen["dispatch"]
    # The resolved dtype is still handed to dispatch explicitly.
    assert seen["dispatch"]["dtype"] is torch.bfloat16
    assert seen.get("tie_weights_called")

    # With no dtype anywhere, the float16 fallback must actually fire: the
    # deprecated alias returns None rather than being absent.
    assert getattr(config, "torch_dtype", torch.float16) is None
    seen.clear()
    with accel_plugin.init_quantized_weights(mtq.INT8_DEFAULT_CFG):
        transformers.AutoModelForCausalLM.from_pretrained("stub-checkpoint", config=config)
    assert seen["from_config"]["dtype"] is torch.float16
    assert seen["dispatch"]["dtype"] is torch.float16
