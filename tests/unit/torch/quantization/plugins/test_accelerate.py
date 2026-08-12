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


def test_init_quantized_weights_dtype_resolution():
    """dtype/torch_dtype must not leak into load_checkpoint_and_dispatch().

    Both are model-construction kwargs: `load_checkpoint_and_dispatch()`
    accepts `dtype` but not `torch_dtype`, so forwarding kwargs verbatim
    raised TypeError for callers using the legacy alias. The fallback also has
    to survive `config.torch_dtype` being a deprecated alias that returns None
    instead of being absent, which defeats a `getattr(..., default)` fallback.
    """
    from transformers import PretrainedConfig

    def resolve(kwargs: dict, config) -> torch.dtype:
        """Mirror of the resolution order in accelerate.patched_from_pretrained."""
        dtype_kwarg = kwargs.pop("dtype", None)
        legacy_dtype_kwarg = kwargs.pop("torch_dtype", None)
        config_dtype = getattr(config, "dtype", None)
        if config_dtype is None:
            config_dtype = getattr(config, "torch_dtype", None)
        return (
            dtype_kwarg
            if dtype_kwarg is not None
            else legacy_dtype_kwarg
            if legacy_dtype_kwarg is not None
            else config_dtype
            if config_dtype is not None
            else torch.float16
        )

    config = PretrainedConfig()

    # Explicit kwargs win, in precedence order, and are consumed.
    kwargs = {"dtype": torch.bfloat16, "attn_implementation": "sdpa"}
    assert resolve(kwargs, config) is torch.bfloat16
    assert "dtype" not in kwargs and "torch_dtype" not in kwargs

    kwargs = {"torch_dtype": torch.bfloat16}
    assert resolve(kwargs, config) is torch.bfloat16
    assert "torch_dtype" not in kwargs

    # A config carrying no dtype must fall back to float16, not None.
    assert resolve({}, config) is torch.float16

    # A config that does carry one is respected.
    config_with_dtype = PretrainedConfig()
    config_with_dtype.dtype = torch.bfloat16
    assert resolve({}, config_with_dtype) is torch.bfloat16
