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
from modelopt.torch.quantization.ggml import GGML_FORMAT_REGISTRY
from modelopt.torch.quantization.nn import QuantLinearConvBase, TensorQuantizer
from modelopt.torch.quantization.utils.core_utils import enable_weight_access_and_writeback

try:
    from accelerate import cpu_offload
    from accelerate.hooks import AlignDevicesHook, ModelHook, add_hook_to_module
    from accelerate.utils import set_module_tensor_to_device
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


@pytest.mark.parametrize("offload_mode", ["direct_hook", "cpu_offload"])
def test_buffer_registered_during_weight_access_survives_buffer_offload(offload_mode):
    """GPTQ's payload pin is registered while the offloaded weight is materialized; the hook must
    still be able to reload it after offloading the module again."""
    torch.manual_seed(0)
    model = nn.Linear(512, 8, bias=False)
    inputs = torch.randn(128, 512)
    iq1_s_weights = {"num_bits": "iq1_s", "block_sizes": {-1: 256}, "backend": "ggml"}
    quant_cfg = [
        {"quantizer_name": "*", "enable": False},
        {"quantizer_name": "*weight_quantizer", "cfg": iq1_s_weights, "enable": True},
    ]
    mtq.quantize(model, {"quant_cfg": quant_cfg, "algorithm": "max"}, lambda m: m(inputs))
    weights_map = {name: value.detach().clone() for name, value in model.state_dict().items()}
    if offload_mode == "cpu_offload":
        cpu_offload(model, execution_device="cpu", offload_buffers=True, state_dict=weights_map)
    else:
        hook = AlignDevicesHook(
            execution_device="cpu",
            offload=True,
            offload_buffers=True,
            place_submodules=True,
            weights_map=weights_map,
        )
        add_hook_to_module(model, hook)

    gptq = {"method": "gptq", "block_size": 256, "perc_damp": 0.3}
    mtq.calibrate(model, algorithm=gptq, forward_loop=lambda m: m(inputs))

    (pin_key,) = [key for key in weights_map if key.endswith("_ggml_pinned_iq1_s")]
    decoded = GGML_FORMAT_REGISTRY["iq1_s"].dequantize(
        weights_map[pin_key], torch.tensor((8, 512)), dtype=torch.float32
    )
    torch.testing.assert_close(model(inputs), inputs @ decoded.T)


@pytest.mark.parametrize("offload_mode", ["direct_hook", "cpu_offload"])
def test_weight_access_restores_owners_after_later_onload_failure(offload_mode):
    model = nn.Sequential(*(nn.Linear(4, 4, bias=False) for _ in range(4)))
    for owner in model:
        owner.register_buffer("scale", torch.ones(4))
    originals = [
        {name: value.detach().clone() for name, value in owner.state_dict().items()}
        for owner in model
    ]
    if offload_mode == "cpu_offload":
        weights_map = {name: value.detach().clone() for name, value in model.state_dict().items()}
        cpu_offload(model, execution_device="cpu", offload_buffers=True, state_dict=weights_map)
        backings = [(weights_map, f"{index}.") for index in range(len(model))]
    else:
        add_hook_to_module(model, ModelHook())
        backings = []
        for owner, original in zip(model, originals, strict=True):
            weights_map = {name: value.clone() for name, value in original.items()}
            add_hook_to_module(
                owner,
                AlignDevicesHook(
                    execution_device="cpu",
                    offload=True,
                    offload_buffers=True,
                    weights_map=weights_map,
                ),
            )
            backings.append((weights_map, ""))
    hooks = [owner._hf_hook for owner in model]

    # The caller already owns this resident update; entering must not reload its stale map.
    for name, value in originals[0].items():
        set_module_tensor_to_device(model[0], name, "cpu", value=value.clone().add_(1))
    resident_weight = model[0].weight.detach().clone()
    mixed_name = "weight" if offload_mode == "direct_hook" else "scale"
    updated = originals[2][mixed_name].clone().add_(2)
    set_module_tensor_to_device(model[2], mixed_name, "cpu", value=updated)
    devices = {name: tensor.device for name, tensor in model.state_dict().items()}
    with (
        pytest.raises(ValueError),
        enable_weight_access_and_writeback(model, model, writeback=False),
    ):
        pass
    assert {name: tensor.device for name, tensor in model.state_dict().items()} == devices
    assert all(hook.offload for hook in hooks)
    torch.testing.assert_close(getattr(model[2], mixed_name), updated, rtol=0, atol=0)
    set_module_tensor_to_device(model[2], mixed_name, "meta")

    # Owner 1 onloads successfully; owner 2 loads its weight then fails on its buffer.
    backing, prefix = backings[2]
    missing_key = prefix + "scale"
    missing = backing.pop(missing_key)
    devices = {name: tensor.device for name, tensor in model.state_dict().items()}
    try:
        with pytest.raises(KeyError) as error, enable_weight_access_and_writeback(model, model):
            pytest.fail("Weight access yielded after an incomplete onload")
        assert error.value.args == (missing_key,)
        assert {name: tensor.device for name, tensor in model.state_dict().items()} == devices
        assert all(hook.offload for hook in hooks)
        torch.testing.assert_close(model[0].weight, resident_weight, rtol=0, atol=0)
        for index, (owner_backing, owner_prefix) in enumerate(backings):
            for name, value in originals[index].items():
                if index != 2 or name != "scale":
                    torch.testing.assert_close(
                        owner_backing[owner_prefix + name], value, rtol=0, atol=0
                    )
    finally:
        backing[missing_key] = missing

    with enable_weight_access_and_writeback(model, model):
        assert all(not tensor.is_meta for tensor in model.state_dict().values())
        torch.testing.assert_close(model[0].weight, resident_weight, rtol=0, atol=0)
    assert {name: tensor.device for name, tensor in model.state_dict().items()} == devices
    assert all(hook.offload for hook in hooks)
    inputs = torch.randn(2, 4)
    expected = inputs @ resident_weight.T
    for original in originals[1:]:
        expected = expected @ original["weight"].T
    torch.testing.assert_close(model(inputs), expected)
