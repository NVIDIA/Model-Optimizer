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
    from accelerate import hooks as accelerate_hooks
    from accelerate.hooks import AlignDevicesHook, ModelHook, add_hook_to_module
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


def test_buffer_registered_during_weight_access_survives_buffer_offload():
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


@pytest.mark.parametrize("buffer_case", ["local", "sparse", "new_nested", "new_nested_resident"])
def test_weight_access_keeps_buffer_persistence_local_to_its_owner(buffer_case, monkeypatch):
    model = nn.Linear(4, 4, bias=False)
    model.register_buffer("scale", torch.ones(4))
    model.child = nn.Module()
    if buffer_case == "local":
        model.child.register_buffer("scale", torch.full((4,), 2.0), persistent=False)
    elif buffer_case == "sparse":
        model.child.register_buffer(
            "scale", torch.sparse_coo_tensor([[1, 3]], [2.0, 2.0], (4,), check_invariants=True)
        )
    weights = model.state_dict() if buffer_case != "sparse" else {"weight": model.weight.detach()}
    add_hook_to_module(
        model,
        AlignDevicesHook(
            execution_device="cpu",
            offload=True,
            offload_buffers=buffer_case != "sparse",
            place_submodules=buffer_case.startswith("new_nested"),
            weights_map=weights,
        ),
    )
    if buffer_case == "new_nested_resident":
        model._hf_hook.pre_forward(model)
    devices = {name: tensor.device for name, tensor in model.named_buffers()}
    keys = model.state_dict().keys()
    persistence = {owner: owner._non_persistent_buffers_set.copy() for owner in model.modules()}
    sets = {owner: owner._non_persistent_buffers_set for owner in model.modules()}
    inputs = torch.ones(1, 4)
    with enable_weight_access_and_writeback(model, model):
        torch.testing.assert_close(model.scale, torch.ones(4))
        if buffer_case == "local":
            torch.testing.assert_close(model.child.scale, torch.full((4,), 2.0))
        elif buffer_case == "sparse":
            torch.testing.assert_close(
                model.child.scale.to_dense(), torch.tensor([0.0, 2.0, 0.0, 2.0])
            )
        else:
            model.child.register_buffer("scratch", torch.full((4,), 2.0), persistent=False)
            model.child.register_buffer("keep", torch.full((4,), 3.0))
            scratch = model.child.scratch
            persistence[model.child].add("scratch")
            devices.update(
                {
                    "child.scratch": torch.device("cpu"),
                    "child.keep": model.weight.device
                    if buffer_case.endswith("resident")
                    else torch.device("meta"),
                }
            )
            keys = set(keys) | {"child.keep"}
        expected = model(inputs).detach()
    assert model.state_dict().keys() == keys
    assert {name: tensor.device for name, tensor in model.named_buffers()} == devices
    assert {owner: owner._non_persistent_buffers_set for owner in model.modules()} == persistence
    assert all(owner._non_persistent_buffers_set is saved for owner, saved in sets.items())
    torch.testing.assert_close(model(inputs), expected)
    if buffer_case.startswith("new_nested"):
        assert model.child.scratch is scratch
        assert "child.scratch" not in weights
        torch.testing.assert_close(weights["child.keep"], torch.full((4,), 3.0))
        set_tensor = accelerate_hooks.set_module_tensor_to_device
        armed = False

        def fail_placement(owner, name, device, *args, **kwargs):
            nonlocal armed
            if owner is model and name == "scale" and str(device) == failure_device and armed:
                armed = False
                raise RuntimeError("native placement failed")
            return set_tensor(owner, name, device, *args, **kwargs)

        for failure_device in ("cpu", "meta"):
            armed = True
            with monkeypatch.context() as patch:
                patch.setattr(accelerate_hooks, "set_module_tensor_to_device", fail_placement)
                with pytest.raises(RuntimeError, match="native placement failed"):
                    model(inputs)
            assert all(owner._non_persistent_buffers_set is saved for owner, saved in sets.items())
            assert {
                owner: owner._non_persistent_buffers_set for owner in model.modules()
            } == persistence
            torch.testing.assert_close(model(inputs), expected)
