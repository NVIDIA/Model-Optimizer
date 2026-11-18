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
from collections import UserDict
from collections.abc import Mapping

import pytest
import torch
import torch.nn as nn

import modelopt.torch.quantization as mtq
from modelopt.torch.quantization.ggml import GGML_FORMAT_REGISTRY
from modelopt.torch.quantization.ggml.common import pin_packed_weight, pinned_packed_weight
from modelopt.torch.quantization.nn import TensorQuantizer
from modelopt.torch.quantization.repair import IQRepairPolicy, plan_iq_repair
from modelopt.torch.quantization.utils.core_utils import enable_weight_access_and_writeback

try:
    from accelerate import cpu_offload, disk_offload
    from accelerate import hooks as accelerate_hooks
    from accelerate.hooks import AlignDevicesHook, ModelHook, add_hook_to_module
    from accelerate.utils import set_module_tensor_to_device
except ImportError:
    pytest.skip("accelerate not available", allow_module_level=True)


def test_linear_with_accelerate_monkey_patched_forward():
    calls = []

    class RecordingHook(ModelHook):
        def pre_forward(self, module, *args, **kwargs):
            calls.append("pre")
            return (args[0] + 0.5,), kwargs

        def post_forward(self, module, output):
            calls.append("post")
            return output * 2

    module_test = nn.Linear(16, 16)
    add_hook_to_module(module_test, RecordingHook())

    mtq.replace_quant_module(module_test)

    module_ref = nn.Linear(16, 16)
    mtq.replace_quant_module(module_ref)
    module_ref.load_state_dict(module_test.state_dict())
    for module in (module_test, module_ref):
        module.input_quantizer.enable_calib()
        module.weight_quantizer.enable_calib()

    x = torch.randn(1, 16, requires_grad=True)
    ref_x = x.detach().clone().requires_grad_()
    out1 = module_test(x)
    out2 = module_ref(ref_x + 0.5) * 2
    torch.testing.assert_close(out1, out2)
    assert calls == ["pre", "post"]
    actual = torch.autograd.grad(out1.sum(), (x, module_test.weight, module_test.bias))
    expected = torch.autograd.grad(out2.sum(), (ref_x, module_ref.weight, module_ref.bias))
    for actual_grad, expected_grad in zip(actual, expected):
        torch.testing.assert_close(actual_grad, expected_grad)

    for name in ("input_quantizer", "weight_quantizer"):
        quantizer, reference = getattr(module_test, name), getattr(module_ref, name)
        quantizer.load_calib_amax()
        reference.load_calib_amax()
        assert quantizer.amax is not None
        torch.testing.assert_close(quantizer.amax, reference.amax)


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


@pytest.mark.parametrize("offload_mode", ["cpu_offload", "disk_offload"])
@pytest.mark.parametrize("failure_stage", ["body", "writeback", "cleanup", "partial_cleanup"])
def test_weight_access_writeback_failure_preserves_backing_maps(
    offload_mode, failure_stage, tmp_path, monkeypatch
):
    model = nn.Sequential(*(nn.Linear(4, 4, bias=False) for _ in range(3)))
    model[2].weight = model[1].weight
    for owner in model:
        owner.register_buffer("scale", torch.ones(4))
    originals = {name: value.detach().clone() for name, value in model.state_dict().items()}
    if offload_mode == "cpu_offload":
        cpu_offload(model, execution_device="cpu", offload_buffers=True)
    else:
        disk_offload(model, tmp_path / "offload", execution_device="cpu", offload_buffers=True)
    hooks = [owner._hf_hook for owner in model]
    dataset = hooks[0].weights_map.dataset
    backing = getattr(dataset, "state_dict", dataset)
    original_backing = dict(backing)
    tied_map = {}
    for hook in hooks[1:]:
        hook.tied_params_map = tied_map
        hook.tied_params_names.add("weight")

    # An already resident owner can share the CPU backing, as CPU pre_forward does.
    hooks[0].pre_forward(model[0])
    resident_weight = model[0].weight
    devices = {name: value.device for name, value in model.state_dict().items()}
    failure = RuntimeError("late writeback failure")

    def fail_writeback(module, state_dict, prefix, local_metadata):
        if failure_stage == "writeback":
            raise failure

    handle = model[1].register_state_dict_post_hook(fail_writeback)
    post_forward = None
    set_tensor = accelerate_hooks.set_module_tensor_to_device
    armed = True

    def fail_before_last_tensor(owner, name, device, *args, **kwargs):
        nonlocal armed
        if owner is model[1] and name == "scale" and device == "meta" and armed:
            armed = False
            raise failure
        return set_tensor(owner, name, device, *args, **kwargs)

    try:
        with (
            pytest.raises(RuntimeError) as error,
            enable_weight_access_and_writeback(model, model, rollback=True),
        ):
            post_forward = hooks[1].post_forward

            def fail_cleanup(owner, output):
                post_forward(owner, output)
                if failure_stage == "cleanup":
                    raise failure
                if failure_stage == "writeback":
                    raise RuntimeError("later cleanup failure")

            hooks[1].post_forward = fail_cleanup
            if failure_stage == "partial_cleanup":
                monkeypatch.setattr(
                    accelerate_hooks, "set_module_tensor_to_device", fail_before_last_tensor
                )
            with torch.no_grad():
                for index, owner in enumerate(model):
                    owner.weight.copy_(originals[f"{index}.weight"] + 1)
                    owner.scale.copy_(originals[f"{index}.scale"] + 2)
            if failure_stage != "partial_cleanup":
                model[-1].register_buffer(
                    "_ggml_pinned_iq1_s", torch.tensor([3], dtype=torch.uint8)
                )
                model[-1].register_buffer("scale", model[-1].scale + 3, persistent=False)
            if failure_stage == "body":
                raise failure
        assert error.value is failure
    finally:
        handle.remove()
        if post_forward is not None:
            hooks[1].post_forward = post_forward
        monkeypatch.setattr(accelerate_hooks, "set_module_tensor_to_device", set_tensor)

    after = model.state_dict()
    assert after.keys() == originals.keys()
    assert {name: after[name].device for name in devices} == devices
    assert all(hook.offload for hook in hooks)
    assert model[0].weight is resident_weight
    assert not any(tied_map.values())
    assert backing.keys() == original_backing.keys()
    for name, value in originals.items():
        torch.testing.assert_close(dataset[name], value, rtol=0, atol=0)
    for name, value in original_backing.items():
        assert backing[name] is value
    if offload_mode == "cpu_offload":
        assert backing["1.weight"].data_ptr() == backing["2.weight"].data_ptr()

    with enable_weight_access_and_writeback(model, model):
        with torch.no_grad():
            for index, owner in enumerate(model):
                owner.weight.copy_(originals[f"{index}.weight"] + 1)
                owner.scale.copy_(originals[f"{index}.scale"] + 2)
        model[-1].register_buffer("_ggml_pinned_iq1_s", torch.tensor([3], dtype=torch.uint8))
    for name, value in originals.items():
        torch.testing.assert_close(dataset[name], value + (1 if name.endswith("weight") else 2))
    torch.testing.assert_close(
        dataset["2._ggml_pinned_iq1_s"], torch.tensor([3], dtype=torch.uint8)
    )
    inputs = torch.ones(2, 4)
    expected = inputs
    for index in range(3):
        expected = expected @ (originals[f"{index}.weight"] + 1).T
    torch.testing.assert_close(model(inputs), expected)


@pytest.mark.parametrize("offload_mode", ["cpu_offload", "disk_offload"])
@pytest.mark.parametrize("failure_stage", ["local_writeback", "peer"])
def test_iq_finalize_writeback_failure_restores_state_for_retry(
    offload_mode, failure_stage, tmp_path
):
    model = nn.ModuleList([nn.Linear(256, 2, bias=False) for _ in range(2)])
    mtq.quantize(
        model,
        {
            "quant_cfg": [
                {"quantizer_name": "*", "enable": False},
                {
                    "quantizer_name": "*weight_quantizer",
                    "cfg": {"num_bits": "iq1_m", "backend": "ggml"},
                },
            ],
            "algorithm": None,
        },
    )
    for owner in model:
        quantizer = owner.weight_quantizer
        quantizer.pre_quant_scale = torch.linspace(0.5, 1.5, 256)
        source = owner.weight * quantizer.pre_quant_scale
        pin_packed_weight(
            quantizer, "iq1_m", GGML_FORMAT_REGISTRY["iq1_m"].pack(source, quantizer), source=source
        )
    original = {name: value.detach().clone() for name, value in model.state_dict().items()}
    if offload_mode == "cpu_offload":
        # Native CPU onload casts these master weights back to FP32, preventing aliases from
        # disguising a peer rollback that only restores the temporary onloaded tensors.
        masters = {
            name: value.double() if value.is_floating_point() else value.clone()
            for name, value in original.items()
        }
        cpu_offload(model, execution_device="cpu", offload_buffers=True, state_dict=masters)
    else:
        disk_offload(model, tmp_path / "offload", execution_device="cpu", offload_buffers=True)
    dataset = model[0]._hf_hook.weights_map.dataset
    backing = getattr(dataset, "state_dict", dataset)
    original_backing = dict(backing)
    policy = IQRepairPolicy()
    plan = plan_iq_repair([model])

    def weight_access(**kwargs):
        return enable_weight_access_and_writeback(model, model, **kwargs)

    failure = RuntimeError("late IQ writeback failure")
    armed = True

    def fail_writeback(module, state_dict, prefix, local_metadata):
        nonlocal armed
        if armed and failure_stage == "local_writeback":
            armed = False
            raise failure

    def synchronize_error(error):
        nonlocal armed
        if armed and failure_stage == "peer":
            armed = False
            raise failure

    try:
        with enable_weight_access_and_writeback(model, model, writeback=False):
            fields = policy.prepare_plan(plan, 0)
        with torch.no_grad():
            for field in fields:
                field.add_(0.25)
        inputs = torch.randn(3, 256)
        expected = [owner(inputs).detach() for owner in model]
        native = [owner.weight_quantizer._local_repair_state.native_payload() for owner in model]
        devices = {name: value.device for name, value in model.state_dict().items()}
        handle = model[0].register_state_dict_post_hook(fail_writeback)
        try:
            with pytest.raises(RuntimeError) as error:
                policy.finalize_(weight_access=weight_access, synchronize_error=synchronize_error)
            assert error.value is failure
        finally:
            handle.remove()
        assert {name: value.device for name, value in model.state_dict().items()} == devices
        assert all(owner._hf_hook.offload for owner in model)
        assert all(hasattr(owner.weight_quantizer, "_local_repair_state") for owner in model)
        assert all(owner.weight_quantizer._enable_pre_quant_scale for owner in model)
        assert backing.keys() == original_backing.keys()
        for name, value in original_backing.items():
            assert backing[name] is value
        for name, value in original.items():
            torch.testing.assert_close(dataset[name], value, check_dtype=False, rtol=0, atol=0)
        with enable_weight_access_and_writeback(model, model, writeback=False):
            state = model.state_dict()
            assert state.keys() == original.keys()
            for name, value in original.items():
                torch.testing.assert_close(state[name], value, rtol=0, atol=0)
        for owner, output in zip(model, expected, strict=True):
            torch.testing.assert_close(owner(inputs), output, rtol=0, atol=0)
        policy.finalize_(weight_access=weight_access)
        with enable_weight_access_and_writeback(model, model, writeback=False):
            for owner, payload in zip(model, native, strict=True):
                torch.testing.assert_close(
                    pinned_packed_weight(owner.weight_quantizer, "iq1_m"), payload, rtol=0, atol=0
                )
        for owner, output in zip(model, expected, strict=True):
            torch.testing.assert_close(owner(inputs), output, rtol=0, atol=0)
        assert all(not owner.weight_quantizer._enable_pre_quant_scale for owner in model)
        assert all(not hasattr(owner.weight_quantizer, "_local_repair_state") for owner in model)
        assert all(parameter.requires_grad for parameter in model.parameters())
    finally:
        policy.close()


@pytest.mark.parametrize(
    "mapping_mode",
    [
        "shared_userdict",
        "independent_userdict",
        "unknown",
        "custom_userdict",
        "custom_dict",
        "conflicting_hooks",
        "same_backing_hooks",
    ],
)
def test_iq_plan_checks_resident_offload_mapping_backings(mapping_mode):
    model = nn.ModuleList([nn.Linear(256, 2, bias=False) for _ in range(2)])
    mtq.quantize(
        model,
        {
            "quant_cfg": [
                {"quantizer_name": "*", "enable": False},
                {
                    "quantizer_name": "*weight_quantizer",
                    "cfg": {"num_bits": "iq1_m", "backend": "ggml"},
                },
            ],
            "algorithm": None,
        },
    )

    class LazyWeights(Mapping):
        def __getitem__(self, key):
            pytest.fail("Ownership inspection read an unknown lazy backing")

        def __iter__(self):
            return iter(("weight",))

        def __len__(self):
            return 1

    class RedirectedWeights(UserDict):
        def __getitem__(self, key):
            return self.data["cpu." + key]

        def __contains__(self, key):
            return "cpu." + key in self.data

    class RedirectedDict(dict):
        def __getitem__(self, key):
            return super().__getitem__("cpu." + key)

        def __contains__(self, key):
            return super().__contains__("cpu." + key)

    shared = model[0].weight.detach()
    mappings = [UserDict(owner.state_dict()) for owner in model]
    if mapping_mode in ("conflicting_hooks", "same_backing_hooks"):
        parent_weights = {
            f"{index}.{name}": value.clone() if mapping_mode == "conflicting_hooks" else value
            for index, weights in enumerate(mappings)
            for name, value in weights.items()
        }
        add_hook_to_module(
            model,
            AlignDevicesHook(
                execution_device="cpu",
                offload=True,
                place_submodules=True,
                weights_map=parent_weights,
            ),
        )
    for owner, weights in zip(model, mappings, strict=True):
        if mapping_mode == "shared_userdict":
            weights["weight"] = shared
        if mapping_mode in ("custom_userdict", "custom_dict"):
            mapping_class = (
                RedirectedWeights if mapping_mode == "custom_userdict" else RedirectedDict
            )
            weights = mapping_class(
                {"cpu." + name: value.double() for name, value in weights.items()}
            )
        add_hook_to_module(
            owner,
            AlignDevicesHook(
                execution_device="cpu",
                offload=True,
                weights_map=LazyWeights() if mapping_mode == "unknown" else weights,
            ),
        )
    if mapping_mode in ("independent_userdict", "same_backing_hooks"):
        plan_iq_repair([model[0]], root_model=model)
    else:
        with pytest.raises(ValueError):
            plan_iq_repair([model[0]], root_model=model)
        if mapping_mode in ("custom_userdict", "custom_dict"):
            with pytest.raises(ValueError), enable_weight_access_and_writeback(model, model):
                pytest.fail("Weight access accepted a mapping with uninspectable lookup semantics")


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


def test_weight_access_rollback_snapshots_are_opt_in():
    model = nn.Linear(256, 256, bias=False)
    original = model.weight.detach().clone()
    cpu_offload(model, execution_device="cpu")
    for rollback in (False, True):
        with (
            torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU]) as profile,
            enable_weight_access_and_writeback(
                model, model, **({"rollback": True} if rollback else {})
            ) as undo,
            torch.no_grad(),
        ):
            model.weight.add_(1)
        clones = sum(event.count for event in profile.key_averages() if event.key == "aten::clone")
        assert bool(clones) == rollback
        if rollback:
            undo()
        with enable_weight_access_and_writeback(model, model, writeback=False):
            torch.testing.assert_close(model.weight, original + 1, rtol=0, atol=0)
