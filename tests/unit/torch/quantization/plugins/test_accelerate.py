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

import gc
import pickle
import threading
import weakref
from collections import OrderedDict, UserDict
from collections.abc import Mapping
from contextlib import contextmanager, nullcontext

import pytest
import torch
import torch.nn as nn

import modelopt.torch.quantization as mtq
from modelopt.torch.quantization.ggml import GGML_FORMAT_REGISTRY
from modelopt.torch.quantization.ggml.common import pin_packed_weight, pinned_packed_weight
from modelopt.torch.quantization.nn import TensorQuantizer
from modelopt.torch.quantization.nn.modules.quant_linear import RealQuantLinear
from modelopt.torch.quantization.qtensor import INT4QTensor, QTensorWrapper
from modelopt.torch.quantization.repair import IQRepairPolicy, plan_iq_repair
from modelopt.torch.quantization.utils.core_utils import enable_weight_access_and_writeback

try:
    from accelerate import cpu_offload, disk_offload
    from accelerate import hooks as accelerate_hooks
    from accelerate.hooks import AlignDevicesHook, ModelHook, add_hook_to_module
    from accelerate.utils import set_module_tensor_to_device

    from modelopt.torch.quantization.plugins import accelerate as accelerate_plugin
    from modelopt.torch.quantization.plugins.accelerate import (
        weight_access_and_writeback_context as offload_weight_access,
    )
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
        offload_weight_access(model, writeback=False, rollback=True),
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
        with pytest.raises(KeyError) as error, offload_weight_access(model):
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

    with offload_weight_access(model):
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
@pytest.mark.parametrize(
    "failure_stage",
    [
        "body",
        "writeback",
        "cleanup",
        "partial_cleanup",
        "scope_before",
        "scope_after",
        "body_scope",
        "scope_restore",
        "scope_restore_intervening",
        "scope_restore_rejected_entry",
    ],
)
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
    retry_scope = failure_stage.startswith("scope_restore")
    if retry_scope and offload_mode == "disk_offload":
        backing["0.weight"] = resident_weight.detach()
        original_backing = dict(backing)
    restore_target = backing["0.weight"] if retry_scope else None
    sibling = nn.Linear(16, 16, bias=False)
    mtq.replace_quant_module(sibling)
    RealQuantLinear._setup(sibling)
    packed = INT4QTensor(
        sibling.weight.shape, sibling.weight.dtype, torch.zeros((16, 8), dtype=torch.uint8)
    )
    sibling.weight = QTensorWrapper(packed)
    registry = sibling._parameters
    root = nn.ModuleDict({"layers": model, "packed_sibling": sibling})
    devices = {name: value.device for name, value in model.state_dict().items()}
    failure = RuntimeError("late writeback failure")
    scope_failure = RuntimeError("receipt scope inspection failed")
    scope_failure_stage = "scope" in failure_stage
    rollback_scope = accelerate_plugin._rollback_scope
    if scope_failure_stage:
        pending = nn.Linear(4, 4, bias=False)
        cpu_offload(pending, execution_device="cpu")
        with offload_weight_access(pending, rollback=True) as pending_undo:
            pass

    def fail_writeback(module, state_dict, prefix, local_metadata):
        if failure_stage == "writeback":
            raise failure

    handle = model[1].register_state_dict_post_hook(fail_writeback)
    post_forward = None
    set_tensor = accelerate_hooks.set_module_tensor_to_device
    armed = True
    scans = 0
    restores = 0
    copy_tensor = torch.Tensor.copy_

    def fail_first_restore(tensor, source, *args, **kwargs):
        nonlocal restores
        if tensor is restore_target:
            restores += 1
            if restores == 1:
                raise RuntimeError("one-shot snapshot restoration failure")
        return copy_tensor(tensor, source, *args, **kwargs)

    def fail_exit_scope(*args, **kwargs):
        nonlocal scans
        scans += 1
        if (
            failure_stage == "body_scope"
            or (failure_stage == "scope_before" and scans == 1)
            or (failure_stage == "scope_after" and scans == 2)
            or (retry_scope and scans == 2)
        ):
            raise scope_failure
        return rollback_scope(*args, **kwargs)

    def fail_before_last_tensor(owner, name, device, *args, **kwargs):
        nonlocal armed
        if owner is model[1] and name == "scale" and device == "meta" and armed:
            armed = False
            raise failure
        return set_tensor(owner, name, device, *args, **kwargs)

    try:
        with (
            pytest.raises(RuntimeError) as error,
            offload_weight_access(root, rollback=True) as undo,
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
            if scope_failure_stage:
                monkeypatch.setattr(accelerate_plugin, "_rollback_scope", fail_exit_scope)
            if retry_scope:
                monkeypatch.setattr(torch.Tensor, "copy_", fail_first_restore)
            if failure_stage in ("body", "body_scope"):
                raise failure
        assert error.value is (
            scope_failure
            if failure_stage in ("scope_before", "scope_after") or retry_scope
            else failure
        )
        if failure_stage == "body_scope":
            assert any("receipt scope inspection failed" in note for note in error.value.__notes__)
    finally:
        handle.remove()
        if post_forward is not None:
            hooks[1].post_forward = post_forward
        monkeypatch.setattr(accelerate_hooks, "set_module_tensor_to_device", set_tensor)
        monkeypatch.setattr(accelerate_plugin, "_rollback_scope", rollback_scope)

    if scope_failure_stage:
        with pytest.raises(RuntimeError):
            pending_undo()
    if retry_scope:
        assert restores == 1
        torch.testing.assert_close(restore_target, originals["0.weight"] + 1, rtol=0, atol=0)
        if failure_stage in ("scope_restore_intervening", "scope_restore_rejected_entry"):
            if failure_stage == "scope_restore_rejected_entry":
                pending_parameter = pending.weight

                def refuse_entry_scope(*args, **kwargs):
                    raise RuntimeError("refused entry scope inspection")

                with monkeypatch.context() as entry_patch:
                    entry_patch.setattr(accelerate_plugin, "_rollback_scope", refuse_entry_scope)
                    with (
                        pytest.raises(RuntimeError, match="refused entry"),
                        offload_weight_access(pending, writeback=False),
                    ):
                        pytest.fail("A failed entry scan exposed offloaded weights")
                assert pending.weight is pending_parameter and pending.weight.is_meta
            else:
                unrelated = nn.Linear(4, 4, bias=False)
                with enable_weight_access_and_writeback(unrelated, unrelated, writeback=False):
                    pass
            with pytest.raises(RuntimeError):
                undo()
            assert restores == 1
            torch.testing.assert_close(restore_target, originals["0.weight"] + 1, rtol=0, atol=0)
            copy_tensor(restore_target, originals["0.weight"])
        else:
            undo()
            assert restores == 2
            torch.testing.assert_close(restore_target, originals["0.weight"], rtol=0, atol=0)
            undo()
            assert restores == 2

    assert sibling._parameters is registry
    metadata = sibling.weight.metadata
    sibling.weight = nn.Parameter(sibling.weight.detach().clone(), requires_grad=False)
    assert isinstance(sibling.weight, QTensorWrapper)
    assert sibling.weight.metadata == metadata
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

    with offload_weight_access(model):
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


@pytest.mark.parametrize("writeback", [False, True])
@pytest.mark.parametrize("failed", [False, True])
@pytest.mark.parametrize("rollback", [False, True])
def test_weight_access_rollback_snapshots_are_opt_in(writeback, failed, rollback):
    model = nn.Linear(4, 4, bias=False)
    original = model.weight.detach().clone()
    weights = {"weight": model.weight.detach()}
    backing = weights["weight"]
    cpu_offload(model, execution_device="cpu", state_dict=weights)
    registry, parameter = model._parameters, model.weight
    with (
        pytest.raises(RuntimeError, match="body failed") if failed else nullcontext(),
        enable_weight_access_and_writeback(
            model, model, writeback=writeback, **({"rollback": True} if rollback else {})
        ) as undo,
        torch.no_grad(),
    ):
        assert model.weight.data_ptr() == backing.data_ptr()
        model.weight.add_(1)
        if failed:
            raise RuntimeError("body failed after an alias write")

    expected = original if failed and rollback else original + 1
    torch.testing.assert_close(weights["weight"], expected, rtol=0, atol=0)
    if rollback:
        # Success commits first; either explicit undo or failure restores the original alias.
        undo()
        expected = original
        assert model._parameters is registry and model.weight is parameter
        assert weights["weight"] is backing
        torch.testing.assert_close(backing, original, rtol=0, atol=0)
        undo()
        assert model._parameters is registry and model.weight is parameter
        assert weights["weight"] is backing
        torch.testing.assert_close(backing, original, rtol=0, atol=0)
    else:
        assert undo is None
    inputs = torch.ones(2, 4)
    with enable_weight_access_and_writeback(model, model, writeback=False):
        torch.testing.assert_close(model.weight, expected, rtol=0, atol=0)
        torch.testing.assert_close(model(inputs), inputs @ expected.T, rtol=0, atol=0)


@pytest.mark.parametrize("failed", [False, True])
@pytest.mark.parametrize("registry_type", [dict, OrderedDict])
def test_weight_access_rollback_preserves_buffer_registry(failed, registry_type):
    class BufferRegistry(registry_type):
        pass

    model = nn.Linear(4, 4, bias=False)
    model.register_buffer("scale", torch.ones(4))
    model.register_buffer("excluded", torch.zeros(4), persistent=False)
    weights = dict(model.state_dict())
    originals = {name: value.clone() for name, value in weights.items()}
    cpu_offload(model, execution_device="cpu", offload_buffers=True, state_dict=weights)
    registry = BufferRegistry(model._buffers)
    registry.metadata = {"owner": object()}
    metadata = registry.metadata
    model._buffers = registry
    buffers = dict(registry)
    persistence = model._non_persistent_buffers_set
    exclusions = persistence.copy()
    with (
        pytest.raises(RuntimeError, match="body failed") if failed else nullcontext(),
        offload_weight_access(model, rollback=True) as undo,
        torch.no_grad(),
    ):
        model.scale.add_(2)
        model.register_buffer("scale", model.scale + 3)
        model.register_buffer("scratch", torch.full((4,), 7.0), persistent=False)
        del model.excluded
        if failed:
            raise RuntimeError("body failed after buffer registration")
    if not failed:
        torch.testing.assert_close(weights["scale"], originals["scale"] + 5)
        undo()
    assert model._non_persistent_buffers_set is persistence
    assert persistence == exclusions
    assert model._buffers is registry
    assert isinstance(model._buffers, BufferRegistry) and registry.metadata is metadata
    assert registry.keys() == buffers.keys()
    assert list(registry) == list(buffers)
    assert all(registry[name] is tensor for name, tensor in buffers.items())
    assert list(model.state_dict()) == list(originals)
    for name, value in originals.items():
        torch.testing.assert_close(weights[name], value, rtol=0, atol=0)
    with offload_weight_access(model, writeback=False):
        torch.testing.assert_close(model.scale, originals["scale"], rtol=0, atol=0)


@pytest.mark.parametrize("intervening", [False, True])
def test_weight_access_failed_rollback_can_retry_before_becoming_idempotent(
    monkeypatch, intervening
):
    model = nn.Linear(4, 4, bias=False)
    original = model.weight.detach().clone()
    weights = {"weight": model.weight.detach()}
    backing = weights["weight"]
    cpu_offload(model, execution_device="cpu", state_dict=weights)
    registry, parameter = model._parameters, model.weight
    with offload_weight_access(model, rollback=True) as undo, torch.no_grad():
        model.weight.add_(1)
    torch.testing.assert_close(weights["weight"], original + 1, rtol=0, atol=0)

    copy_tensor = torch.Tensor.copy_
    restores = 0
    failure = RuntimeError("one-shot snapshot restore failure")

    def fail_first_restore(tensor, source, *args, **kwargs):
        nonlocal restores
        if tensor is backing:
            restores += 1
            if restores == 1:
                raise failure
        return copy_tensor(tensor, source, *args, **kwargs)

    monkeypatch.setattr(torch.Tensor, "copy_", fail_first_restore)
    with pytest.raises(RuntimeError) as error:
        undo()
    assert error.value is failure
    torch.testing.assert_close(backing, original + 1, rtol=0, atol=0)
    if intervening:
        with offload_weight_access(model, writeback=False):
            pass
        try:
            with pytest.raises(RuntimeError):
                undo()
        finally:
            torch.testing.assert_close(backing, original + 1, rtol=0, atol=0)
        assert restores == 1
        return
    undo()
    assert restores == 2
    assert weights["weight"] is backing
    assert model._parameters is registry and model.weight is parameter
    torch.testing.assert_close(backing, original, rtol=0, atol=0)
    undo()
    assert restores == 2
    assert weights["weight"] is backing
    assert model._parameters is registry and model.weight is parameter
    with offload_weight_access(model, writeback=False):
        torch.testing.assert_close(model.weight, original, rtol=0, atol=0)


@pytest.mark.parametrize(
    ("overlap", "later_mode"),
    [
        (overlap, later_mode)
        for overlap in (
            "same_owner",
            "shared_mapping",
            "shared_storage",
            "shared_ties",
            "independent",
            "resident_storage",
            "resident_independent",
        )
        for later_mode in ("write", "read_only", "failed")
    ]
    + [
        ("redirected_dict", "read_only"),
        ("opaque_userdict", "read_only"),
        ("external_dict", "read_only"),
        ("resident_storage", "late_alias"),
        ("resident_storage", "nested_receipt"),
        ("shared_mapping", "rejected_entry"),
        ("shared_storage", "rejected_mapping"),
        ("same_owner", "empty_mapping"),
        ("shared_mapping", "empty_mapping"),
        ("shared_storage", "concurrent_undo"),
        ("resident_storage", "concurrent_undo"),
    ],
)
def test_weight_access_undo_expires_before_later_overlapping_access(
    overlap, later_mode, monkeypatch
):
    redirected = overlap in ("redirected_dict", "opaque_userdict", "external_dict")
    shared_storage = overlap in ("shared_storage", "resident_storage") or redirected
    independent = overlap in ("independent", "resident_independent")
    resident = overlap in ("resident_storage", "resident_independent")
    empty_mapping = later_mode == "empty_mapping"
    rejected = later_mode in ("rejected_entry", "rejected_mapping") or empty_mapping
    first = nn.Linear(4, 4, bias=False)
    if independent:
        first.register_buffer("empty", torch.empty(0), persistent=False)
    original = first.weight.detach().clone()
    first_weights = {"weight": first.weight.detach()}
    ties = {}
    if overlap == "shared_ties":
        add_hook_to_module(
            first,
            AlignDevicesHook(
                execution_device="cpu",
                offload=True,
                weights_map=first_weights,
                tied_params_map=ties,
            ),
        )
    else:
        cpu_offload(first, execution_device="cpu", state_dict=first_weights)
    with offload_weight_access(first, rollback=True) as undo, torch.no_grad():
        first.weight.add_(1)

    if overlap == "same_owner":
        later, later_weights = first, first_weights
    else:
        later = nn.Linear(4, 4, bias=False)
        if independent:
            later.register_buffer("empty", torch.empty(0), persistent=False)
        later_weights = (
            first_weights
            if overlap == "shared_mapping"
            else {
                "weight": first_weights["weight"].view(4, 4)
                if shared_storage and later_mode != "late_alias"
                else first_weights["weight"].clone()
            }
        )
        if shared_storage and later_mode in ("read_only", "rejected_mapping") and not resident:

            class NativeWeights(dict):
                def get(self, *args, **kwargs):
                    pytest.fail("Scope discovery called a dict-subclass get override")

            later_weights = NativeWeights(later_weights)
            if overlap == "external_dict":

                class ExternalWeights(NativeWeights):
                    def __getitem__(self, key):
                        return self.tensor

                external = ExternalWeights({"weight": later_weights["weight"].clone()})
                external.tensor = later_weights["weight"]
                later_weights = external
            elif redirected:

                class RedirectedWeights(
                    UserDict if overlap == "opaque_userdict" else NativeWeights
                ):
                    def __getitem__(self, key):
                        return super().__getitem__("cpu." + key)

                    def __contains__(self, key):
                        return super().__contains__("cpu." + key)

                later_weights = RedirectedWeights({"cpu.weight": later_weights["weight"]})
        if resident:
            later.weight = nn.Parameter(later_weights["weight"])
        elif overlap == "shared_ties":
            add_hook_to_module(
                later,
                AlignDevicesHook(
                    execution_device="cpu",
                    offload=True,
                    weights_map=later_weights,
                    tied_params_map=ties,
                ),
            )
        else:
            cpu_offload(later, execution_device="cpu", state_dict=later_weights)
    if later_mode == "concurrent_undo":
        replaying, release = threading.Event(), threading.Event()
        copy_tensor = torch.Tensor.copy_
        errors = []

        def paused_copy(tensor, source, *args, **kwargs):
            if tensor is first_weights["weight"] and threading.current_thread() is worker:
                replaying.set()
                assert release.wait(5), "Timed out waiting to release undo replay"
            return copy_tensor(tensor, source, *args, **kwargs)

        def replay():
            try:
                undo()
            except BaseException as exc:
                errors.append(exc)

        monkeypatch.setattr(torch.Tensor, "copy_", paused_copy)
        worker = threading.Thread(target=replay)
        worker.start()
        try:
            assert replaying.wait(5), "Undo did not reach snapshot replay"
            with pytest.raises(RuntimeError, match="serial"):
                undo()
            with (
                pytest.raises(RuntimeError, match="serial"),
                enable_weight_access_and_writeback(later, later)
                if resident
                else offload_weight_access(later),
                torch.no_grad(),
            ):
                later.weight.add_(2)
        finally:
            release.set()
            worker.join(5)
        assert not worker.is_alive()
        assert not errors
        torch.testing.assert_close(first_weights["weight"], original, rtol=0, atol=0)
        undo()  # Completed replay remains idempotent after the guard is released.
        return
    if later_mode == "rejected_entry":
        later.bias = nn.Parameter(torch.zeros(4))
    if empty_mapping:
        saved_weights = later_weights.copy()
        later_weights.clear()
    with (
        pytest.raises(AssertionError)
        if empty_mapping
        else pytest.raises(ValueError)
        if rejected
        else pytest.raises(RuntimeError)
        if later_mode == "failed"
        else nullcontext(),
        enable_weight_access_and_writeback(later, later, writeback=later_mode != "read_only")
        if resident
        else offload_weight_access(
            later, writeback=later_mode != "read_only", rollback=later_mode == "rejected_entry"
        ),
        torch.no_grad(),
    ):
        assert not rejected, "Preflight accepted the rejected owner state or mapping"
        if later_mode == "late_alias":
            later_weights["weight"] = first_weights["weight"].view(4, 4)
            later.weight = nn.Parameter(later_weights["weight"])
        if later_mode == "nested_receipt":
            with offload_weight_access(first, rollback=True) as undo:
                first.weight.add_(3)
        if shared_storage and later_mode == "read_only":
            try:
                with pytest.raises(RuntimeError):
                    undo()
            finally:
                torch.testing.assert_close(later.weight, original + 1, rtol=0, atol=0)
        if later_mode != "read_only" or redirected:
            later.weight.add_(2)
        if overlap == "shared_ties":
            ties[later.weight.data_ptr()] = {"cpu": later.weight}
        if later_mode == "failed":
            raise RuntimeError("later body failed")
    if empty_mapping:
        later_weights.update(saved_weights)
    if rejected:
        later_weights["weight"].add_(2)
    expected = original + 1
    if later_mode == "nested_receipt":
        expected = expected + 3
    if later_mode != "read_only" or redirected:
        expected = expected + 2
    torch.testing.assert_close(later_weights["weight"], expected, rtol=0, atol=0)
    registry, parameter = later._parameters, later.weight
    backing = later_weights["weight"]
    tied_state = {pointer: entries.copy() for pointer, entries in ties.items()}
    try:
        with nullcontext() if independent else pytest.raises(RuntimeError):
            undo()
    finally:
        # Even the refusal must precede any journal replay or alias copy.
        torch.testing.assert_close(later_weights["weight"], expected, rtol=0, atol=0)
        assert later_weights["weight"] is backing
        assert later._parameters is registry and later.weight is parameter
        assert ties.keys() == tied_state.keys()
        assert all(
            ties[pointer][device] is tensor
            for pointer, entries in tied_state.items()
            for device, tensor in entries.items()
        )
    if independent:
        torch.testing.assert_close(first_weights["weight"], original, rtol=0, atol=0)


@pytest.mark.parametrize("stage", ["while_open", "outer_publication"])
def test_weight_access_undo_is_valid_only_after_its_own_exit(stage):
    model = nn.Linear(4, 4, bias=False)
    original = model.weight.detach().clone()
    weights = {"weight": model.weight.detach()}
    cpu_offload(model, execution_device="cpu", state_dict=weights)
    with (
        offload_weight_access(model) if stage == "outer_publication" else nullcontext(),
        torch.no_grad(),
    ):
        with offload_weight_access(model, rollback=True) as undo:
            model.weight.add_(1)
            if stage == "while_open":
                try:
                    with pytest.raises(RuntimeError):
                        undo()
                finally:
                    torch.testing.assert_close(weights["weight"], original + 1, rtol=0, atol=0)
        if stage == "while_open":
            undo()
    expected = original if stage == "while_open" else original + 1
    try:
        with pytest.raises(RuntimeError) if stage == "outer_publication" else nullcontext():
            undo()
    finally:
        torch.testing.assert_close(weights["weight"], expected, rtol=0, atol=0)


def test_weight_access_undo_tracking_does_not_retain_abandoned_models():
    model = nn.Linear(4, 4, bias=False)
    cpu_offload(model, execution_device="cpu")
    with offload_weight_access(model, rollback=True) as undo:
        pass
    owner_ref, receipt_ref = weakref.ref(model), weakref.ref(undo)
    del model, undo
    assert receipt_ref() is None
    gc.collect()
    assert owner_ref() is None and receipt_ref() is None


@pytest.mark.parametrize("mapping_mode", ["unknown", "custom_dict", "custom_userdict"])
@pytest.mark.parametrize("writeback", [False, True])
def test_weight_access_refuses_custom_mapping_before_onload(mapping_mode, writeback, monkeypatch):
    model = nn.Linear(4, 4, bias=False)

    class LazyWeights(Mapping):
        def __getitem__(self, key):
            pytest.fail("Preflight read an unknown lazy mapping")

        def __iter__(self):
            return iter(("weight",))

        def __len__(self):
            return 1

    class RedirectedDict(dict):
        def __getitem__(self, key):
            return super().__getitem__("cpu." + key)

        def __contains__(self, key):
            return super().__contains__("cpu." + key)

    class RedirectedWeights(UserDict):
        def __getitem__(self, key):
            return self.data["cpu." + key]

        def __contains__(self, key):
            return "cpu." + key in self.data

    mapping = (
        LazyWeights()
        if mapping_mode == "unknown"
        else (RedirectedDict if mapping_mode == "custom_dict" else RedirectedWeights)(
            {"cpu.weight": model.weight.detach().double()}
        )
    )
    add_hook_to_module(
        model, AlignDevicesHook(execution_device="cpu", offload=True, weights_map=mapping)
    )
    hook, parameter, registry = model._hf_hook, model.weight, model._parameters
    onloads = []
    pre_forward = hook.pre_forward

    def observe_onload(*args, **kwargs):
        onloads.append(True)
        return pre_forward(*args, **kwargs)

    monkeypatch.setattr(hook, "pre_forward", observe_onload)
    with (
        pytest.raises(ValueError),
        offload_weight_access(model, writeback=writeback, rollback=not writeback),
    ):
        pytest.fail("Uninspectable mapping yielded")
    assert not onloads and model.weight is parameter and model._parameters is registry
    assert model.weight.is_meta and hook.offload


@pytest.mark.parametrize("buffer_only", [False, True, "resident_child"])
def test_weight_access_dispatches_hookless_composite(buffer_only):
    model = nn.Sequential(nn.Linear(4, 4, bias=False), nn.Linear(4, 4, bias=False))
    if buffer_only:
        for owner in model:
            weight = owner.weight.detach()
            del owner.weight
            owner.register_buffer("weight", weight)
    expected = [owner.weight.detach().clone() + 1 for owner in model]
    for owner in model:
        cpu_offload(owner, execution_device="cpu", offload_buffers=bool(buffer_only))
        if buffer_only == "resident_child":
            owner.resident = nn.Linear(4, 4, bias=False)
    assert not hasattr(model, "_hf_hook")
    with enable_weight_access_and_writeback(model, model), torch.no_grad():
        assert all(not owner.weight.is_meta for owner in model)
        for owner in model:
            owner.weight.add_(1)
    assert all(owner.weight.is_meta for owner in model)
    with enable_weight_access_and_writeback(model, model, writeback=False):
        for owner, weight in zip(model, expected, strict=True):
            torch.testing.assert_close(owner.weight, weight, rtol=0, atol=0)
    assert all(owner.weight.is_meta for owner in model)


@pytest.mark.parametrize("later_failure", [False, True])
def test_weight_access_rollback_preserves_values_before_onload(later_failure):
    class MutatingOnload(AlignDevicesHook):
        def pre_forward(self, module, *args, **kwargs):
            result = super().pre_forward(module, *args, **kwargs)
            with torch.no_grad():
                module.weight.add_(3)
            return result

    model = nn.Sequential(nn.Linear(4, 4, bias=False), nn.Linear(4, 4, bias=False))
    model[1].register_buffer("scale", torch.ones(4))
    backings = [dict(owner.state_dict()) for owner in model]
    originals = [backing["weight"].clone() for backing in backings]
    for index, owner in enumerate(model):
        hook_class = AlignDevicesHook if index else MutatingOnload
        add_hook_to_module(
            owner, hook_class(execution_device="cpu", offload=True, weights_map=backings[index])
        )
    if later_failure:
        missing = backings[1].pop("weight")
        with pytest.raises(KeyError), offload_weight_access(model, rollback=True):
            pytest.fail("Incomplete onload yielded")
        torch.testing.assert_close(backings[0]["weight"], originals[0], rtol=0, atol=0)
        assert all(owner.weight.is_meta and owner._hf_hook.offload for owner in model)
        backings[1]["weight"] = missing
    with offload_weight_access(model, rollback=True) as undo:
        torch.testing.assert_close(model[0].weight, originals[0] + 3, rtol=0, atol=0)
    undo()
    undo()
    for owner, backing, original in zip(model, backings, originals, strict=True):
        torch.testing.assert_close(backing["weight"], original, rtol=0, atol=0)
        assert owner.weight.is_meta and owner._hf_hook.offload


@pytest.mark.parametrize("inner_composite", [False, True])
def test_weight_access_dispatches_nested_active_accelerate(inner_composite):
    owner = nn.Linear(4, 4, bias=False)
    model = nn.Sequential(owner)
    cpu_offload(owner, execution_device="cpu")
    with enable_weight_access_and_writeback(model, model), torch.no_grad():
        owner.weight.add_(2)
        original = owner.weight
        expected = original.clone()
        inner = model if inner_composite else owner
        with enable_weight_access_and_writeback(inner, model, rollback=True) as undo:
            assert callable(undo)
            assert owner.weight is original
            torch.testing.assert_close(owner.weight, expected, rtol=0, atol=0)
            owner.weight = nn.Parameter(torch.zeros_like(original))
        undo()
        assert owner.weight is original
        torch.testing.assert_close(owner.weight, expected, rtol=0, atol=0)
        plain_module = nn.Linear(4, 4)
        with enable_weight_access_and_writeback(plain_module, plain_module, rollback=True) as plain:
            assert plain is None
    assert owner.weight.is_meta


def test_weight_access_dispatcher_preserves_body_error_during_receipt_inspection():
    pending = nn.Linear(4, 4, bias=False)
    cpu_offload(pending, execution_device="cpu")
    with offload_weight_access(pending, rollback=True) as undo:
        pass
    plain = nn.Module()
    failure = RuntimeError("caller failed")
    with pytest.raises(Exception) as error, enable_weight_access_and_writeback(plain, plain):
        plain.register_parameter("lazy", nn.parameter.UninitializedParameter())
        raise failure
    assert error.value is failure
    assert len(error.value.__notes__) >= 1
    with pytest.raises(RuntimeError):
        undo()


def test_weight_access_publishes_duplicate_hook_slots_before_parent_reload(monkeypatch):
    model = nn.Sequential(nn.Linear(4, 4, bias=False), nn.Linear(4, 4, bias=False))
    mappings = [UserDict({"weight": owner.weight.detach().double()}) for owner in model]
    parent = {f"{index}.weight": values["weight"] for index, values in enumerate(mappings)}
    add_hook_to_module(
        model,
        AlignDevicesHook(
            execution_device="cpu", offload=True, place_submodules=True, weights_map=parent
        ),
    )
    for owner, mapping in zip(model, mappings, strict=True):
        add_hook_to_module(
            owner, AlignDevicesHook(execution_device="cpu", offload=True, weights_map=mapping)
        )
    onloads = []
    for owner in model.modules():
        hook = owner._hf_hook
        pre_forward = hook.pre_forward

        def observe_onload(mod, *args, _pre=pre_forward, **kwargs):
            onloads.append(mod)
            return _pre(mod, *args, **kwargs)

        monkeypatch.setattr(hook, "pre_forward", observe_onload)
    inputs = torch.randn(2, 4)
    originals = [mapping["weight"].clone() for mapping in mappings]
    with offload_weight_access(model), torch.no_grad():
        assert all(owner.weight.dtype == torch.float32 for owner in model)
        model[0].weight.add_(1)
        model[1].weight.sub_(2)
        assert onloads == [model]
        expected = model(inputs).detach()
    for index, delta in enumerate((1, -2)):
        torch.testing.assert_close(mappings[index]["weight"], originals[index] + delta)
        torch.testing.assert_close(parent[f"{index}.weight"], originals[index] + delta)
        assert mappings[index]["weight"].dtype == parent[f"{index}.weight"].dtype == torch.float64
    model._hf_hook.pre_forward(model)
    try:
        actual = inputs @ model[0].weight.T @ model[1].weight.T
        torch.testing.assert_close(actual, expected)
    finally:
        model._hf_hook.post_forward(model, None)
    torch.testing.assert_close(model(inputs), expected)


@pytest.mark.parametrize("writeback", [False, True])
@pytest.mark.parametrize("rollback", [False, True])
@pytest.mark.parametrize("resident_field", ["buffer", "weight"])
def test_weight_access_mixed_quantizer_buffer_keeps_default_export_path(
    writeback, rollback, resident_field
):
    model = nn.Linear(4, 4, bias=False)
    mtq.replace_quant_module(model)
    for quantizer in model.modules():
        if isinstance(quantizer, TensorQuantizer):
            quantizer.disable()
    model.input_quantizer._enable_pre_quant_scale = True
    model.input_quantizer.pre_quant_scale = torch.ones(4)
    weights = dict(model.state_dict())
    expected_weight = weights["weight"].clone() + (resident_field == "weight")
    add_hook_to_module(
        model,
        AlignDevicesHook(
            execution_device="cpu",
            offload=True,
            offload_buffers=True,
            place_submodules=True,
            weights_map=weights,
        ),
    )
    field = "input_quantizer._pre_quant_scale" if resident_field == "buffer" else "weight"
    set_module_tensor_to_device(
        model,
        field,
        "cpu",
        value=weights[field].clone() if resident_field == "buffer" else expected_weight,
    )
    resident = model.input_quantizer.pre_quant_scale if resident_field == "buffer" else model.weight
    with (
        pytest.raises(ValueError) if rollback else nullcontext(),
        enable_weight_access_and_writeback(model, model, writeback=writeback, rollback=rollback),
    ):
        torch.testing.assert_close(model.weight, expected_weight, rtol=0, atol=0)
        if resident_field == "buffer":
            torch.testing.assert_close(model.input_quantizer.pre_quant_scale, torch.ones(4))
            inputs = torch.ones(2, 4)
            torch.testing.assert_close(
                model(inputs), torch.nn.functional.linear(inputs, expected_weight)
            )
        else:
            assert model.input_quantizer.pre_quant_scale.is_meta
    if rollback:
        if resident_field == "buffer":
            assert model.weight.is_meta and model.input_quantizer.pre_quant_scale is resident
        else:
            assert model.weight is resident and model.input_quantizer.pre_quant_scale.is_meta
    elif resident_field == "weight":
        assert model.weight is resident


@pytest.mark.parametrize(
    "alias_kind",
    [
        "nonpersistent",
        "overlapping",
        "stepped",
        "strided_overlap",
        "retained",
        "duplicate_buffer",
        "duplicate_parameter",
    ],
)
def test_weight_access_rollback_checks_backing_aliases_before_body(alias_kind):
    model = nn.Linear(4, 4, bias=False)
    scale = torch.ones(4)
    if alias_kind == "duplicate_buffer":
        model.register_buffer("scratch", scale, persistent=False)
    model.register_buffer("scale", scale)
    if alias_kind == "duplicate_parameter":
        model.alias = model.weight
    weights = {name: value.clone() for name, value in model.state_dict().items()}
    if alias_kind == "duplicate_buffer":
        weights["scale"] = model.scale.detach()
    elif alias_kind == "duplicate_parameter":
        weights["weight"] = weights["weight"].double()
        weights["alias"] = model.alias.detach()
    if alias_kind == "overlapping":
        weights["scale"] = torch.ones(1).expand(4)
    elif alias_kind == "stepped":
        weights["scale"] = torch.arange(8.0)[::2]
    elif alias_kind == "strided_overlap":
        model.scale = torch.ones(2, 2)
        weights["scale"] = torch.ones(3).as_strided((2, 2), (1, 1))

    class RetainedMasterHook(AlignDevicesHook):
        def pre_forward(self, module, *args, **kwargs):
            result = super().pre_forward(module, *args, **kwargs)
            self.master = module.scale.data
            module.scale.data = self.master.clone()
            return result

        def post_forward(self, module, output):
            output = super().post_forward(module, output)
            self.master.copy_(module.scale)
            module.scale.data = self.master
            return output

    if alias_kind == "retained":
        model.scale = weights["scale"].detach()
    hook_class = RetainedMasterHook if alias_kind == "retained" else AlignDevicesHook
    add_hook_to_module(
        model,
        hook_class(
            execution_device="cpu",
            offload=True,
            offload_buffers=alias_kind
            not in ("retained", "duplicate_buffer", "duplicate_parameter"),
            weights_map=weights,
        ),
    )
    if alias_kind == "duplicate_parameter":
        model.weight = model.alias  # restore both resident registrations of the same parameter
    if alias_kind == "nonpersistent":
        # Resident managed tensors have independent storage; only scratch aliases the map.
        model.weight = nn.Parameter(weights["weight"].clone())
        model.scale = weights["scale"].clone()
        model.register_buffer("scratch", weights["scale"], persistent=False)
    backing = weights["alias" if alias_kind == "duplicate_parameter" else "scale"]
    original = backing.clone()
    if alias_kind in ("overlapping", "strided_overlap"):
        with (
            pytest.raises(ValueError, match="overlapping"),
            offload_weight_access(model, rollback=True),
        ):
            pytest.fail("Unsupported overlapping backing yielded")
    else:
        with (
            pytest.raises(RuntimeError, match="body failed"),
            offload_weight_access(model, rollback=True),
            torch.no_grad(),
        ):
            if alias_kind == "duplicate_parameter":
                model.weight.add_(5)
            elif alias_kind == "nonpersistent":
                model.scratch[0] += 5
            else:
                model.scale[0] += 5
            raise RuntimeError("body failed")
    torch.testing.assert_close(backing, original, rtol=0, atol=0)
    assert "scratch" not in weights


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
        "sequential_shared_slot",
        "resident_shared_slot",
        "resident_shared_slot_missing",
        "resident_shared_backing",
        "resident_independent",
        "active_shared_slot",
        "active_independent",
    ],
)
def test_iq_plan_checks_resident_offload_mapping_backings(mapping_mode, monkeypatch):
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
    if mapping_mode == "same_backing_hooks":
        for weights in mappings:
            weights["weight"] = weights["weight"].double()
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
    resident_values = [owner.weight.detach().clone() for owner in model]
    if mapping_mode in (
        "resident_shared_slot",
        "resident_shared_slot_missing",
        "active_shared_slot",
    ):
        mappings[1] = mappings[0]
    for index, (owner, weights) in enumerate(zip(model, mappings, strict=True)):
        if mapping_mode in ("shared_userdict", "resident_shared_backing"):
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
        if mapping_mode.startswith("resident_"):
            owner.weight = nn.Parameter(resident_values[index])
    if mapping_mode == "sequential_shared_slot":
        # The first hook is independent; the second publishes through the frozen owner.
        from accelerate.hooks import SequentialHook

        hidden = AlignDevicesHook(execution_device="cpu", offload=True, weights_map=mappings[1])
        hidden.original_devices = {"weight": torch.device("cpu")}
        model[0]._hf_hook = SequentialHook(model[0]._hf_hook, SequentialHook(hidden))
    if mapping_mode == "resident_shared_slot_missing":
        del mappings[0]["weight"]
    active = mapping_mode.startswith("active_")
    context = enable_weight_access_and_writeback(model, model) if active else nullcontext()
    with context:
        if active:
            assert all(not owner._hf_hook.offload for owner in model)
            for owner, value in zip(model, resident_values, strict=True):
                owner.weight = nn.Parameter(value)
        if mapping_mode in (
            "independent_userdict",
            "same_backing_hooks",
            "resident_independent",
            "active_independent",
        ):
            plan_iq_repair([model[0]], root_model=model)
        else:
            with pytest.raises(ValueError):
                plan_iq_repair([model[0]], root_model=model)


@pytest.mark.parametrize(
    ("offload_mode", "failure_stage"),
    [
        (mode, stage)
        for mode in ("cpu_offload", "disk_offload")
        for stage in ("local_writeback", "peer", "peer_recovery_cleanup")
    ]
    # Only resident CPU backing snapshots use copy_ during owner undo.
    + [("cpu_offload", "peer_owner_rollback")],
)
def test_iq_finalize_writeback_failure_restores_state_for_retry(
    offload_mode, failure_stage, tmp_path, monkeypatch
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

    window = 0
    events = []
    checked_originals = False
    in_owner_undo = False
    owner_undo_fault_armed = failure_stage == "peer_owner_rollback"

    @contextmanager
    def weight_access(**kwargs):
        nonlocal window, checked_originals, in_owner_undo
        window += 1
        events.append(("access", window))
        with offload_weight_access(model, **kwargs) as undo:
            if failure_stage == "peer_recovery_cleanup" and window == 4:
                torch.testing.assert_close(model.state_dict(), original, rtol=0, atol=0)
                checked_originals = True

            def owner_undo():
                nonlocal in_owner_undo
                in_owner_undo = True
                events.append(("undo_begin", owner_undo, undo))
                try:
                    undo()
                except BaseException as exc:
                    events.append(("undo_error", owner_undo, undo, str(exc)))
                    raise
                else:
                    events.append(("undo_ok", owner_undo, undo))
                finally:
                    in_owner_undo = False

            yield owner_undo if failure_stage == "peer_owner_rollback" else undo

    failure = RuntimeError("late IQ writeback failure")
    armed = True

    def fail_writeback(module, state_dict, prefix, local_metadata):
        nonlocal armed
        if armed and failure_stage == "local_writeback":
            armed = False
            raise failure

    def synchronize_error(error):
        nonlocal armed
        events.append(("sync", error))
        if armed and failure_stage.startswith("peer"):
            armed = False
            raise failure

    set_tensor = accelerate_hooks.set_module_tensor_to_device
    recovery_fault_armed = failure_stage == "peer_recovery_cleanup"
    quantizer = model[0].weight_quantizer

    def fail_partial_recovery(owner, name, device, *args, **kwargs):
        nonlocal recovery_fault_armed
        if (
            recovery_fault_armed
            and window == 2
            and owner is quantizer
            and name == "_ggml_pinned_iq1_m_source"
            and str(device) == "meta"
        ):
            recovery_fault_armed = False
            owner.register_buffer("recovery_scratch", torch.ones(1), persistent=False)
            raise RuntimeError("one-shot recovery cleanup failure")
        return set_tensor(owner, name, device, *args, **kwargs)

    copy_tensor = torch.Tensor.copy_

    def fail_owner_undo_copy(tensor, *args, **kwargs):
        nonlocal owner_undo_fault_armed
        if in_owner_undo and owner_undo_fault_armed:
            owner_undo_fault_armed = False
            raise RuntimeError("one-shot owner undo copy failure")
        return copy_tensor(tensor, *args, **kwargs)

    try:
        with offload_weight_access(model, writeback=False):
            fields = policy.prepare_plan(plan, 0)
        with torch.no_grad():
            for field in fields:
                field.add_(0.25)
        inputs = torch.randn(3, 256)
        expected = [owner(inputs).detach() for owner in model]
        native = [owner.weight_quantizer._local_repair_state.native_payload() for owner in model]
        devices = {name: value.device for name, value in model.state_dict().items()}
        registrations = {
            owner.weight_quantizer: (
                owner.weight_quantizer._buffers.copy(),
                owner.weight_quantizer._non_persistent_buffers_set.copy(),
                {name: tensor.device for name, tensor in owner.weight_quantizer._buffers.items()},
            )
            for owner in model
        }
        monkeypatch.setattr(accelerate_hooks, "set_module_tensor_to_device", fail_partial_recovery)
        monkeypatch.setattr(torch.Tensor, "copy_", fail_owner_undo_copy)
        handle = model[0].register_state_dict_post_hook(fail_writeback)
        try:
            with pytest.raises(RuntimeError) as error:
                policy.finalize_(weight_access=weight_access, synchronize_error=synchronize_error)
            assert error.value is failure
        finally:
            handle.remove()
        if failure_stage == "peer_recovery_cleanup":
            assert not recovery_fault_armed
            assert policy._pending_rollback is not None
            for quantizer, (buffers, nonpersistent, placement) in registrations.items():
                saved_buffers, saved_nonpersistent, *_ = policy._pending_rollback[1][quantizer]
                assert all(saved_buffers[name] is tensor for name, tensor in buffers.items())
                assert saved_buffers.keys() == buffers.keys()
                assert {name: tensor.device for name, tensor in saved_buffers.items()} == placement
                assert saved_nonpersistent == nonpersistent
            monkeypatch.setattr(accelerate_hooks, "set_module_tensor_to_device", set_tensor)
        if failure_stage == "peer_owner_rollback":
            assert not owner_undo_fault_armed
            assert policy._pending_rollback is not None
            callback = policy._pending_rollback[2]
            undo_error_index = next(i for i, event in enumerate(events) if event[0] == "undo_error")
            failed_undo = events[undo_error_index]
            assert failed_undo[1] is callback
            assert events[undo_error_index + 1] == (
                "sync",
                "RuntimeError: one-shot owner undo copy failure",
            )
            assert not any(event[0] == "access" for event in events[undo_error_index + 1 :])
            with pytest.raises(RuntimeError, match="Retry IQ finalization recovery"):
                policy.close()
        else:
            assert {name: value.device for name, value in model.state_dict().items()} == devices
            assert all(owner._hf_hook.offload for owner in model)
            assert all(hasattr(owner.weight_quantizer, "_local_repair_state") for owner in model)
            assert all(owner.weight_quantizer._enable_pre_quant_scale for owner in model)
            assert backing.keys() == original_backing.keys()
            for name, value in original_backing.items():
                assert backing[name] is value
            for name, value in original.items():
                torch.testing.assert_close(dataset[name], value, check_dtype=False, rtol=0, atol=0)
            with offload_weight_access(model, writeback=False):
                state = model.state_dict()
                assert state.keys() == original.keys()
                for name, value in original.items():
                    torch.testing.assert_close(state[name], value, rtol=0, atol=0)
            for owner, output in zip(model, expected, strict=True):
                torch.testing.assert_close(owner(inputs), output, rtol=0, atol=0)
        retry_start = len(events)
        policy.finalize_(
            weight_access=weight_access,
            synchronize_error=synchronize_error if failure_stage == "peer_owner_rollback" else None,
        )
        if failure_stage == "peer_owner_rollback":
            # Retry must invoke the original receipt, succeed, and agree before new access.
            assert events[retry_start] == ("undo_begin", callback, failed_undo[2])
            assert events[retry_start + 1] == ("undo_ok", callback, failed_undo[2])
            assert events[retry_start + 2] == ("sync", None)
            assert events[retry_start + 3][0] == "access"
            assert sum(event[0] == "undo_begin" for event in events) == 2
        if failure_stage == "peer_recovery_cleanup":
            assert checked_originals
            assert policy._pending_rollback is None
        with offload_weight_access(model, writeback=False):
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
