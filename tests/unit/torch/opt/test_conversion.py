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

# Unit tests for modelopt/torch/opt/conversion.py: the ModeloptStateManager contract,
# apply_mode bookkeeping, and the modelopt_state/save/restore machinery.
#
# Scope note (to avoid duplicating sibling tests): test_chaining.py covers real-mode chaining and
# in-memory modelopt_state/restore_from_modelopt_state round-trips; test_load_modelopt_state.py
# covers schema validation of corrupt state files. This file tests the manager/apply/save-restore
# contract directly with synthetic modes in a dedicated registry, plus file-based
# mto.save()/mto.restore() round-trips which no sibling exercises.

import copy
import io
import warnings

import pytest
import torch
import torch.nn as nn
from _test_utils.torch.misc import compare_outputs, set_seed
from pydantic import ValidationError

import modelopt.torch.opt as mto
import modelopt.torch.quantization as mtq  # noqa: F401  # registers the "quantize" mode
from modelopt import __version__
from modelopt.torch.opt.config import ModeloptBaseConfig, ModeloptField
from modelopt.torch.opt.conversion import ModelLikeModule, ModeloptStateManager, get_mode
from modelopt.torch.opt.mode import ModeDescriptor, _ModeRegistryCls
from modelopt.torch.utils.distributed import _serialize

TC_REGISTRY_NAME = "tc_test_conversion"


class ToyLinearModel(nn.Module):
    # same toy architecture as SimpleLinearModel in test_chaining.py
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(4, 4)

    def forward(self, x):
        return self.linear(x)


class TCConfig(ModeloptBaseConfig):
    channels: int = ModeloptField(default=8)
    label: str = ModeloptField(default="default")


def _tc_convert(model, config, **kwargs):
    # leave a trace on the model so tests can verify the entrypoint was invoked
    model._tc_convert_count = getattr(model, "_tc_convert_count", 0) + 1
    model._tc_convert_kwargs = dict(kwargs)
    return model, {"channels_at_convert": config.channels}


def _tc_restore(model, config, metadata):
    model._tc_restore_count = getattr(model, "_tc_restore_count", 0) + 1
    model._tc_restored_metadata = copy.deepcopy(metadata)
    return model


class _TCBaseDescriptor(ModeDescriptor):
    @property
    def config_class(self):
        return TCConfig

    @property
    def convert(self):
        return _tc_convert

    @property
    def restore(self):
        return _tc_restore


class TCSimpleDescriptor(_TCBaseDescriptor):
    @property
    def name(self):
        return "tc_simple"


class TCFollowerDescriptor(_TCBaseDescriptor):
    @property
    def name(self):
        return "tc_follower"


class TCStrictDescriptor(_TCBaseDescriptor):
    @property
    def name(self):
        return "tc_strict"

    @property
    def next_modes(self):
        return {"tc_follower"}


class TCProhibitDescriptor(_TCBaseDescriptor):
    @property
    def name(self):
        return "tc_prohibit"

    @property
    def next_prohibited_modes(self):
        return {"tc_simple"}


class TCExportDescriptor(_TCBaseDescriptor):
    @property
    def name(self):
        return "tc_export"

    @property
    def is_export_mode(self):
        return True


class TCNeedsExportDescriptor(_TCBaseDescriptor):
    @property
    def name(self):
        return "tc_needs_export"

    @property
    def export_mode(self):
        return "tc_export"


class TCNoSaveDescriptor(_TCBaseDescriptor):
    @property
    def name(self):
        return "tc_nosave"

    @property
    def save_mode_in_state(self):
        return False


class TCUpdateDescriptor(_TCBaseDescriptor):
    @property
    def name(self):
        return "tc_update"

    @property
    def update_for_new_mode(self):
        def _update(model, config, metadata):
            metadata["updated_for_new_mode"] = True
            config.label = "updated_for_new_mode"

        return _update

    @property
    def update_for_save(self):
        def _update(model, config, metadata):
            metadata["save_count"] = metadata.get("save_count", 0) + 1

        return _update


_TC_DESCRIPTOR_CLASSES = [
    TCSimpleDescriptor, TCFollowerDescriptor, TCStrictDescriptor, TCProhibitDescriptor,
    TCExportDescriptor, TCNeedsExportDescriptor, TCNoSaveDescriptor, TCUpdateDescriptor,
]  # fmt: skip


@pytest.fixture(scope="module")
def tc_modes():
    """Register the synthetic test modes for this module and clean them up afterwards."""
    registry = _ModeRegistryCls(TC_REGISTRY_NAME)
    for cls in _TC_DESCRIPTOR_CLASSES:
        registry.register_mode(cls)
    yield registry
    for cls in _TC_DESCRIPTOR_CLASSES:
        registry.remove_mode(cls().name)
    # Zero global residue: _ModeRegistryCls.__del__ is unreachable (the
    # class-level _all_registries list holds a strong reference), so drop
    # the emptied registry from the global list explicitly.
    _ModeRegistryCls._all_registries.remove(registry)


def _cfg(mode, **overrides):
    """Get a validated config instance for the given mode."""
    return ModeloptStateManager.get_config_class(mode, overrides)


def _mode_names(manager):
    return [m_str for m_str, _ in manager.state_dict()]


def _bumped_minor_version():
    major, minor = __version__.split(".")[:2]
    return f"{major}.{int(minor) + 1}.0"


# ---------------------------------------------------------------------------------------------
# ModeloptStateManager: construction, is_converted, remove_state
# ---------------------------------------------------------------------------------------------


def test_fake_module_manager_defaults():
    manager = ModeloptStateManager()  # no model -> fake module with fresh state
    assert not manager.has_state
    assert manager.state_dict() == []
    assert manager.last_mode is None
    assert manager.state_version == __version__


def test_init_state_on_model():
    model = ToyLinearModel()
    manager = ModeloptStateManager(model, init_state=True)
    assert ModeloptStateManager.is_converted(model)
    assert ModeloptStateManager.is_converted(model, is_root=True)
    assert not manager.has_state
    # the manager operates on the state stored on the model (shared instance)
    assert manager.state_dict() is getattr(model, ModeloptStateManager._state_key)


def test_init_state_twice_raises():
    model = ToyLinearModel()
    ModeloptStateManager(model, init_state=True)
    with pytest.raises(AssertionError, match="already has modelopt state"):
        ModeloptStateManager(model, init_state=True)


def test_manager_requires_existing_state():
    with pytest.raises(AssertionError, match="Model must have modelopt state"):
        ModeloptStateManager(ToyLinearModel(), init_state=False)


def test_is_converted_plain_model_false():
    assert not ModeloptStateManager.is_converted(ToyLinearModel())


def test_is_converted_finds_non_root_state():
    parent = nn.Sequential(nn.Linear(2, 2))
    ModeloptStateManager(parent[0], init_state=True)
    # state on a submodule counts as converted ...
    assert ModeloptStateManager.is_converted(parent)
    # ... but not as a root-level conversion
    with pytest.raises(AssertionError, match="not the root"):
        ModeloptStateManager.is_converted(parent, is_root=True)


def test_is_converted_multiple_states_raises():
    parent = nn.Sequential(nn.Linear(2, 2), nn.Linear(2, 2))
    ModeloptStateManager(parent[0], init_state=True)
    ModeloptStateManager(parent[1], init_state=True)
    with pytest.raises(AssertionError, match="multiple modelopt states"):
        ModeloptStateManager.is_converted(parent)


def test_remove_state():
    model = ToyLinearModel()
    ModeloptStateManager(model, init_state=True)
    ModeloptStateManager.remove_state(model)
    assert not ModeloptStateManager.is_converted(model)


# ---------------------------------------------------------------------------------------------
# ModeloptStateManager: add_mode bookkeeping and accessors
# ---------------------------------------------------------------------------------------------


def test_add_mode_records_state_structure(tc_modes):
    manager = ModeloptStateManager()
    manager.add_mode("tc_simple", _cfg("tc_simple", channels=4), {"foo": 1})

    assert manager.has_state
    state = manager.state_dict()
    assert len(state) == 1
    entry = state[0]
    assert isinstance(entry, tuple)
    assert entry[0] == "tc_simple"
    assert set(entry[1].keys()) == {"config", "metadata"}
    # config is stored as a plain serialized dict (model_dump), not a config instance
    assert entry[1]["config"] == {"channels": 4, "label": "default"}
    assert entry[1]["metadata"] == {"foo": 1}


def test_add_mode_preserves_order_and_last_accessors(tc_modes):
    manager = ModeloptStateManager()
    manager.add_mode("tc_simple", _cfg("tc_simple"), {"first": True})
    manager.add_mode("tc_follower", _cfg("tc_follower"), {"second": True})

    assert _mode_names(manager) == ["tc_simple", "tc_follower"]
    assert str(manager.last_mode) == "tc_follower"
    assert isinstance(manager.last_mode, TCFollowerDescriptor)
    assert manager._last_metadata == {"second": True}


def test_last_metadata_is_stored_by_reference(tc_modes):
    # update_for_new_mode/update_for_save hooks rely on in-place mutation of the stored
    # metadata, so add_mode must keep the caller-provided dict instance.
    manager = ModeloptStateManager()
    metadata = {"foo": 1}
    manager.add_mode("tc_simple", _cfg("tc_simple"), metadata)
    assert manager._last_metadata is metadata


def test_last_config_roundtrip_and_setter(tc_modes):
    manager = ModeloptStateManager()
    manager.add_mode("tc_simple", _cfg("tc_simple", channels=4), {})

    config = manager._last_config
    assert isinstance(config, TCConfig)
    assert config.channels == 4

    manager._last_config = TCConfig(channels=7)
    assert manager.state_dict()[0][1]["config"]["channels"] == 7


def test_modes_with_states_yields_registered_descriptors(tc_modes):
    manager = ModeloptStateManager()
    manager.add_mode("tc_simple", _cfg("tc_simple", channels=5), {"m": 0})
    manager.add_mode("tc_follower", _cfg("tc_follower"), {"m": 1})

    yielded = list(manager.modes_with_states())
    assert [str(m) for m, _, _ in yielded] == ["tc_simple", "tc_follower"]
    for m, config, metadata in yielded:
        assert isinstance(m, ModeDescriptor)
        assert isinstance(config, TCConfig)
        assert isinstance(metadata, dict)
    assert yielded[0][1].channels == 5
    assert yielded[1][2] == {"m": 1}


def test_add_mode_respects_next_modes(tc_modes):
    manager = ModeloptStateManager()
    manager.add_mode("tc_strict", _cfg("tc_strict"), {})
    with pytest.raises(AssertionError, match="Cannot add tc_simple after tc_strict"):
        manager.add_mode("tc_simple", _cfg("tc_simple"), {})
    # the failed add must not have been recorded
    assert _mode_names(manager) == ["tc_strict"]
    # an allowed next mode still works
    manager.add_mode("tc_follower", _cfg("tc_follower"), {})
    assert _mode_names(manager) == ["tc_strict", "tc_follower"]


def test_add_mode_respects_next_prohibited_modes(tc_modes):
    manager = ModeloptStateManager()
    manager.add_mode("tc_prohibit", _cfg("tc_prohibit"), {})
    with pytest.raises(AssertionError, match="does not allow tc_simple to be its next mode"):
        manager.add_mode("tc_simple", _cfg("tc_simple"), {})
    # any mode not on the prohibited list is fine
    manager.add_mode("tc_follower", _cfg("tc_follower"), {})
    assert _mode_names(manager) == ["tc_prohibit", "tc_follower"]


def test_export_stack_bookkeeping(tc_modes):
    manager = ModeloptStateManager()
    assert len(manager._export_stack) == 0

    manager.add_mode("tc_needs_export", _cfg("tc_needs_export"), {})
    assert list(manager._export_stack) == [("tc_needs_export", "tc_export")]

    # stacking a second export-requiring mode grows the stack
    manager.add_mode("tc_needs_export", _cfg("tc_needs_export"), {})
    assert list(manager._export_stack) == [("tc_needs_export", "tc_export")] * 2

    # applying the matching export mode pops the stack
    manager.add_mode("tc_export", _cfg("tc_export"), {})
    assert list(manager._export_stack) == [("tc_needs_export", "tc_export")]
    manager.add_mode("tc_export", _cfg("tc_export"), {})
    assert len(manager._export_stack) == 0


def test_export_mode_requires_open_stack(tc_modes):
    manager = ModeloptStateManager()
    with pytest.raises(AssertionError, match="current export stack"):
        manager.check_mode("tc_export")


# ---------------------------------------------------------------------------------------------
# ModeloptStateManager: load_state_dict and versioning
# ---------------------------------------------------------------------------------------------


def _donor_manager():
    manager = ModeloptStateManager()
    manager.add_mode("tc_simple", _cfg("tc_simple", channels=3), {"nested": {"value": 1}})
    manager.add_mode("tc_follower", _cfg("tc_follower"), {"idx": 1})
    return manager


def test_load_state_dict_rejects_existing_state(tc_modes):
    manager = _donor_manager()
    with pytest.raises(AssertionError, match="Cannot load state_dict if there is already one"):
        manager.load_state_dict(_donor_manager().state_dict(), __version__)


def test_load_state_dict_replays_modes(tc_modes):
    donor = _donor_manager()
    manager = ModeloptStateManager()
    manager.load_state_dict(donor.state_dict(), __version__)

    assert manager.state_dict() == donor.state_dict()
    assert _mode_names(manager) == ["tc_simple", "tc_follower"]
    assert manager.state_version == __version__


def test_load_state_dict_version_mismatch_warns(tc_modes):
    manager = ModeloptStateManager()
    version = _bumped_minor_version()
    with pytest.warns(UserWarning, match="Compatibility of checkpoint with current version"):
        manager.load_state_dict(_donor_manager().state_dict(), version)
    # the loaded version must be preserved exactly (not overwritten with the current one)
    assert manager.state_version == version


def test_load_state_dict_patch_version_mismatch_does_not_warn(tc_modes):
    major, minor = __version__.split(".")[:2]
    patch_version = f"{major}.{minor}.999"
    manager = ModeloptStateManager()
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        manager.load_state_dict(_donor_manager().state_dict(), patch_version)
    assert not any("Compatibility of checkpoint" in str(w.message) for w in caught)
    assert manager.state_version == patch_version


def test_load_state_dict_is_isolated_from_input(tc_modes):
    donor = _donor_manager()
    donor_state = donor.state_dict()
    manager = ModeloptStateManager()
    manager.load_state_dict(donor_state, __version__)

    # the target must own a deepcopy: mutating the donor state does not leak into the target
    assert manager.state_dict() is not donor_state
    donor_state[0][1]["metadata"]["nested"]["value"] = 999
    assert manager.state_dict()[0][1]["metadata"]["nested"]["value"] == 1


def test_load_state_dict_unknown_mode_raises(tc_modes):
    state = [("tc_does_not_exist", {"config": {}, "metadata": {}})]
    manager = ModeloptStateManager()
    with pytest.raises(KeyError, match="not found in any registry"):
        manager.load_state_dict(state, __version__)


def test_load_state_dict_invalid_config_raises(tc_modes):
    state = [("tc_simple", {"config": {"bogus_field": 1}, "metadata": {}})]
    manager = ModeloptStateManager()
    with pytest.raises(ValidationError):
        manager.load_state_dict(state, __version__)


# ---------------------------------------------------------------------------------------------
# ModeloptStateManager: transfer_state_dict and has_state_for_mode_type
# ---------------------------------------------------------------------------------------------


def test_transfer_state_dict_moves_state_instance(tc_modes):
    model_from = mto.apply_mode(ToyLinearModel(), "tc_simple")
    state_list = ModeloptStateManager(model_from).state_dict()

    model_to = ToyLinearModel()
    ModeloptStateManager.transfer_state_dict(model_from, model_to)

    assert not ModeloptStateManager.is_converted(model_from)
    assert ModeloptStateManager.is_converted(model_to, is_root=True)
    manager_to = ModeloptStateManager(model_to)
    # the docstring promises the *same instance* is transferred
    assert manager_to.state_dict() is state_list
    assert _mode_names(manager_to) == ["tc_simple"]


def test_transfer_state_dict_rejects_converted_target(tc_modes):
    model_from = mto.apply_mode(ToyLinearModel(), "tc_simple")
    model_to = mto.apply_mode(ToyLinearModel(), "tc_simple")
    with pytest.raises(AssertionError, match="already has modelopt state"):
        ModeloptStateManager.transfer_state_dict(model_from, model_to)


def test_has_state_for_mode_type(tc_modes):
    model = mto.apply_mode(ToyLinearModel(), "tc_simple")
    assert ModeloptStateManager.has_state_for_mode_type(TC_REGISTRY_NAME, model=model)

    state = mto.modelopt_state(model)
    assert ModeloptStateManager.has_state_for_mode_type(TC_REGISTRY_NAME, state=state)
    # the state contains no quantization modes
    assert not ModeloptStateManager.has_state_for_mode_type("quantization", state=state)

    with pytest.raises(ValueError, match="Must provide either model or state"):
        ModeloptStateManager.has_state_for_mode_type(TC_REGISTRY_NAME, model=model, state=state)
    with pytest.raises(ValueError, match="Must provide either model or state"):
        ModeloptStateManager.has_state_for_mode_type(TC_REGISTRY_NAME)


# ---------------------------------------------------------------------------------------------
# apply_mode
# ---------------------------------------------------------------------------------------------


def test_apply_mode_empty_mode_is_noop():
    model = ToyLinearModel()
    out = mto.apply_mode(model, [])
    assert out is model
    assert not ModeloptStateManager.is_converted(model)


def test_apply_mode_empty_mode_with_model_like():
    # NOTE: documents current behavior; arguably a bug because apply_mode's "vanilla case" is
    # supposed to just initialize and return the model, but ModelLikeModule.init_modellike()
    # unconditionally calls transfer_state_dict() on a module whose state was never initialized.
    with pytest.raises(AssertionError, match="Model must have modelopt state"):
        mto.apply_mode((ToyLinearModel, (), {}), [])


def test_apply_mode_descriptor_instance_not_supported(tc_modes):
    # NOTE: documents current behavior; arguably a bug because apply_mode's docstring says the
    # mode "may be specified as a string or as the actual ModeDescriptor class", but
    # get_mode_config() only handles strings and (name, config) tuples and tries to subscript
    # the descriptor instead.
    descriptor = _ModeRegistryCls.get_from_any("tc_simple")
    with pytest.raises(TypeError, match="not subscriptable"):
        mto.apply_mode(ToyLinearModel(), descriptor)


def test_apply_mode_infers_init_state_and_appends(tc_modes):
    model = ToyLinearModel()
    model = mto.apply_mode(model, "tc_simple")  # init_state inferred as True
    assert ModeloptStateManager.is_converted(model, is_root=True)

    model = mto.apply_mode(model, "tc_follower")  # init_state inferred as False -> appends
    assert _mode_names(ModeloptStateManager(model)) == ["tc_simple", "tc_follower"]
    assert model._tc_convert_count == 2


def test_apply_mode_multiple_modes_in_one_call(tc_modes):
    model = mto.apply_mode(ToyLinearModel(), ["tc_simple", "tc_follower"])
    assert _mode_names(ModeloptStateManager(model)) == ["tc_simple", "tc_follower"]
    assert model._tc_convert_count == 2


def test_apply_mode_init_state_true_on_converted_raises(tc_modes):
    model = mto.apply_mode(ToyLinearModel(), "tc_simple")
    with pytest.raises(AssertionError, match="already has modelopt state"):
        mto.apply_mode(model, "tc_follower", init_state=True)


def test_apply_mode_records_config_override_and_metadata(tc_modes):
    model = mto.apply_mode(ToyLinearModel(), [("tc_simple", {"channels": 16})])
    entry = ModeloptStateManager(model).state_dict()[0]
    assert entry[1]["config"]["channels"] == 16
    assert entry[1]["config"]["label"] == "default"  # non-overridden fields keep their default
    # metadata returned by the convert entrypoint is recorded verbatim
    assert entry[1]["metadata"] == {"channels_at_convert": 16}


def test_apply_mode_kwargs_passed_but_not_persisted(tc_modes):
    model = mto.apply_mode(ToyLinearModel(), "tc_simple", mode_kwargs={"custom_fn": "sentinel"})
    # kwargs reach the convert entrypoint ...
    assert model._tc_convert_kwargs == {"custom_fn": "sentinel"}
    # ... but are not saved in the modelopt state
    entry = ModeloptStateManager(model).state_dict()[0]
    assert "custom_fn" not in entry[1]["config"]
    assert "custom_fn" not in entry[1]["metadata"]


def test_apply_mode_kwargs_broadcast_and_length_mismatch(tc_modes):
    # a single dict is broadcast to all modes
    model = mto.apply_mode(ToyLinearModel(), ["tc_simple", "tc_follower"], mode_kwargs={"k": 1})
    assert model._tc_convert_kwargs == {"k": 1}

    with pytest.raises(AssertionError, match="Number of mode kwargs must match"):
        mto.apply_mode(ToyLinearModel(), "tc_simple", mode_kwargs=[{}, {}])


def test_apply_mode_with_explicit_registry(tc_modes):
    model = mto.apply_mode(ToyLinearModel(), "tc_simple", registry=tc_modes)
    assert str(ModeloptStateManager(model).last_mode) == "tc_simple"

    # a mode outside the provided registry must not be found, even if globally registered
    with pytest.raises(KeyError):
        mto.apply_mode(ToyLinearModel(), "quantize", registry=tc_modes)


def test_apply_mode_update_for_new_mode_hook(tc_modes):
    model = mto.apply_mode(ToyLinearModel(), "tc_update")
    entry = ModeloptStateManager(model).state_dict()[0]
    assert "updated_for_new_mode" not in entry[1]["metadata"]

    # adding the next mode triggers the previous mode's update_for_new_mode hook
    model = mto.apply_mode(model, "tc_simple")
    entry = ModeloptStateManager(model).state_dict()[0]
    assert entry[1]["metadata"]["updated_for_new_mode"] is True
    # the hook's in-place config change must be written back to the stored state
    assert entry[1]["config"]["label"] == "updated_for_new_mode"


def test_get_mode(tc_modes):
    model = ToyLinearModel()
    assert get_mode(model) is None
    model = mto.apply_mode(model, ["tc_simple", "tc_follower"])
    assert str(get_mode(model)) == "tc_follower"


# ---------------------------------------------------------------------------------------------
# modelopt_state
# ---------------------------------------------------------------------------------------------


def test_modelopt_state_structure_and_version(tc_modes):
    model = mto.apply_mode(ToyLinearModel(), "tc_simple")
    state = mto.modelopt_state(model)

    assert set(state.keys()) == {"modelopt_state_dict", "modelopt_version"}
    assert state["modelopt_version"] == __version__
    assert [m_str for m_str, _ in state["modelopt_state_dict"]] == ["tc_simple"]


def test_modelopt_state_on_unconverted_model():
    model = ToyLinearModel()
    state = mto.modelopt_state(model)
    assert state["modelopt_state_dict"] == []
    assert state["modelopt_version"] == __version__
    # querying the state must not convert the model as a side effect
    assert not ModeloptStateManager.is_converted(model)


def test_modelopt_state_skips_unsaved_modes(tc_modes):
    model = mto.apply_mode(ToyLinearModel(), ["tc_simple", "tc_nosave"])
    # the mode is tracked in the manager ...
    assert _mode_names(ModeloptStateManager(model)) == ["tc_simple", "tc_nosave"]
    # ... but excluded from the exported state (save_mode_in_state=False)
    state = mto.modelopt_state(model)
    assert [m_str for m_str, _ in state["modelopt_state_dict"]] == ["tc_simple"]


def test_modelopt_state_update_for_save_hook(tc_modes):
    model = mto.apply_mode(ToyLinearModel(), "tc_update")
    state = mto.modelopt_state(model)
    assert state["modelopt_state_dict"][0][1]["metadata"]["save_count"] == 1
    # the hook runs on every save
    state = mto.modelopt_state(model)
    assert state["modelopt_state_dict"][0][1]["metadata"]["save_count"] == 2


def test_modelopt_state_aliases_internal_state(tc_modes):
    model = mto.apply_mode(ToyLinearModel(), "tc_simple")
    manager = ModeloptStateManager(model)

    state = mto.modelopt_state(model)
    state["modelopt_state_dict"][0][1]["metadata"]["injected"] = "corruption"
    # NOTE: documents current behavior; arguably a bug because modelopt_state() returns
    # references to the manager's internal state entries, so callers mutating the returned
    # dict silently corrupt the modelopt state recorded on the model.
    assert manager.state_dict()[0][1]["metadata"]["injected"] == "corruption"

    # a deepcopy of the returned state is properly isolated (the caller's job today)
    state_copy = copy.deepcopy(mto.modelopt_state(model))
    state_copy["modelopt_state_dict"][0][1]["metadata"]["injected2"] = "x"
    assert "injected2" not in manager.state_dict()[0][1]["metadata"]


# ---------------------------------------------------------------------------------------------
# save / restore round-trips through actual files
# ---------------------------------------------------------------------------------------------


def test_save_restore_roundtrip_custom_mode(tc_modes, tmp_path):
    set_seed()
    model = mto.apply_mode(ToyLinearModel(), [("tc_simple", {"channels": 3})])
    with torch.no_grad():
        model.linear.weight.add_(1.0)  # make weights distinguishable from a fresh init

    ckpt = tmp_path / "ckpt.pt"
    mto.save(model, ckpt)
    model2 = mto.restore(ToyLinearModel(), ckpt)

    # the restore entrypoint was called exactly once with the saved metadata
    assert model2._tc_restore_count == 1
    assert model2._tc_restored_metadata == {"channels_at_convert": 3}
    assert not hasattr(model2, "_tc_convert_count")  # restore path must not call convert

    # modelopt state and weights survive the file round-trip
    manager, manager2 = ModeloptStateManager(model), ModeloptStateManager(model2)
    assert manager2.state_dict() == manager.state_dict()
    assert manager2.state_version == __version__
    assert torch.equal(model2.linear.weight, model.linear.weight)

    model.eval()
    model2.eval()
    x = torch.randn(2, 4)
    compare_outputs(model(x), model2(x))


def test_save_restore_binary_io(tc_modes):
    model = mto.apply_mode(ToyLinearModel(), "tc_simple")
    buffer = io.BytesIO()
    mto.save(model, buffer)
    buffer.seek(0)
    model2 = mto.restore(ToyLinearModel(), buffer)
    assert str(ModeloptStateManager(model2).last_mode) == "tc_simple"


def test_save_restore_roundtrip_quantize(tmp_path):
    set_seed()
    model = mto.apply_mode(ToyLinearModel(), "quantize", init_state=True)

    ckpt = tmp_path / "quantized.pt"
    mto.save(model, ckpt)
    model2 = mto.restore(ToyLinearModel(), ckpt)

    manager, manager2 = ModeloptStateManager(model), ModeloptStateManager(model2)
    assert torch.equal(_serialize(manager.state_dict()), _serialize(manager2.state_dict()))

    model.eval()
    model2.eval()
    x = torch.randn(2, 4)
    compare_outputs(model(x), model2(x))


def test_restore_from_modelopt_state_requires_exactly_one_source(tc_modes, tmp_path):
    model = mto.apply_mode(ToyLinearModel(), "tc_simple")
    state = mto.modelopt_state(model)
    path = tmp_path / "state.pt"
    torch.save(state, path)

    with pytest.raises(AssertionError, match="but not both"):
        mto.restore_from_modelopt_state(ToyLinearModel())
    with pytest.raises(AssertionError, match="but not both"):
        mto.restore_from_modelopt_state(
            ToyLinearModel(), modelopt_state=state, modelopt_state_path=path
        )


def test_restore_from_modelopt_state_path(tc_modes, tmp_path):
    model = mto.apply_mode(ToyLinearModel(), [("tc_simple", {"channels": 11})])
    path = tmp_path / "state.pt"
    torch.save(mto.modelopt_state(model), path)

    model2 = mto.restore_from_modelopt_state(ToyLinearModel(), modelopt_state_path=path)
    assert model2._tc_restore_count == 1
    assert model2._tc_restored_metadata == {"channels_at_convert": 11}
    assert ModeloptStateManager(model2).state_dict()[0][1]["config"]["channels"] == 11


def test_restore_version_mismatch_warns(tc_modes):
    model = mto.apply_mode(ToyLinearModel(), "tc_simple")
    state = copy.deepcopy(mto.modelopt_state(model))
    state["modelopt_version"] = "0.0.1"

    with pytest.warns(UserWarning, match="Compatibility of checkpoint with current version"):
        model2 = mto.restore_from_modelopt_state(ToyLinearModel(), modelopt_state=state)
    assert ModeloptStateManager(model2).state_version == "0.0.1"


def test_restore_unknown_mode_raises(tc_modes):
    model = mto.apply_mode(ToyLinearModel(), "tc_simple")
    state = copy.deepcopy(mto.modelopt_state(model))
    state["modelopt_state_dict"][0] = ("tc_does_not_exist", state["modelopt_state_dict"][0][1])
    with pytest.raises(KeyError, match="not found in any registry"):
        mto.restore_from_modelopt_state(ToyLinearModel(), modelopt_state=state)


def test_restore_corrupt_config_raises(tc_modes):
    model = mto.apply_mode(ToyLinearModel(), "tc_simple")
    state = copy.deepcopy(mto.modelopt_state(model))
    state["modelopt_state_dict"][0][1]["config"]["bogus_field"] = 1
    with pytest.raises(ValidationError):
        mto.restore_from_modelopt_state(ToyLinearModel(), modelopt_state=state)


def test_restore_corrupt_state_entry_raises(tc_modes):
    model = mto.apply_mode(ToyLinearModel(), "tc_simple")
    state = copy.deepcopy(mto.modelopt_state(model))
    del state["modelopt_state_dict"][0][1]["config"]
    with pytest.raises(KeyError, match="config"):
        mto.restore_from_modelopt_state(ToyLinearModel(), modelopt_state=state)


def test_restore_is_isolated_from_caller_state(tc_modes):
    model = mto.apply_mode(ToyLinearModel(), "tc_simple")
    state = copy.deepcopy(mto.modelopt_state(model))
    model2 = mto.restore_from_modelopt_state(ToyLinearModel(), modelopt_state=state)

    # mutating the caller's state after restore must not leak into the restored model
    state["modelopt_state_dict"][0][1]["metadata"]["poison"] = True
    assert "poison" not in ModeloptStateManager(model2).state_dict()[0][1]["metadata"]


def test_restore_empty_state():
    state = {"modelopt_state_dict": [], "modelopt_version": __version__}
    model = mto.restore_from_modelopt_state(ToyLinearModel(), modelopt_state=state)
    assert ModeloptStateManager.is_converted(model, is_root=True)
    assert not ModeloptStateManager(model).has_state
    model(torch.randn(2, 4))  # the model is fully functional


def test_model_like_module_rejects_nn_module():
    with pytest.raises(AssertionError, match=r"should not be a nn\.Module"):
        ModelLikeModule(ToyLinearModel())
