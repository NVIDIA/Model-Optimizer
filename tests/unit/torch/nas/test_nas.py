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

import copy
import io
import types
import warnings
from collections import deque

import pytest
import torch
from _test_utils.torch.vision_models import TinyMobileNetFeatures
from torch import nn
from torch.nn.modules.batchnorm import _BatchNorm
from torchvision.models.mobilenetv2 import InvertedResidual

import modelopt.torch.nas as mtn
import modelopt.torch.opt as mto
from modelopt.torch.nas.autonas import AutoNASPatchManager
from modelopt.torch.nas.patch import prep_for_eval
from modelopt.torch.opt.utils import is_dynamic
from modelopt.torch.utils import run_forward_loop
from modelopt.torch.utils.random import _set_deterministic_seed, centroid


@pytest.fixture(scope="module")
def test_config():
    return {
        "nn.Sequential": {"min_depth": 1},
        "nn.Conv2d": {
            "channels_ratio": (0.4, 0.6, 0.8, 1.0),
            "kernel_size": (1, 3),
            "channel_divisor": 16,
        },
        "nn.BatchNorm2d": {},
        "nn.SyncBatchNorm": {},
    }


def get_data_loader(num_batches):
    """Yield some fake data that's consistent over the test."""
    for _ in range(num_batches):
        yield torch.rand(2, 3, 56, 56), int(torch.randint(1000, (1,)))


def test_automode(test_config):
    # Use model class to create a model inside mtn.convert
    model = mtn.convert((TinyMobileNetFeatures, (), {}), mode=[("autonas", test_config)])
    num_batches = 3
    auto_data = AutoNASPatchManager(model).patch_data
    queue = auto_data["queue"]

    # check that queue stays empty during eval mode and config remains the same
    model.eval()
    config = mtn.sample(model)
    for img, _ in get_data_loader(num_batches):
        model(img)
    assert len(queue) == 0
    assert config == mtn.get_subnet_config(model)

    # check that queue fills up in train mode and config changes
    model.train()
    for img, _ in get_data_loader(num_batches):
        model(img)
    assert len(queue) == num_batches
    assert config != mtn.get_subnet_config(model)

    # check that queue fills up in eval mode with fill=True but config remains the same
    model.eval()
    config = mtn.get_subnet_config(model)
    auto_data["fill"] = True
    for img, _ in get_data_loader(num_batches):
        model(img)
    auto_data["fill"] = False
    assert len(queue) == 2 * num_batches
    assert config == mtn.get_subnet_config(model)

    # check whether anything changes when automode is turned off
    mtn.set_modelopt_patches_enabled(False)
    model.train()
    for img, _ in get_data_loader(num_batches):
        model(img)
    assert len(queue) == 2 * num_batches
    assert config == mtn.get_subnet_config(model)

    # check whether export successfully calibrates the model during export
    fake_queue = deque(auto_data["queue"], maxlen=len(auto_data["queue"]))
    auto_data["queue"] = fake_queue  # fake filled queue
    for img, _ in get_data_loader(1):
        break
    model.eval()  # this does not calib BN since we turned it off above

    out_dyn = model(img)
    mtn.set_modelopt_patches_enabled(True)

    model2 = copy.deepcopy(model)  # deepcopy for 2nd export below...

    # export, no calib
    model = mtn.export(model, calib=False)
    out = model(img)
    assert torch.allclose(out_dyn, out)

    # export, calib
    model2 = mtn.export(model2, calib=True)
    out2 = model2(img)
    assert not torch.allclose(out_dyn, out2)


@pytest.mark.parametrize("native_eval", [False, True])
def test_prep_for_eval(native_eval, test_config):
    """Test all four ways of calling prep_for_eval.

    Specifically, we can check all four combinations of calling prep_for_eval with
    automode on/off and with/without dataloader

    We check this using both calling prep_for_eval directly or using the monkey-patched
    train function.
    """
    model = TinyMobileNetFeatures()
    model = mtn.convert(model, mode=[("autonas", test_config)])
    num_batches = 3
    auto_data = AutoNASPatchManager(model).patch_data
    queue = auto_data["queue"]

    # define eval function to be tested
    def eval_for_testing(model, data_loader=None):
        if data_loader is None and native_eval:
            model.eval()
        elif data_loader is None:
            prep_for_eval(model)
        else:
            prep_for_eval(model, lambda m: run_forward_loop(m, data_loader=data_loader))

    # grab one bn layer that we can later monitor
    bn_layer = None
    for mod in model.modules():
        if isinstance(mod, _BatchNorm) and mod.track_running_stats:
            bn_layer = mod
            break
    assert bn_layer is not None, "Couldn't detect BN layer."

    def _get_current_bn_stats():
        return bn_layer.running_mean.detach().clone(), bn_layer.running_var.detach().clone()

    # run inference once with random config
    mtn.sample(model)
    with pytest.warns(Warning):
        model.eval()
    model(next(get_data_loader(1))[0])

    # 1. automode off and no dataloader
    # expected behavior --> simply switch to eval mode and that's it
    # config should remain unchanged
    bn_mean, bn_var = _get_current_bn_stats()
    config = mtn.get_subnet_config(model)

    mtn.set_modelopt_patches_enabled(False)
    model.train()
    eval_for_testing(model)

    assert not model.training
    bn_mean2, bn_var2 = _get_current_bn_stats()
    assert torch.allclose(bn_mean, bn_mean2)
    assert torch.allclose(bn_var, bn_var2)
    assert len(queue) == 0
    assert config == mtn.get_subnet_config(model)

    # 2. automode off and dataloader
    # expected behavior --> reset_bn but do not touch queue, config unchanged
    bn_mean, bn_var = _get_current_bn_stats()
    config = mtn.get_subnet_config(model)

    mtn.set_modelopt_patches_enabled(False)
    model.train()
    eval_for_testing(model, get_data_loader(num_batches))

    assert not model.training
    bn_mean2, bn_var2 = _get_current_bn_stats()
    assert not torch.allclose(bn_mean, bn_mean2)
    assert not torch.allclose(bn_var, bn_var2)
    assert len(queue) == 0
    assert config == mtn.get_subnet_config(model)

    # 3 automode on and no dataloader
    # expected behavior --> reset_bn with empty queue and issue warning, config unchanged
    bn_mean, bn_var = _get_current_bn_stats()
    config = mtn.get_subnet_config(model)
    mtn.select(model, config)

    mtn.set_modelopt_patches_enabled(True)
    model.train()
    with pytest.warns(Warning):
        eval_for_testing(model)

    assert not model.training
    bn_mean2, bn_var2 = _get_current_bn_stats()
    assert torch.allclose(bn_mean, bn_mean2)
    assert torch.allclose(bn_var, bn_var2)
    assert len(queue) == 0
    assert config == mtn.get_subnet_config(model)

    # 4 automode on and dataloader
    # expected behavior --> reset_bn with queue and no warning, config unchanged
    bn_mean, bn_var = _get_current_bn_stats()
    config = mtn.get_subnet_config(model)

    mtn.set_modelopt_patches_enabled(True)
    model.train()
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        eval_for_testing(model, get_data_loader(num_batches))

    assert not model.training
    bn_mean2, bn_var2 = _get_current_bn_stats()
    assert not torch.allclose(bn_mean, bn_mean2)
    assert not torch.allclose(bn_var, bn_var2)
    assert len(queue) == num_batches
    assert not auto_data["fill"]
    assert config == mtn.get_subnet_config(model)

    # TODO: retest and assert no warning when reset_bn is called again since we
    # just calibrated BN and did NOT resample

    # 4.5 automode on and no dataloader
    # expected behavior --> we call eval_for_testing after sampling
    # --> warning, then we call with eplicit data loader --> no warning
    # then call it without dataloader --> no warning, although queue is too short since
    # we just calibrated BN without resampling
    mtn.set_modelopt_patches_enabled(True)
    model.train()
    mtn.sample(model)

    bn_mean, bn_var = _get_current_bn_stats()
    config = mtn.get_subnet_config(model)

    with pytest.warns(Warning):
        eval_for_testing(model)

    with warnings.catch_warnings():
        warnings.simplefilter("error")
        eval_for_testing(model, get_data_loader(num_batches))
        eval_for_testing(model)

    num_batches += num_batches
    assert not model.training
    bn_mean2, bn_var2 = _get_current_bn_stats()
    assert not torch.allclose(bn_mean, bn_mean2)
    assert not torch.allclose(bn_var, bn_var2)
    assert len(queue) == num_batches
    assert not auto_data["fill"]
    assert config == mtn.get_subnet_config(model)

    # 5 automode on and no dataloader after some train forward passes
    # expected behavior --> reset_bn with queue, warning, and config to max
    bn_mean, bn_var = _get_current_bn_stats()
    max_config = mtn.sample(model, sample_func=max)

    mtn.set_modelopt_patches_enabled(True)
    model.train()
    for i, (img, _) in enumerate(get_data_loader(100)):
        model(img)
        if max_config != mtn.get_subnet_config(model):
            break
    len_queue = num_batches + i + 1

    assert len(queue) == len_queue

    with pytest.warns(Warning):
        eval_for_testing(model)

    assert not model.training
    bn_mean2, bn_var2 = _get_current_bn_stats()
    assert not torch.allclose(bn_mean, bn_mean2)
    assert not torch.allclose(bn_var, bn_var2)
    assert len(queue) == len_queue
    assert max_config == mtn.get_subnet_config(model)


def test_set_auto_mode(test_config):
    """Test whether we can correctly set automode on/off."""
    # setup model in autonas mode and training mode
    model = TinyMobileNetFeatures()
    model = mtn.convert(model, mode=[("autonas", test_config)])
    model.train()

    # setup sample functions
    f_fixed = centroid
    _set_deterministic_seed()

    # random input
    img = torch.randn(2, 3, 12, 12)

    def _compare_to_fixed(auto_enabled: bool):
        # get and set fixed config
        config_fixed = mtn.sample(model, sample_func=f_fixed)

        # run forward and retrieve config
        model(img)
        config_random = mtn.get_subnet_config(model)

        # check if config changed
        if auto_enabled:
            assert config_fixed != config_random
        else:
            assert config_fixed == config_random

        # check if global automode flag is on/off
        assert mtn.is_modelopt_patches_enabled() == auto_enabled

    def compare_before_during_after_dec(ctx_manager, ctx_args, before, inside):
        def _context_compare():
            with ctx_manager(*ctx_args):
                _compare_to_fixed(inside)

        fxs = [_context_compare]

        if ctx_manager != mtn.set_modelopt_patches_enabled:
            # define decorated function
            @ctx_manager(*ctx_args)
            def _decorated_compare():
                return _compare_to_fixed(inside)

            fxs.append(_decorated_compare)

        for fx in fxs:
            # try before
            _compare_to_fixed(before)
            # try modified
            fx()
            # try after
            _compare_to_fixed(before)

    for auto_enabled in [True, False]:
        # set automode globally
        mtn.set_modelopt_patches_enabled(auto_enabled)

        # 1. automode on
        _compare_to_fixed(auto_enabled)

        # 2. disable automode via no_modelopt/set_modelopt_enabled --> config shouldn't change
        compare_before_during_after_dec(mtn.no_modelopt_patches, (), auto_enabled, False)
        compare_before_during_after_dec(
            mtn.set_modelopt_patches_enabled, (False,), auto_enabled, False
        )

        # 3. enable automode via enable_modelopt/set_modelopt_enabled --> config should change
        compare_before_during_after_dec(mtn.enable_modelopt_patches, (), auto_enabled, True)
        compare_before_during_after_dec(
            mtn.set_modelopt_patches_enabled, (True,), auto_enabled, True
        )

        # 4. dual automode on/off
        with mtn.set_modelopt_patches_enabled(not auto_enabled):
            compare_before_during_after_dec(
                mtn.set_modelopt_patches_enabled, (auto_enabled,), not auto_enabled, auto_enabled
            )
        _compare_to_fixed(auto_enabled)


@pytest.mark.parametrize(
    ("cls", "args", "submodule", "use_config", "dummy_input", "mode"),
    [
        (
            InvertedResidual,
            (16, 32, 1, 6),
            "",
            False,
            torch.randn(1, 16, 8, 8),
            ["autonas", "export_nas"],
        ),
        (
            InvertedResidual,
            (16, 32, 1, 6),
            "conv.1",
            True,
            torch.randn(1, 16, 8, 8),
            ["autonas", "export_nas"],
        ),
        (
            InvertedResidual,
            (16, 32, 1, 6),
            "conv.1",
            True,
            torch.randn(1, 16, 8, 8),
            ["autonas"],
        ),
        (
            InvertedResidual,
            (16, 32, 1, 6),
            "conv.0",
            False,
            torch.randn(1, 16, 8, 8),
            [],
        ),
        (
            InvertedResidual,
            (16, 32, 1, 6),
            "",
            True,
            torch.randn(1, 16, 8, 8),
            ["autonas", "export_nas"],
        ),
        (
            TinyMobileNetFeatures,
            (),
            "",
            False,
            torch.randn(1, 3, 64, 64),
            ["autonas", "export_nas"],
        ),
        (TinyMobileNetFeatures, (), "", False, torch.randn(1, 3, 64, 64), ["autonas"]),
        (TinyMobileNetFeatures, (), "", False, torch.randn(1, 3, 64, 64), []),
    ],
)
def test_save_restore_whole(
    cls: type[nn.Module], args, submodule, use_config, dummy_input, mode, test_config
):
    """Test whether we can save and restore a model."""
    # setup model
    model = cls(*args)

    # check for "export_nas"
    if "export_nas" in mode:
        mode.remove("export_nas")
        use_export = True
    else:
        use_export = False

    config = test_config if use_config else None
    mode_with_config = [(m, config or {}) if m in ["autonas", "fastnas"] else m for m in mode]

    # should be fine since it is in-place
    mtn.convert(model.get_submodule(submodule), mode=mode_with_config)
    assert not mode or is_dynamic(model), "Model should be dynamic after convert."

    mtn.sample(model)

    # check for export
    if use_export:
        if submodule:
            # vanilla export shouldn't work here (use deepcopy to avoid weird model modifications)
            with pytest.raises(AssertionError):
                mtn.export(copy.deepcopy(model))

        # export is in-place, so we can just do it like this...
        mtn.export(model.get_submodule(submodule))

        # check that it's not dynamic anymore when we use export
        assert not is_dynamic(model), "Model should not be dynamic after export."

    # check saving
    if submodule and mode:
        with pytest.raises(AssertionError):
            mto.modelopt_state(model)
    modelopt_state = mto.modelopt_state(model.get_submodule(submodule))

    # restore model or submodule in-place
    model2 = cls(*args)
    mto.restore_from_modelopt_state(model2.get_submodule(submodule), modelopt_state)
    if not use_export and mode:
        assert is_dynamic(model2), "Model2 should be dynamic after restore."
    model2.load_state_dict(model.state_dict())

    # there shouldn't be any state or state in submodule, hence we cannot init state manager
    if not mode or submodule:
        with pytest.raises(AssertionError):
            manager = mto.ModeloptStateManager(model)
            manager2 = mto.ModeloptStateManager(model2)

    # check modelopt state now
    if mode:
        manager = mto.ModeloptStateManager(model.get_submodule(submodule))
        manager2 = mto.ModeloptStateManager(model2.get_submodule(submodule))
        assert manager.state_dict() == manager2.state_dict()
        assert mto.ModeloptStateManager.has_state_for_mode_type("nas", state=modelopt_state)
        assert mto.ModeloptStateManager.has_state_for_mode_type(
            "nas", model=model2.get_submodule(submodule)
        )

    # run comparison in eval mode since there might be model randomization in train mode
    model.eval()
    model2.eval()
    output = model(dummy_input)
    output2 = model2(dummy_input)
    if isinstance(output, torch.Tensor):
        assert torch.allclose(output, output2)
    elif isinstance(output, dict):
        for key in output:
            assert torch.allclose(output[key], output2[key])


@pytest.mark.parametrize(("cls", "args", "config"), [(InvertedResidual, (16, 32, 1, 6), None)])
def test_warning_raised(cls, args, config):
    model = cls(*args)
    model = mtn.convert(model, mode=[("fastnas", config)])
    mtn.sample(model)
    model = mtn.export(model)

    buffer = io.BytesIO()
    mto.save(model, buffer)
    buffer.seek(0)

    original_model = cls(*args)
    with pytest.raises(RuntimeError):
        original_model.forward_old = original_model.forward
        original_model.forward = types.MethodType(
            original_model, lambda self, x: self.forward_old(x)
        )
        mto.restore(original_model, buffer)
