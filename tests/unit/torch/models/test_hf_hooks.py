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

"""How the hook dispatcher resolves a model type's support plugins."""

import sys
import types

import pytest

from modelopt.torch.models import hf


def _plugin(monkeypatch, name, **hooks):
    module = types.ModuleType(f"modelopt.torch.models.{name}")
    module.__all__ = list(hooks)
    for hook_name, hook in hooks.items():
        setattr(module, hook_name, hook)
    monkeypatch.setitem(sys.modules, module.__name__, module)


def test_a_model_type_takes_its_hooks_from_several_plugins(monkeypatch):
    _plugin(monkeypatch, "loading_plugin", has_mtp_weights=lambda path: True)
    _plugin(monkeypatch, "calibration_plugin", forward_calibration_batch=lambda model, batch: True)
    monkeypatch.setitem(hf._PLUGINS, "several", ("loading_plugin", "calibration_plugin"))
    model = types.SimpleNamespace(config=types.SimpleNamespace(model_type="several"))

    assert hf.checkpoint_has_mtp("several", "ckpt") is True
    assert hf.forward_calibration_batch(model, {}) is True
    assert hf.prefers_bundled_code("several", None, "ckpt", {}) is False


def test_two_plugins_defining_one_hook_are_refused(monkeypatch):
    _plugin(monkeypatch, "first_plugin", has_mtp_weights=lambda path: True)
    _plugin(monkeypatch, "second_plugin", has_mtp_weights=lambda path: False)
    monkeypatch.setitem(hf._PLUGINS, "clashing", ("first_plugin", "second_plugin"))

    with pytest.raises(ValueError, match="has_mtp_weights"):
        hf.checkpoint_has_mtp("clashing", "ckpt")
