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

"""Dispatch optional Hugging Face loading and calibration hooks to per-model packages."""

import importlib
from contextlib import nullcontext

__all__ = [
    "checkpoint_has_mtp",
    "prepare_model_for_calibration",
    "prepare_model_for_loading",
]

# Keep modeling dependencies lazy: importing model specs must not import calibration code.
_PLUGINS = {
    "nemotron_h": "nemotron_h.mtp",
    "nemotron_h_omni": "nemotron_h.mtp",
}


def _get_plugin(model_type):
    """Return the optional ModelOpt support plugin for a Hugging Face model type."""
    name = _PLUGINS.get(model_type)
    return None if name is None else importlib.import_module(f".{name}", __package__)


def prepare_model_for_loading(
    model_type, checkpoint_path: str, trust_remote_code: bool, *, model_class=None
):
    """Prepare checkpoint-only modules, optionally on the loader's selected concrete class."""
    plugin = _get_plugin(model_type)
    if plugin is None:
        return nullcontext()
    return plugin.prepare_for_loading(checkpoint_path, trust_remote_code, model_class=model_class)


def checkpoint_has_mtp(model_type, checkpoint_path: str) -> bool:
    """Use the model plugin to detect MTP tensors before selecting a supported loading path."""
    plugin = _get_plugin(model_type)
    return plugin is not None and plugin.has_mtp_weights(checkpoint_path)


def prepare_model_for_calibration(model) -> None:
    """Run the optional model-specific hook that augments the default calibration forward."""
    plugin = _get_plugin(getattr(getattr(model, "config", None), "model_type", None))
    if plugin is not None:
        plugin.prepare_for_calibration(model)
