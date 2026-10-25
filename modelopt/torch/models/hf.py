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
    "prefers_bundled_code",
    "prepare_model_for_calibration",
    "prepare_model_for_loading",
]

# Keep modeling dependencies lazy: importing model specs must not import calibration code.
# A plugin defines any of prepare_for_loading, has_mtp_weights and prepare_for_calibration.
_PLUGINS = {
    "nemotron_h": "nemotron_h.mtp",
    "nemotron_h_omni": "nemotron_h.mtp",
    "qwen4_exp": "qwen4_exp.qsa_indexer",
    "qwen4_exp_text": "qwen4_exp.qsa_indexer",
}


def _get_plugin(model_type):
    """Return the optional ModelOpt support plugin for a Hugging Face model type."""
    name = _PLUGINS.get(model_type)
    return None if name is None else importlib.import_module(f".{name}", __package__)


def prepare_model_for_loading(
    model_type, checkpoint_path: str, trust_remote_code: bool, *, model_class=None
):
    """Prepare checkpoint-only modules, optionally on the loader's selected concrete class."""
    hook = getattr(_get_plugin(model_type), "prepare_for_loading", None)
    if hook is None:
        return nullcontext()
    return hook(checkpoint_path, trust_remote_code, model_class=model_class)


def checkpoint_has_mtp(model_type, checkpoint_path: str) -> bool:
    """Use the model plugin to detect MTP tensors before selecting a supported loading path."""
    hook = getattr(_get_plugin(model_type), "has_mtp_weights", None)
    return hook is not None and hook(checkpoint_path)


def prefers_bundled_code(model_type, hf_config, checkpoint_path: str, config_kwargs: dict) -> bool:
    """Use the model plugin to tell whether the checkpoint's bundled code must replace the built-in class."""
    plugin = _get_plugin(model_type)
    return plugin is not None and plugin.prefers_bundled_code(
        hf_config, checkpoint_path, config_kwargs
    )


def prepare_model_for_calibration(model) -> None:
    """Run the optional model-specific hook that augments the default calibration forward."""
    plugin = _get_plugin(getattr(getattr(model, "config", None), "model_type", None))
    hook = getattr(plugin, "prepare_for_calibration", None)
    if hook is not None:
        hook(model)
