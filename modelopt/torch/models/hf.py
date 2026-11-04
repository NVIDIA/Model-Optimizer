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

from .specs import hf_model_type

__all__ = [
    "checkpoint_has_mtp",
    "forward_calibration_batch",
    "prefers_bundled_code",
    "prepare_model_for_calibration",
    "prepare_model_for_loading",
]

# Keep modeling dependencies lazy: importing model specs must not import calibration code.
# A plugin defines any of these hooks and lists exactly those in its __all__ (see README.md).
_HOOKS = frozenset(
    {
        "prepare_for_loading",
        "has_mtp_weights",
        "prefers_bundled_code",
        "prepare_for_calibration",
        "forward_calibration_batch",
    }
)
# A model type's plugins are one module or several; no two of them may define the same hook.
_PLUGINS: dict[str, str | tuple[str, ...]] = {
    "deepseek": "deepseek.bundled_code",
    "deepseek_v2": "deepseek.bundled_code",
    "deepseek_v3": "deepseek.bundled_code",
    "deepseek_v32": "deepseek.bundled_code",
    "deepseek_v4": "deepseek.bundled_code",
    "kimi_k2": "deepseek.bundled_code",
    "nemotron_h": "nemotron_h.mtp",
    "nemotron_h_omni": ("nemotron_h.mtp", "nemotron_h_nano_vl_v2.calibration"),
    "NemotronH_Nano_VL_V2": "nemotron_h_nano_vl_v2.calibration",
    "Llama_Nemotron_Nano_VL": "nemotron_h_nano_vl_v2.calibration",
    # The InternVL wrappers the calibration helper's InternVL branch was written against.
    "internvl_chat": "nemotron_h_nano_vl_v2.calibration",
    "qwen4_exp": "qwen4_exp.qsa_indexer",
    "qwen4_exp_text": "qwen4_exp.qsa_indexer",
}


def _get_hook(model_type, hook_name):
    """Return ``hook_name`` from a Hugging Face model type's support plugins, or None if none has it."""
    names = _PLUGINS.get(model_type, ())
    hooks = []
    for name in (names,) if isinstance(names, str) else names:
        plugin = importlib.import_module(f".{name}", __package__)
        # Hooks are optional, so a misspelled one would be skipped silently: refuse it instead.
        unknown = set(getattr(plugin, "__all__", ())) - _HOOKS
        if unknown:
            raise ValueError(
                f"{plugin.__name__} exports {sorted(unknown)}, not hooks of {sorted(_HOOKS)}"
            )
        hook = getattr(plugin, hook_name, None)
        if hook is not None:
            hooks.append(hook)
    # A second definition would be skipped just as silently.
    if len(hooks) > 1:
        raise ValueError(f"{model_type} has {len(hooks)} plugins defining {hook_name}")
    return hooks[0] if hooks else None


def prepare_model_for_loading(
    model_type, checkpoint_path: str, trust_remote_code: bool, *, model_class=None
):
    """Prepare checkpoint-only modules, optionally on the loader's selected concrete class."""
    hook = _get_hook(model_type, "prepare_for_loading")
    if hook is None:
        return nullcontext()
    return hook(checkpoint_path, trust_remote_code, model_class=model_class)


def checkpoint_has_mtp(model_type, checkpoint_path: str) -> bool:
    """Use the model plugin to detect MTP tensors before selecting a supported loading path."""
    hook = _get_hook(model_type, "has_mtp_weights")
    return hook is not None and hook(checkpoint_path)


def prefers_bundled_code(model_type, hf_config, checkpoint_path: str, config_kwargs: dict) -> bool:
    """Use the model plugin to tell whether the checkpoint's bundled code must replace the built-in class."""
    hook = _get_hook(model_type, "prefers_bundled_code")
    return hook is not None and hook(hf_config, checkpoint_path, config_kwargs)


def prepare_model_for_calibration(model) -> None:
    """Run the optional model-specific hook that augments the default calibration forward."""
    hook = _get_hook(
        getattr(getattr(model, "config", None), "model_type", None), "prepare_for_calibration"
    )
    if hook is not None:
        hook(model)


def forward_calibration_batch(model, batch: dict) -> bool:
    """Run one calibration batch through the model plugin's own forward.

    Returns False when the model has no such forward, or its plugin declines this model: the
    caller then runs the model's plain forward.
    """
    hook = _get_hook(hf_model_type(model), "forward_calibration_batch")
    return hook is not None and hook(model, batch)
