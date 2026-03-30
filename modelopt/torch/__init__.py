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

"""Model optimization and deployment subpackage for torch."""

import sys as _sys
import warnings as _warnings

from packaging.version import Version as _Version
from torch import __version__ as _torch_version
from torch import device as _device
from torch import dtype as _dtype

from . import distill, nas, opt, peft, prune, quantization, sparsity, speculative, utils

if _Version(_torch_version) < _Version("2.9"):
    _warnings.warn(
        "nvidia-modelopt will drop torch<2.9 support in a future release.", DeprecationWarning
    )


# Since `hf` dependencies are optional and users have pre-installed transformers, we need to ensure
# correct version is installed to avoid incompatibility issues.
def _patch_transformers_compat(mod) -> None:
    """Compatibility shims for names removed in transformers 5.0."""
    import torch.nn as _nn

    # AutoModelForVision2Seq -> AutoModelForImageTextToText
    if not hasattr(mod, "AutoModelForVision2Seq") and hasattr(mod, "AutoModelForImageTextToText"):
        mod.AutoModelForVision2Seq = mod.AutoModelForImageTextToText

    # get_parameter_device and get_parameter_dtype were removed in transformers 5.0
    modeling_utils = _sys.modules.get("transformers.modeling_utils")
    if modeling_utils is not None:
        if not hasattr(modeling_utils, "get_parameter_device"):

            def get_parameter_device(parameter: _nn.Module) -> _device:
                return next(parameter.parameters()).device

            modeling_utils.get_parameter_device = get_parameter_device  # type: ignore[attr-defined]

        if not hasattr(modeling_utils, "get_parameter_dtype"):

            def get_parameter_dtype(parameter: _nn.Module) -> _dtype:
                return next(parameter.parameters()).dtype

            modeling_utils.get_parameter_dtype = get_parameter_dtype  # type: ignore[attr-defined]

        if not hasattr(modeling_utils, "load_sharded_checkpoint"):
            try:
                from transformers.trainer_utils import (
                    load_sharded_checkpoint as _load_sharded_checkpoint,
                )

                modeling_utils.load_sharded_checkpoint = _load_sharded_checkpoint  # type: ignore[attr-defined]
            except ImportError:
                pass

    # AutoConfig.register raises ValueError when a model type is already built into
    # transformers (e.g. exaone_moe added in 5.0). Older packages like TRT-LLM call
    # register without exist_ok=True. Patch CONFIG_MAPPING.register to silently skip.
    try:
        from transformers.models.auto.configuration_auto import CONFIG_MAPPING as _CONFIG_MAPPING

        _orig_cfg_register = _CONFIG_MAPPING.register

        def _patched_cfg_register(key, value, exist_ok=False):
            _orig_cfg_register(key, value, exist_ok=True)

        _CONFIG_MAPPING.register = _patched_cfg_register
    except Exception:
        pass


try:
    from transformers import __version__ as _transformers_version

    if _Version(_transformers_version) < _Version("4.56"):
        _warnings.warn(
            f"transformers {_transformers_version} is not tested with current version of modelopt and may cause issues."
            " Please install recommended version with `pip install -U nvidia-modelopt[hf]` if working with HF models.",
        )
    elif _Version(_transformers_version) >= _Version("5.0"):
        _warnings.warn(
            "transformers>=5.0 support is experimental. Unified Hugging Face checkpoint export for quantized "
            "checkpoints may not work for some models yet.",
        )

    # Temporary workaround until TRT-LLM container supports transformers 5.0
    if "transformers" in _sys.modules:
        _patch_transformers_compat(_sys.modules["transformers"])
    else:

        class _TransformersCompatFinder:
            def find_module(self, fullname, path=None):
                if fullname == "transformers":
                    _sys.meta_path.remove(self)  # type: ignore[arg-type]
                    import importlib as _importlib

                    _patch_transformers_compat(_importlib.import_module(fullname))

        _sys.meta_path.insert(0, _TransformersCompatFinder())  # type: ignore[arg-type]
except ImportError:
    pass

# Initialize modelopt_internal if available
with utils.import_plugin(
    "modelopt_internal", success_msg="modelopt_internal successfully initialized", verbose=True
):
    import modelopt_internal
