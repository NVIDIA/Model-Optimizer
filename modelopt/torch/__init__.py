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

import importlib
import warnings as _warnings

from packaging.version import Version as _Version
from torch import __version__ as _torch_version

# Pre-initialize torch._dynamo to prevent double-registration with peft's torch.compile() call
importlib.import_module("torch._dynamo")
from . import (  # noqa: E402
    distill,
    nas,
    opt,
    peft,
    prune,
    quantization,
    sparsity,
    speculative,
    utils,
)

if _Version(_torch_version) < _Version("2.9"):
    _warnings.warn(
        "nvidia-modelopt will drop torch<2.9 support in a future release.", DeprecationWarning
    )


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
except ImportError:
    pass

# Initialize modelopt_internal if available
with utils.import_plugin(
    "modelopt_internal", success_msg="modelopt_internal successfully initialized", verbose=True
):
    import modelopt_internal
