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

"""Plugins for sparse attention integration with various frameworks."""

from modelopt.torch.utils import import_plugin

# Set of model plugins called during conversion.  A set (rather than a list)
# keeps re-imports idempotent — the same callback inserted multiple times
# stays registered once.  Matches the convention used by quantization and peft.
CUSTOM_MODEL_PLUGINS: set = set()


def register_custom_model_plugins_on_the_fly(model):
    """Apply every registered custom model plugin to ``model``."""
    for callback in CUSTOM_MODEL_PLUGINS:
        callback(model)


# Built-in plugins
from . import huggingface  # noqa: E402

# Model-specific plugins for VSA.  Guarded by ``import_plugin`` so the
# module-level imports stay soft — a missing dependency in one plugin must
# not break the core sparse-attention API.
with import_plugin("ltx2"):
    from . import ltx2

with import_plugin("wan22"):
    from . import wan22

__all__ = [
    "CUSTOM_MODEL_PLUGINS",
    "register_custom_model_plugins_on_the_fly",
]
