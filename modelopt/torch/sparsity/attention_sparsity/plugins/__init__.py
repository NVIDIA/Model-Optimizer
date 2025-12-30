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

# List of model plugins that are called during conversion
# Each plugin is a callable that takes (model) and performs validation/setup
CUSTOM_MODEL_PLUGINS: set = set()


def register_custom_model_plugins_on_the_fly(model):
    """Applies all registered custom model plugins."""
    for callback in CUSTOM_MODEL_PLUGINS:
        callback(model)


from . import huggingface  # noqa: E402

__all__ = [
    "CUSTOM_MODEL_PLUGINS",
    "register_custom_model_plugins_on_the_fly",
]
