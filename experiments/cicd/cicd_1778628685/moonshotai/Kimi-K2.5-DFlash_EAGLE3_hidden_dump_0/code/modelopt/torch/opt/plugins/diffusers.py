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

"""ModelOpt plugin for enabling automatic save/restore of ModelOpt state for `diffusers` library."""

import types

from diffusers import ModelMixin

from .huggingface import _new_save_pretrained, _patch_model_init_for_modelopt, register_for_patching

__all__ = []


def _new_from_pretrained(cls, /, pretrained_model_name_or_path, *args, **kwargs):
    """Patch for `cls.from_pretrained` method to restore ModelOpt state."""
    with _patch_model_init_for_modelopt(cls, pretrained_model_name_or_path):
        model = types.MethodType(cls._modelopt_cache["from_pretrained"].__func__, cls)(
            pretrained_model_name_or_path, *args, **kwargs
        )
    return model


modelmixin_patch_methods = [
    ("from_pretrained", classmethod(_new_from_pretrained)),
    ("save_pretrained", _new_save_pretrained),
]
register_for_patching("diffusers", ModelMixin, modelmixin_patch_methods)
