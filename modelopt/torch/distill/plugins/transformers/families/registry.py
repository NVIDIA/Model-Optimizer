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

"""Resolve HF family descriptors through text configuration metadata."""

from .base import TransformersFamilyAdapter

__all__ = ["decoder_layers", "register_transformers_family", "transformers_family"]

_FAMILIES = {}
_DEFAULT = TransformersFamilyAdapter()


def register_transformers_family(adapter, *model_types):
    """Register a thin compatibility adapter for one or more HF model types."""
    names = model_types or (adapter.name,)
    duplicates = set(names).intersection(_FAMILIES)
    if duplicates:
        raise ValueError(f"HF repair families already registered: {sorted(duplicates)}")
    _FAMILIES.update(dict.fromkeys(names, adapter))


def transformers_family(model):
    """Use the text model's own type, keeping wrapper and tower types separate."""
    config = getattr(model, "config", None)
    family = _FAMILIES.get(getattr(config, "model_type", None))
    if family is not None:
        return family
    get_text_config = getattr(config, "get_text_config", None)
    if callable(get_text_config):
        config = get_text_config()
    return _FAMILIES.get(getattr(config, "model_type", None), _DEFAULT)


def decoder_layers(model):
    """Return registered transformer layers using the model's family descriptor."""
    return transformers_family(model).decoder_layers(model)
