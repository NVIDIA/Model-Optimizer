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

"""Tests for resolve_descriptor_from_pretrained dynamic-module caching.

Verifies that resolve_descriptor_from_pretrained calls force_cache_dynamic_modules
so that decoder_layer_cls() works for models with custom code (e.g. Nemotron-H).
"""

from unittest.mock import MagicMock, patch

import pytest

pytest.importorskip("transformers")

import modelopt.torch.puzzletron as mtpz

MODEL_ID = "nvidia/NVIDIA-Nemotron-Nano-12B-v2-Base"

FACTORY_MODULE = "modelopt.torch.puzzletron.anymodel.model_descriptor.model_descriptor_factory"


class TestResolveDescriptorCachesDynamicModules:
    """resolve_descriptor_from_pretrained must call force_cache_dynamic_modules."""

    @patch(f"{FACTORY_MODULE}.force_cache_dynamic_modules")
    @patch(f"{FACTORY_MODULE}.AutoConfig")
    def test_force_cache_called(self, mock_auto_config_cls, mock_force_cache):
        mock_config = MagicMock()
        mock_config.model_type = "llama"
        mock_auto_config_cls.from_pretrained.return_value = mock_config

        mtpz.anymodel.resolve_descriptor_from_pretrained("/fake/path", trust_remote_code=True)

        mock_force_cache.assert_called_once_with(mock_config, "/fake/path", trust_remote_code=True)

    @patch(f"{FACTORY_MODULE}.force_cache_dynamic_modules")
    @patch(f"{FACTORY_MODULE}.AutoConfig")
    def test_force_cache_called_without_trust_remote_code(
        self, mock_auto_config_cls, mock_force_cache
    ):
        mock_config = MagicMock()
        mock_config.model_type = "llama"
        mock_auto_config_cls.from_pretrained.return_value = mock_config

        mtpz.anymodel.resolve_descriptor_from_pretrained("/fake/path")

        mock_force_cache.assert_called_once_with(mock_config, "/fake/path", trust_remote_code=False)


def test_resolve_descriptor_caches_dynamic_modules():
    """End-to-end: resolve_descriptor_from_pretrained must cache dynamic modules so decoder_layer_cls works."""
    pytest.importorskip("mamba_ssm")

    descriptor = mtpz.anymodel.resolve_descriptor_from_pretrained(MODEL_ID, trust_remote_code=True)

    layer_classes = descriptor.decoder_layer_cls()
    assert layer_classes, (
        "decoder_layer_cls() returned empty after resolve_descriptor_from_pretrained"
    )
    print(
        f"  Descriptor: {descriptor.__name__}, decoder classes: {[c.__name__ for c in layer_classes]}"
    )


@pytest.mark.parametrize(
    ("model_type", "descriptor_name"),
    [
        ("qwen3_5_text", "Qwen3P5TextModelDescriptor"),
        ("qwen3_5", "Qwen3P5VLModelDescriptor"),
        ("qwen3_6_text", "Qwen3P5TextModelDescriptor"),
        ("qwen3_6", "Qwen3P5VLModelDescriptor"),
    ],
)
@patch(f"{FACTORY_MODULE}.force_cache_dynamic_modules")
@patch(f"{FACTORY_MODULE}.AutoConfig")
def test_resolve_qwen3_5_and_qwen3_6_model_types(
    mock_auto_config_cls, mock_force_cache, model_type, descriptor_name
):
    pytest.importorskip("transformers.models.qwen3_5.modeling_qwen3_5")
    mock_config = MagicMock()
    mock_config.model_type = model_type
    mock_auto_config_cls.from_pretrained.return_value = mock_config

    descriptor = mtpz.anymodel.resolve_descriptor_from_pretrained("/fake/path")

    assert descriptor.__name__ == descriptor_name
    mock_force_cache.assert_called_once_with(mock_config, "/fake/path", trust_remote_code=False)
