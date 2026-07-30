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

"""End-to-end test that resolve_descriptor_from_pretrained caches dynamic modules."""

import pytest

pytest.importorskip("mamba_ssm")

import modelopt.torch.puzzletron as mtpz

MODEL_ID = "nvidia/NVIDIA-Nemotron-Nano-12B-v2-Base"


def test_resolve_descriptor_caches_dynamic_modules():
    """resolve_descriptor_from_pretrained must cache dynamic modules so decoder_layer_cls works."""
    descriptor = mtpz.anymodel.resolve_descriptor_from_pretrained(MODEL_ID, trust_remote_code=True)

    layer_classes = descriptor.decoder_layer_cls()
    assert layer_classes, (
        "decoder_layer_cls() returned empty after resolve_descriptor_from_pretrained"
    )
    print(
        f"  Descriptor: {descriptor.__name__}, decoder classes: {[c.__name__ for c in layer_classes]}"
    )
