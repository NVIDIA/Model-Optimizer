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

"""MLlama family export specs."""

from ...model_config import AttentionConfig
from ..base import ModelSpec
from ..hooks import ModelHooks
from ..registry import register


class MLlamaHooks(ModelHooks):
    """MLlama interleaves self- and cross-attention layers; route by class name."""

    def place_submodule(self, name, module, built, layer_config, ctx):
        """Route attention into the self- or cross-attention slot by module class name."""
        if isinstance(built, AttentionConfig):
            if "cross" in type(module).__name__.lower():
                layer_config.cross_attention = built
            else:
                layer_config.self_attention = built
            return True
        return False


register(
    ModelSpec(
        name="mllama",
        decoder_types=("mllama",),
        hooks=MLlamaHooks(),
    )
)
