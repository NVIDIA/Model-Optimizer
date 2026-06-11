# Vendored from NVIDIA-NeMo/Automodel @ e42584e3 (Apache-2.0):
#   https://github.com/NVIDIA-NeMo/Automodel/blob/e42584e303397e9bd34643407b8a57d7def88ce9/tools/diffusion/processors/__init__.py
# Trimmed to the Qwen-Image image path: drops the flux / wan / hunyuan processors (and their
# extra deps, e.g. nemo_automodel.shared.transformers_patches) that the fastgen example does
# not use. The Qwen-Image processor self-registers with ProcessorRegistry on import.
# Original license below.
#
# Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

from .base import BaseModelProcessor
from .base_video import BaseVideoProcessor
from .caption_loaders import (
    CaptionLoader,
    CaptionLoadingStats,
    JSONLCaptionLoader,
    JSONSidecarCaptionLoader,
    MetaJSONCaptionLoader,
    get_caption_loader,
)
from .qwen_image import QwenImageProcessor
from .registry import ProcessorRegistry

__all__ = [
    "BaseModelProcessor",
    "BaseVideoProcessor",
    "CaptionLoader",
    "CaptionLoadingStats",
    "JSONLCaptionLoader",
    "JSONSidecarCaptionLoader",
    "MetaJSONCaptionLoader",
    "ProcessorRegistry",
    "QwenImageProcessor",
    "get_caption_loader",
]
