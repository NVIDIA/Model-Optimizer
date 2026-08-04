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

import warnings

from .adapter import LTX2TrainingForwardAdapter, create_ltx2_adapter
from .loader import LTX2ModelLoader
from .pipeline import LTX2InferencePipeline

warnings.warn(
    "LTX-2 packages (ltx-core, ltx-pipelines, ltx-trainer) are provided by Lightricks and are "
    "NOT covered by the Apache 2.0 license governing NVIDIA Model Optimizer. You MUST comply "
    "with the LTX Community License Agreement when installing and using LTX-2 with NVIDIA Model "
    "Optimizer. Any derivative models or fine-tuned weights from LTX-2 (including quantized or "
    "distilled checkpoints) remain subject to the LTX Community License Agreement, not Apache "
    "2.0. See: https://github.com/Lightricks/LTX-2/blob/main/LICENSE",
    UserWarning,
    stacklevel=1,
)

__all__ = [
    "LTX2InferencePipeline",
    "LTX2ModelLoader",
    "LTX2TrainingForwardAdapter",
    "create_ltx2_adapter",
]
