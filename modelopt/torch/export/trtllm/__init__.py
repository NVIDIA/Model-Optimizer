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

"""Checkpoint export logic for the TensorRT-LLM specific format.

**Deprecation Notice**: The ``export_tensorrt_llm_checkpoint`` API is deprecated as of 0.48.0 and
will be removed in 0.49.0. Users are encouraged to transition to the unified HF export API
(:meth:`export_hf_checkpoint <modelopt.torch.export.unified_export_hf.export_hf_checkpoint>`),
which provides enhanced functionality and flexibility for exporting models to multiple inference
frameworks including TensorRT-LLM, vLLM, and SGLang.
"""

from .model_config import *
from .model_config_export import *
