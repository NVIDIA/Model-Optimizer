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

"""Export to the TensorRT-LLM checkpoint format.

Everything reachable only from :func:`export_tensorrt_llm_checkpoint
<modelopt.torch.export.trtllm.model_config_export.export_tensorrt_llm_checkpoint>` lives
here, so the framework-agnostic export code in :mod:`modelopt.torch.export` stays free of
TensorRT-LLM checkpoint concerns. This subpackage is deliberately *not* re-exported from
:mod:`modelopt.torch.export`; import from ``modelopt.torch.export.trtllm`` directly.
"""

from .model_config import *
from .model_config_export import *
