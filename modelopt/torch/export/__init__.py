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

"""Export package for Hugging Face and Megatron-based models."""

from .convert_hf_config import *
from .model_utils import *
from .moe_utils import *
from .plugins import *
from .quant_format import *
from .registry import *
from .shard_cast_utils import *
from .transformer_engine import *

# Deprecated: kept only to satisfy the migration period in the deprecation policy (README.md),
# which requires a deprecated feature to keep working while warning for one release. The
# TensorRT-LLM checkpoint export moved to ``modelopt.torch.export.trtllm`` in 0.48.0; these two
# names are its previously documented import path. Both warn on call. Remove this re-export in
# 0.49.0 -- nothing inside this package may depend on it.
from .trtllm.model_config_export import (
    export_tensorrt_llm_checkpoint,
    torch_to_tensorrt_llm_checkpoint,
)
from .unified_export_hf import *
from .unified_export_megatron import *
