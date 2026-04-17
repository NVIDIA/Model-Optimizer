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
# mypy: ignore-errors

"""AnyModel: Architecture-agnostic model compression for HuggingFace models.

This module provides a declarative approach to model compression that works with
any HuggingFace model without requiring custom modeling code. Instead of duplicating
HuggingFace modeling classes, AnyModel uses ModelDescriptors that define:

1. Which decoder layer class(es) to patch for heterogeneous configs
2. How to map BlockConfig to layer-specific overrides
3. Weight name patterns for subblock checkpointing

Example usage:
    >>> from modelopt.torch.puzzletron.anymodel import convert_model
    >>> convert_model(
    ...     input_dir="path/to/hf_checkpoint",
    ...     output_dir="path/to/anymodel_checkpoint",
    ...     converter="llama",
    ... )

Supported models:
    - llama: Llama 2, Llama 3, Llama 3.1, Llama 3.2
    - (more to come: qwen2, mistral_small, etc.)
"""

from . import models  # trigger factory registration
from .converter import *
from .model_descriptor import *
from .puzzformer import *
