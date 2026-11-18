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

"""Quantization package."""

# Initialize mode and plugins
from . import mode, plugins, utils

# Add methods to mtq namespace
from .compress import *
from .config import *
from .conversion import *
from .model_quant import *
from .nn.modules.quant_module import QuantModuleRegistry
from .utils import update_quant_cfg_with_kv_cache_quant
