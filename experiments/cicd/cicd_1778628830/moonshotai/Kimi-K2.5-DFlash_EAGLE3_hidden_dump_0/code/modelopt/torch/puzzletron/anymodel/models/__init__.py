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

from packaging.version import Version as _Version
from transformers import __version__ as _transformers_version

# Import models to trigger factory registration
from .gpt_oss import *
from .llama import *
from .mistral_small import *
from .nemotron_h import *
from .nemotron_h_v2 import *
from .qwen2 import *
from .qwen3 import *

if _Version(_transformers_version) >= _Version("4.57.0"):
    from .qwen3_vl import *
