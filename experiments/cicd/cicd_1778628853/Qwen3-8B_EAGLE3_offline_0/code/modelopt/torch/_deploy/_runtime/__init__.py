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

from modelopt.torch.utils.import_utils import import_plugin

from .registry import *
from .runtime_client import *

# no runtime_client_impl will be available if 'deploy' is not installed
with import_plugin("ort_client", verbose=False):
    from .ort_client import *


with import_plugin("trt_client", verbose=False):
    # ImportError if tensorrt is not installed
    # ModuleNotFoundError if .tensorrt/ is not available
    from .trt_client import *
