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

"""A simplified API for :meth:`modelopt.torch.nas<modelopt.torch.nas>` for pruning algorithms.

This module provides a simplified API for pruning that is based on the NAS infrastructure but
simplifies the overall workflow to accommodate for the simpler nature of pruning algorithms.
"""

# nas is a required - so let's check if it's available
import modelopt.torch.nas
from modelopt.torch.utils import import_plugin

from . import fastnas, gradnas, plugins
from .pruning import *

with import_plugin("mcore_minitron", verbose=False):
    from .plugins import mcore_minitron
