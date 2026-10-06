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

"""Module for general-purpose model optimization infrastructure.

``modelopt.torch.opt`` contains tooling to:

* ingest a user-provided model and set it up for optimization;
* define and implement search and optimization procedures;
* export a model back to a regular pytorch model after optimization;
* save, restore, and manage checkpoints from which the model modifications can be resumed.

Please refer to each individual sub-module to learn more about the various concepts wihin
``modelopt.torch.opt`` and how to use them to implement a model optimization algorithm.
"""

from modelopt.torch.utils.import_utils import import_plugin

with import_plugin("opt_hooks", verbose=False):
    from . import _hooks

with import_plugin("huggingface", verbose=False):
    from .plugins.huggingface import *

from . import plugins, utils
from .config import *
from .conversion import *
from .mode import *
from .searcher import *
