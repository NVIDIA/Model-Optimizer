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

"""Distillation API subpackage for torch."""

from modelopt.torch.utils import import_plugin as _import_plugin

from . import mode
from .config import *
from .distillation import *
from .distillation_model import *
from .layerwise_distillation_model import *
from .loss_balancers import *
from .losses import *
from .registry import *

with _import_plugin("megatron", verbose=False):
    from .doge_megatron_loss import *

# isort: off
# Import plugins last to avoid circular imports
from . import plugins
