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

"""API for sparsification algorithms.

This module provides access to both weight sparsity and attention sparsity algorithms through
:meth:`sparsify <modelopt.torch.sparsity.sparsification.sparsify>`.
"""

from . import attention_sparsity
from .attention_sparsity.config import *
from .sparsification import *

# Import weight sparsity for backward compatibility
from .weight_sparsity import mode, module, plugins
