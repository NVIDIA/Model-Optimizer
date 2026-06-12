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

"""Handles speculative plugins for third-party modules.

Please check out the source code of this module for examples of how plugins work and how you can
write your own one. Currently, we support plugins for

- :meth:`transformers<modelopt.torch.speculative.plugins.transformers>`
"""

from modelopt.torch.utils import import_plugin

with import_plugin("megatron_eagle"):
    from .megatron_eagle import *

with import_plugin("megatron_medusa"):
    from .megatron_medusa import *

with import_plugin("transformers"):
    from .transformers import *
