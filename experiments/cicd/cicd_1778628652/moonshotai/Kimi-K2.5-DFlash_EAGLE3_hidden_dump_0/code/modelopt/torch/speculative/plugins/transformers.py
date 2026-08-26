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

"""Deprecated shim — moved to :mod:`modelopt.torch.speculative.plugins.hf_eagle`."""

import warnings

warnings.warn(
    "modelopt.torch.speculative.plugins.transformers has been renamed to "
    "modelopt.torch.speculative.plugins.hf_eagle. Update your imports; this "
    "shim will be removed in a future release.",
    DeprecationWarning,
    stacklevel=2,
)

from .hf_eagle import *  # noqa: E402, F403
from .hf_medusa import *  # noqa: E402, F403
