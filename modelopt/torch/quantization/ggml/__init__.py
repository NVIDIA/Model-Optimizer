# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

"""GGML-compatible block quantization formats."""

# Importing the backend installs its TensorQuantizer dispatch entry.
from . import backend as _backend
from .iq1_s import *
from .iq1_s import __all__ as _iq1_s_all
from .iq2_xs import *
from .iq2_xs import __all__ as _iq2_xs_all
from .iq2_xxs import *
from .iq2_xxs import __all__ as _iq2_xxs_all
from .q8_0 import *
from .q8_0 import __all__ as _q8_0_all
from .registry import GGML_FORMAT_REGISTRY, IQ_FORMAT_REGISTRY, GGMLFormat, IQFormat

__all__ = [  # noqa: PLE0604
    *_iq1_s_all,
    *_iq2_xs_all,
    *_iq2_xxs_all,
    *_q8_0_all,
    "GGML_FORMAT_REGISTRY",
    "GGMLFormat",
    "IQ_FORMAT_REGISTRY",
    "IQFormat",
]
