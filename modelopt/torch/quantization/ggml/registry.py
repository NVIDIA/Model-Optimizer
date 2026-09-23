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

"""The GGML IQ formats, listed once for backend dispatch and export."""

from .common import IQFormat
from .iq1_s import IQ1_S_FORMAT
from .iq2_xs import IQ2_XS_FORMAT
from .iq2_xxs import IQ2_XXS_FORMAT

__all__ = ["IQ_FORMAT_REGISTRY", "IQFormat"]

# Every IQ format, keyed by the name a quantizer's num_bits carries, in increasing bits per
# weight. Backend dispatch and both exporters read this mapping and export's IQ_FORMATS is derived
# from it, so adding a format is one entry here rather than a row in several parallel tables.
#
# It is an explicit list rather than formats registering themselves on import, so its contents
# never depend on which modules happen to have been imported first.
IQ_FORMAT_REGISTRY: dict[str, IQFormat] = {
    fmt.name: fmt for fmt in (IQ1_S_FORMAT, IQ2_XXS_FORMAT, IQ2_XS_FORMAT)
}
