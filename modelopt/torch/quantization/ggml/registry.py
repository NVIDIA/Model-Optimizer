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

"""GGML block formats, listed once for backend dispatch and export."""

from .common import GGMLFormat, IQFormat
from .iq1_s import IQ1_S_FORMAT
from .iq2_xs import IQ2_XS_FORMAT
from .iq2_xxs import IQ2_XXS_FORMAT
from .q8_0 import Q8_0_FORMAT

__all__ = ["GGML_FORMAT_REGISTRY", "IQ_FORMAT_REGISTRY", "GGMLFormat", "IQFormat"]

# Every GGML block format, keyed by the name a quantizer's num_bits carries, in increasing bits
# per weight. Backend dispatch reads this mapping, and export will derive its supported formats
# from it rather than maintaining parallel metadata and packer tables.
#
# It is an explicit list rather than formats registering themselves on import, so its contents
# never depend on which modules happen to have been imported first.
GGML_FORMAT_REGISTRY: dict[str, GGMLFormat] = {
    fmt.name: fmt for fmt in (IQ1_S_FORMAT, IQ2_XXS_FORMAT, IQ2_XS_FORMAT, Q8_0_FORMAT)
}

# Compatibility view for the already-merged IQ export path. Export migrates to the general
# registry in the Q8_0 export PR; IQ-specific conformance tests continue to use this subset.
IQ_FORMAT_REGISTRY: dict[str, GGMLFormat] = {
    name: fmt for name, fmt in GGML_FORMAT_REGISTRY.items() if name.startswith("iq")
}
