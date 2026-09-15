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

"""TensorQuantizer backend dispatch for GGML-compatible IQ formats."""

import torch

from ..nn.modules.tensor_quantizer import register_quant_backend
from .iq1_s import iq1_s_fake_quant
from .iq2_xs import iq2_xs_fake_quant


def ggml_fake_quant(inputs: torch.Tensor, quantizer) -> torch.Tensor:
    """Dispatch an IQ quantizer to its format-specific implementation."""
    num_bits = getattr(quantizer, "num_bits", None)
    if num_bits == "iq1_s":
        return iq1_s_fake_quant(inputs, quantizer)
    if num_bits == "iq2_xs":
        return iq2_xs_fake_quant(inputs, quantizer)
    raise ValueError("The psx_luts backend requires num_bits='iq1_s' or 'iq2_xs'")


register_quant_backend("psx_luts", ggml_fake_quant)
