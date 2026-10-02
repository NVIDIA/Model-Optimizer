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

"""TensorQuantizer backend dispatch for GGML-compatible weight-only IQ formats."""

import torch

from ..nn.modules.tensor_quantizer import register_quant_backend
from .registry import IQ_FORMAT_REGISTRY


def ggml_fake_quant(inputs: torch.Tensor, quantizer) -> torch.Tensor:
    """Dispatch an IQ quantizer to its format-specific implementation."""
    num_bits = getattr(quantizer, "num_bits", None)
    extra_args = getattr(quantizer, "backend_extra_args", None) or {}
    unknown_args = set(extra_args) - {"block_chunk_size", "decode_chunk_size"}
    if unknown_args:
        raise ValueError(f"Unsupported ggml backend_extra_args: {sorted(unknown_args)}")
    # num_bits arrives untyped from the quantizer and is a tuple for scalar formats,
    # so narrow before the lookup rather than relying on the dict to reject it.
    iq_format = IQ_FORMAT_REGISTRY.get(num_bits) if isinstance(num_bits, str) else None
    if iq_format is None:
        supported = ", ".join(repr(name) for name in sorted(IQ_FORMAT_REGISTRY))
        raise ValueError(f"The ggml backend requires num_bits in ({supported})")
    return iq_format.fake_quant(inputs, quantizer, **extra_args)


register_quant_backend("ggml", ggml_fake_quant)
