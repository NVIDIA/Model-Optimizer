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

"""Capabilities of the initial fused GDN fake-quant path."""

from ..nn import TensorQuantizer

__all__ = []


def validate_gdn_quantizer(
    quantizer: TensorQuantizer, *, state: bool, name: str | None = None
) -> None:
    name = name or ("gdn_state_quantizer" if state else "gdn_w_quantizer")
    axis = (0, 1) if state else (0, 1, 2)
    if not isinstance(quantizer, TensorQuantizer):
        raise ValueError(f"{name} requires a single TensorQuantizer")
    integer_state = (
        state and quantizer.num_bits == 8 and not quantizer.unsigned and quantizer.narrow_range
    )
    formats = "dynamic E4M3 or signed narrow-range INT8" if state else "dynamic E4M3"
    if not (
        quantizer._dynamic
        and (quantizer.num_bits == (4, 3) or integer_state)
        and quantizer.axis == axis
        and quantizer.block_sizes is None
        and quantizer.fake_quant
        and quantizer._pass_through_bwd
        and not quantizer.rotate_is_enabled
        and quantizer.pre_quant_scale is None
        and quantizer.backend is None
        and not quantizer._bias
        and not quantizer._use_constant_amax
    ):
        raise ValueError(
            f"{name} supports only {formats} fake quantization with axis={axis}, "
            "pass_through_bwd=True, no block_sizes, rotation, pre-scaling, bias, constant "
            "amax, or custom backend. Other gradient rules and formats are not implemented."
        )
