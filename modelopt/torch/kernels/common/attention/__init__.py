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

"""Shared Triton kernels for modelopt (attention, quantization, etc.)."""

import torch

from modelopt.torch.utils import import_plugin

IS_AVAILABLE = False
attention = None
attention_calibrate = None
register_triton_attention = None

if torch.cuda.is_available():
    with import_plugin(
        "triton",
        msg_if_missing=(
            "Your device is potentially capable of using the triton attention "
            "kernel. Try to install triton with `pip install triton`."
        ),
    ):
        from .triton_fa import attention as _attention

        attention = _attention
        IS_AVAILABLE = True
        from .hf_triton_attention import register_triton_attention as _register_triton_attention

        register_triton_attention = _register_triton_attention

        # Calibration lives in the sparsity subpackage (skip-softmax specific).
        # Imported here so ``from modelopt.torch.kernels.common.attention import
        # attention_calibrate`` keeps working.
        from modelopt.torch.kernels.sparsity.attention.calibrate import (
            attention_calibrate as _attention_calibrate,
        )

        attention_calibrate = _attention_calibrate

__all__ = [
    "IS_AVAILABLE",
    "attention",
    "attention_calibrate",
    "register_triton_attention",
]
