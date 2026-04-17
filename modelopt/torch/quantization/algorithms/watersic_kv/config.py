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

"""Configuration for the WaterSIC KV-cache quantization algorithm."""

from __future__ import annotations

from typing import Literal

from modelopt.torch.opt.config import ModeloptField
from modelopt.torch.quantization.config import QuantizeAlgorithmConfig


class WaterSICKVCalibConfig(QuantizeAlgorithmConfig):
    """Configuration for WaterSIC KV-cache quantization.

    WaterSIC (Water-filling Successive Interference Cancellation) is a
    rate-adaptive quantization method for KV-cache compression.  It
    applies the ZSIC algorithm with optional KL-aware importance
    weighting and LMMSE shrinkage correction to minimize attention-output
    distortion at a target bits-per-element budget.

    Reference: "WaterSIC: Water-filling Successive Interference
    Cancellation for KV-Cache Quantization" (2024).
    """

    method: Literal["watersic_kv"] = ModeloptField(
        "watersic_kv",
        title="Calibration algorithm identifier.",
        description="Fixed identifier for the WaterSIC KV-cache calibration method.",
    )

    target_rate: float = ModeloptField(
        default=2.0,
        gt=0.0,
        title="Target bits per element.",
        description=(
            "Average number of bits per quantized KV-cache element.  The binary "
            "search over the ZSIC damping parameter c is driven to hit this rate."
        ),
    )

    kl_aware: bool = ModeloptField(
        default=False,
        title="Enable KL-aware importance weighting.",
        description=(
            "When True, per-token importance weights derived from the attention "
            "distribution are folded into the Hessian so that tokens with higher "
            "attention mass receive tighter quantization."
        ),
    )

    importance_clip: float = ModeloptField(
        default=50.0,
        gt=0.0,
        title="Importance weight clipping ratio.",
        description=(
            "Maximum ratio by which a single token's importance weight may exceed "
            "the mean weight.  Clips extreme outlier tokens to prevent them from "
            "dominating the Hessian estimate."
        ),
    )

    use_lmmse: bool = ModeloptField(
        default=True,
        title="Apply LMMSE shrinkage correction.",
        description=(
            "When True, the LMMSE (Linear Minimum Mean-Squared Error) shrinkage "
            "correction is applied after ZSIC quantization to partially undo "
            "quantization bias and reduce reconstruction NMSE."
        ),
    )

    n_rescaler_iters: int = ModeloptField(
        default=0,
        ge=0,
        title="Diagonal rescaler optimization iterations.",
        description=(
            "Number of coordinate-descent iterations for the diagonal rescaler "
            "that adjusts per-column scale factors after LMMSE.  Set to 0 to "
            "disable the rescaler (faster but slightly higher distortion)."
        ),
    )

    sample_frac: float | None = ModeloptField(
        default=None,
        title="Row subsampling fraction for binary search.",
        description=(
            "If set, only this fraction of rows (KV heads) are used during the "
            "binary search for c.  Full rows are then quantized with the found c.  "
            "Speeds up calibration on large KV caches at a small accuracy cost."
        ),
    )

    use_sequential: bool = ModeloptField(
        default=False,
        title="Enable sequential layer-by-layer calibration.",
        description=(
            "Must be False for WaterSIC. Unlike weight quantization, KV-cache "
            "quantization does not have progressive error accumulation between "
            "layers, so sequential calibration is not needed."
        ),
    )
