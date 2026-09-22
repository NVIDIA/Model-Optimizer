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


"""Backward-compatible alias; the helpers moved to ``modelopt.torch.kernels.common.attention``.

They are building blocks of the shared flash-attention kernel, so they live next to it;
the sparsity package depends on the common kernel package, never the other way round.
"""

from modelopt.torch.kernels.common.attention.skip_softmax_helpers import (
    _apply_sparse_nm_to_qk_tile,
    _skip_softmax_decision,
    _sparse_nm_masks_m4,
)

__all__ = ["_apply_sparse_nm_to_qk_tile", "_skip_softmax_decision", "_sparse_nm_masks_m4"]
