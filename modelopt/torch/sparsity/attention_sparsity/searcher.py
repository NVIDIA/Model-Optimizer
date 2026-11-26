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

"""Searcher that calibrates sparse attention thresholds."""

from modelopt.torch.opt.searcher import BaseSearcher, SearchStateDict

from .calibration import calibrate_sparse_attention
from .conversion import get_sparse_attention_mode_state

__all__ = ["SparseAttentionSearcher"]


class SparseAttentionSearcher(BaseSearcher):
    """Calibrate sparse attention thresholds with the forward loop of the search config.

    Calibration is configured by ``sparse_cfg["calibration"]`` of the model's ``sparse_attention``
    mode config and is skipped without it. Without a ``forward_loop`` or ``data_loader``, a RULER
    calibration dataset is generated from the model's tokenizer.
    """

    @property
    def default_state_dict(self) -> SearchStateDict:
        """Return default state dict."""
        return {}

    def run_search(self) -> None:
        """Calibrate the sparse attention modules of the model."""
        mode_state = get_sparse_attention_mode_state(self.model)
        assert mode_state is not None, "Model has no sparse_attention mode to calibrate!"
        calibrate_sparse_attention(self.model, mode_state[0], forward_loop=self.forward_loop)
