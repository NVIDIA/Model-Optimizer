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

"""N:M sparse softmax method for attention scores via Triton kernel."""

from contextlib import contextmanager

from .registry import SparseAttentionMethod, register_sparse_method
from .triton_skip_softmax import (
    _clear_triton_backend_config,
    _diffusers_backend_context,
    _set_triton_backend_config,
)


@register_sparse_method("triton_sparse_softmax")
class TritonSparseSoftmaxMethod(SparseAttentionMethod):
    """N:M sparse softmax applied to attention scores via Triton kernel.

    Sparsity is applied inside the fused Triton flash attention kernel,
    not as a separate pre/post-processing step. For every M consecutive
    K positions, the top-N attention scores are kept; the other M-N are
    set to -inf before softmax.

    Config params:
        sparsity_n: Keep top-N of every M attention scores (0 to disable).
        sparsity_m: Group size (4 or 8).
        dense_sink_tokens: Leading KV tokens excluded from N:M sparsity and kept dense.
        dense_recent_tokens: Recent KV tokens excluded from N:M sparsity and kept dense.
    """

    def __init__(self, method_config=None):
        """Initialize with N:M sparsity parameters from config."""
        super().__init__()
        method_config = method_config or {}
        self.sparsity_n = method_config.get("sparsity_n", 2)
        self.sparsity_m = method_config.get("sparsity_m", 4)
        self.dense_sink_tokens = method_config.get("dense_sink_tokens", 0)
        self.dense_recent_tokens = method_config.get("dense_recent_tokens", 64)

    @property
    def name(self) -> str:
        """Method name identifier."""
        return "triton_sparse_softmax"

    def get_threshold_info(self) -> dict:
        """Return fixed N:M pattern info for display/debugging."""
        return {
            "type": "fixed",
            "pattern": f"{self.sparsity_n}:{self.sparsity_m}",
            "dense_sink_tokens": self.dense_sink_tokens,
            "dense_recent_tokens": self.dense_recent_tokens,
        }

    # calculate_sparsity and apply_sparsity use base class defaults
    # (no-op mask and NotImplementedError) — sparsity is fused into the Triton kernel.

    def get_sparse_context(self, module):
        """Return context manager that activates N:M sparse softmax during forward.

        Sets ``module._apply_sparse_nm`` for the HF (modelopt_triton) backend,
        which reads the N:M parameters from this method instance, and pushes the
        parameters to the diffusers/LTX Triton backends via their thread-local
        configs (those backends have no module handle at dispatch time).
        """

        @contextmanager
        def _sparse_nm_context():
            module._apply_sparse_nm = True
            _set_triton_backend_config(
                sparsity_n=self.sparsity_n,
                sparsity_m=self.sparsity_m,
                dense_sink_tokens=self.dense_sink_tokens,
                dense_recent_tokens=self.dense_recent_tokens,
            )
            with _diffusers_backend_context():
                try:
                    yield
                finally:
                    module._apply_sparse_nm = False
                    _clear_triton_backend_config()

        return _sparse_nm_context()
