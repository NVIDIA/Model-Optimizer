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
        num_sink_tokens: KV positions before this index kept dense (attention sinks).
        dense_window_size: Tokens near diagonal kept dense (absolute token count).
    """

    def __init__(self, method_config=None):
        """Initialize with N:M sparsity parameters from config."""
        super().__init__()
        method_config = method_config or {}
        self.sparsity_n = method_config.get("sparsity_n", 2)
        self.sparsity_m = method_config.get("sparsity_m", 4)
        self.num_sink_tokens = method_config.get("num_sink_tokens", 0)
        self.dense_window_size = method_config.get("dense_window_size", 64)

    @property
    def name(self) -> str:
        """Method name identifier."""
        return "triton_sparse_softmax"

    # calculate_sparsity and apply_sparsity use base class defaults
    # (no-op mask and NotImplementedError) — sparsity is fused into the Triton kernel.

    def get_sparse_context(self, module):
        """Return context manager that activates N:M sparse softmax during forward."""

        @contextmanager
        def _sparse_nm_context():
            module._apply_sparse_nm = True
            try:
                yield
            finally:
                module._apply_sparse_nm = False

        return _sparse_nm_context()
