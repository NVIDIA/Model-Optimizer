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

"""Skip-softmax method for attention via Triton kernel tile skipping."""

from contextlib import contextmanager

from .registry import SparseAttentionMethod, register_sparse_method


@register_sparse_method("triton_skip_softmax")
class TritonSkipSoftmaxMethod(SparseAttentionMethod):
    """Skip-softmax tile skipping via the Triton flash attention kernel.

    During prefill, KV tiles whose max attention score is far below the
    running softmax max are skipped entirely — no V load, no softmax
    update, no accumulation. This is a long-context optimization that
    benefits sequences with strong attention locality.

    Config params:
        skip_softmax_threshold: Tiles contributing less than this fraction
            are skipped. Typical values: 1e-3 to 1e-1. Set to 0 to disable.
    """

    def __init__(self, method_config=None):
        """Initialize with skip-softmax threshold from config."""
        super().__init__()
        method_config = method_config or {}
        self.skip_softmax_threshold = method_config.get("skip_softmax_threshold", 0.1)

    @property
    def name(self) -> str:
        """Method name identifier."""
        return "triton_skip_softmax"

    def get_sparse_context(self, module):
        """Return context manager that activates skip-softmax during forward."""

        @contextmanager
        def _skip_softmax_context():
            module._apply_skip_softmax = True
            try:
                yield
            finally:
                module._apply_skip_softmax = False

        return _skip_softmax_context()
