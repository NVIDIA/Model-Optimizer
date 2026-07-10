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

"""Unit tests for TritonSparseSoftmaxMethod (no GPU required)."""

import pytest
import torch

from modelopt.torch.sparsity.attention_sparsity.methods.triton_sparse_softmax import (
    TritonSparseSoftmaxMethod,
)


class TestInit:
    def test_default_config(self):
        m = TritonSparseSoftmaxMethod()
        assert m.sparsity_n == 2
        assert m.sparsity_m == 4
        assert m.dense_sink_tokens == 0
        assert m.dense_recent_tokens == 64

    def test_custom_config(self):
        m = TritonSparseSoftmaxMethod(
            {
                "sparsity_n": 4,
                "sparsity_m": 8,
                "dense_sink_tokens": 16,
                "dense_recent_tokens": 0,
            }
        )
        assert m.sparsity_n == 4
        assert m.sparsity_m == 8
        assert m.dense_sink_tokens == 16
        assert m.dense_recent_tokens == 0

    def test_name(self):
        assert TritonSparseSoftmaxMethod().name == "triton_sparse_softmax"

    def test_threshold_info_reports_pattern(self):
        m = TritonSparseSoftmaxMethod({"sparsity_n": 2, "sparsity_m": 4})
        info = m.get_threshold_info()
        assert info["type"] == "fixed"
        assert info["pattern"] == "2:4"


class TestSparseContext:
    def test_sets_module_flag(self):
        m = TritonSparseSoftmaxMethod()
        module = torch.nn.Linear(4, 4)
        with m.get_sparse_context(module):
            assert module._apply_sparse_nm is True
        assert module._apply_sparse_nm is False

    def test_pushes_config_to_diffusers_backend(self):
        """The context forwards N:M params to the diffusers thread-local config."""
        pytest.importorskip("diffusers.models.attention_dispatch")
        from modelopt.torch.kernels.sparsity.attention import (
            diffusers_triton_attention as diffusers_mod,
        )

        m = TritonSparseSoftmaxMethod({"sparsity_n": 2, "sparsity_m": 4})
        module = torch.nn.Linear(4, 4)
        with m.get_sparse_context(module):
            assert diffusers_mod._thread_local.sparsity_n == 2
            assert diffusers_mod._thread_local.sparsity_m == 4
        assert diffusers_mod._thread_local.sparsity_n == 0

    def test_pushes_config_to_ltx_backend(self):
        """The context forwards N:M params to the LTX thread-local config."""
        from modelopt.torch.kernels.sparsity.attention import ltx_triton_attention as ltx_mod

        m = TritonSparseSoftmaxMethod({"sparsity_n": 4, "sparsity_m": 8})
        module = torch.nn.Linear(4, 4)
        with m.get_sparse_context(module):
            assert ltx_mod._thread_local.active is True
            assert ltx_mod._thread_local.sparsity_n == 4
            assert ltx_mod._thread_local.sparsity_m == 8
        assert ltx_mod._thread_local.active is False
        assert ltx_mod._thread_local.sparsity_n == 0

    def test_context_clears_on_exception(self):
        m = TritonSparseSoftmaxMethod()
        module = torch.nn.Linear(4, 4)
        try:
            with m.get_sparse_context(module):
                raise RuntimeError("boom")
        except RuntimeError:
            pass
        assert module._apply_sparse_nm is False
