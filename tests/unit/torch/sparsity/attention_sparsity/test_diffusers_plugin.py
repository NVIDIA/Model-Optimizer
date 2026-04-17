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

"""Unit tests for the diffusers WAN sparse attention via the modelopt_triton backend.

Tests cover:
- Config validation: "triton" backend is accepted; "diffusers_triton" is rejected
- quantize_p is NOT in SparseAttentionAttributeConfig (it moved to quantization.sage_attention)
- triton_sparse_softmax and triton_skip_softmax do NOT have quantize_p attribute
- diffusers_triton_attention: set_triton_skip_softmax_config does NOT touch quantize_p
- diffusers_triton_attention: set_sage_attention_config / clear_sage_attention_config
- clear_triton_skip_softmax_config does NOT reset quantize_p (composability)
"""

import pytest

pytest.importorskip("diffusers")

import torch.nn as nn

# ---------------------------------------------------------------------------
# Configs (no quantize_p — that's now a quantization feature)
# ---------------------------------------------------------------------------


_SPARSE_CFG = {
    "sparse_cfg": {
        "*": {
            "method": "triton_sparse_softmax",
            "sparsity_n": 2,
            "sparsity_m": 4,
            "num_sink_tokens": 0,
            "dense_window_size": 0,
            "backend": "triton",
            "enable": True,
        },
        "default": {"enable": False},
    }
}

_SKIP_CFG = {
    "sparse_cfg": {
        "*": {
            "method": "triton_skip_softmax",
            "skip_softmax_threshold": 0.1,
            "backend": "triton",
            "enable": True,
        },
        "default": {"enable": False},
    }
}


# ---------------------------------------------------------------------------
# Tests: config validation
# ---------------------------------------------------------------------------


class TestTritonBackend:
    """Validate that the "triton" backend is accepted and "diffusers_triton" is rejected."""

    def test_backend_accepted_sparse(self):
        from modelopt.torch.sparsity.attention_sparsity.config import SparseAttentionAttributeConfig

        cfg = SparseAttentionAttributeConfig(backend="triton", method="triton_sparse_softmax")
        assert cfg.backend == "triton"

    def test_backend_accepted_skip(self):
        from modelopt.torch.sparsity.attention_sparsity.config import SparseAttentionAttributeConfig

        cfg = SparseAttentionAttributeConfig(backend="triton", method="triton_skip_softmax")
        assert cfg.backend == "triton"

    def test_diffusers_triton_backend_rejected(self):
        from pydantic import ValidationError

        from modelopt.torch.sparsity.attention_sparsity.config import SparseAttentionAttributeConfig

        with pytest.raises(ValidationError):
            SparseAttentionAttributeConfig(backend="diffusers_triton")

    def test_invalid_backend_still_rejected(self):
        from pydantic import ValidationError

        from modelopt.torch.sparsity.attention_sparsity.config import SparseAttentionAttributeConfig

        with pytest.raises(ValidationError):
            SparseAttentionAttributeConfig(backend="unknown_backend")

    def test_quantize_p_not_in_sparse_config(self):
        """quantize_p moved to quantization.sage_attention — should not appear in sparse config."""
        from modelopt.torch.sparsity.attention_sparsity.config import SparseAttentionAttributeConfig

        cfg = SparseAttentionAttributeConfig(backend="triton", method="triton_sparse_softmax")
        assert not hasattr(cfg, "quantize_p"), (
            "quantize_p should NOT be a field on SparseAttentionAttributeConfig; "
            "it belongs to modelopt.torch.quantization.sage_attention"
        )


# ---------------------------------------------------------------------------
# Tests: sparse methods do NOT have quantize_p
# ---------------------------------------------------------------------------


class TestMethodNoQuantizeP:
    """triton_skip_softmax and triton_sparse_softmax must NOT expose quantize_p."""

    def test_skip_softmax_no_quantize_p(self):
        from modelopt.torch.sparsity.attention_sparsity.methods.triton_skip_softmax import (
            TritonSkipSoftmaxMethod,
        )

        m = TritonSkipSoftmaxMethod(method_config={"skip_softmax_threshold": 0.05})
        assert not hasattr(m, "quantize_p"), (
            "TritonSkipSoftmaxMethod must not have a quantize_p attribute; "
            "NVFP4 quantization is managed by modelopt.torch.quantization.sage_attention"
        )
        assert m.skip_softmax_threshold == pytest.approx(0.05)

    def test_sparse_softmax_no_quantize_p(self):
        from modelopt.torch.sparsity.attention_sparsity.methods.triton_sparse_softmax import (
            TritonSparseSoftmaxMethod,
        )

        m = TritonSparseSoftmaxMethod(method_config={"sparsity_n": 2, "sparsity_m": 4})
        assert not hasattr(m, "quantize_p"), (
            "TritonSparseSoftmaxMethod must not have a quantize_p attribute; "
            "NVFP4 quantization is managed by modelopt.torch.quantization.sage_attention"
        )
        assert m.sparsity_n == 2


# ---------------------------------------------------------------------------
# Tests: diffusers_triton_attention thread-local config
# ---------------------------------------------------------------------------


class TestDiffusersTritonAttentionConfig:
    """set/clear functions for sparse params and sage_attention params work correctly."""

    def test_set_and_get_threshold(self):
        from modelopt.torch.sparsity.attention_sparsity.kernels.diffusers_triton_attention import (
            _thread_local,
            clear_triton_skip_softmax_config,
            set_triton_skip_softmax_config,
        )

        set_triton_skip_softmax_config(threshold=0.05)
        assert _thread_local.skip_threshold == pytest.approx(0.05)
        clear_triton_skip_softmax_config()

    def test_set_triton_skip_softmax_config_no_quantize_p_param(self):
        """set_triton_skip_softmax_config must NOT accept quantize_p."""
        import inspect

        from modelopt.torch.sparsity.attention_sparsity.kernels.diffusers_triton_attention import (
            set_triton_skip_softmax_config,
        )

        sig = inspect.signature(set_triton_skip_softmax_config)
        assert "quantize_p" not in sig.parameters, (
            "set_triton_skip_softmax_config must not have a quantize_p parameter; "
            "use set_sage_attention_config() instead"
        )

    def test_set_and_get_sparsity_params(self):
        from modelopt.torch.sparsity.attention_sparsity.kernels.diffusers_triton_attention import (
            _thread_local,
            clear_triton_skip_softmax_config,
            set_triton_skip_softmax_config,
        )

        set_triton_skip_softmax_config(
            sparsity_n=2,
            sparsity_m=4,
            num_sink_tokens=16,
            dense_window_size=128,
        )
        assert _thread_local.sparsity_n == 2
        assert _thread_local.sparsity_m == 4
        assert _thread_local.num_sink_tokens == 16
        assert _thread_local.dense_window_size == 128
        clear_triton_skip_softmax_config()
        assert _thread_local.sparsity_n == 0
        assert _thread_local.sparsity_m == 4
        assert _thread_local.num_sink_tokens == 0
        assert _thread_local.dense_window_size == 64

    def test_clear_sparse_does_not_reset_quantize_p(self):
        """clear_triton_skip_softmax_config must NOT reset quantize_p.

        This is the key composability guarantee: SageAttention sets quantize_p=True
        once for the whole transformer forward; each per-layer sparsity context
        can clear its own sparse params without clobbering the outer quantize_p.
        """
        from modelopt.torch.sparsity.attention_sparsity.kernels.diffusers_triton_attention import (
            _thread_local,
            clear_sage_attention_config,
            clear_triton_skip_softmax_config,
            set_sage_attention_config,
            set_triton_skip_softmax_config,
        )

        # SageAttention outer wrapper sets quantize_p=True
        set_sage_attention_config(quantize_p=True)
        assert _thread_local.quantize_p is True

        # Sparsity per-layer context sets threshold then clears
        set_triton_skip_softmax_config(threshold=0.1)
        clear_triton_skip_softmax_config()

        # quantize_p must survive the sparsity clear
        assert _thread_local.quantize_p is True, (
            "clear_triton_skip_softmax_config() must not reset quantize_p; "
            "SageAttention controls quantize_p independently"
        )

        # SageAttention outer wrapper clears quantize_p
        clear_sage_attention_config()
        assert _thread_local.quantize_p is False

    def test_set_sage_attention_config(self):
        from modelopt.torch.sparsity.attention_sparsity.kernels.diffusers_triton_attention import (
            _thread_local,
            clear_sage_attention_config,
            set_sage_attention_config,
        )

        set_sage_attention_config(quantize_p=True)
        assert _thread_local.quantize_p is True
        clear_sage_attention_config()
        assert _thread_local.quantize_p is False


# ---------------------------------------------------------------------------
# Tests: apply_sage_attention API
# ---------------------------------------------------------------------------


class TestApplySageAttention:
    """apply_sage_attention wraps the transformer forward and marks the module."""

    def _make_dummy_transformer(self):
        """A minimal nn.Module whose forward returns a tensor."""

        class DummyTransformer(nn.Module):
            def forward(self, x):
                return x * 2

        return DummyTransformer()

    def test_marks_transformer(self):
        """apply_sage_attention sets _modelopt_sage_attention=True on the module."""
        pytest.importorskip("triton")

        from modelopt.torch.quantization import apply_sage_attention

        model = self._make_dummy_transformer()
        assert not hasattr(model, "_modelopt_sage_attention")
        apply_sage_attention(model)
        assert getattr(model, "_modelopt_sage_attention", False) is True

    def test_wraps_forward(self):
        """apply_sage_attention replaces forward with a wrapper function."""
        pytest.importorskip("triton")

        from modelopt.torch.quantization import apply_sage_attention

        model = self._make_dummy_transformer()
        original = model.forward
        apply_sage_attention(model)
        assert model.forward is not original

    def test_import_from_mtq(self):
        """apply_sage_attention is accessible via modelopt.torch.quantization."""
        import modelopt.torch.quantization as mtq

        assert hasattr(mtq, "apply_sage_attention")
        assert callable(mtq.apply_sage_attention)
