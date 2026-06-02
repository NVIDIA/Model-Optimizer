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

"""GPU tests for skip-softmax calibration via the Triton backend on HF models.

These exercise the HuggingFace (``modelopt_triton``) wiring that routes the
calibration forward pass through the fused ``attention_calibrate`` kernel and
feeds the collected multi-threshold tile-skip statistics into the same
exponential-model fit used by the PyTorch path.
"""

import pytest
import torch
from _test_utils.torch.transformers_models import create_tiny_llama_dir
from transformers import AutoModelForCausalLM

import modelopt.torch.sparsity.attention_sparsity as mtsa
from modelopt.torch.sparsity.attention_sparsity.config import SKIP_SOFTMAX_TRITON_CALIB
from modelopt.torch.sparsity.attention_sparsity.sparse_attention import SparseAttentionModule

pytestmark = [
    pytest.mark.filterwarnings("ignore::UserWarning"),
    pytest.mark.filterwarnings("ignore::RuntimeWarning"),
]

from modelopt.torch.kernels.common.attention import IS_AVAILABLE as TRITON_KERNEL_AVAILABLE

# Thresholds spanning a wide range so the collected sparsity covers the (10%, 90%)
# window the exponential fit relies on.
THRESHOLD_TRIALS = [1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 5e-2, 1e-1, 3e-1, 5e-1, 7e-1, 9e-1]


@pytest.fixture(scope="module")
def tiny_llama_dir(tmp_path_factory):
    """Create a minimal Llama model directory."""
    return create_tiny_llama_dir(
        tmp_path_factory.mktemp("tiny_llama_triton_calib"),
        num_hidden_layers=2,
        hidden_size=64,
        intermediate_size=128,
        num_attention_heads=4,
        num_key_value_heads=2,
        max_position_embeddings=1024,
    )


def _load_eager(tiny_llama_dir):
    return AutoModelForCausalLM.from_pretrained(
        tiny_llama_dir, attn_implementation="eager", device_map="cuda"
    )


def _make_forward_loop(vocab_size, lengths=(128, 256, 384, 512)):
    """Forward loop that runs several full-prefill passes of varying length.

    Each pass triggers one ``attention_calibrate`` call per layer, producing one
    per-sample calibration record per length.
    """

    def forward_loop(model):
        torch.manual_seed(0)
        for seq_len in lengths:
            input_ids = torch.randint(0, vocab_size, (1, seq_len), device="cuda")
            with torch.no_grad():
                model(input_ids, use_cache=False)

    return forward_loop


@pytest.mark.skipif(not TRITON_KERNEL_AVAILABLE, reason="Need CUDA + triton")
class TestTritonCalibrationHF:
    """End-to-end calibration via the Triton backend on a tiny HF model."""

    def test_sparsify_triton_calib_sets_params(self, tiny_llama_dir):
        """Running SKIP_SOFTMAX_TRITON_CALIB fits a finite exponential model."""
        import copy

        model = _load_eager(tiny_llama_dir)

        # Use the calibrator's default (dense) threshold trials so the collected
        # sparsity densely covers the (10%, 90%) window the fit filters on.
        config = copy.deepcopy(SKIP_SOFTMAX_TRITON_CALIB)

        forward_loop = _make_forward_loop(model.config.vocab_size)
        sparse_model = mtsa.sparsify(model, config, forward_loop=forward_loop)

        # Backend dispatched to the Triton kernel.
        assert sparse_model.config._attn_implementation == "modelopt_triton"

        sparse_modules = [
            m for m in sparse_model.modules() if isinstance(m, SparseAttentionModule)
        ]
        assert len(sparse_modules) == 2

        # Calibration produced finite, in-bounds (a, b) for the prefill phase.
        for module in sparse_modules:
            method = module._sparse_method_instance
            assert method.name == "triton_skip_softmax"
            params = method.calibration_params
            assert params is not None and "prefill" in params
            a, b = params["prefill"]["a"], params["prefill"]["b"]
            assert a > 0 and torch.isfinite(torch.tensor(a))
            assert 0.0 <= b <= 20.0
            # Prefill-only: decode must not be calibrated.
            assert "decode" not in params

    def test_calibrated_model_inference(self, tiny_llama_dir):
        """A model calibrated through the Triton path still runs inference cleanly."""
        import copy

        model = _load_eager(tiny_llama_dir)
        config = copy.deepcopy(SKIP_SOFTMAX_TRITON_CALIB)

        forward_loop = _make_forward_loop(model.config.vocab_size)
        sparse_model = mtsa.sparsify(model, config, forward_loop=forward_loop)

        sparse_model.eval()
        input_ids = torch.randint(0, model.config.vocab_size, (1, 64), device="cuda")
        with torch.no_grad():
            out = sparse_model(input_ids, use_cache=False)
        assert out.logits is not None
        assert not torch.isnan(out.logits).any()


@pytest.mark.skipif(not TRITON_KERNEL_AVAILABLE, reason="Need CUDA + triton")
class TestHFBackendCalibrationCounters:
    """Lower-level checks on the HF backend's calibration branch."""

    def test_counters_monotonic_in_threshold(self):
        """Skipped-tile counts are non-decreasing as the threshold grows."""
        from modelopt.torch.kernels.common.attention.hf_triton_attention import (
            clear_hf_triton_skip_softmax_config,
            get_calibration_counters,
            get_calibration_seq_k,
            set_hf_triton_skip_softmax_config,
            triton_attention_forward,
        )

        batch, num_heads, seq_len, head_dim = 1, 4, 256, 64
        torch.manual_seed(0)
        q = torch.randn(batch, num_heads, seq_len, head_dim, device="cuda", dtype=torch.float16)
        k = torch.randn_like(q)
        v = torch.randn_like(q)

        # A bare module stand-in; the calibration branch returns before touching
        # any sparse-method attributes.
        module = torch.nn.Module()

        set_hf_triton_skip_softmax_config(
            calibration_mode=True, threshold_trials=THRESHOLD_TRIALS
        )
        try:
            out, _ = triton_attention_forward(
                module, q, k, v, attention_mask=None, scaling=1.0 / (head_dim**0.5)
            )
            counters = get_calibration_counters()
            seq_k = get_calibration_seq_k()
        finally:
            clear_hf_triton_skip_softmax_config()

        assert out.shape == (batch, seq_len, num_heads, head_dim)
        assert seq_k == seq_len
        assert counters is not None
        assert counters.shape == (len(THRESHOLD_TRIALS), 2)

        totals = counters[:, 0]
        skipped = counters[:, 1]
        assert torch.all(totals == totals[0])  # same tile count for every threshold
        assert torch.all(skipped[1:] >= skipped[:-1])  # monotonic non-decreasing
        assert torch.all(skipped <= totals)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
