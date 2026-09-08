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

import copy
import itertools

import pytest
import torch
from _test_utils.torch.transformers_models import create_tiny_llama_dir
from transformers import AutoModelForCausalLM

import modelopt.torch.sparsity.attention_sparsity as mtsa
from modelopt.torch.kernels.common.attention import IS_AVAILABLE as TRITON_KERNEL_AVAILABLE
from modelopt.torch.kernels.common.attention.hf_triton_attention import triton_attention_forward
from modelopt.torch.sparsity.attention_sparsity.config import SKIP_SOFTMAX_TRITON_CALIB
from modelopt.torch.sparsity.attention_sparsity.methods.triton_skip_softmax import (
    TritonSkipSoftmaxMethod,
)

pytestmark = [
    pytest.mark.filterwarnings("ignore::UserWarning"),
    pytest.mark.filterwarnings("ignore::RuntimeWarning"),
]

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


def _calibration_module(threshold_trials):
    """Build a bare module whose ``_sparse_method_instance`` is in calibration mode.

    The HF backend reads its calibration config from (and writes counters back
    to) ``module._sparse_method_instance``, so this is the minimal stand-in for
    driving ``triton_attention_forward`` through the calibration branch.
    """
    method = TritonSkipSoftmaxMethod()
    method.set_calibration_mode(True)
    method._threshold_trials = threshold_trials

    module = torch.nn.Module()
    module._sparse_method_instance = method
    return module


@pytest.mark.skipif(not TRITON_KERNEL_AVAILABLE, reason="Need CUDA + triton")
class TestTritonCalibrationHF:
    """End-to-end calibration via the Triton backend on a tiny HF model."""

    def test_calibrated_model_inference(self, tiny_llama_dir):
        """SKIP_SOFTMAX_TRITON_CALIB dispatches to the Triton backend and the
        calibrated model runs inference cleanly."""
        model = _load_eager(tiny_llama_dir)
        config = copy.deepcopy(SKIP_SOFTMAX_TRITON_CALIB)
        # Prefill-only (custom forward_loop can't drive RULER decode calibration).
        config["sparse_cfg"]["calibration"]["target_sparse_ratio"] = {"prefill": 0.5}

        forward_loop = _make_forward_loop(model.config.vocab_size)
        sparse_model = mtsa.sparsify(model, config, forward_loop=forward_loop)
        assert sparse_model.config._attn_implementation == "modelopt_triton"

        sparse_model.eval()
        input_ids = torch.randint(0, model.config.vocab_size, (1, 64), device="cuda")
        with torch.no_grad():
            out = sparse_model(input_ids, use_cache=False)
        assert out.logits is not None
        assert not torch.isnan(out.logits).any()

    def test_decode_branch_reports_decode_phase(self):
        """The HF calibration branch routes decode-shaped calls through the kernel
        and surfaces its counters as a ``decode``-phase stats record.

        This is the HF-only counter path in ``_collect_calibration_stats``; the
        kernel's skip-count behavior itself is covered in the kernel test suite.
        """
        num_heads, seq_k, head_dim = 4, 512, 64
        torch.manual_seed(0)
        q = torch.randn(1, num_heads, 1, head_dim, device="cuda", dtype=torch.float16)
        k = torch.randn(1, num_heads, seq_k, head_dim, device="cuda", dtype=torch.float16)
        v = torch.randn(1, num_heads, seq_k, head_dim, device="cuda", dtype=torch.float16)

        module = _calibration_module(THRESHOLD_TRIALS)
        method = module._sparse_method_instance
        triton_attention_forward(module, q, k, v, attention_mask=None, scaling=1.0 / head_dim**0.5)
        assert method._hf_calibration_is_decode is True
        assert method._hf_calibration_counters is not None

        method._collect_calibration_stats(module)
        assert module._last_stats["phase"] == "decode"
        assert len(module._last_stats["sparsity"]) == len(THRESHOLD_TRIALS)

    def test_decode_calibration_measures_full_cache_with_sink(self):
        """Decode calibration must scan the whole KV cache and report real sparsity.

        A dominant sink at position 0 makes the distant KV tiles negligible, so a
        correct decode measurement skips almost all of them. This guards the two
        decode bugs that random inputs don't expose:
          * causal-offset ``kv_bound`` — without it the loop stops after the first
            ``BLOCK_M`` tokens, so ``total`` would be a fraction of the cache.
          * padding-row exclusion — without it the 127 padding rows veto every
            tile and sparsity is 0%.
        """
        num_heads, seq_k, head_dim = 4, 2048, 64
        block_n = 128  # the calibration kernel measures at 128x128
        q = torch.ones(1, num_heads, 1, head_dim, device="cuda", dtype=torch.float16)
        k = torch.zeros(1, num_heads, seq_k, head_dim, device="cuda", dtype=torch.float16)
        k[:, :, 0] = 20.0  # attention sink dominates every query
        v = torch.randn(1, num_heads, seq_k, head_dim, device="cuda", dtype=torch.float16)

        module = _calibration_module(THRESHOLD_TRIALS)
        method = module._sparse_method_instance
        triton_attention_forward(module, q, k, v, attention_mask=None, scaling=1.0 / head_dim**0.5)

        counters = method._hf_calibration_counters
        total = int(counters[0, 0])
        # Full cache scanned (not truncated to the first block).
        assert total == num_heads * (seq_k // block_n), total
        sparsity = (counters[:, 1].float() / counters[:, 0].clamp(min=1)).tolist()
        # Sink => the vast majority of tiles are negligible and skippable (not 0%).
        assert max(sparsity) > 0.8, sparsity
        # Skipped-tile fraction is non-decreasing as the threshold grows.
        assert all(later >= earlier for earlier, later in itertools.pairwise(sparsity)), sparsity


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
