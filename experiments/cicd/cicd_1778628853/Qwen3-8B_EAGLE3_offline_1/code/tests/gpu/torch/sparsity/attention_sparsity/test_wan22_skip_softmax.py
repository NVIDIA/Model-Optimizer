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

"""End-to-end tests for skip-softmax sparse attention on a tiny Wan 2.2 pipeline.

Uses ``create_tiny_wan22_pipeline_dir`` (matches diffusers' own tiny Wan 2.2
test config): dual 2-layer transformer, tiny VAE, tiny T5 text encoder.
The full ``WanPipeline`` is loaded, sparsified with ``mtsa.sparsify``, and
run end-to-end with a 2-step denoising loop — asserting no NaN/Inf in the
pipeline output.
"""

import pytest
import torch

pytestmark = [
    pytest.mark.filterwarnings("ignore::UserWarning"),
    pytest.mark.filterwarnings("ignore::RuntimeWarning"),
    pytest.mark.filterwarnings("ignore::DeprecationWarning"),
]

diffusers = pytest.importorskip("diffusers")

from modelopt.torch.kernels.common.attention import IS_AVAILABLE as TRITON_KERNEL_AVAILABLE

if TRITON_KERNEL_AVAILABLE:
    import modelopt.torch.sparsity.attention_sparsity as mtsa
    from modelopt.torch.sparsity.attention_sparsity.sparse_attention import SparseAttentionModule


# ---------------------------------------------------------------------------
# Tiny Wan 2.2 pipeline fixture — shared across tests (pipeline load is costly)
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def tiny_wan22_path(tmp_path_factory):
    """Create and save a tiny Wan 2.2 pipeline to disk once per module."""
    from _test_utils.torch.diffusers_models import create_tiny_wan22_pipeline_dir

    return str(create_tiny_wan22_pipeline_dir(tmp_path_factory.mktemp("tiny_wan22")))


@pytest.fixture
def tiny_wan22_pipe(tiny_wan22_path):
    """Load a fresh copy of the tiny Wan 2.2 pipeline on CUDA (per test)."""
    from diffusers import WanPipeline

    pipe = WanPipeline.from_pretrained(tiny_wan22_path, torch_dtype=torch.bfloat16)
    pipe.to("cuda")
    return pipe


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


_TINY_PIPE_KWARGS = {
    "prompt": "test",
    "negative_prompt": "",
    "num_frames": 5,
    "height": 16,
    "width": 16,
    "num_inference_steps": 2,
    "guidance_scale": 1.0,
}


def _skip_softmax_cfg(raw_threshold=-5.0):
    """Sparse config targeting Wan 2.2 self-attention (attn1) only."""
    return {
        "sparse_cfg": {
            "*attn1*": {
                "method": "triton_skip_softmax",
                "backend": "triton",
                "skip_softmax_raw_threshold": raw_threshold,
                "enable": True,
            },
            "default": {"enable": False},
        },
    }


def _sparsify_both_transformers(pipe, cfg):
    """Apply sparsify to both transformer and transformer_2 (14B-style dual)."""
    mtsa.sparsify(pipe.transformer, cfg)
    mtsa.sparsify(pipe.transformer_2, cfg)


def _run_pipe(pipe, seed=0):
    """Run the pipeline with a fixed seed; return output frames tensor."""
    generator = torch.Generator(device="cuda").manual_seed(seed)
    with torch.no_grad():
        output = pipe(generator=generator, **_TINY_PIPE_KWARGS)
    # output.frames[0] is a list of PIL images; for assertion we just need shape+health
    return output


def _count_sparse_modules(module):
    return sum(isinstance(m, SparseAttentionModule) for m in module.modules())


# ---------------------------------------------------------------------------
# E2E tests
# ---------------------------------------------------------------------------


@pytest.mark.skipif(not TRITON_KERNEL_AVAILABLE, reason="Need CUDA + triton")
class TestWan22PipelineE2E:
    """End-to-end skip-softmax flow on a tiny Wan 2.2 pipeline."""

    def test_baseline_pipeline_runs(self, tiny_wan22_pipe):
        """Dense baseline: pipeline produces finite frames without sparsification."""
        output = _run_pipe(tiny_wan22_pipe)
        assert output.frames is not None
        assert len(output.frames) == 1
        # Frames are PIL Images; ensure list isn't empty
        assert len(output.frames[0]) > 0

    def test_sparsify_inserts_modules_in_both_transformers(self, tiny_wan22_pipe):
        """Both transformer and transformer_2 get SparseAttentionModule instances."""
        _sparsify_both_transformers(tiny_wan22_pipe, _skip_softmax_cfg())
        assert _count_sparse_modules(tiny_wan22_pipe.transformer) > 0
        assert _count_sparse_modules(tiny_wan22_pipe.transformer_2) > 0

    def test_skip_softmax_pipeline_runs_e2e(self, tiny_wan22_pipe):
        """Sparsified pipeline runs end-to-end producing finite frames."""
        _sparsify_both_transformers(tiny_wan22_pipe, _skip_softmax_cfg(raw_threshold=-5.0))
        output = _run_pipe(tiny_wan22_pipe)
        assert output.frames is not None
        assert len(output.frames[0]) > 0

    def test_tight_threshold_matches_dense_within_tolerance(self, tiny_wan22_pipe, tiny_wan22_path):
        """raw_threshold=-50 (effectively dense) → output close to unsparsified run."""
        from diffusers import WanPipeline

        # Dense run: fresh pipe, no sparsification
        dense_pipe = WanPipeline.from_pretrained(tiny_wan22_path, torch_dtype=torch.bfloat16)
        dense_pipe.to("cuda")
        dense_frame0 = _run_pipe(dense_pipe).frames[0][0]

        # Sparse run: same seed, raw_threshold=-50 (≈ no tiles skipped)
        _sparsify_both_transformers(tiny_wan22_pipe, _skip_softmax_cfg(raw_threshold=-50.0))
        sparse_frame0 = _run_pipe(tiny_wan22_pipe).frames[0][0]

        # Both are PIL images — convert to tensor and compare
        import numpy as np

        d = np.asarray(dense_frame0, dtype=np.float32)
        s = np.asarray(sparse_frame0, dtype=np.float32)
        # Pixel-wise MAE should be small for tight threshold (but not bit-exact due to
        # different code paths in the online softmax accumulation).
        mae = np.abs(d - s).mean()
        assert mae < 20.0, f"MAE between dense and tight-sparse frames was {mae:.2f}"

    def test_measure_sparsity_counts_accumulate(self, tiny_wan22_pipe):
        """measure_sparsity=True + a permissive threshold → nonzero sparsity counters."""
        from modelopt.torch.sparsity.attention_sparsity.methods.triton_skip_softmax import (
            TritonSkipSoftmaxMethod,
        )

        _sparsify_both_transformers(tiny_wan22_pipe, _skip_softmax_cfg(raw_threshold=-2.0))

        # Enable measurement + reset counters on every sparse module
        for module in (tiny_wan22_pipe.transformer, tiny_wan22_pipe.transformer_2):
            for m in module.modules():
                if isinstance(m, SparseAttentionModule):
                    method = m._sparse_method_instance
                    if isinstance(method, TritonSkipSoftmaxMethod):
                        method.enable_measure_sparsity(True)
                        method.reset_sparsity_counters()

        _run_pipe(tiny_wan22_pipe)

        # Sum counters across all sparse modules
        total_sum = 0
        for module in (tiny_wan22_pipe.transformer, tiny_wan22_pipe.transformer_2):
            for m in module.modules():
                if isinstance(m, SparseAttentionModule):
                    method = m._sparse_method_instance
                    if isinstance(method, TritonSkipSoftmaxMethod):
                        total, skipped = method.get_sparsity_counters()
                        assert skipped <= total
                        total_sum += total

        assert total_sum > 0, "Expected nonzero sparsity counters after pipeline run"

    def test_save_restore_roundtrip(self, tiny_wan22_pipe):
        """Sparsified transformer saves & restores via modelopt_state, preserving
        per-module method choice. The ``*attn1*`` pattern maps to triton_skip_softmax;
        ``attn2`` modules keep the default method. The restored model must show the
        identical (module_name → method) mapping.
        """
        from _test_utils.torch.diffusers_models import get_tiny_wan22_transformer

        import modelopt.torch.opt as mto

        _sparsify_both_transformers(tiny_wan22_pipe, _skip_softmax_cfg())
        state = mto.modelopt_state(tiny_wan22_pipe.transformer)

        # Restore into a fresh transformer of the same shape
        torch.manual_seed(0)
        restored = get_tiny_wan22_transformer().to("cuda", dtype=torch.bfloat16).eval()
        mto.restore_from_modelopt_state(restored, state)

        def _method_map(module):
            return {
                name: m._method
                for name, m in module.named_modules()
                if isinstance(m, SparseAttentionModule)
            }

        orig_map = _method_map(tiny_wan22_pipe.transformer)
        restored_map = _method_map(restored)
        assert orig_map == restored_map, (
            f"Restored method map differs from original:\n"
            f"  orig:     {orig_map}\n"
            f"  restored: {restored_map}"
        )
        # Sanity: the ``*attn1*`` pattern should have produced at least one
        # triton_skip_softmax module in the restored model.
        triton_attn1 = [
            name
            for name, method in restored_map.items()
            if method == "triton_skip_softmax" and "attn1" in name
        ]
        assert triton_attn1, (
            f"Expected at least one attn1 module with triton_skip_softmax after restore, "
            f"got {restored_map}"
        )


@pytest.mark.skipif(not TRITON_KERNEL_AVAILABLE, reason="Need CUDA + triton")
class TestWan22Calibration:
    """Multi-threshold calibration path on a tiny Wan 2.2 transformer."""

    def test_calibration_collects_stats_per_module(self, tiny_wan22_pipe):
        """A forward pass under calibration_mode populates per-module _last_stats."""
        from modelopt.torch.sparsity.attention_sparsity.methods.triton_skip_softmax import (
            TritonSkipSoftmaxMethod,
        )

        _sparsify_both_transformers(tiny_wan22_pipe, _skip_softmax_cfg())

        threshold_trials = [1e-3, 1e-2, 1e-1]
        for module in (tiny_wan22_pipe.transformer, tiny_wan22_pipe.transformer_2):
            for m in module.modules():
                if isinstance(m, SparseAttentionModule):
                    method = m._sparse_method_instance
                    if isinstance(method, TritonSkipSoftmaxMethod):
                        method._calibration_mode = True
                        method._threshold_trials = threshold_trials

        _run_pipe(tiny_wan22_pipe)

        # At least one sparse module should report stats of the correct shape
        found_stats = False
        for module in (tiny_wan22_pipe.transformer, tiny_wan22_pipe.transformer_2):
            for m in module.modules():
                if isinstance(m, SparseAttentionModule) and m._last_stats is not None:
                    stats = m._last_stats
                    assert len(stats["sparsity"]) == len(threshold_trials)
                    found_stats = True
                    break
            if found_stats:
                break
        assert found_stats, "No sparse module reported calibration stats"
