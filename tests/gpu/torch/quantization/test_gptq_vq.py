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

"""Test _blockwise_vector_update against gptq_quantize_scaled_vq from adaptive_rounding.py."""
# ruff: noqa: N803, N806 — uppercase names (W, A, Q, H, etc.) match math notation in the reference.

import tempfile
from types import SimpleNamespace

import torch
import torch.nn.functional as F

from modelopt.torch.quantization.utils.calib_utils import GPTQHelper

# ---------------------------------------------------------------------------
# Exact copies from modelopt-internal/docs/adaptive_rounding.py
# ---------------------------------------------------------------------------


def compute_output_nmse(
    W: torch.Tensor,
    W_q: torch.Tensor,
    A: torch.Tensor,
) -> float:
    """
    Compute the output Normalized MSE:

        NMSE = || (W - W_q) @ A ||_F^2  /  || W @ A ||_F^2

    where @ denotes matrix multiplication. This measures how much
    weight quantization error propagates through the activations.

    Args:
        W:   Original weight matrix, shape (out_features, in_features)
        W_q: Quantized weight matrix, same shape as W
        A:   Activation matrix, shape (in_features, n_samples)

    Returns:
        Scalar NMSE value
    """
    error_output = (W - W_q) @ A
    ref_output = W @ A
    nmse = error_output.norm() ** 2 / ref_output.norm() ** 2
    return nmse.item()


def _round_to_e4m3(s: torch.Tensor) -> torch.Tensor:
    """Round a float32 scale tensor to the nearest E4M3 representable value."""
    return s.to(dtype=torch.float8_e4m3fn).to(dtype=torch.float32)


def _vq_nearest(vecs: torch.Tensor, codebook: torch.Tensor, chunk_size: int = 1) -> torch.Tensor:
    """
    Find the nearest codeword for each row-vector in vecs.

    Args:
        vecs: (N, D) float tensor
        codebook: (K, D) float tensor
        chunk_size: Process this many vectors at a time to bound memory.
                    Default 1 to avoid torch.cdist batch precision issues.

    Returns:
        (N, D) tensor of nearest codewords
    """
    N = vecs.shape[0]
    result = torch.empty_like(vecs)
    for start in range(0, N, chunk_size):
        end = min(start + chunk_size, N)
        dists = torch.cdist(vecs[start:end], codebook)
        result[start:end] = codebook[dists.argmin(dim=1)]
    return result


def scaled_vector_quantize(
    W: torch.Tensor,
    codebook: torch.Tensor,
    quant_block_size: int,
    n_scale_iters: int = 3,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Block-scaled vector quantization with iterative scale optimisation.

    For each row-block of quant_block_size elements we find a scalar
    scale s and codeword assignments that (approximately) minimise

        || block - s * reconstruct(block / s) ||^2

    via alternating minimisation:
      1. Fix s  -> assign each sub-vector to nearest codeword in codebook
      2. Fix assignments -> solve for optimal s in closed form:
            s* = sum_i <x_i, c_i> / sum_i ||c_i||^2

    Initialised with s = max(|block|) / max(|codebook|) (amax), then
    refined for n_scale_iters iterations.

    Effective bitrate: log2(n_codewords) / codebook_dim + 16 / quant_block_size

    Args:
        W: (out_features, in_features)
        codebook: (n_codewords, codebook_dim) — must evenly divide quant_block_size
        quant_block_size: Elements per scaling block
        n_scale_iters: Number of assign-then-rescale iterations (default 3)

    Returns:
        W_q: Dequantised weight matrix
        scales: (out_features, n_blocks)
    """
    codebook_dim = codebook.shape[1]
    out_features, in_features = W.shape
    assert quant_block_size % codebook_dim == 0, (
        f"quant_block_size ({quant_block_size}) must be a multiple of codebook_dim ({codebook_dim})"
    )
    assert in_features % codebook_dim == 0, (
        f"in_features ({in_features}) must be a multiple of codebook_dim ({codebook_dim})"
    )

    cb_absmax = codebook.abs().max().item()
    n_blocks = (in_features + quant_block_size - 1) // quant_block_size

    # Initial scales: amax heuristic
    pad = n_blocks * quant_block_size - in_features
    if pad > 0:
        W_padded = torch.nn.functional.pad(W, (0, pad))
    else:
        W_padded = W
    W_blocks = W_padded.reshape(out_features, n_blocks, quant_block_size)
    block_max = W_blocks.abs().amax(dim=2)
    scales = (block_max / cb_absmax).clamp(min=1e-10)

    W_q = torch.empty_like(W)

    for b in range(n_blocks):
        start = b * quant_block_size
        end = min(start + quant_block_size, in_features)
        block = W[:, start:end]  # (out_features, block_len)
        s = scales[:, b]  # (out_features,)

        for _ in range(n_scale_iters):
            # Step 1: assign sub-vectors to nearest codeword at current scale
            normalized = block / s.unsqueeze(1)
            vecs = normalized.reshape(-1, codebook_dim)
            q_vecs = _vq_nearest(vecs, codebook)
            q_block = q_vecs.reshape(out_features, end - start)

            # Step 2: optimal scale given assignments
            #   s* = sum_i <x_i, c_i> / sum_i ||c_i||^2
            # where x_i are original sub-vectors, c_i are assigned codewords
            numerator = (block * q_block).sum(dim=1)  # (out_features,)
            denominator = (q_block * q_block).sum(dim=1)  # (out_features,)
            s = (numerator / denominator.clamp(min=1e-20)).clamp(min=1e-10)

        # Round converged scale to E4M3 and do final assignment
        s = _round_to_e4m3(s)
        normalized = block / s.unsqueeze(1)
        vecs = normalized.reshape(-1, codebook_dim)
        q_vecs = _vq_nearest(vecs, codebook)
        W_q[:, start:end] = q_vecs.reshape(out_features, end - start) * s.unsqueeze(1)
        scales[:, b] = s

    return W_q, scales


def gptq_quantize_scaled_vq(
    W: torch.Tensor,
    A: torch.Tensor,
    codebook: torch.Tensor,
    quant_block_size: int,
    gptq_block_size: int = 128,
    damp_pct: float = 0.01,
    n_scale_iters: int = 3,
    h_inv: torch.Tensor | None = None,
) -> tuple[torch.Tensor, float]:
    """
    GPTQ with block-scaled vector quantization.

    Scales are pre-computed via the same iterative optimisation as
    scaled_vector_quantize. During the GPTQ pass, columns are processed
    in groups of codebook_dim: each group is VQ-quantized to the nearest
    codeword, and per-column errors are compensated into remaining columns
    using the inverse Hessian.

    Args:
        W: (out_features, in_features)
        A: (in_features, n_samples)
        codebook: (n_codewords, codebook_dim)
        quant_block_size: Elements per scaling block
        gptq_block_size: Columns per GPTQ lazy batch update (must be
                         a multiple of codebook_dim)
        damp_pct: Dampening fraction
        n_scale_iters: Iterations for scale optimisation
        h_inv: Optional pre-computed upper-triangular Cholesky factor of the
               damped inverse Hessian. If None, computed from A.

    Returns:
        W_q: Quantized weight matrix (dequantised)
        nmse: Output NMSE achieved
    """
    codebook_dim = codebook.shape[1]
    assert gptq_block_size % codebook_dim == 0, (
        f"gptq_block_size ({gptq_block_size}) must be a multiple of codebook_dim ({codebook_dim})"
    )

    W_orig = W.clone()
    W = W.clone()
    out_features, in_features = W.shape

    _, scales = scaled_vector_quantize(
        W_orig, codebook, quant_block_size, n_scale_iters=n_scale_iters
    )

    if h_inv is None:
        H = 2.0 * (A @ A.T)
        damp = damp_pct * torch.diag(H).mean()
        H.diagonal().add_(damp)

        H_inv = torch.linalg.inv(H)
        try:
            L = torch.linalg.cholesky(H_inv)
        except torch.linalg.LinAlgError:
            H.diagonal().add_(damp * 10)
            H_inv = torch.linalg.inv(H)
            L = torch.linalg.cholesky(H_inv)
        Hinv = L.T
    else:
        Hinv = h_inv

    Q = torch.zeros_like(W)

    for i in range(0, in_features, gptq_block_size):
        j_end = min(i + gptq_block_size, in_features)
        E = torch.zeros(out_features, j_end - i, dtype=W.dtype, device=W.device)

        for j in range(i, j_end, codebook_dim):
            d = min(codebook_dim, j_end - j)
            sb = j // quant_block_size
            s = scales[:, sb]  # (out_features,)

            sub_vec = W[:, j : j + d] / s.unsqueeze(1)
            q_vecs = _vq_nearest(
                sub_vec.reshape(-1, codebook_dim) if d == codebook_dim else sub_vec.reshape(-1, d),
                codebook[:, :d],
            )
            q_block = q_vecs.reshape(out_features, d) * s.unsqueeze(1)
            Q[:, j : j + d] = q_block

            for k in range(d):
                col = j + k
                err = (W[:, col] - Q[:, col]) / Hinv[col, col]
                E[:, col - i] = err
                W[:, col:j_end] -= err.unsqueeze(1) * Hinv[col, col:j_end].unsqueeze(0)

        if j_end < in_features:
            W[:, j_end:] -= E @ Hinv[i:j_end, j_end:]

    nmse = compute_output_nmse(W_orig, Q, A)
    return Q, nmse


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_codebook_file(tmp_dir, name, n_codewords, vector_size):
    """Save a codebook .pt file in the format load_vector_lut_codebook expects."""
    codebook = torch.randn(n_codewords, vector_size)
    torch.save({"sorted_values": codebook}, f"{tmp_dir}/{name}.pt")
    return codebook.cuda().float()


def _make_hessian(activations):
    """Build Hessian H = 2 * X @ X^T from activations (batch, seq, features)."""
    X = activations.reshape(-1, activations.shape[-1]).t().float()
    return 2.0 * (X @ X.t())


def _attach_mock_quantizer(module, encode_path, encode_format, quant_block_size, scale_type):
    """Attach a mock weight_quantizer with the right attributes for _blockwise_vector_update."""
    module.weight_quantizer = SimpleNamespace(
        num_bits=encode_format,
        backend="psx_luts",
        backend_extra_args={
            "encode_path": encode_path,
            "lut_type": "vector_lut",
            "block_sizes": quant_block_size,
            "scale_type": scale_type,
        },
    )


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

SEED = 42
OUT_FEATURES = 64
IN_FEATURES = 128
GPTQ_BLOCK_SIZE = 128
QUANT_BLOCK_SIZE = 16
VECTOR_SIZE = 8
N_CODEWORDS = 256
PERC_DAMP = 0.01
SCALE_TYPE = "e4m3"


def test_clip_vector_prescaled_vs_vq_nearest():
    """clip_vector_prescaled and _vq_nearest must produce identical quantized vectors."""
    from luts import clip_vector_prescaled, clip_vector_scalesign_fast

    torch.manual_seed(SEED)

    with tempfile.TemporaryDirectory() as tmp_dir:
        codebook = _make_codebook_file(tmp_dir, "test_sorted", N_CODEWORDS, VECTOR_SIZE)

        W = torch.randn(OUT_FEATURES, IN_FEATURES, device="cuda", dtype=torch.float32)

        # Compute scales with sign_scale=True (scales can be negative)
        _, scales = clip_vector_scalesign_fast(
            W,
            codebook,
            QUANT_BLOCK_SIZE,
            SCALE_TYPE,
            scale_algo="max",
            sign_scale=True,
            return_scales=True,
        )
        scales_2d = scales.reshape(OUT_FEATURES, -1)

        for j in range(0, IN_FEATURES, VECTOR_SIZE):
            d = min(VECTOR_SIZE, IN_FEATURES - j)
            s = scales_2d[:, j // QUANT_BLOCK_SIZE].contiguous()
            sub = W[:, j : j + d].contiguous()

            # clip_vector_prescaled path
            if d < VECTOR_SIZE:
                sub_padded = F.pad(sub, (0, VECTOR_SIZE - d))
            else:
                sub_padded = sub
            q_luts = clip_vector_prescaled(sub_padded, codebook, s)[:, :d]

            # _vq_nearest path (manual normalize -> lookup -> denormalize)
            normalized = sub / s.unsqueeze(1)
            vecs = (
                normalized.reshape(-1, VECTOR_SIZE)
                if d == VECTOR_SIZE
                else normalized.reshape(-1, d)
            )
            cb_slice = codebook if d == VECTOR_SIZE else codebook[:, :d]
            q_vecs = _vq_nearest(vecs, cb_slice)
            q_ref = q_vecs.reshape(OUT_FEATURES, d) * s.unsqueeze(1)

            assert torch.allclose(q_luts, q_ref, atol=1e-5), (
                f"VQ mismatch at col {j}: max diff={(q_luts - q_ref).abs().max().item():.2e}"
            )


def test_blockwise_vector_update_vs_gptq_quantize_scaled_vq():
    """_blockwise_vector_update must produce identical weights to gptq_quantize_scaled_vq.

    Both paths share the same pre-computed scales (from clip_vector_scalesign_fast)
    and the same h_inv, so the only variable is the GPTQ loop itself.
    The reference's scaled_vector_quantize is patched to return the shared scales.
    """
    from unittest.mock import patch

    from luts import clip_vector_scalesign_fast

    torch.manual_seed(SEED)

    with tempfile.TemporaryDirectory() as tmp_dir:
        encode_format = "test_sorted"
        codebook = _make_codebook_file(tmp_dir, encode_format, N_CODEWORDS, VECTOR_SIZE)

        W = torch.randn(OUT_FEATURES, IN_FEATURES, device="cuda", dtype=torch.float32)
        A = torch.randn(4, 16, IN_FEATURES, device="cuda", dtype=torch.float32)
        A_flat = A.reshape(-1, IN_FEATURES).t()  # (in_features, n_samples) for reference
        hessian = _make_hessian(A)

        # Shared scales — computed once, used by both paths
        Q_rtn, scales = clip_vector_scalesign_fast(
            W,
            codebook,
            QUANT_BLOCK_SIZE,
            SCALE_TYPE,
            scale_algo="max",
            sign_scale=True,
            return_scales=True,
        )
        scales_2d = scales.reshape(OUT_FEATURES, -1)

        # --- Reference: exact gptq_quantize_scaled_vq from adaptive_rounding.py ---
        # Patch scaled_vector_quantize to return the shared scales (Q_rtn unused by caller)
        with patch(
            f"{__name__}.scaled_vector_quantize",
            return_value=(Q_rtn, scales_2d),
        ):
            Q_ref, _ = gptq_quantize_scaled_vq(
                W,
                A_flat,
                codebook,
                QUANT_BLOCK_SIZE,
                GPTQ_BLOCK_SIZE,
                damp_pct=PERC_DAMP,
            )

        # --- Modelopt: _blockwise_vector_update ---
        module = torch.nn.Linear(IN_FEATURES, OUT_FEATURES, bias=False).cuda()
        module.weight.data = W.clone()
        _attach_mock_quantizer(module, tmp_dir, encode_format, QUANT_BLOCK_SIZE, SCALE_TYPE)

        helper = GPTQHelper(module, "test_layer")
        helper.hessian = hessian.clone()
        helper.n_samples = 1
        helper.update_weights(GPTQ_BLOCK_SIZE, PERC_DAMP)
        Q_modelopt = module.weight.data.float()

        # Same scales + same GPTQ loop structure => weights must match
        assert torch.allclose(Q_modelopt, Q_ref, atol=1e-5), (
            f"Weights differ: max diff={(Q_modelopt - Q_ref).abs().max().item():.2e}"
        )


def test_mtq_quantize_gptq_vs_gptq_quantize_scaled_vq():
    """End-to-end: mtq.quantize with GPTQ vs gptq_quantize_scaled_vq.

    Both paths compute their own hessian and h_inv independently.
    Shared scales ensure the only variable is the GPTQ loop. The hessian is
    injected into GPTQHelper so both paths compute identical h_inv.
    """
    from unittest.mock import patch

    from luts import clip_vector_scalesign_fast

    # Register psx_luts backend (workaround for _default_disabled_quantizer_cfg list/dict change)
    import modelopt.torch.quantization.config as _mtq_cfg

    _orig = _mtq_cfg._default_disabled_quantizer_cfg
    if isinstance(_orig, list):
        _mtq_cfg._default_disabled_quantizer_cfg = {}
    import modelopt_internal.torch.quantization.plugins.psx_luts  # noqa: F401

    _mtq_cfg._default_disabled_quantizer_cfg = _orig

    import modelopt.torch.quantization as mtq

    torch.manual_seed(SEED)

    with tempfile.TemporaryDirectory() as tmp_dir:
        encode_format = "test_sorted"
        codebook = _make_codebook_file(tmp_dir, encode_format, N_CODEWORDS, VECTOR_SIZE)

        W = torch.randn(OUT_FEATURES, IN_FEATURES, device="cuda", dtype=torch.float32)
        A = torch.randn(4, 16, IN_FEATURES, device="cuda", dtype=torch.float32)
        A_flat = A.reshape(-1, IN_FEATURES).t()

        # --- Modelopt: mtq.quantize with GPTQ (list-based config) ---
        config = {
            "quant_cfg": [
                {"quantizer_name": "*", "enable": False},
                {
                    "quantizer_name": "*weight_quantizer",
                    "cfg": {
                        "backend": "psx_luts",
                        "num_bits": encode_format,
                        "pass_through_bwd": True,
                        "backend_extra_args": {
                            "encode_path": tmp_dir,
                            "lut_type": "vector_lut",
                            "block_sizes": QUANT_BLOCK_SIZE,
                            "scale_type": SCALE_TYPE,
                            "scale_algo": "max",
                            "round_mode": "rne",
                            "sign_scale": True,
                        },
                    },
                },
            ],
            "algorithm": {
                "method": "gptq",
                "use_sequential": False,
                "perc_damp": PERC_DAMP,
                "block_size": GPTQ_BLOCK_SIZE,
            },
        }

        model = torch.nn.Linear(IN_FEATURES, OUT_FEATURES, bias=False).cuda()
        model.weight.data = W.clone()

        def forward_loop(m):
            m(A)

        # Capture h_inv from the modelopt GPTQ run
        captured = {}
        _orig_update = GPTQHelper.update_weights

        def _capturing_update(self, *args, **kwargs):
            _orig_update(self, *args, **kwargs)
            captured["h_inv"] = self.h_inv.clone()

        with patch.object(GPTQHelper, "update_weights", _capturing_update):
            mtq.quantize(model, config, forward_loop=forward_loop)
        Q_modelopt = model.weight.data.float()

        # --- Reference: gptq_quantize_scaled_vq with shared scales + captured h_inv ---
        Q_rtn, scales = clip_vector_scalesign_fast(
            W,
            codebook,
            QUANT_BLOCK_SIZE,
            SCALE_TYPE,
            scale_algo="max",
            sign_scale=True,
            return_scales=True,
        )
        scales_2d = scales.reshape(OUT_FEATURES, -1)

        with patch(
            f"{__name__}.scaled_vector_quantize",
            return_value=(Q_rtn, scales_2d),
        ):
            Q_ref, _ = gptq_quantize_scaled_vq(
                W,
                A_flat,
                codebook,
                QUANT_BLOCK_SIZE,
                GPTQ_BLOCK_SIZE,
                damp_pct=PERC_DAMP,
                h_inv=captured["h_inv"],
            )

        assert torch.allclose(Q_modelopt, Q_ref, atol=1e-5), (
            f"Weights differ: max diff={(Q_modelopt - Q_ref).abs().max().item():.2e}"
        )
