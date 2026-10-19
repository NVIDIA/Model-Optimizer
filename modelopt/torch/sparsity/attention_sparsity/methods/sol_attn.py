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

"""Sol-Attn training-free sparse attention for image and video diffusion models.

Sol-Attn (https://arxiv.org/abs/2607.24027) routes each query block to exact attention
over high-scoring KV blocks during online softmax and approximates the remaining blocks
with proxy scores. Kernels come from the external ``sol_attn`` package (CuTe DSL on
SM89/SM90/SM100/SM120, Triton otherwise), imported lazily at runtime.

Integration:
    After ``mtsa.sparsify(model, SOL_ATTN_DEFAULT)``, eligible
    ``F.scaled_dot_product_attention`` calls inside matched layers run Sol-Attn; other
    calls (cross-attention, masks, unsupported dtype/head dim/device) stay dense.
"""

import math

import torch

from . import SparseAttentionMethod, register_sparse_method

_HEAD_DIM = 128
# Sequence length from which the SM90 CuTe kernel benefits from 4-way KV splitting.
_SM90_KV_SPLIT_MIN_TOKENS = 65536


def _import_sol_attn():
    try:
        import sol_attn
    except ImportError as e:
        raise ImportError(
            "The 'sol_attn' sparse attention method requires the 'sol_attn' package. Install it "
            "with: pip install 'sol-attn @ git+https://github.com/NVlabs/Sana@sol-engine"
            f"#subdirectory=techniques/sparse_backends' (error: {e})"
        ) from e
    return sol_attn


@register_sparse_method("sol_attn")
class SolAttn(SparseAttentionMethod):
    """Sol-Attn sparse attention that replaces eligible SDPA calls.

    Configuration Parameters:
        - tau: Routing threshold multiplier; higher values select fewer exact KV blocks.
        - thresh_type: ``"diag"`` or ``"exact"`` (full-covariance) routing threshold.
        - kv_splits: KV splits for the SM90 kernel (1, 2, 4) or ``"auto"``.
    """

    patches_sdpa = True

    def __init__(self, method_config: dict | None = None):
        """Initialize Sol-Attn method."""
        super().__init__()
        config = method_config or {}
        self.tau = float(config.get("tau", 1.0))
        self.thresh_type = config.get("thresh_type", "diag")
        self.kv_splits = config.get("kv_splits", "auto")
        if self.thresh_type not in ("diag", "exact"):
            raise ValueError(f"thresh_type must be 'diag' or 'exact', got {self.thresh_type!r}")
        if self.kv_splits not in ("auto", 1, 2, 4):
            raise ValueError(f"kv_splits must be 'auto', 1, 2, or 4, got {self.kv_splits!r}")

    def can_replace_sdpa(
        self,
        query,
        key,
        value,
        attn_mask=None,
        dropout_p=0.0,
        is_causal=False,
        scale=None,
        enable_gqa=False,
        **kwargs,
    ) -> bool:
        """Non-causal, unmasked BF16 self-attention with head dim 128 on SM80+."""
        return (
            attn_mask is None
            and dropout_p == 0.0
            and not is_causal
            and not enable_gqa
            and not kwargs
            and (scale is None or math.isclose(scale, _HEAD_DIM**-0.5))
            and query.ndim == 4
            and query.shape == key.shape == value.shape
            and query.shape[-1] == _HEAD_DIM
            and query.dtype == key.dtype == value.dtype == torch.bfloat16
            and query.is_cuda
            and torch.cuda.get_device_capability(query.device) >= (8, 0)
        )

    def _resolve_kv_splits(self, sol_attn, query: torch.Tensor) -> int:
        if self.kv_splits != "auto":
            return self.kv_splits
        long_seq = query.shape[2] >= _SM90_KV_SPLIT_MIN_TOKENS
        return 4 if long_seq and sol_attn.get_sol_attn_backend(query.device) == "cute_sm90" else 1

    def forward_attention(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        **kwargs,
    ) -> tuple[torch.Tensor, dict]:
        """Run Sol-Attn on SDPA-layout ``[batch, heads, seq_len, 128]`` tensors."""
        sol_attn = _import_sol_attn()
        q, k, v = (x.transpose(1, 2).contiguous() for x in (query, key, value))
        out = sol_attn.sol_attn(
            q,
            k,
            v,
            tau=self.tau,
            thresh_type=self.thresh_type,
            kv_splits=self._resolve_kv_splits(sol_attn, query),
        )
        return out.transpose(1, 2), {"phase": "sol_attn"}

    def get_threshold_info(self) -> dict:
        """Get Sol-Attn configuration info."""
        return {"type": "sol_attn", "tau": self.tau, "thresh_type": self.thresh_type}

    @property
    def name(self) -> str:
        """Method identifier."""
        return "sol_attn"
