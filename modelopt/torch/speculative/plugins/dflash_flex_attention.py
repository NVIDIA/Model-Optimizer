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

"""Block-sparse FlexAttention path for the DFlash/DSpark draft.

Query block ``b`` with anchor ``a_b`` sees only the context prefix ``kv < a_b`` and its own
draft block, so most of the dense ``[B, 1, Q, KV]`` mask is empty. SDPA computes all of it,
on its memory-efficient fallback (the fused kernels reject arbitrary masks and cap
``head_dim`` at 256); FlexAttention takes the mask as a predicate and skips empty tiles.

K/V are repeated to the query head count rather than using ``enable_gqa=True``, whose
backward is ~10x slower.
"""

import torch

__all__ = ["build_draft_block_mask", "flex_attention_forward", "is_block_mask"]

# head_dim > 256 cannot use FlexAttention's default tiles (SMEM overflow); 32x32 with
# two pipeline stages is the only combination measured to both fit and run.
_LARGE_HEAD_DIM_KERNEL_OPTIONS = {
    "BLOCK_M": 32,
    "BLOCK_N": 32,
    "BLOCK_M1": 32,
    "BLOCK_N1": 32,
    "BLOCK_M2": 32,
    "BLOCK_N2": 32,
    "num_stages": 2,
    "num_warps": 4,
}
_MAX_DEFAULT_TILE_HEAD_DIM = 256

_PINNED_TILE_MASK_BLOCK_SIZE = 64
_DEFAULT_TILE_MASK_BLOCK_SIZE = 128


def _mask_block_size(head_dim):
    """BlockMask block size, which must be divisible by the kernel's BLOCK_M/BLOCK_N.

    The pinned 32x32 tiles allow the finer 64; the autotuner may pick tiles up to 128.
    """
    if head_dim > _MAX_DEFAULT_TILE_HEAD_DIM:
        return _PINNED_TILE_MASK_BLOCK_SIZE
    return _DEFAULT_TILE_MASK_BLOCK_SIZE


_flex_attention_compiled = None
_create_block_mask_compiled = None


def _flex_ops():
    """Resolve and compile the FlexAttention entry points once per process."""
    global _flex_attention_compiled, _create_block_mask_compiled
    if _flex_attention_compiled is None:
        from torch.nn.attention.flex_attention import create_block_mask, flex_attention

        # dynamic=False: shapes are fixed within a run, and dynamic shapes slow the kernel.
        _flex_attention_compiled = torch.compile(flex_attention, dynamic=False)
        _create_block_mask_compiled = torch.compile(create_block_mask, dynamic=False)
    return _flex_attention_compiled, _create_block_mask_compiled


def is_block_mask(mask) -> bool:
    """True if ``mask`` is a FlexAttention ``BlockMask`` rather than a dense tensor."""
    if mask is None or torch.is_tensor(mask):
        return False
    try:
        from torch.nn.attention.flex_attention import BlockMask
    except ImportError:
        return False
    return isinstance(mask, BlockMask)


def build_draft_block_mask(
    seq_len,
    anchor_positions,
    block_keep_mask,
    n_blocks,
    block_size,
    window,
    device,
    head_dim,
    causal=False,
):
    """BlockMask equivalent of ``HFDFlashModel._build_draft_attention_mask``.

    Every term of the dense predicate must appear here too, or the two paths silently train
    different models.
    """
    _, create_block_mask = _flex_ops()
    bsz = anchor_positions.shape[0]
    q_len = n_blocks * block_size
    kv_len = seq_len + q_len
    # Indexed inside mask_mod, which runs under vmap -- keep them on-device and integral.
    anchors = anchor_positions.to(device=device, dtype=torch.int32)
    keep = block_keep_mask.to(device=device, dtype=torch.bool)

    def mask_mod(b, h, q_idx, kv_idx):
        q_block = q_idx // block_size
        anchor = anchors[b, q_block]
        is_ctx = kv_idx < seq_len
        ctx_ok = is_ctx & (kv_idx < anchor)
        if window is not None:
            # Window from the query's real position (anchor + position in block).
            ctx_ok = ctx_ok & (kv_idx > anchor + (q_idx % block_size) - window)
        draft_ok = (~is_ctx) & (q_block == (kv_idx - seq_len) // block_size)
        if causal:
            # Block-causal: block position i sees draft positions <= i.
            draft_ok = draft_ok & (((kv_idx - seq_len) % block_size) <= (q_idx % block_size))
        return (ctx_ok | draft_ok) & keep[b, q_block]

    return create_block_mask(
        mask_mod,
        bsz,
        None,
        q_len,
        kv_len,
        device=device,
        BLOCK_SIZE=_mask_block_size(head_dim),
    )


def _repeat_kv(x, n_rep):
    """HF's ``repeat_kv``: [B, n_kv, S, D] -> [B, n_kv * n_rep, S, D]."""
    if n_rep == 1:
        return x
    b, h, s, d = x.shape
    return x[:, :, None].expand(b, h, n_rep, s, d).reshape(b, h * n_rep, s, d)


def flex_attention_forward(query, key, value, block_mask, scaling):
    """FlexAttention with the draft's BlockMask. Returns ``[B, q_len, n_heads, head_dim]``.

    The layout matches what HF's ``sdpa_attention_forward`` returns so the caller's
    ``reshape(bsz, q_len, -1)`` is unchanged.
    """
    flex_attention, _ = _flex_ops()
    head_dim = query.shape[-1]
    n_rep = query.shape[1] // key.shape[1]
    kernel_options = (
        _LARGE_HEAD_DIM_KERNEL_OPTIONS if head_dim > _MAX_DEFAULT_TILE_HEAD_DIM else None
    )
    attn_output = flex_attention(
        query,
        _repeat_kv(key, n_rep),
        _repeat_kv(value, n_rep),
        block_mask=block_mask,
        scale=scaling,
        kernel_options=kernel_options,
    )
    return attn_output.transpose(1, 2).contiguous()
