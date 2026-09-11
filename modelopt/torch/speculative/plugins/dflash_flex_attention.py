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

The draft's attention mask is dense in shape but sparse in content. Query block ``b``
with anchor ``a_b`` attends to

    context:  kv < a_b                     -- a prefix, and anchors are sorted, so
                                              across blocks this is a staircase
    draft:    kv in [S + Bb, S + Bb + B)   -- the block diagonal, ``block_size`` wide

Handing SDPA a materialized ``[B, 1, Q, KV]`` float mask makes it compute all of it,
and disqualifies every fused backend on the way: PyTorch's FlashAttention kernels
reject arbitrary masks *and* cap ``head_dim`` at 256, while a Gemma-4 draft's
``global_head_dim`` is 512. What is left is the cutlass memory-efficient backend,
whose only kernel is **sm80** -- an Ampere kernel on Blackwell. Profiling the
Gemma-4-E4B DSpark run on B300 put that single backward kernel at 59% of the whole
training step, running at roughly 2.5% of the GPU's bf16 peak.

FlexAttention instead takes the mask as a predicate, compiles it into the kernel, and
skips fully-masked tiles. Measured on B300 at the production shape
(q[4,16,4096,512], kv[4,1->16,8192,512], 512 anchors x block 8, 33% mask density):

    dense-mask SDPA   fwd 24.80 ms   fwd+bwd 225.37 ms
    flex (this file)  fwd  6.84 ms   fwd+bwd  56.40 ms      4.0x

Both are within bf16 rounding of each other on out/dq/dk/dv, including the
fully-masked rows that invalid blocks produce.

Two non-obvious requirements, both established by measurement rather than docs:

* ``head_dim`` 512 overflows shared memory at FlexAttention's default tiles (263 KB
  required against a 232 KB limit), so the tiles are pinned above ``head_dim`` 256.
  Of the shapes that fit, only 32x32 runs at all on torch 2.11 / sm103: 64x32 and
  64x64 fault with "misaligned address" and 128x32 with "unspecified launch
  failure". At ``head_dim`` <= 256 the library defaults are far better than anything
  pinned (9.8x over SDPA at 256), so they are left alone.
* ``enable_gqa=True`` must NOT be used. Its forward matches the pre-repeated path
  exactly, but its backward takes 568 ms -- 10x slower, and 2.5x worse than the SDPA
  baseline it is meant to replace. K/V are repeated to the query head count first,
  which is what HF's sdpa path does anyway.
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

# Block granularity of the BlockMask itself. Finer granularity tracks the true 33%
# density more closely (64 -> 65.7% sparsity vs 128 -> 64.2%) and measured 58.0 ms
# against 61.4 ms; 32 is marginally better again but doubles the metadata for ~1 ms.
_PINNED_TILE_MASK_BLOCK_SIZE = 64
_DEFAULT_TILE_MASK_BLOCK_SIZE = 128


def _mask_block_size(head_dim):
    """Mask granularity, which is coupled to the kernel tiles and cannot be chosen freely.

    FlexAttention requires the BlockMask's block size to be divisible by the kernel's
    BLOCK_M/BLOCK_N, and raises "Q and KV block size must be divisible by BLOCK_M and
    BLOCK_N" otherwise. So:

    * head_dim > 256 pins 32x32 tiles (SMEM), so 64 is safe -- and finer granularity
      tracks the true mask density better (65.7% sparsity vs 64.2% at 128, measured
      58.0 ms vs 61.4 ms).
    * head_dim <= 256 leaves the tiles to Inductor's autotuner, which may pick up to
      128. Only FlexAttention's own default of 128 is guaranteed compatible with every
      choice it can make.
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

        # dynamic=False: shapes are fixed by (batch, num_anchors, block_size, seq_len)
        # within a run, and dynamic shapes measurably deoptimize the generated kernel.
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
    seq_len, anchor_positions, block_keep_mask, n_blocks, block_size, window, device,
    head_dim,
):
    """BlockMask equivalent of ``HFDFlashModel._build_draft_attention_mask``.

    Same predicate, expressed for FlexAttention instead of materialized. Rebuilt every
    step because anchors are resampled on every forward; the compiled builder costs
    ~0.1 ms, against ~1.5 ms to materialize the dense mask it replaces.
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
            # Same sliding window as the dense path: measured against the query's REAL
            # position (anchor + position-in-block), not its index in the draft block.
            ctx_ok = ctx_ok & (kv_idx > anchor + (q_idx % block_size) - window)
        draft_ok = (~is_ctx) & (q_block == (kv_idx - seq_len) // block_size)
        return (ctx_ok | draft_ok) & keep[b, q_block]

    return create_block_mask(
        mask_mod, bsz, None, q_len, kv_len, device=device,
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
