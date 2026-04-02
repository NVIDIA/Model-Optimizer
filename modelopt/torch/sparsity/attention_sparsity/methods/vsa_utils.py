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

"""Utility functions for Video Sparse Attention (VSA).

This module provides 3D block operations for video sparse attention,
including reshaping tensors into video blocks and variable block size computation.
"""

import functools
import math

import torch


@functools.lru_cache(maxsize=10)
def get_tile_partition_indices(
    video_shape: tuple[int, int, int],
    tile_size: tuple[int, int, int],
    device: torch.device,
) -> torch.LongTensor:
    """Get indices to partition video tokens into tiles.

    Args:
        video_shape: Video dimensions (T, H, W) after patchification.
        tile_size: Tile dimensions (tile_T, tile_H, tile_W).
        device: Device for the output tensor.

    Returns:
        LongTensor of indices to rearrange tokens into tile order.
    """
    vid_t, vid_h, vid_w = video_shape
    ts, hs, ws = tile_size
    indices = torch.arange(vid_t * vid_h * vid_w, device=device, dtype=torch.long).reshape(
        vid_t, vid_h, vid_w
    )

    tiles = []
    for t in range(math.ceil(vid_t / ts)):
        for h in range(math.ceil(vid_h / hs)):
            for w in range(math.ceil(vid_w / ws)):
                tile = indices[
                    t * ts : min(t * ts + ts, vid_t),
                    h * hs : min(h * hs + hs, vid_h),
                    w * ws : min(w * ws + ws, vid_w),
                ]
                tiles.append(tile.flatten())

    return torch.cat(tiles, dim=0)


@functools.lru_cache(maxsize=10)
def get_reverse_tile_partition_indices(
    video_shape: tuple[int, int, int],
    tile_size: tuple[int, int, int],
    device: torch.device,
) -> torch.LongTensor:
    """Get indices to reverse tile partitioning back to original order.

    Args:
        video_shape: Video dimensions (T, H, W) after patchification.
        tile_size: Tile dimensions (tile_T, tile_H, tile_W).
        device: Device for the output tensor.

    Returns:
        LongTensor of indices to reverse the tile rearrangement.
    """
    forward_indices = get_tile_partition_indices(video_shape, tile_size, device)
    return torch.argsort(forward_indices)


@functools.lru_cache(maxsize=10)
def construct_variable_block_sizes(
    video_shape: tuple[int, int, int],
    num_tiles: tuple[int, int, int],
    tile_size: tuple[int, int, int],
    device: torch.device,
) -> torch.LongTensor:
    """Compute valid (non-padded) token count for each tile.

    Since video dimensions may not divide evenly by tile size, edge tiles
    will have fewer valid tokens. This function computes the actual valid
    token count for each tile.

    Args:
        video_shape: Video dimensions (T, H, W) after patchification.
        num_tiles: Number of tiles in each dimension (n_T, n_H, n_W).
        tile_size: Tile dimensions (tile_T, tile_H, tile_W).
        device: Device for the output tensor.

    Returns:
        LongTensor of shape [num_tiles_total] with valid tokens per tile.
    """
    t, h, w = video_shape
    ts_t, ts_h, ts_w = tile_size
    n_t, n_h, n_w = num_tiles

    def _sizes(dim_len: int, tile: int, n_tiles: int) -> torch.LongTensor:
        """Compute size of each tile along one dimension."""
        sizes = torch.full((n_tiles,), tile, dtype=torch.long, device=device)
        remainder = dim_len - (n_tiles - 1) * tile
        sizes[-1] = remainder if remainder > 0 else tile
        return sizes

    t_sizes = _sizes(t, ts_t, n_t)  # [n_t]
    h_sizes = _sizes(h, ts_h, n_h)  # [n_h]
    w_sizes = _sizes(w, ts_w, n_w)  # [n_w]

    # Broadcast multiply to get tokens per tile
    block_sizes = (
        t_sizes[:, None, None] * h_sizes[None, :, None] * w_sizes[None, None, :]
    ).reshape(-1)

    return block_sizes


@functools.lru_cache(maxsize=10)
def get_non_pad_index(
    variable_block_sizes: torch.LongTensor,
    max_block_size: int,
) -> torch.LongTensor:
    """Get indices of non-padded tokens in the padded layout.

    When tiles have variable sizes, we pad to max_block_size. This function
    returns indices to extract only valid (non-padded) tokens.

    Args:
        variable_block_sizes: Tensor of valid token counts per tile.
        max_block_size: Maximum tile size (usually tile_T * tile_H * tile_W).

    Returns:
        LongTensor of indices for valid tokens.
    """
    n_win = variable_block_sizes.shape[0]
    device = variable_block_sizes.device

    starts_pad = torch.arange(n_win, device=device) * max_block_size
    index_pad = starts_pad[:, None] + torch.arange(max_block_size, device=device)[None, :]
    index_mask = (
        torch.arange(max_block_size, device=device)[None, :] < variable_block_sizes[:, None]
    )

    return index_pad[index_mask]
