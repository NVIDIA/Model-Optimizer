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

"""DMS KV cache: HF-compatible wrapper, combined cache layer, and contiguous cache layer.

The paged cache layer (DMSPagedCacheLayer) is in dms.cache_paged.
"""

import functools
from enum import Enum
from typing import Any

import torch
from transformers import CacheLayerMixin
from transformers.cache_utils import Cache

from dms.cache_paged import DMSPagedCacheLayer

# =============================================================================
# Contiguous (non-paged) DMS cache layer
# =============================================================================


class DMSContCacheLayer(CacheLayerMixin):
    """Used for storing contiguous (non-paged) cache."""

    def __init__(
        self,
        dms_window_size: int,
        max_context_length: int,
        block_size: int = 256,
        growth_factor: float = 1.5,
        accommodate_min_initial_context_length: int = 4096,
        disable_eviction: bool = False,
    ):
        """Initialize contiguous cache layer."""
        super().__init__()
        self.dms_window_size = dms_window_size
        self.max_context_length = max_context_length
        self.block_size = block_size
        self.growth_factor = growth_factor
        self.min_initial_context_length = accommodate_min_initial_context_length
        self.disable_eviction = disable_eviction

        self.key_cache = None
        self.value_cache = None
        self.eviction_info = None
        self.cache_seq_lengths = None
        self.cumulative_length = 0

        self.device = None

    def offload(self):
        """Offload cache tensors to CPU."""
        if self.key_cache is not None:
            self.key_cache = self.key_cache.to("cpu", non_blocking=True)
            self.value_cache = self.value_cache.to("cpu", non_blocking=True)
            self.eviction_info = self.eviction_info.to("cpu", non_blocking=True)
            self.cache_seq_lengths = self.cache_seq_lengths.to("cpu", non_blocking=True)

    def prefetch(self):
        """Prefetch cache tensors back to the original device."""
        if self.key_cache is not None and self.key_cache.device != self.device:
            self.key_cache = self.key_cache.to(self.device, non_blocking=True)
            self.value_cache = self.value_cache.to(self.device, non_blocking=True)
            self.eviction_info = self.eviction_info.to(self.device, non_blocking=True)
            self.cache_seq_lengths = self.cache_seq_lengths.to(self.device, non_blocking=True)

    def reset(self):
        """Reset cache to uninitialized state."""
        self.key_cache = None
        self.value_cache = None
        self.eviction_info = None
        self.cache_seq_lengths = None
        self.cumulative_length = 0

    def reorder_cache(self, beam_idx: torch.LongTensor):
        """Reorder cache for beam search (not supported)."""
        raise NotImplementedError("Beam search is not supported")

    def lazy_initialization(self, key_states: torch.Tensor):
        """Lazy initialization placeholder."""
        return None

    def update(self):
        """Update cache (not implemented for contiguous cache)."""
        raise NotImplementedError("update method is not implemented")

    def is_initialized(self) -> bool:
        """Check if the cache has been initialized."""
        return self.key_cache is not None

    def replace(
        self, key_states: torch.Tensor, value_states: torch.Tensor, cache_kwargs: dict[str, Any]
    ):
        """Replace the entire cache contents."""
        if self.device is None:
            self.device = key_states.device
        eviction_info = cache_kwargs["eviction_info"]
        seq_lengths = cache_kwargs["sequence_lengths"]
        cumulative_length = cache_kwargs["cumulative_length"]

        assert key_states is not None, "key_states is None"
        assert value_states is not None, "value_states is None"
        assert eviction_info is not None, "eviction_info is None"
        assert seq_lengths is not None, "seq_lengths is None"

        self.cumulative_length = cumulative_length
        self.key_cache = key_states
        self.value_cache = value_states
        self.eviction_info = eviction_info
        self.cache_seq_lengths = seq_lengths

    def get_cache(
        self,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Get the cached key, value, sequence lengths, and eviction info."""
        return (
            self.key_cache,
            self.value_cache,
            self.cache_seq_lengths,
            self.eviction_info,
        )

    def get_mask_sizes(self, cache_position: torch.Tensor) -> tuple[int, int]:
        """Return the length and offset of the cache, used to generate the mask."""
        kv_offset = 0
        query_length = cache_position.shape[0]
        past_seen_tokens = self.get_seq_length()
        kv_length = query_length + past_seen_tokens
        return kv_length, kv_offset

    def get_seq_length(self) -> int:
        """Returns the sequence length of the cached states."""
        return self.cumulative_length

    def get_max_cache_shape(self) -> int:
        """Returns the maximum sequence length of the cache object."""
        return self.max_context_length


# =============================================================================
# Combined contiguous + paged cache layer
# =============================================================================


class Mode(Enum):
    """Cache operation modes."""

    START = 0
    PREFILL = 1
    INFERENCE = 2


class DMSCombinedCacheLayer(CacheLayerMixin):
    """Used for handling prefill along with inference.

    Contiguous cache is used for recent tokens, paged cache is used for tokens outside of the sliding window.
    """

    def __init__(
        self,
        dms_window_size: int,
        max_context_length: int,
        block_size: int = 256,
        growth_factor: float = 1.5,
        accommodate_min_initial_context_length: int = 4096,
        disable_eviction: bool = False,
    ):
        """Initialize combined cache with contiguous and paged sub-caches."""
        super().__init__()
        self.dms_window_size = dms_window_size
        self.block_size = block_size
        self.disable_eviction = disable_eviction
        self.paged_cache = DMSPagedCacheLayer(
            dms_window_size=dms_window_size,
            max_context_length=max_context_length,
            block_size=block_size,
            growth_factor=growth_factor,
            accommodate_min_initial_context_length=accommodate_min_initial_context_length,
            disable_eviction=True,
        )  # For prefill & inference
        self.cont_cache = DMSContCacheLayer(
            dms_window_size=dms_window_size,
            max_context_length=max_context_length,
            block_size=block_size,
            growth_factor=growth_factor,
            accommodate_min_initial_context_length=accommodate_min_initial_context_length,
            disable_eviction=True,
        )  # For prefill

        self.max_context_length = max_context_length

        self.current_mode = Mode.START
        self.cumulative_length = 0

    def offload(self):
        """Offload cache tensors to CPU."""
        self.paged_cache.offload()
        self.cont_cache.offload()

    def prefetch(self):
        """Prefetch cache tensors back to the original device."""
        self.paged_cache.prefetch()
        self.cont_cache.prefetch()

    def reset(self):
        """Reset both sub-caches and return to start mode."""
        self.paged_cache.reset()
        self.cont_cache.reset()
        self.current_mode = Mode.START
        self.cumulative_length = 0

    def reorder_cache(self, beam_idx: torch.LongTensor):
        """Reorder cache for beam search (not supported)."""
        raise NotImplementedError("Beam search is not supported")

    def lazy_initialization(self, key_states: torch.Tensor):
        """Lazy initialization placeholder."""
        return None

    def update(
        self,
        key_states: torch.Tensor,
        value_states: torch.Tensor,
        cache_kwargs: dict[str, Any],
    ):
        """Update the cache with new key-value states and eviction info."""
        assert self.current_mode != Mode.START

        batch, head, seq_len, _head_dim = key_states.size()
        seq_lengths = cache_kwargs["sequence_lengths"]
        eviction_info = cache_kwargs["eviction_info"]
        cumulative_length = cache_kwargs["cumulative_length"]
        self.cumulative_length += cumulative_length

        assert value_states.size() == key_states.size(), (
            f"value_states.size: {value_states.size()} != key_states.size: {key_states.size()}"
        )
        assert seq_lengths is None or seq_lengths.size() == (batch, head), (
            f"seq_lengths.size: {seq_lengths.size()} != (batch, head): {(batch, head)}"
        )
        assert eviction_info.size() == (
            batch,
            head,
            seq_len,
        ), (
            f"eviction info size: {eviction_info.size()} should be {(batch, head, seq_len)}"
        )  # Eviction info is right shifted by 1

        if self.current_mode == Mode.PREFILL:
            assert seq_lengths is not None

            keys_recent = key_states[:, :, -self.cont_cache.dms_window_size :, :]
            keys_to_paged_cache = key_states[:, :, : -self.cont_cache.dms_window_size, :]

            values_recent = value_states[:, :, -self.cont_cache.dms_window_size :, :]
            values_to_paged_cache = value_states[:, :, : -self.cont_cache.dms_window_size, :]

            eviction_info_recent = eviction_info[..., -self.cont_cache.dms_window_size :]

            seq_lengths_recent = torch.clamp(seq_lengths, max=self.cont_cache.dms_window_size)
            seq_lengths_to_paged_cache = (seq_lengths - self.cont_cache.dms_window_size).clamp(
                min=0
            )

            # move what we can to the paged cache

            cumulative_length_to_paged_cache = keys_to_paged_cache.shape[2]

            if cumulative_length_to_paged_cache > 0 and seq_lengths_to_paged_cache.max() > 0:
                self.paged_cache.fast_update_ignore_eviction(
                    key_states=keys_to_paged_cache,
                    value_states=values_to_paged_cache,
                    sequence_lengths=seq_lengths_to_paged_cache,
                )

            self.cont_cache.replace(
                key_states=keys_recent,
                value_states=values_recent,
                cache_kwargs={
                    "eviction_info": eviction_info_recent,
                    "sequence_lengths": seq_lengths_recent,
                    "cumulative_length": keys_recent.shape[2],
                },
            )
        elif self.current_mode == Mode.INFERENCE:
            assert seq_lengths is None, "seq_lengths is not None in inference mode"
            assert cumulative_length == 1, f"cumulative_length: {cumulative_length} != 1"
            assert self.cont_cache.cumulative_length == 0

            self.paged_cache.update(
                key_states=key_states,
                value_states=value_states,
                cache_kwargs={
                    "eviction_info": eviction_info,
                    "sequence_lengths": None,
                    "cumulative_length": 1,
                },
            )
        else:
            raise ValueError(f"Invalid mode: {self.current_mode}")

    def prefill_mode(self):
        """Switch to prefill mode."""
        if self.current_mode == Mode.PREFILL:
            pass
        elif self.current_mode == Mode.INFERENCE:
            # Revert last self.window_size keys and values to contiguous cache
            raise NotImplementedError("Cannot revert to prefill mode from inference mode")
        elif self.current_mode == Mode.START:
            pass
        else:
            raise ValueError(f"Invalid mode: {self.current_mode}")

        self.paged_cache.enable_prefill_mode()
        self.current_mode = Mode.PREFILL

    def inference_mode(self):
        """Switch to inference mode."""
        if self.current_mode == Mode.INFERENCE:
            pass
        elif self.current_mode == Mode.PREFILL:
            key_states, value_states, seq_lengths, eviction_info = self.cont_cache.get_cache()

            self.current_mode = Mode.INFERENCE

            self.paged_cache.disable_prefill_mode(disable_eviction=self.disable_eviction)

            self.paged_cache.update(
                key_states=key_states,
                value_states=value_states,
                cache_kwargs={
                    "eviction_info": eviction_info,
                    "sequence_lengths": seq_lengths,
                    "cumulative_length": self.cont_cache.cumulative_length,
                },
            )
            self.cont_cache.reset()

        elif self.current_mode == Mode.START:
            self.current_mode = Mode.INFERENCE
        else:
            raise ValueError(f"Invalid mode: {self.current_mode}")

    def start_mode(self):
        """Assert that the cache is in start mode."""
        assert self.current_mode == Mode.START, (
            f"current_mode: {self.current_mode} is not Mode.START"
        )

    def get_recent_cache(self):
        """Get the recent contiguous cache contents."""
        assert self.current_mode == Mode.PREFILL, (
            f"current_mode: {self.current_mode} is not Mode.PREFILL"
        )
        return self.cont_cache.get_cache()

    def get_recent_cache_csize(self):
        """Get the cumulative length of the recent cache."""
        return self.cont_cache.cumulative_length

    def get_paged_cache_csize(self):
        """Get the cumulative length of the paged cache."""
        return self.paged_cache.cumulative_length

    def get_paged_cache(self):
        """Get the paged cache layer."""
        return self.paged_cache

    def is_inference_mode(self):
        """Check if in inference mode."""
        return self.current_mode == Mode.INFERENCE

    def is_prefill_mode(self):
        """Check if in prefill mode."""
        return self.current_mode == Mode.PREFILL

    def is_start_mode(self):
        """Check if in start mode."""
        return self.current_mode == Mode.START

    def get_mask_sizes(self, cache_position: torch.Tensor) -> tuple[int, int]:
        """Return the length and offset of the cache, used to generate the mask."""
        kv_offset = 0
        query_length = cache_position.shape[0]
        past_seen_tokens = self.get_seq_length()
        kv_length = query_length + past_seen_tokens
        return kv_length, kv_offset

    def get_seq_length(self) -> int:
        """Returns the sequence length of the cached states."""
        return self.cumulative_length

    def get_max_cache_shape(self) -> int:
        """Returns the maximum sequence length of the cache object."""
        return self.paged_cache.max_context_length


# =============================================================================
# HuggingFace-compatible DMS cache wrapper
# =============================================================================


class DMSCache(Cache):
    """HuggingFace Cache implementation for DMS with combined cache layers."""

    def __init__(
        self,
        dms_window_size: int,
        max_context_length: int,
        offloading: bool = False,
        offload_only_non_sliding: bool = False,
        accommodate_min_initial_context_length: int = 4096,
        disable_eviction: bool = False,
        block_size: int = 256,
    ):
        """Initialize the DMS cache."""
        super().__init__(
            layer_class_to_replicate=functools.partial(
                DMSCombinedCacheLayer,
                dms_window_size=dms_window_size,
                max_context_length=max_context_length,
                accommodate_min_initial_context_length=accommodate_min_initial_context_length,
                disable_eviction=disable_eviction,
                block_size=block_size,
            ),
            offloading=offloading,
            offload_only_non_sliding=offload_only_non_sliding,
        )

        self.current_mode = Mode.START

    def to_legacy_cache(self) -> tuple[tuple[torch.Tensor, torch.Tensor]]:
        """Convert to legacy cache format (not supported)."""
        raise NotImplementedError("Not Supported")

    def _match_single_layer_mode(self, layer: DMSCombinedCacheLayer):
        if self.current_mode == Mode.PREFILL:
            layer.prefill_mode()
        elif self.current_mode == Mode.INFERENCE:
            layer.inference_mode()
        elif self.current_mode == Mode.START:
            layer.start_mode()
        else:
            raise ValueError(f"Invalid mode: {self.current_mode}")

    def _match_all_layers_mode(self):
        for layer in self.layers:
            assert isinstance(layer, DMSCombinedCacheLayer)
            self._match_single_layer_mode(layer)

    def prefill_mode(self):
        """Set all layers to prefill mode."""
        self.current_mode = Mode.PREFILL
        self._match_all_layers_mode()

    def inference_mode(self):
        """Set all layers to inference mode."""
        self.current_mode = Mode.INFERENCE
        self._match_all_layers_mode()

    def is_prefill_mode(self):
        """Check if in prefill mode."""
        return self.current_mode == Mode.PREFILL

    def is_inference_mode(self):
        """Check if in inference mode."""
        return self.current_mode == Mode.INFERENCE

    def is_start_mode(self):
        """Check if in start mode."""
        return self.current_mode == Mode.START

    @classmethod
    def from_legacy_cache(cls, past_key_values: tuple[tuple[torch.Tensor, torch.Tensor]]):
        """Create from legacy cache (not supported)."""
        raise NotImplementedError("Not Supported")

    def early_initialization(
        self,
        batch_size: int,
        num_heads: int,
        head_dim: int,
        dtype: torch.dtype,
        device: torch.device,
    ):
        """Perform early initialization (not supported)."""
        return None  # not supported

    def __iter__(self):
        raise NotImplementedError("Not Supported")

    def __getitem__(self, layer_idx: int):
        while layer_idx >= len(self.layers):
            self.layers.append(self.layer_class_to_replicate())
            self._match_single_layer_mode(self.layers[-1])
        return self.layers[layer_idx]

    def __setitem__(self, layer_idx: int, value: DMSCombinedCacheLayer):
        while layer_idx >= len(self.layers):
            self.layers.append(self.layer_class_to_replicate())
            self._match_single_layer_mode(self.layers[-1])
        self.layers[layer_idx] = value
        self._match_single_layer_mode(self.layers[layer_idx])

    def get_cr(self, get_per_layer_cr: bool = False) -> float | tuple[float, list[float]]:
        """Compute the compression ratio across all cache layers."""
        per_elem = []
        for layer in self.layers:
            assert isinstance(layer, DMSCombinedCacheLayer)
            cum_seq_len = layer.get_seq_length()
            if layer.paged_cache.cache_seq_lengths is None:
                return 1.0
            sizes = layer.paged_cache.cache_seq_lengths.cpu()

            frac = sizes / max(cum_seq_len, 1)
            per_elem.append(frac)

        per_elem = torch.stack(per_elem, dim=0)
        total_cr = 1 / per_elem.mean()

        if get_per_layer_cr:
            per_layer_cr = 1 / per_elem.mean(dim=-1)
            return total_cr.item(), per_layer_cr.tolist()
        else:
            return total_cr.item()
