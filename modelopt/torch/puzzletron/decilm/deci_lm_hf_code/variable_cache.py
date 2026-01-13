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

# mypy: ignore-errors
from copy import deepcopy
from typing import Any

import torch
from transformers.cache_utils import (
    Cache,  # used to let GenerationMixin know that we use a Cache object
)

from .configuration_decilm import DeciLMConfig
from .transformers_4_44_2__cache_utils import Cache as Cache_4_44_2
from .transformers_4_44_2__cache_utils import SinkCache, SlidingWindowCache, StaticCache
from .transformers_4_51_3__cache_utils import HybridChunkedCache

LayerIndex = tuple[
    int, ...
]  # supports both regular transformer blocks and parallel transformer multi-blocks


class VariableCache(Cache_4_44_2, Cache):
    """
    A Cache object that supports a different Cache implementation for every layer,
    including layers without any kv-cache.
    Implemented using a list of Cache objects, each represents a "model" with 1 layer.
    The default implementation for the layer caches is StaticCache.
    The cache of each layer is allocated to the same gpu as the layer itself.
    """

    def __init__(
        self,
        *,  # key-word only, no positional args allowed to avoid mix-ups with newer transformers versions
        config: DeciLMConfig,
        batch_size: int | None = None,
        max_cache_len: int | None = None,
        dtype: torch.dtype = torch.get_default_dtype(),
        max_batch_size: int | None = None,
        **kwargs,
    ) -> None:
        Cache_4_44_2.__init__(self)

        self.config = deepcopy(config)
        self.max_batch_size = batch_size or max_batch_size
        self.batch_size = self.max_batch_size
        self.max_cache_len = (
            config.max_position_embeddings if (max_cache_len is None) else max_cache_len
        )
        self.dtype = dtype

        self.layer_caches: dict[LayerIndex, Cache_4_44_2] = {}
        self.layer_devices: dict[LayerIndex, torch.device] = {}

    def __repr__(self):
        return (
            f"VariableCache:\n"
            f"==============\n"
            f"max_batch_size={self.max_batch_size}\n"
            f"batch_size={self.batch_size}\n"
            f"max_cache_len={self.max_cache_len}\n"
            f"dtype={self.dtype}\n"
            f"layer_caches={self.layer_caches}\n"
            f"layer_devices={self.layer_devices}\n"
            f"==============\n"
        )

    def update(
        self,
        key_states: torch.Tensor,
        value_states: torch.Tensor,
        layer_idx: int | LayerIndex,
        cache_kwargs: dict[str, Any] | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if isinstance(layer_idx, int):
            layer_idx = _int_to_layer_index(layer_idx)

        if layer_idx not in self.layer_caches:
            self.layer_devices[layer_idx] = key_states.device
            self._init_layer_cache(layer_idx)

        layer_cache = self.layer_caches[layer_idx]
        assert layer_cache is not None, (
            f"Trying to update the cache of a cache-less layer: {layer_idx=}"
        )

        k_out, v_out = layer_cache.update(
            key_states=key_states, value_states=value_states, layer_idx=0, cache_kwargs=cache_kwargs
        )

        input_seq_len = key_states.shape[2]  # [batch_size, num_kv_heads, seq_len, hidden_size]
        cache_seq_len = self.get_seq_length(layer_idx)
        seq_len = max(input_seq_len, cache_seq_len)

        k_out = k_out[:, :, :seq_len, :]
        v_out = v_out[:, :, :seq_len, :]
        return k_out, v_out

    def _init_layer_cache(self, layer_idx: LayerIndex) -> None:
        block_config = self.config.get_block_config(layer_idx)
        attention_config = block_config.attention

        if attention_config.no_op or attention_config.replace_with_linear:
            return None

        device = self.layer_devices[layer_idx]
        assert device is not None, f"Trying to init layer cache for {layer_idx=} without device"

        config = deepcopy(self.config)
        config.num_hidden_layers = 1
        config.num_key_value_heads = (
            self.config.num_attention_heads // attention_config.n_heads_in_group
        )

        if attention_config.is_llama4:
            attention_chunk_size = attention_config.llama4.attention_chunk_size
            is_chunked = attention_chunk_size is not None
            config.no_rope_layers = [int(is_chunked)]
            config.attention_chunk_size = (
                attention_chunk_size if is_chunked else config.get_min_attention_chunk_size()
            )
            self.layer_caches[layer_idx] = HybridChunkedCache(
                config=config,
                max_batch_size=self.max_batch_size,
                max_cache_len=self.max_cache_len,
                dtype=self.dtype,
            )
            return

        if attention_config.window_length is not None:
            if not attention_config.is_sink:
                config.sliding_window = attention_config.window_length
                self.layer_caches[layer_idx] = SlidingWindowCache(
                    config=config,
                    max_batch_size=self.max_batch_size,
                    max_cache_len=self.max_cache_len,
                    device=device,
                    dtype=self.dtype,
                )
                return
            elif not attention_config.unshifted_sink:
                self.layer_caches[layer_idx] = SinkCache(
                    window_length=attention_config.window_length,
                    num_sink_tokens=attention_config.num_sink_tokens,
                )
                return

        self.layer_caches[layer_idx] = StaticCache(
            config=config,
            max_batch_size=self.max_batch_size,
            max_cache_len=self.max_cache_len,
            device=device,
            dtype=self.dtype,
        )

    def _get_arbitrary_cache(self) -> Cache_4_44_2:
        if len(self.layer_caches) == 0:
            raise NoCacheFoundError()
        layer_cache = next(iter(self.layer_caches.values()))
        return layer_cache

    def get_seq_length(self, layer_idx: int | LayerIndex | None = 0) -> int:
        """default 0 to match standard HF implementation"""
        if (layer_idx is None) or (
            layer_idx == 0 and _int_to_layer_index(0) not in self.layer_caches
        ):
            try:
                layer_cache = self._get_arbitrary_cache()
                return layer_cache.get_seq_length()
            except NoCacheFoundError:
                return 0

        if isinstance(layer_idx, int):
            layer_idx = _int_to_layer_index(layer_idx)

        layer_cache = self.layer_caches[layer_idx]
        return layer_cache.get_seq_length()

    def get_max_length(self) -> int | None:
        """Returns the maximum sequence length of the cached states."""
        return self.max_cache_len

    def get_max_cache_shape(self) -> int | None:
        return self.max_cache_len

    def reset(self):
        for layer_idx, layer_cache in self.layer_caches.items():
            if hasattr(layer_cache, "reset"):
                layer_cache.reset()
            else:
                self.layer_caches[layer_idx] = None
                self.layer_devices[layer_idx] = None
                # self._init_layer_cache(layer_idx)


class NoCacheFoundError(Exception):
    pass


def _int_to_layer_index(layer_idx: int) -> LayerIndex:
    return (layer_idx,)
