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

# Architecture reference: https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash-1017/blob/1bae0e1088993b9b82de08f59de5be5a34e42170/inference/engram.py
# Copyright (c) 2023 DeepSeek. MIT License.
# SPDX-License-Identifier: Apache-2.0 AND MIT

"""Stateless full-sequence Engram hashing and differentiable memory reads."""

import math

import numpy as np
import torch
from tokenizers import Regex, normalizers
from torch import nn

from ._hf_math import V41Linear

__all__ = []


def _next_prime(start, used):
    candidate = start
    while True:
        if (
            candidate not in used
            and candidate >= 2
            and all(candidate % divisor for divisor in range(2, math.isqrt(candidate) + 1))
        ):
            used.add(candidate)
            return candidate
        candidate += 1


def normalized_token_map(tokenizer):
    normalizer = normalizers.Sequence(
        [
            normalizers.NFKC(),
            normalizers.NFD(),
            normalizers.StripAccents(),
            normalizers.Lowercase(),
            normalizers.Replace(Regex(r"[ \t\r\n]+"), " "),
            normalizers.Replace(Regex(r"^ $"), "\ue000"),
            normalizers.Strip(),
            normalizers.Replace("\ue000", " "),
        ]
    )
    backend = tokenizer.backend_tokenizer
    compressed, vocabulary = [], {}
    for token_id in range(len(tokenizer)):
        text = backend.decode([token_id], skip_special_tokens=False)
        key = (
            backend.id_to_token(token_id)
            if "\ufffd" in text
            else (normalizer.normalize_str(text) or text)
        )
        if key not in vocabulary:
            vocabulary[key] = len(vocabulary)
        compressed.append(vocabulary[key])
    return compressed, len(vocabulary)


class V41NgramHash(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        primes, offsets, used = [], [], set()
        for row_count in config.engram_num_embeddings:
            layer, sizes = [], []
            for _ in range(config.engram_max_ngram_size - 1):
                current, heads = config.engram_vocab_size, []
                for _ in range(config.engram_n_heads):
                    current = _next_prime(current, used)
                    heads.append(current)
                    sizes.append(current)
                    current += 1
                layer.append(heads)
            if sum(sizes) != row_count:
                raise ValueError(
                    "Engram table row count does not match its published prime bucket layout."
                )
            primes.append(layer)
            offsets.append(np.cumsum([0, *sizes[:-1]]))
        bound = max(1, (np.iinfo(np.int64).max // config.engram_compressed_vocab_size) // 2)
        multipliers = [
            np.random.default_rng(10007 * layer).integers(
                0, bound, size=config.engram_max_ngram_size, dtype=np.int64
            )
            * 2
            + 1
            for layer in config.engram_layer_ids
        ]
        self._derived_values = (primes, np.array(offsets).tolist(), np.array(multipliers).tolist())
        self.register_buffer("primes", torch.tensor(primes), persistent=False)
        self.register_buffer("offsets", torch.tensor(np.array(offsets)), persistent=False)
        self.register_buffer("multipliers", torch.tensor(np.array(multipliers)), persistent=False)
        mapping = config.engram_token_map
        if mapping is not None:
            self._validate_map(mapping)
        self.register_buffer(
            "token_map", torch.tensor(mapping if mapping is not None else [0] * config.vocab_size)
        )
        self.ready = mapping is not None

    def reset_hash_buffers(self):
        for name, values in zip(("primes", "offsets", "multipliers"), self._derived_values):
            setattr(self, name, torch.tensor(values, device=self.token_map.device))

    def _validate_map(self, mapping):
        if (
            len(mapping) != self.config.vocab_size
            or min(mapping) < 0
            or max(mapping) >= self.config.engram_compressed_vocab_size
        ):
            raise ValueError(
                "Engram token map must cover the vocabulary with valid compressed IDs."
            )

    def set_tokenizer(self, tokenizer):
        mapping, vocabulary_size = normalized_token_map(tokenizer)
        if vocabulary_size != self.config.engram_compressed_vocab_size:
            raise ValueError(
                "Tokenizer normalization disagrees with the checkpoint's compressed vocabulary."
            )
        self._validate_map(mapping)
        self.token_map.copy_(torch.tensor(mapping, device=self.token_map.device))
        self.config.engram_token_map = mapping
        self.ready = True

    def forward(self, input_ids, token_mask=None):
        if not self.ready:
            raise ValueError(
                "Engram requires the published tokenizer; call set_engram_tokenizer or provide engram_token_map."
            )
        tokens = self.token_map[input_ids]
        if token_mask is not None:
            tokens = tokens.masked_fill(~token_mask, -1)
        position = torch.arange(tokens.shape[1], device=tokens.device)
        blocked = torch.zeros_like(tokens, dtype=torch.bool)
        history = []
        pad = self.token_map[self.config.engram_pad_token_id]
        for shift in range(self.config.engram_max_ngram_size):
            source = tokens[:, (position - shift).clamp_min(0)]
            blocked = blocked | (position < shift) | (source == -1)
            history.append(torch.where(blocked, pad, source))
        products = torch.stack(history, -1).unsqueeze(-2) * self.multipliers
        accumulated, hashes = products[..., 0], []
        for shift in range(1, self.config.engram_max_ngram_size):
            accumulated = accumulated.bitwise_xor(products[..., shift])
            hashes.append(accumulated.unsqueeze(-1) % self.primes[:, shift - 1])
        return torch.cat(hashes, -1) + self.offsets


class V41Engram(nn.Module):
    def __init__(self, config, layer_id):
        super().__init__()
        self.config = config
        self.layer_hash_index = config.engram_layer_ids.index(layer_id)
        self.embed = nn.Embedding(
            config.engram_num_embeddings[self.layer_hash_index], config.engram_head_dim
        )
        width = (config.engram_max_ngram_size - 1) * config.engram_n_heads * config.engram_head_dim
        self.wkv = V41Linear(width, config.hidden_size * (config.hc_mult + 1))
        self.q_weight = nn.Parameter(torch.ones(config.hc_mult, config.hidden_size))
        self.k_weight = nn.Parameter(torch.ones(config.hc_mult, config.hidden_size))

    def forward(self, hidden_states, hashes, token_mask):
        kv = self.wkv(self.embed(hashes).flatten(-2))
        key, value = kv.split(
            [self.config.hc_mult * self.config.hidden_size, self.config.hidden_size], -1
        )
        key = key.float().unflatten(-1, (self.config.hc_mult, self.config.hidden_size))
        stream = hidden_states.float()
        normalization = torch.rsqrt(stream.square().mean(-1) + self.config.rms_norm_eps)
        normalization = normalization * torch.rsqrt(
            key.square().mean(-1) + self.config.rms_norm_eps
        )
        dot = (stream * self.q_weight.float() * self.k_weight.float() * key).sum(-1)
        dot = dot * normalization / math.sqrt(self.config.hidden_size)
        gate = torch.copysign(dot.abs().clamp_min(1e-6).sqrt(), dot).sigmoid()
        if token_mask is not None:
            gate = gate.masked_fill(~token_mask.unsqueeze(-1), 0)
        return (stream + gate.unsqueeze(-1) * value.float().unsqueeze(-2)).to(hidden_states.dtype)
