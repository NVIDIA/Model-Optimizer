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

"""Lightweight fake base model for offline speculative decoding training."""

import json

import torch
import torch.nn as nn
from huggingface_hub import hf_hub_download
from safetensors.torch import load_file as safetensors_load_file
from transformers import PretrainedConfig, PreTrainedModel


class FakeBaseConfig(PretrainedConfig):
    """Minimal config for FakeBaseModel that supports offline speculative decoding training."""

    model_type = "fake_base_model"

    def __init__(
        self,
        num_hidden_layers=None,
        hidden_size=None,
        vocab_size=None,
        max_position_embeddings=None,
        dtype=torch.bfloat16,
        tie_word_embeddings=False,
        **kwargs,
    ):
        """Initialize FakeBaseConfig with minimal model configuration parameters."""
        super().__init__(tie_word_embeddings=tie_word_embeddings, **kwargs)
        self.num_hidden_layers = num_hidden_layers
        self.hidden_size = hidden_size
        self.vocab_size = vocab_size
        self.max_position_embeddings = max_position_embeddings
        self.dtype = dtype


class FakeBaseModel(PreTrainedModel):
    """Minimal base model for offline speculative decoding.

    Contains only lm_head, embed_tokens, and necessary configs.

    This lightweight class should works ootb for convert, train, save/reload, and
    export in offline speculative decoding workflow, while allowing:
    1. Faster initialization and loading by omitting full model layers.
    2. Compatibility with cases where standard HuggingFace loading is incomplete or unsupported.

    Subclasses should override/define the following attributes:
        SOURCE_HF_REPO (str): HuggingFace repository ID for weight retrieval.
        INDEX_FILENAME (str): Name of the JSON file listing sharded weight files.
        LM_HEAD_KEY (str): Key for the language modeling head in the safetensors state dict.
        EMBED_TOKENS_KEY (str): Key for the embedding tokens in the safetensors state dict.
    """

    config_class = FakeBaseConfig

    # Default values; subclasses should override as needed.
    SOURCE_HF_REPO: str = None
    INDEX_FILENAME: str = "model.safetensors.index.json"
    LM_HEAD_KEY: str = "lm_head.weight"
    EMBED_TOKENS_KEY: str = "model.embed_tokens.weight"

    def __init__(self, config: FakeBaseConfig):
        """Initialize FakeBaseModel and download lm_head/embed_tokens weights from HuggingFace.

        Args:
            config (FakeBaseConfig): Model configuration.
        """
        super().__init__(config)
        self.config = config
        self.model = nn.Module()
        self.model.layers = nn.ModuleList()
        self.model.dtype = config.dtype
        self.embed_tokens = nn.Embedding(self.config.vocab_size, self.config.hidden_size)
        self.lm_head = nn.Linear(self.config.hidden_size, self.config.vocab_size, bias=False)

        try:
            lm_head_w, embed_tokens_w = self._download_lm_head_and_embed_tokens()
            assert lm_head_w.shape == (self.config.vocab_size, self.config.hidden_size)
            assert embed_tokens_w.shape == (self.config.vocab_size, self.config.hidden_size)
            self.lm_head.weight.data.copy_(lm_head_w)
            self.embed_tokens.weight.data.copy_(embed_tokens_w)
        except Exception as e:
            raise ValueError(f"Failed to initialize lm_head and embed_tokens: {e}")

    @classmethod
    def from_base_config(cls, base_config: PretrainedConfig):
        """Create a FakeBaseModel instance using a configuration from a full, real model.

        Args:
            base_config (PretrainedConfig): The original model configuration.

        Returns:
            FakeBaseModel: A new instance with the minimal configuration.
        """
        config_params = {
            "num_hidden_layers": getattr(base_config, "num_hidden_layers", None),
            "hidden_size": getattr(base_config, "hidden_size", None),
            "vocab_size": getattr(base_config, "vocab_size", None),
            "max_position_embeddings": getattr(base_config, "max_position_embeddings", None),
            "dtype": getattr(base_config, "dtype", torch.bfloat16),
            "tie_word_embeddings": getattr(base_config, "tie_word_embeddings", False),
        }
        return cls(FakeBaseConfig(**config_params))

    def _download_lm_head_and_embed_tokens(self):
        if self.SOURCE_HF_REPO is None:
            raise ValueError("Set SOURCE_HF_REPO as a class attribute or in a subclass.")

        index_json_file = hf_hub_download(
            repo_id=self.SOURCE_HF_REPO,
            filename=self.INDEX_FILENAME,
        )
        with open(index_json_file) as f:
            index_data = json.load(f)

        weight_map = index_data.get("weight_map", {})
        lm_head_file = weight_map.get(self.LM_HEAD_KEY)
        embed_tokens_file = weight_map.get(self.EMBED_TOKENS_KEY)

        if not lm_head_file or not embed_tokens_file:
            raise RuntimeError(f"{self.LM_HEAD_KEY} or {self.EMBED_TOKENS_KEY} not found in index!")

        lm_head_shard_file = hf_hub_download(repo_id=self.SOURCE_HF_REPO, filename=lm_head_file)
        embed_tokens_shard_file = hf_hub_download(
            repo_id=self.SOURCE_HF_REPO, filename=embed_tokens_file
        )

        lm_head_state = safetensors_load_file(lm_head_shard_file, device="cpu")
        embed_tokens_state = safetensors_load_file(embed_tokens_shard_file, device="cpu")

        lm_head_weight = lm_head_state[self.LM_HEAD_KEY]
        embed_tokens_weight = embed_tokens_state[self.EMBED_TOKENS_KEY]

        return lm_head_weight, embed_tokens_weight

    def forward(self, *args, **kwargs):
        """Not implemented: FakeBaseModel omits full model weights and cannot run inference."""
        raise NotImplementedError("FakeBaseModel forward is not implemented.")


class KimiK25FakeBaseModel(FakeBaseModel):
    """FakeBaseModel subclass tailored for Kimi-K2.5."""

    SOURCE_HF_REPO = "moonshotai/Kimi-K2.5"
    LM_HEAD_KEY = "language_model.lm_head.weight"
    EMBED_TOKENS_KEY = "language_model.model.embed_tokens.weight"
