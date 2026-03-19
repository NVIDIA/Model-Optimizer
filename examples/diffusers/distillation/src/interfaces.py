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

"""Pluggable interfaces for model-specific components.

Three protocols define the contract between the unified trainer and model backends:
- ModelLoader: load the transformer architecture from checkpoint
- TrainingForwardAdapter: noise application, model forward, task loss (stateful)
- InferencePipeline: validation video generation + data preprocessing (optional)
"""

from __future__ import annotations

import gc
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol, runtime_checkable

import torch
import torch.nn as nn
from torch import Tensor


@dataclass
class StrategyOutputs:
    """Output from TrainingForwardAdapter.prepare_inputs()."""

    noisy_input: Any
    targets: Tensor
    loss_mask: Tensor
    forward_kwargs: dict = field(default_factory=dict)


@dataclass
class CachedEmbeddings:
    """Pre-computed text embeddings for a single prompt.

    Contents are model-specific (e.g., LTX-2 stores video/audio context
    separately; Wan stores a single T5 embedding tensor).
    """

    positive: dict[str, Tensor]
    negative: dict[str, Tensor]


# ---------------------------------------------------------------------------
# Protocols
# ---------------------------------------------------------------------------


@runtime_checkable
class ModelLoader(Protocol):
    def load_transformer(
        self,
        path: str | Path,
        device: str = "cpu",
        dtype: torch.dtype = torch.bfloat16,
    ) -> nn.Module: ...


@runtime_checkable
class TrainingForwardAdapter(Protocol):
    """Model-specific training forward logic.

    Adapters are *stateful* -- they may hold references to model-specific
    components (patchifiers, embedding connectors, etc.) that are initialized
    by the model backend's factory function.
    """

    def prepare_inputs(
        self,
        batch: dict[str, Tensor],
        noise: Tensor,
        timesteps: Tensor,
    ) -> StrategyOutputs:
        """Apply noise to latents, build model-specific inputs, compute targets."""
        ...

    def forward_model(self, model: nn.Module, inputs: StrategyOutputs) -> Tensor:
        """Run the model and return prediction as a single Tensor."""
        ...

    def compute_task_loss(self, pred: Tensor, targets: Tensor, loss_mask: Tensor) -> Tensor:
        """Compute flow-matching task loss (typically masked MSE)."""
        ...


@runtime_checkable
class InferencePipeline(Protocol):
    """Model-specific inference for validation video generation and data preprocessing.

    Manages the lifecycle of non-training components (text encoder, VAE):
    - Text encoder: loaded once, used to cache embeddings, then deleted.
    - VAE: kept on CPU during training, moved to GPU briefly during generate().
    """

    def load_components(self, model_config: Any, device: str, dtype: torch.dtype) -> None:
        """Load text encoder, VAE, and other inference components.

        Args:
            model_config: A ModelConfig object (has .model_path, .text_encoder_path, etc.)
                          or a plain path string for simple backends.
        """
        ...

    def encode_prompts(
        self,
        prompts: list[str],
        negative_prompt: str,
        device: str,
    ) -> list[CachedEmbeddings]:
        """Encode prompts via the text encoder. Must be called before unload_text_encoder()."""
        ...

    def unload_text_encoder(self) -> None:
        """Permanently delete the text encoder to free GPU/CPU memory."""
        ...

    def offload_to_cpu(self) -> None:
        """Move remaining heavy components (VAE) to CPU."""
        ...

    def encode_videos(
        self,
        videos: list[Tensor],
        device: str,
    ) -> list[Tensor]:
        """Encode pixel-space videos [C,F,H,W] to latent space via VAE encoder."""
        ...

    def generate(
        self,
        model: nn.Module,
        cached_embeds: list[CachedEmbeddings],
        config: dict,
        device: str,
    ) -> list[Tensor]:
        """Run denoising inference and return decoded videos [C,F,H,W] in [0,1]."""
        ...


def free_gpu_memory() -> None:
    """Force-release GPU memory."""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
