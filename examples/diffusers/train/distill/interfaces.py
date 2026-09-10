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
from typing import TYPE_CHECKING, Any, Protocol, runtime_checkable

if TYPE_CHECKING:
    from pathlib import Path

import torch
import torch.nn as nn
from torch import Tensor


@dataclass
class TextEmbeddings:
    """Processed text embeddings ready for the transformer."""

    video_context: Tensor  # [B, L, D] text context for video modality
    audio_context: Tensor | None = None  # [B, L, D] text context for audio modality (None if N/A)


@dataclass
class BackboneInputs:
    """Output from TrainingForwardAdapter.prepare_inputs()."""

    noisy_input: Any
    targets: Tensor
    loss_mask: Tensor
    forward_kwargs: dict = field(default_factory=dict)

    # Audio (optional, for audio-video models like LTX-2)
    audio_targets: Tensor | None = None
    audio_loss_mask: Tensor | None = None


@dataclass
class CachedEmbeddings:
    """Pre-computed text embeddings for a single prompt.

    Contents are model-specific (e.g., LTX-2 stores video/audio context
    separately; Wan stores a single T5 embedding tensor).
    """

    positive: dict[str, Tensor]
    negative: dict[str, Tensor]


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
        pipeline: Any = None,
    ) -> BackboneInputs:
        """Apply noise to latents, build model-specific inputs, compute targets.

        Args:
            pipeline: The inference pipeline, used to process raw text embeddings
                into model-ready context (e.g. run connectors for LTX-2).
        """
        ...

    def forward_model(self, model: nn.Module, inputs: BackboneInputs) -> Any:
        """Run the model and return the raw model output.

        The return type is model-specific (e.g. Tensor for Wan, tuple for LTX-2).
        It is passed opaquely to compute_task_loss / compute_distillation_loss.
        """
        ...

    def compute_task_loss(self, model_output: Any, inputs: BackboneInputs) -> Tensor:
        """Compute flow-matching task loss from model output and targets.

        Handles all modalities (video, audio if applicable).
        """
        ...

    def compute_distillation_loss(
        self, student_output: Any, teacher_output: Any, inputs: BackboneInputs
    ) -> Tensor:
        """Compute output-level distillation loss between student and teacher.

        Handles all modalities (video, audio if applicable).
        """
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

    def process_text_embeddings(
        self,
        raw_embeds: Tensor,
        attention_mask: Tensor,
    ) -> TextEmbeddings:
        """Transform raw cached text embeddings into model-ready context.

        For models with text embedding connectors (e.g. LTX-2), this runs the
        lightweight connector to produce video and audio context. For models
        without connectors (e.g. Wan), this is an identity operation.

        Called during both training (by the adapter) and inference (by generate).
        """
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
