# Copyright (c) 2025 Lightricks. All rights reserved.
# Created by Andrew Kvochko


import torch

from ltx_core.guidance.perturbations import BatchedPerturbationConfig
from ltx_core.model.transformer.adaln import AdaLayerNormSingle
from ltx_core.model.transformer.attention import AttentionCallable, AttentionFunction
from ltx_core.model.transformer.modality import Modality
from ltx_core.model.transformer.rope import LTXRopeType
from ltx_core.model.transformer.text_projection import PixArtAlphaTextProjection
from ltx_core.model.transformer.transformer import BasicAVTransformerBlock, TransformerConfig
from ltx_core.model.transformer.transformer_args import TransformerArgs, TransformerArgsPreprocessor

import os
enable_vfly = os.environ.get("ENABLE_VFLY", "false").lower() == "true"
if enable_vfly:
    from vfly.utils.parallel import (
        dit_sp_gather,
        dit_sp_split,
    )

class LTXModel(torch.nn.Module):
    """
    LTX model transformer implementation.

    This class implements the transformer blocks for the LTX model.
    """

    def __init__(  # noqa: PLR0913
        self,
        num_attention_heads: int = 32,
        attention_head_dim: int = 128,
        in_channels: int = 128,
        out_channels: int = 128,
        num_layers: int = 48,
        cross_attention_dim: int = 4096,
        norm_eps: float = 1e-06,
        attention_type: AttentionFunction | AttentionCallable = AttentionFunction.DEFAULT,
        caption_channels: int = 3840,
        positional_embedding_theta: float = 10000.0,
        positional_embedding_max_pos: list[int] | None = None,
        timestep_scale_multiplier: int = 1000,
        use_middle_indices_grid: bool = True,
        audio_num_attention_heads: int = 32,
        audio_attention_head_dim: int = 64,
        audio_in_channels: int = 128,
        audio_out_channels: int = 128,
        audio_cross_attention_dim: int = 2048,
        audio_positional_embedding_max_pos: list[int] | None = None,
        av_ca_timestep_scale_multiplier: int = 1,
        rope_type: LTXRopeType = LTXRopeType.INTERLEAVED,
        double_precision_rope: bool = False,
    ):
        super().__init__()
        self.use_middle_indices_grid = use_middle_indices_grid
        self.num_attention_heads = num_attention_heads
        self.positional_embedding_theta = positional_embedding_theta

        if positional_embedding_max_pos is None:
            positional_embedding_max_pos = [20, 2048, 2048]
        self.positional_embedding_max_pos = positional_embedding_max_pos

        self.rope_type = rope_type
        self.double_precision_rope = double_precision_rope
        self.timestep_scale_multiplier = timestep_scale_multiplier
        self.av_ca_timestep_scale_multiplier = av_ca_timestep_scale_multiplier
        self.audio_cross_attention_dim = audio_cross_attention_dim
        if audio_positional_embedding_max_pos is None:
            audio_positional_embedding_max_pos = [20]
        self.audio_positional_embedding_max_pos = audio_positional_embedding_max_pos
        self.audio_num_attention_heads = audio_num_attention_heads
        self.audio_inner_dim = self.audio_num_attention_heads * audio_attention_head_dim

        # Common dimensions
        self.inner_dim = num_attention_heads * attention_head_dim

        # Initializfe model-specific components
        self._init_video(
            in_channels=in_channels,
            out_channels=out_channels,
            caption_channels=caption_channels,
            norm_eps=norm_eps,
            num_scale_shift_values=4,
            cross_pe_max_pos=max(self.positional_embedding_max_pos[0], self.audio_positional_embedding_max_pos[0]),
        )
        self._init_audio(
            in_channels=audio_in_channels,
            out_channels=audio_out_channels,
            caption_channels=caption_channels,
            norm_eps=norm_eps,
            num_scale_shift_values=4,
            cross_pe_max_pos=max(self.positional_embedding_max_pos[0], self.audio_positional_embedding_max_pos[0]),
        )

        # Initialize transformer blocks
        self._init_transformer_blocks(
            num_layers=num_layers,
            attention_head_dim=attention_head_dim,
            cross_attention_dim=cross_attention_dim,
            audio_attention_head_dim=audio_attention_head_dim,
            norm_eps=norm_eps,
            attention_type=attention_type,
        )

    def _init_video(
        self,
        in_channels: int,
        out_channels: int,
        caption_channels: int,
        norm_eps: float,
        num_scale_shift_values: int,
        cross_pe_max_pos: int,
    ) -> None:
        """Initialize video-specific components."""
        # Video input components
        self.patchify_proj = torch.nn.Linear(in_channels, self.inner_dim, bias=True)

        self.adaln_single = AdaLayerNormSingle(self.inner_dim)

        self.av_ca_video_scale_shift_adaln_single = AdaLayerNormSingle(
            self.inner_dim,
            embedding_coefficient=num_scale_shift_values,
        )

        self.av_ca_a2v_gate_adaln_single = AdaLayerNormSingle(
            self.inner_dim,
            embedding_coefficient=1,
        )

        # Video caption projection
        self.caption_projection = PixArtAlphaTextProjection(
            in_features=caption_channels,
            hidden_size=self.inner_dim,
        )

        # Video output components
        self.scale_shift_table = torch.nn.Parameter(torch.empty(2, self.inner_dim))
        self.norm_out = torch.nn.LayerNorm(self.inner_dim, elementwise_affine=False, eps=norm_eps)
        self.proj_out = torch.nn.Linear(self.inner_dim, out_channels)

        self.video_args_preprocessor = TransformerArgsPreprocessor(
            patchify_proj=self.patchify_proj,
            adaln=self.adaln_single,
            caption_projection=self.caption_projection,
            cross_scale_shift_adaln=self.av_ca_video_scale_shift_adaln_single,
            cross_gate_adaln=self.av_ca_a2v_gate_adaln_single,
            inner_dim=self.inner_dim,
            max_pos=self.positional_embedding_max_pos,
            num_attention_heads=self.num_attention_heads,
            cross_pe_max_pos=cross_pe_max_pos,
            use_middle_indices_grid=self.use_middle_indices_grid,
            audio_cross_attention_dim=self.audio_cross_attention_dim,
            timestep_scale_multiplier=self.timestep_scale_multiplier,
            double_precision_rope=self.double_precision_rope,
            positional_embedding_theta=self.positional_embedding_theta,
            rope_type=self.rope_type,
            av_ca_timestep_scale_multiplier=self.av_ca_timestep_scale_multiplier,
        )

    def _init_audio(
        self,
        in_channels: int,
        out_channels: int,
        caption_channels: int,
        norm_eps: float,
        num_scale_shift_values: int,
        cross_pe_max_pos: int,
    ) -> None:
        """Initialize audio-specific components."""

        # Audio input components
        self.audio_patchify_proj = torch.nn.Linear(in_channels, self.audio_inner_dim, bias=True)

        self.audio_adaln_single = AdaLayerNormSingle(
            self.audio_inner_dim,
        )

        self.av_ca_audio_scale_shift_adaln_single = AdaLayerNormSingle(
            self.audio_inner_dim,
            embedding_coefficient=num_scale_shift_values,
        )

        self.av_ca_v2a_gate_adaln_single = AdaLayerNormSingle(
            self.audio_inner_dim,
            embedding_coefficient=1,
        )

        # Audio caption projection
        self.audio_caption_projection = PixArtAlphaTextProjection(
            in_features=caption_channels,
            hidden_size=self.audio_inner_dim,
        )

        # Audio output components
        self.audio_scale_shift_table = torch.nn.Parameter(torch.empty(2, self.audio_inner_dim))
        self.audio_norm_out = torch.nn.LayerNorm(self.audio_inner_dim, elementwise_affine=False, eps=norm_eps)
        self.audio_proj_out = torch.nn.Linear(self.audio_inner_dim, out_channels)

        self.audio_args_preprocessor = TransformerArgsPreprocessor(
            patchify_proj=self.audio_patchify_proj,
            adaln=self.audio_adaln_single,
            caption_projection=self.audio_caption_projection,
            cross_scale_shift_adaln=self.av_ca_audio_scale_shift_adaln_single,
            cross_gate_adaln=self.av_ca_v2a_gate_adaln_single,
            inner_dim=self.audio_inner_dim,
            max_pos=self.audio_positional_embedding_max_pos,
            num_attention_heads=self.audio_num_attention_heads,
            cross_pe_max_pos=cross_pe_max_pos,
            use_middle_indices_grid=self.use_middle_indices_grid,
            audio_cross_attention_dim=self.audio_cross_attention_dim,
            timestep_scale_multiplier=self.timestep_scale_multiplier,
            double_precision_rope=self.double_precision_rope,
            positional_embedding_theta=self.positional_embedding_theta,
            rope_type=self.rope_type,
            av_ca_timestep_scale_multiplier=self.av_ca_timestep_scale_multiplier,
        )

    def _init_transformer_blocks(
        self,
        num_layers: int,
        attention_head_dim: int,
        cross_attention_dim: int,
        audio_attention_head_dim: int,
        norm_eps: float,
        attention_type: AttentionFunction | AttentionCallable,
    ) -> None:
        """Initialize transformer blocks for LTX."""
        video_config = TransformerConfig(
            dim=self.inner_dim,
            heads=self.num_attention_heads,
            d_head=attention_head_dim,
            context_dim=cross_attention_dim,
        )
        audio_config = TransformerConfig(
            dim=self.audio_inner_dim,
            heads=self.audio_num_attention_heads,
            d_head=audio_attention_head_dim,
            context_dim=self.audio_cross_attention_dim,
        )
        self.transformer_blocks = torch.nn.ModuleList(
            [
                BasicAVTransformerBlock(
                    idx=idx,
                    video=video_config,
                    audio=audio_config,
                    rope_type=self.rope_type,
                    norm_eps=norm_eps,
                    attention_function=attention_type,
                )
                for idx in range(num_layers)
            ]
        )

    def _process_transformer_blocks(
        self,
        video: TransformerArgs,
        audio: TransformerArgs,
        perturbations: BatchedPerturbationConfig,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Process transformer blocks for LTXAV."""

        # Process transformer blocks
        for block in self.transformer_blocks:
            vx, ax = block(
                video=video,
                audio=audio,
                perturbations=perturbations,
            )

        return vx, ax

    def _process_output(
        self,
        scale_shift_table: torch.Tensor,
        norm_out: torch.nn.LayerNorm,
        proj_out: torch.nn.Linear,
        x: torch.Tensor,
        embedded_timestep: torch.Tensor,
    ) -> torch.Tensor:
        """Process output for LTXV."""
        # Apply scale-shift modulation
        scale_shift_values = (
            scale_shift_table[None, None].to(device=x.device, dtype=x.dtype) + embedded_timestep[:, :, None]
        )
        shift, scale = scale_shift_values[:, :, 0], scale_shift_values[:, :, 1]

        x = norm_out(x)
        x = x * (1 + scale) + shift
        x = proj_out(x)
        return x

    def forward(
        self, video: Modality | None, audio: Modality | None, perturbations: BatchedPerturbationConfig
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass for LTX models.

        Returns:
            Processed output tensors
        """

        video_args = self.video_args_preprocessor.prepare(video)
        audio_args = self.audio_args_preprocessor.prepare(audio)

        if enable_vfly:
            assert video_args.context_mask is None, "only support when context mask is None"
            assert audio_args.context_mask is None, "only support when context mask is None"
            # only split the video inputs for cp, audio inputs are not split since its seq len is too small and is odd(121)
            video_args.x = dit_sp_split(video_args.x, dim=1)
            video_args.context = dit_sp_split(video_args.context, dim=1)
            video_args.timesteps = dit_sp_split(video_args.timesteps, dim=1)
            video_args.embedded_timestep = dit_sp_split(video_args.embedded_timestep, dim=1)
            _positional_embeddings = []
            for emb in video_args.positional_embeddings:
                _positional_embeddings.append(dit_sp_split(emb, dim=1))
            video_args.positional_embeddings = tuple(_positional_embeddings)
            _cross_positional_embeddings = []
            for emb in video_args.cross_positional_embeddings:
                _cross_positional_embeddings.append(dit_sp_split(emb, dim=1))
            video_args.cross_positional_embeddings = tuple(_cross_positional_embeddings)
            video_args.cross_scale_shift_timestep = dit_sp_split(video_args.cross_scale_shift_timestep, dim=1)
            video_args.cross_gate_timestep = dit_sp_split(video_args.cross_gate_timestep, dim=1)
            # only split the audio context for cp, its seq len is 1024
            audio_args.context = dit_sp_split(audio_args.context, dim=1)

        # print("video args")
        # print(f"video x: {video_args.x.shape}")
        # print(f"video context: {video_args.context.shape}")
        # print(f"video context mask: {video_args.context_mask.shape if video_args.context_mask is not None else None}")
        # print(f"video timesteps: {video_args.timesteps.shape}")
        # print(f"video embedded timestep: {video_args.embedded_timestep.shape}")
        # print(f"video positional embeddings: {len(video_args.positional_embeddings)}, {video_args.positional_embeddings[0].shape}, {video_args.positional_embeddings[1].shape}")
        # print(f"video cross positional embeddings: {len(video_args.cross_positional_embeddings)}, {video_args.cross_positional_embeddings[0].shape}, {video_args.cross_positional_embeddings[1].shape}")
        # print(f"video cross scale shift timestep: {video_args.cross_scale_shift_timestep.shape}")
        # print(f"video cross_gate_timestep: {video_args.cross_gate_timestep.shape}")

        # print("audio args")
        # print(f"audio x: {audio_args.x.shape}")
        # print(f"audio context: {audio_args.context.shape}")
        # print(f"audio context mask: {audio_args.context_mask.shape if audio_args.context_mask is not None else None}")
        # print(f"audio timesteps: {audio_args.timesteps.shape}")
        # print(f"audio embedded timestep: {audio_args.embedded_timestep.shape}")
        # print(f"audio positional embeddings: {len(audio_args.positional_embeddings)}, {audio_args.positional_embeddings[0].shape, audio_args.positional_embeddings[1].shape}")
        # print(f"audio cross positional embeddings: {len(audio_args.cross_positional_embeddings)}, {audio_args.cross_positional_embeddings[0].shape}, {audio_args.cross_positional_embeddings[1].shape}")
        # print(f"audio cross scale shift timestep: {audio_args.cross_scale_shift_timestep.shape}")
        """
        video args
            video x: torch.Size([1, 6144, 4096])
            video context: torch.Size([1, 1024, 4096])
            video context mask: None
            video timesteps: torch.Size([1, 6144, 24576])
            video embedded timestep: torch.Size([1, 6144, 4096])
            video positional embeddings: 2, torch.Size([1, 6144, 4096]), torch.Size([1, 6144, 4096])
            video cross positional embeddings: 2, torch.Size([1, 6144, 2048]), torch.Size([1, 6144, 2048])
            video cross scale shift timestep: torch.Size([1, 6144, 16384])
        audio args
            audio x: torch.Size([1, 121, 2048])
            audio context: torch.Size([1, 1024, 2048])
            audio context mask: None
            audio timesteps: torch.Size([1, 121, 12288])
            audio embedded timestep: torch.Size([1, 121, 2048])
            audio positional embeddings: 2, (torch.Size([1, 121, 2048]), torch.Size([1, 121, 2048]))
            audio cross positional embeddings: 2, torch.Size([1, 121, 2048]), torch.Size([1, 121, 2048])
            audio cross scale shift timestep: torch.Size([1, 121, 8192])
        """

        # Process transformer blocks
        vx, ax = self._process_transformer_blocks(
            video=video_args,
            audio=audio_args,
            perturbations=perturbations,
        )

        # Process output
        vx = self._process_output(
            self.scale_shift_table, self.norm_out, self.proj_out, vx, video_args.embedded_timestep
        )
        ax = self._process_output(
            self.audio_scale_shift_table, self.audio_norm_out, self.audio_proj_out, ax, audio_args.embedded_timestep
        )

        if enable_vfly:
            vx = dit_sp_gather(vx, dim=1)

        return vx, ax
