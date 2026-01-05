# Copyright (c) 2025 Lightricks. All rights reserved.
# Created by Andrew Kvochko

from dataclasses import dataclass

import torch

from ltx_core.guidance.perturbations import BatchedPerturbationConfig, PerturbationType
from ltx_core.model.transformer.attention import Attention, AttentionCallable, AttentionFunction
from ltx_core.model.transformer.feed_forward import FeedForward
from ltx_core.model.transformer.rope import LTXRopeType
from ltx_core.model.transformer.transformer_args import TransformerArgs
from ltx_core.utils import rms_norm

import os
from contextlib import contextmanager
from typing import Optional
enable_vfly = os.environ.get("ENABLE_VFLY", "false").lower() == "true"
if enable_vfly:
    from vfly.utils.parallel import dit_sp_gather
    from vfly.configs.parallel import DiTParallelConfig

    @contextmanager
    def dit_parallel_config_context(
        tp_size: Optional[int] = None,
        ulysses_size: Optional[int] = None,
        ring_size: Optional[int] = None,
        cp_size: Optional[int] = None,
        dp_size: Optional[int] = None,
        cfg_size: Optional[int] = None,
        fsdp_size: Optional[int] = None,
    ):
        """
        Context manager for temporarily modifying DiTParallelConfig settings.
        
        Args:
            tp_size: Tensor parallel degree
            ulysses_size: Ulysses parallel degree  
            ring_size: Ring attention parallel degree
            cp_size: Context parallel degree
            dp_size: Data parallel degree
            cfg_size: Classifier-free guidance parallel degree
            fsdp_size: Fully sharded data parallel degree
            
        Usage:
            with dit_parallel_config_context(tp_size=2, cp_size=4):
                # DiTParallelConfig is temporarily modified
                model_forward()
            # DiTParallelConfig is restored to original values
        """
        # Get the current DiTParallelConfig instance
        config = DiTParallelConfig.get_instance()
        
        # Store original values
        original_values = {
            'tp_size': config._tp_size,
            'ulysses_size': config._ulysses_size,
            'ring_size': config._ring_size,
            'cp_size': config._cp_size,
            'dp_size': config._dp_size,
            'cfg_size': config._cfg_size,
            'fsdp_size': config._fsdp_size,
        }
        
        try:
            # Apply new configuration values (only if provided)
            new_config = {}
            if tp_size is not None:
                new_config['tp_size'] = tp_size
            if ulysses_size is not None:
                new_config['ulysses_size'] = ulysses_size
            if ring_size is not None:
                new_config['ring_size'] = ring_size
            if cp_size is not None:
                new_config['cp_size'] = cp_size
            if dp_size is not None:
                new_config['dp_size'] = dp_size
            if cfg_size is not None:
                new_config['cfg_size'] = cfg_size
            if fsdp_size is not None:
                new_config['fsdp_size'] = fsdp_size
                
            # Apply the new configuration if any values were provided
            if new_config:
                # Merge with original values to ensure all parameters are set
                full_config = original_values.copy()
                full_config.update(new_config)
                config.set_config(**full_config)
            
            yield config
            
        finally:
            # Restore original configuration
            config.set_config(**original_values)    

@dataclass
class TransformerConfig:
    dim: int
    heads: int
    d_head: int
    context_dim: int


class BasicAVTransformerBlock(torch.nn.Module):
    def __init__(
        self,
        idx: int,
        video: TransformerConfig,
        audio: TransformerConfig,
        rope_type: LTXRopeType = LTXRopeType.INTERLEAVED,
        norm_eps: float = 1e-6,
        attention_function: AttentionFunction | AttentionCallable = AttentionFunction.DEFAULT,
    ):
        super().__init__()

        self.idx = idx

        self.attn1 = Attention(
            query_dim=video.dim,
            heads=video.heads,
            dim_head=video.d_head,
            context_dim=None,
            rope_type=rope_type,
            norm_eps=norm_eps,
            attention_function=attention_function,
        )
        self.audio_attn1 = Attention(
            query_dim=audio.dim,
            heads=audio.heads,
            dim_head=audio.d_head,
            context_dim=None,
            rope_type=rope_type,
            norm_eps=norm_eps,
            attention_function=attention_function,
        )

        self.attn2 = Attention(
            query_dim=video.dim,
            context_dim=video.context_dim,
            heads=video.heads,
            dim_head=video.d_head,
            rope_type=rope_type,
            norm_eps=norm_eps,
            attention_function=attention_function,
        )
        self.audio_attn2 = Attention(
            query_dim=audio.dim,
            context_dim=audio.context_dim,
            heads=audio.heads,
            dim_head=audio.d_head,
            rope_type=rope_type,
            norm_eps=norm_eps,
            attention_function=attention_function,
        )

        # Q: Video, K,V: Audio
        self.audio_to_video_attn = Attention(
            query_dim=video.dim,
            context_dim=audio.dim,
            heads=audio.heads,
            dim_head=audio.d_head,
            rope_type=rope_type,
            norm_eps=norm_eps,
            attention_function=attention_function,
        )

        # Q: Audio, K,V: Video
        self.video_to_audio_attn = Attention(
            query_dim=audio.dim,
            context_dim=video.dim,
            heads=audio.heads,
            dim_head=audio.d_head,
            rope_type=rope_type,
            norm_eps=norm_eps,
            attention_function=attention_function,
        )

        self.ff = FeedForward(video.dim, dim_out=video.dim)
        self.audio_ff = FeedForward(audio.dim, dim_out=audio.dim)

        self.scale_shift_table = torch.nn.Parameter(torch.empty(6, video.dim))
        self.audio_scale_shift_table = torch.nn.Parameter(torch.empty(6, audio.dim))

        self.scale_shift_table_a2v_ca_audio = torch.nn.Parameter(torch.empty(5, audio.dim))
        self.scale_shift_table_a2v_ca_video = torch.nn.Parameter(torch.empty(5, video.dim))

        self.norm_eps = norm_eps

    def get_ada_values(
        self,
        scale_shift_table: torch.Tensor,
        batch_size: int,
        timestep: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        num_ada_params = scale_shift_table.shape[0]

        ada_values = (
            scale_shift_table.unsqueeze(0).unsqueeze(0).to(timestep.dtype)
            + timestep.reshape(batch_size, timestep.shape[1], num_ada_params, -1)
        ).unbind(dim=2)
        return ada_values

    def get_av_ca_ada_values(
        self,
        scale_shift_table: torch.Tensor,
        batch_size: int,
        scale_shift_timestep: torch.Tensor,
        gate_timestep: torch.Tensor,
        num_scale_shift_values: int = 4,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        scale_shift_ada_values = self.get_ada_values(
            scale_shift_table[:num_scale_shift_values, :],
            batch_size,
            scale_shift_timestep,
        )
        gate_ada_values = self.get_ada_values(
            scale_shift_table[num_scale_shift_values:, :],
            batch_size,
            gate_timestep,
        )

        scale_shift_chunks = [t.squeeze(2) for t in scale_shift_ada_values]
        gate_ada_values = [t.squeeze(2) for t in gate_ada_values]

        return (*scale_shift_chunks, *gate_ada_values)

    def forward(
        self,
        video: TransformerArgs,
        audio: TransformerArgs,
        perturbations: BatchedPerturbationConfig | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        batch_size = video.x.shape[0]
        if perturbations is None:
            perturbations = BatchedPerturbationConfig.empty(batch_size)

        vx = video.x
        ax = audio.x

        run_vx = video.enabled and vx.numel() > 0
        run_ax = audio.enabled and ax.numel() > 0

        run_a2v = run_vx and ax.numel() > 0
        run_v2a = run_ax and vx.numel() > 0

        if run_vx:
            vshift_msa, vscale_msa, vgate_msa, vshift_mlp, vscale_mlp, vgate_mlp = self.get_ada_values(
                self.scale_shift_table, vx.shape[0], video.timesteps
            )
            if not perturbations.all_in_batch(PerturbationType.SKIP_VIDEO_SELF_ATTN, self.idx):
                norm_vx = rms_norm(vx, eps=self.norm_eps) * (1 + vscale_msa) + vshift_msa
                v_mask = perturbations.mask_like(PerturbationType.SKIP_VIDEO_SELF_ATTN, self.idx, vx)
                vx += self.attn1(norm_vx, pe=video.positional_embeddings) * vgate_msa * v_mask

            vx += self.attn2(rms_norm(vx, eps=self.norm_eps), context=video.context, mask=video.context_mask)

        if run_ax:
            ashift_msa, ascale_msa, agate_msa, ashift_mlp, ascale_mlp, agate_mlp = self.get_ada_values(
                self.audio_scale_shift_table, ax.shape[0], audio.timesteps
            )

            if not perturbations.all_in_batch(PerturbationType.SKIP_AUDIO_SELF_ATTN, self.idx):
                norm_ax = rms_norm(ax, eps=self.norm_eps) * (1 + ascale_msa) + ashift_msa
                a_mask = perturbations.mask_like(PerturbationType.SKIP_AUDIO_SELF_ATTN, self.idx, ax)
                if enable_vfly:
                    # qkv are not splited, not need for comm
                    with dit_parallel_config_context(ulysses_size=1, ring_size=1, cp_size=1):
                        ax += self.audio_attn1(norm_ax, pe=audio.positional_embeddings) * agate_msa * a_mask
                else:
                    ax += self.audio_attn1(norm_ax, pe=audio.positional_embeddings) * agate_msa * a_mask

            # only kv is splited, using ring attn
            if enable_vfly:
                total_cp_size = DiTParallelConfig.get_instance().cp_size() * DiTParallelConfig.get_instance().ring_size() * DiTParallelConfig.get_instance().ulysses_size()
                with dit_parallel_config_context(ulysses_size=1, ring_size=total_cp_size, cp_size=1):
                    ax += self.audio_attn2(rms_norm(ax, eps=self.norm_eps), context=audio.context, mask=audio.context_mask)
            else:
                ax += self.audio_attn2(rms_norm(ax, eps=self.norm_eps), context=audio.context, mask=audio.context_mask)

        # Audio - Video cross attention.
        if run_a2v or run_v2a:
            vx_norm3 = rms_norm(vx, eps=self.norm_eps)
            ax_norm3 = rms_norm(ax, eps=self.norm_eps)

            (
                scale_ca_audio_hidden_states_a2v,
                shift_ca_audio_hidden_states_a2v,
                scale_ca_audio_hidden_states_v2a,
                shift_ca_audio_hidden_states_v2a,
                gate_out_v2a,
            ) = self.get_av_ca_ada_values(
                self.scale_shift_table_a2v_ca_audio,
                ax.shape[0],
                audio.cross_scale_shift_timestep,
                audio.cross_gate_timestep,
            )

            (
                scale_ca_video_hidden_states_a2v,
                shift_ca_video_hidden_states_a2v,
                scale_ca_video_hidden_states_v2a,
                shift_ca_video_hidden_states_v2a,
                gate_out_a2v,
            ) = self.get_av_ca_ada_values(
                self.scale_shift_table_a2v_ca_video,
                vx.shape[0],
                video.cross_scale_shift_timestep,
                video.cross_gate_timestep,
            )

            if run_a2v:
                vx_scaled = vx_norm3 * (1 + scale_ca_video_hidden_states_a2v) + shift_ca_video_hidden_states_a2v
                ax_scaled = ax_norm3 * (1 + scale_ca_audio_hidden_states_a2v) + shift_ca_audio_hidden_states_a2v
                a2v_mask = perturbations.mask_like(PerturbationType.SKIP_A2V_CROSS_ATTN, self.idx, vx)
                
                if enable_vfly:
                    # only q is splited, not need for comm
                    with dit_parallel_config_context(ulysses_size=1, ring_size=1, cp_size=1):
                        vx += (
                            self.audio_to_video_attn(
                                vx_scaled,
                                context=ax_scaled,
                                pe=video.cross_positional_embeddings,
                                k_pe=audio.cross_positional_embeddings,
                            )
                            * gate_out_a2v
                            * a2v_mask
                        )
                else:
                    vx += (
                        self.audio_to_video_attn(
                            vx_scaled,
                            context=ax_scaled,
                            pe=video.cross_positional_embeddings,
                            k_pe=audio.cross_positional_embeddings,
                        )
                        * gate_out_a2v
                        * a2v_mask
                    )

            if run_v2a:
                ax_scaled = ax_norm3 * (1 + scale_ca_audio_hidden_states_v2a) + shift_ca_audio_hidden_states_v2a
                vx_scaled = vx_norm3 * (1 + scale_ca_video_hidden_states_v2a) + shift_ca_video_hidden_states_v2a
                v2a_mask = perturbations.mask_like(PerturbationType.SKIP_V2A_CROSS_ATTN, self.idx, ax)
                
                if enable_vfly:
                    # only kv is splited, using ring attn
                    with dit_parallel_config_context(ulysses_size=1, ring_size=total_cp_size, cp_size=1):
                        ax += (
                            self.video_to_audio_attn(
                                ax_scaled,
                                context=vx_scaled,
                                pe=audio.cross_positional_embeddings,
                                k_pe=video.cross_positional_embeddings,
                            )
                            * gate_out_v2a
                            * v2a_mask
                        )
                else:
                    ax += (
                        self.video_to_audio_attn(
                            ax_scaled,
                            context=vx_scaled,
                            pe=audio.cross_positional_embeddings,
                            k_pe=video.cross_positional_embeddings,
                        )
                        * gate_out_v2a
                        * v2a_mask
                    )

        if run_vx:
            vx_scaled = rms_norm(vx, eps=self.norm_eps) * (1 + vscale_mlp) + vshift_mlp
            vx += self.ff(vx_scaled) * vgate_mlp

        if run_ax:
            ax_scaled = rms_norm(ax, eps=self.norm_eps) * (1 + ascale_mlp) + ashift_mlp
            ax += self.audio_ff(ax_scaled) * agate_mlp

        return vx, ax
