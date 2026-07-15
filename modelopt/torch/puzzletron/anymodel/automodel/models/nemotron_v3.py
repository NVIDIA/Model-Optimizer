# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import copy
import re
from contextlib import contextmanager
from typing import Any

from ...puzzformer.no_op import MatchingZeros, Same
from ..auto_model_descriptor import AutoModelDescriptor, AutoModelDescriptorFactory

__all__ = ["NemotronV3AutoModelDescriptor"]


@AutoModelDescriptorFactory.register_decorator("nemotron_h", "nemotron_v3")
class NemotronV3AutoModelDescriptor(AutoModelDescriptor):
    """Native AutoModel descriptor for Nemotron3/NemotronH custom blocks."""

    @staticmethod
    def decoder_layer_cls():
        from nemo_automodel.components.models.nemotron_v3.layers import NemotronV3Block

        return NemotronV3Block

    @staticmethod
    def block_config_to_config_overrides(block_config):
        overrides = AutoModelDescriptor.block_config_to_config_overrides(block_config)
        moe = block_config.get_subblock("moe")
        mamba = block_config.get_subblock("mamba")
        if moe is not None:
            overrides.pop("num_experts", None)
            if moe.num_experts is not None:
                overrides["n_routed_experts"] = moe.num_experts
            if moe.top_k is not None:
                overrides["num_experts_per_tok"] = moe.top_k
            if moe.expert_intermediate_size is not None:
                overrides["moe_intermediate_size"] = moe.expert_intermediate_size
            if moe.shared_expert_intermediate_size is not None:
                overrides["moe_shared_expert_intermediate_size"] = (
                    moe.shared_expert_intermediate_size
                )
            if moe.latent_dim is not None:
                overrides["moe_latent_size"] = moe.latent_dim
        if mamba is not None:
            if mamba.num_heads is not None:
                overrides["mamba_num_heads"] = mamba.num_heads
            if mamba.head_dim is not None:
                overrides["mamba_head_dim"] = mamba.head_dim
            if mamba.state_dim is not None:
                overrides["ssm_state_size"] = mamba.state_dim
        return overrides

    @staticmethod
    def _copy_moe_geometry(moe_config, moe, *, hidden_size=None):
        patched = copy.copy(moe_config)
        if hidden_size is not None:
            patched.dim = hidden_size
        if moe.num_experts is not None:
            patched.n_routed_experts = moe.num_experts
        if moe.top_k is not None:
            patched.n_activated_experts = min(
                moe.top_k,
                patched.n_routed_experts,
            )
        if moe.expert_intermediate_size is not None:
            patched.moe_inter_dim = moe.expert_intermediate_size
        if moe.shared_expert_intermediate_size is not None:
            patched.shared_expert_inter_dim = moe.shared_expert_intermediate_size
        if moe.latent_dim is not None:
            patched.moe_latent_size = moe.latent_dim
        return patched

    @staticmethod
    def patch_constructor_arguments(arguments, block_config, layer_idx) -> None:
        """Give each native MoE block its own pruned geometry.

        AutoModel's Nemotron model creates one global ``MoEConfig`` and passes
        that same object to every block.  The ordinary per-layer config copy is
        therefore insufficient for heterogeneous MoE widths: the native MoE
        constructor reads its geometry from this separate object.  Copy it
        before applying overrides so constructing a child layer cannot mutate
        the teacher geometry or another expert-parallel layer.
        """

        del layer_idx
        moe = block_config.get_subblock("moe")
        moe_config = arguments.get("moe_config")
        if moe is None or moe_config is None:
            return

        config = arguments["config"]
        arguments["moe_config"] = NemotronV3AutoModelDescriptor._copy_moe_geometry(
            moe_config,
            moe,
            hidden_size=config.hidden_size,
        )

    @classmethod
    @contextmanager
    def native_state_dict_adapter_context(cls, block_configs):
        """Make AutoModel's global Nemotron adapter honor per-layer MoE shapes."""

        from nemo_automodel.components.models.nemotron_v3.state_dict_adapter import (
            NemotronV3StateDictAdapter,
        )

        layer_pattern = re.compile(r"(?:^|\.)layers\.(\d+)\.mixer\.experts\.")
        original_split = (
            NemotronV3StateDictAdapter._convert_single_merged_expert_to_hf_split_experts
        )
        original_merge = NemotronV3StateDictAdapter._from_hf_w_merged_experts

        def layer_geometry(adapter, layer_idx):
            if not 0 <= layer_idx < len(block_configs):
                return None
            moe = block_configs[layer_idx].get_subblock("moe")
            if moe is None or moe.no_op:
                return None
            patched = cls._copy_moe_geometry(adapter.moe_config, moe)
            if (
                patched.n_routed_experts == adapter.moe_config.n_routed_experts
                and patched.moe_inter_dim == adapter.moe_config.moe_inter_dim
            ):
                return None
            return patched

        def patched_split(adapter, fqn, tensor, *args, **kwargs):
            match = layer_pattern.search(fqn)
            geometry = layer_geometry(adapter, int(match.group(1))) if match else None
            if geometry is None:
                return original_split(adapter, fqn, tensor, *args, **kwargs)
            original_geometry = adapter.moe_config
            adapter.moe_config = geometry
            try:
                return original_split(adapter, fqn, tensor, *args, **kwargs)
            finally:
                adapter.moe_config = original_geometry

        def patched_merge(
            adapter,
            hf_state_dict,
            device_mesh=None,
            reset_view_loaded_keys=True,
        ):
            per_layer = {}
            for key in list(hf_state_dict):
                match = layer_pattern.search(key)
                if match is None:
                    continue
                layer_idx = int(match.group(1))
                if layer_geometry(adapter, layer_idx) is None:
                    continue
                per_layer.setdefault(layer_idx, {})[key] = hf_state_dict.pop(key)

            merged = original_merge(
                adapter,
                hf_state_dict,
                device_mesh,
                reset_view_loaded_keys=reset_view_loaded_keys,
            )
            for layer_idx, layer_state in per_layer.items():
                original_geometry = adapter.moe_config
                adapter.moe_config = layer_geometry(adapter, layer_idx)
                try:
                    merged.update(
                        original_merge(
                            adapter,
                            layer_state,
                            device_mesh,
                            reset_view_loaded_keys=False,
                        )
                    )
                finally:
                    adapter.moe_config = original_geometry
            return merged

        NemotronV3StateDictAdapter._convert_single_merged_expert_to_hf_split_experts = (
            patched_split
        )
        NemotronV3StateDictAdapter._from_hf_w_merged_experts = patched_merge
        try:
            yield
        finally:
            NemotronV3StateDictAdapter._convert_single_merged_expert_to_hf_split_experts = (
                original_split
            )
            NemotronV3StateDictAdapter._from_hf_w_merged_experts = original_merge

    @staticmethod
    def _block_no_op_post_init(layer: Any) -> None:
        block_config = layer.config.block_configs[layer.layer_idx]
        attn = block_config.get_subblock("attention")
        ffn = block_config.get_subblock("ffn")
        mamba = block_config.get_subblock("mamba")
        moe = block_config.get_subblock("moe")
        if (
            (attn is None or attn.no_op)
            and (ffn is None or ffn.no_op)
            and (mamba is None or mamba.no_op)
            and (moe is None or moe.no_op)
        ):
            layer.norm = Same()
            layer.mixer = MatchingZeros()
            if hasattr(layer, "self_attn"):
                layer.self_attn = layer.mixer

    @staticmethod
    def attn_no_op_post_init(layer: Any) -> None:
        NemotronV3AutoModelDescriptor._block_no_op_post_init(layer)

    @staticmethod
    def mlp_no_op_post_init(layer: Any) -> None:
        NemotronV3AutoModelDescriptor._block_no_op_post_init(layer)
