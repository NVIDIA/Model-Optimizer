# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import copy
import functools
import inspect
from contextlib import nullcontext
from typing import Any, Callable

from ...block_config import AttentionConfig, BlockConfig, FFNConfig, MambaConfig, MoEConfig

__all__ = [
    "AutoModelDescriptor",
    "AutoModelDescriptorFactory",
    "ContractAutoModelDescriptor",
]


class AutoModelDescriptorFactory:
    _registry: dict[str, type["AutoModelDescriptor"]] = {}

    @classmethod
    def register(cls, **entries: type["AutoModelDescriptor"]) -> None:
        for name, descriptor in entries.items():
            existing = cls._registry.get(name)
            if existing is not None and existing is not descriptor:
                raise KeyError(f"AutoModel descriptor {name!r} already registered")
            cls._registry[name] = descriptor

    @classmethod
    def register_decorator(cls, *names: str) -> Callable:
        def decorator(descriptor: type["AutoModelDescriptor"]):
            cls.register(**{name: descriptor for name in names})
            return descriptor

        return decorator

    @classmethod
    def get(cls, name: str) -> type["AutoModelDescriptor"] | None:
        return cls._registry.get(name)

    @classmethod
    def registered_names(cls) -> tuple[str, ...]:
        return tuple(sorted(cls._registry))


class AutoModelDescriptor:
    """Descriptor for NeMo AutoModel native blocks.

    AutoModel bypass works by making teacher and student native AutoModel builds
    enter the same heterogeneity context. The patcher temporarily wraps each
    decoder block ``__init__`` so layer ``i`` receives overrides derived from
    ``block_configs[i]`` before parameters are allocated; after construction the
    descriptor can replace no-op submodules and patch state-dict adapters. Local
    bypass then captures descriptor-declared block/subblock I/O on the native
    model parts and trains only the candidate submodule, while global KD uses the
    same descriptor to keep teacher/student sharding and per-layer shapes aligned.
    """

    @staticmethod
    def decoder_layer_cls() -> type | tuple[type, ...]:
        raise NotImplementedError

    @staticmethod
    def block_config_to_config_overrides(block_config: BlockConfig) -> dict[str, Any]:
        overrides: dict[str, Any] = {}
        attn = block_config.get_subblock("attention")
        ffn = block_config.get_subblock("ffn")
        moe = block_config.get_subblock("moe")
        mamba = block_config.get_subblock("mamba")
        if isinstance(attn, AttentionConfig):
            if attn.num_kv_heads is not None:
                overrides["num_key_value_heads"] = attn.num_kv_heads
            if attn.num_query_heads is not None:
                overrides["num_attention_heads"] = attn.num_query_heads
        if isinstance(ffn, FFNConfig) and ffn.intermediate_size is not None:
            overrides["intermediate_size"] = ffn.intermediate_size
        if isinstance(moe, MoEConfig):
            if moe.num_experts is not None:
                overrides["num_experts"] = moe.num_experts
            if moe.expert_intermediate_size is not None:
                overrides["moe_intermediate_size"] = moe.expert_intermediate_size
            if moe.top_k is not None:
                overrides["num_experts_per_tok"] = moe.top_k
            if moe.shared_expert_intermediate_size is not None:
                overrides["moe_shared_expert_intermediate_size"] = moe.shared_expert_intermediate_size
            if moe.latent_dim is not None:
                overrides["moe_latent_size"] = moe.latent_dim
        if isinstance(mamba, MambaConfig):
            if mamba.num_heads is not None:
                overrides["mamba_num_heads"] = mamba.num_heads
            if mamba.head_dim is not None:
                overrides["mamba_head_dim"] = mamba.head_dim
            if mamba.state_dim is not None:
                overrides["ssm_state_size"] = mamba.state_dim
        return overrides

    @classmethod
    def patch_layer_config(
        cls,
        config: Any,
        block_config: BlockConfig,
        layer_idx: int,
    ) -> None:
        """Apply structural per-layer fields shared by native model families.

        Scalar projection sizes are handled by ``block_config_to_config_overrides``.
        Windowed attention also changes a layer's attention kind, so it belongs
        here and is applied to the private config copy passed to that layer.
        """

        attention = block_config.get_subblock("attention")
        window = (
            getattr(attention, "sliding_window_size", None)
            if attention is not None
            else None
        )
        if window is None:
            return
        desired_type = "full_attention" if window == "full" else "sliding_attention"
        layer_types = list(getattr(config, "layer_types", ()) or ())
        if layer_types:
            if len(layer_types) <= layer_idx:
                layer_types.extend(
                    ["full_attention"] * (layer_idx + 1 - len(layer_types))
                )
            layer_types[layer_idx] = desired_type
            config.layer_types = layer_types
        config.sliding_window = None if window == "full" else int(window)

    @classmethod
    def patch_constructor_arguments(
        cls,
        arguments: dict[str, Any],
        block_config: BlockConfig,
        layer_idx: int,
    ) -> None:
        """Patch non-config constructor inputs for a native backend.

        Most native blocks derive all geometry from ``config``. Backends that
        pass a separate structural object (for example a sharded MoE config)
        override this hook while retaining the common signature-aware patcher.
        """

    @classmethod
    def _apply_overrides(
        cls,
        config: Any,
        block_config: BlockConfig | None,
        *,
        layer_idx: int | None = None,
    ) -> Any:
        if block_config is None:
            return config
        new_config = copy.deepcopy(config)
        for key, value in cls.block_config_to_config_overrides(block_config).items():
            setattr(new_config, key, value)
        if layer_idx is not None:
            cls.patch_layer_config(new_config, block_config, layer_idx)
        return new_config

    @staticmethod
    def attn_no_op_post_init(layer: Any) -> None:
        raise NotImplementedError("This AutoModel descriptor does not implement attention no-op")

    @staticmethod
    def mlp_no_op_post_init(layer: Any) -> None:
        raise NotImplementedError("This AutoModel descriptor does not implement MLP no-op")

    @staticmethod
    def patch_state_dict_adapter(model: Any) -> bool:
        return False

    @classmethod
    def native_state_dict_adapter_context(cls, block_configs):
        """Temporarily adapt a native backend's checkpoint shape bridge."""

        return nullcontext()

    @staticmethod
    def patch_hf_model_checkpoint_mapping(model: Any) -> bool:
        return False

    @classmethod
    def make_patched_init(
        cls,
        orig_init: Callable,
        block_configs: list[BlockConfig] | tuple[BlockConfig, ...] | None,
    ) -> Callable:
        signature = inspect.signature(orig_init)

        @functools.wraps(orig_init)
        def _patched(self, *args, **kwargs):
            bound = signature.bind(self, *args, **kwargs)
            config = bound.arguments.get("config")
            if config is None:
                raise TypeError(
                    f"{type(self).__name__} native constructor has no 'config' argument"
                )
            layer_idx = bound.arguments.get("layer_idx")
            block_config = (
                block_configs[layer_idx]
                if block_configs
                and isinstance(layer_idx, int)
                and 0 <= layer_idx < len(block_configs)
                else None
            )
            config = cls._apply_overrides(
                config,
                block_config,
                layer_idx=layer_idx if isinstance(layer_idx, int) else None,
            )
            if block_configs:
                config.block_configs = list(block_configs)
            bound.arguments["config"] = config
            if block_config is not None and isinstance(layer_idx, int):
                cls.patch_constructor_arguments(
                    bound.arguments,
                    block_config,
                    layer_idx,
                )
            orig_init(*bound.args, **bound.kwargs)
            if block_config is not None:
                attn = block_config.get_subblock("attention")
                ffn = block_config.get_subblock("ffn")
                mamba = block_config.get_subblock("mamba")
                moe = block_config.get_subblock("moe")
                if isinstance(attn, AttentionConfig) and attn.no_op:
                    cls.attn_no_op_post_init(self)
                if isinstance(ffn, FFNConfig) and ffn.no_op:
                    cls.mlp_no_op_post_init(self)
                if isinstance(mamba, MambaConfig) and mamba.no_op:
                    cls.attn_no_op_post_init(self)
                if isinstance(moe, MoEConfig) and moe.no_op:
                    cls.mlp_no_op_post_init(self)

            # Native no-op hooks need typed configs during construction, but
            # AutoModel's consolidated checkpoint writer serializes the config
            # attached to each pipeline part.  Do not let BlockConfig objects
            # escape that construction window.
            if block_configs:
                serialized = [block.to_dict() for block in block_configs]
                config.block_configs = serialized
                layer_config = getattr(self, "config", None)
                if layer_config is not None:
                    layer_config.block_configs = serialized

        return _patched


class ContractAutoModelDescriptor(AutoModelDescriptor):
    """Native bridge that delegates shape semantics to an AnyModel contract.

    A family-specific subclass only declares the corresponding structural
    descriptor and native block class. Scoring, HF construction, native
    construction, materialization, and vLLM then share one field mapping.
    """

    STRUCTURAL_DESCRIPTOR: Any = None

    @classmethod
    def _structural_descriptor(cls):
        if cls.STRUCTURAL_DESCRIPTOR is None:
            raise RuntimeError(f"{cls.__name__} did not declare STRUCTURAL_DESCRIPTOR")
        return cls.STRUCTURAL_DESCRIPTOR

    @classmethod
    def block_config_to_config_overrides(
        cls, block_config: BlockConfig
    ) -> dict[str, Any]:
        return cls._structural_descriptor().block_config_to_layer_overrides(block_config)

    @classmethod
    def patch_layer_config(
        cls,
        config: Any,
        block_config: BlockConfig,
        layer_idx: int,
    ) -> None:
        cls._structural_descriptor().patch_layer_config(config, block_config, layer_idx)
