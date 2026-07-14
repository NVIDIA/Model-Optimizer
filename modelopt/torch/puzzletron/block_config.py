# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import dataclasses
from dataclasses import dataclass
from typing import Any, Iterable

__all__ = [
    "BaseDataclass",
    "SubblockConfig",
    "SubblockRef",
    "PrunableAxis",
    "VariantAxis",
    "MoEConfig",
    "MambaConfig",
    "Llama4AttentionConfig",
    "AttentionConfig",
    "MLAConfig",
    "FFNConfig",
    "SUBBLOCK_CLS_DICT",
    "BlockConfig",
    "maybe_cast_block_configs",
    "iter_subblocks",
]


def _asdict(obj: Any) -> Any:
    if isinstance(obj, BaseDataclass):
        return obj.to_dict()
    if dataclasses.is_dataclass(obj):
        return dataclasses.asdict(obj)
    if isinstance(obj, dict):
        return {key: _asdict(value) for key, value in obj.items() if value is not None}
    if isinstance(obj, (list, tuple)):
        return [_asdict(value) for value in obj]
    return obj


def _drop_none(data: dict[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in data.items() if value is not None}


@dataclass(frozen=True, kw_only=True)
class BaseDataclass:
    def __eq__(self, other: object) -> bool:
        return str(self) == str(other)

    def __hash__(self) -> int:
        return hash(str(self))

    def __lt__(self, other: "BaseDataclass") -> bool:
        return str(self) < str(other)

    def to_dict(self) -> dict[str, Any]:
        return _drop_none(dataclasses.asdict(self))


@dataclass(frozen=True, kw_only=True)
class SubblockConfig(BaseDataclass):
    kind: str
    name: str
    no_op: bool = False


@dataclass(frozen=True, kw_only=True)
class SubblockRef:
    index: int
    kind: str
    name: str
    config: SubblockConfig


@dataclass(frozen=True, kw_only=True)
class PrunableAxis:
    axis_id: str
    layer_idx: int | None
    subblock: str
    field: str
    size: int | None = None
    group_size: int | None = None
    sort_kind: str = "permutation"
    tensor_bindings: tuple[str, ...] = ()
    parallel_axis: str | None = None


@dataclass(frozen=True, kw_only=True)
class VariantAxis:
    axis_id: str
    layer_idx: int | None
    subblock: str
    field: str
    values: tuple[Any, ...] = ()
    requires_rpc: bool = False
    requires_vllm: bool = False


@dataclass(frozen=True, kw_only=True)
class MoEConfig(SubblockConfig):
    kind: str = "moe"
    name: str = "moe"
    num_experts: int | None = None
    expert_intermediate_size: int | None = None
    shared_expert_intermediate_size: int | None = None
    top_k: int | None = None
    latent_dim: int | None = None

    def __post_init__(self) -> None:
        for field in (
            "num_experts",
            "expert_intermediate_size",
            "shared_expert_intermediate_size",
            "latent_dim",
        ):
            value = getattr(self, field)
            if value is not None and value <= 0:
                raise ValueError(f"{field} must be positive, got {value}")
        if self.top_k is not None and self.top_k <= 0:
            raise ValueError(f"top_k must be positive, got {self.top_k}")
        if self.num_experts is not None and self.top_k is not None and self.top_k > self.num_experts:
            raise ValueError(
                f"top_k ({self.top_k}) cannot be greater than num_experts ({self.num_experts})"
            )


@dataclass(frozen=True, kw_only=True)
class MambaConfig(SubblockConfig):
    kind: str = "mamba"
    name: str = "mamba"
    state_dim: int | None = None
    num_heads: int | None = None
    head_dim: int | None = None
    num_groups: int | None = None
    conv_kernel_size: int | None = 4

    def __post_init__(self) -> None:
        for field in ("state_dim", "num_heads", "head_dim", "num_groups", "conv_kernel_size"):
            value = getattr(self, field)
            if value is not None and value <= 0:
                raise ValueError(f"{field} must be positive, got {value}")
        if (
            self.num_heads is not None
            and self.num_groups is not None
            and self.num_heads % self.num_groups
        ):
            raise ValueError(
                f"num_heads ({self.num_heads}) must be divisible by num_groups ({self.num_groups})"
            )


@dataclass(frozen=True, kw_only=True)
class Llama4AttentionConfig(BaseDataclass):
    attention_chunk_size: int | None = None
    use_rope: bool | None = None
    use_qk_norm: bool | None = None
    attn_scale: float | None = None
    floor_scale: float | None = None
    attn_temperature_tuning: bool | None = None
    attention_dropout: float | None = None


@dataclass(frozen=True, kw_only=True)
class AttentionConfig(SubblockConfig):
    kind: str = "attention"
    name: str = "attention"
    num_kv_heads: int | None = None
    num_query_heads: int | None = None
    qk_head_dim: int | None = None
    v_head_dim: int | None = None
    # ``"full"`` is explicit: ``None`` means the axis is unspecified/inherited.
    sliding_window_size: int | str | None = None
    # Structural metadata for heterogeneous attention families. These fields
    # describe ownership/math and are not independent pruning axes.
    k_eq_v: bool | None = None
    kv_source_layer: int | None = None
    llama4: Llama4AttentionConfig | None = None

    def __post_init__(self) -> None:
        if isinstance(self.llama4, dict):
            object.__setattr__(self, "llama4", Llama4AttentionConfig(**self.llama4))
        if self.no_op:
            for attr in (
                "num_kv_heads",
                "num_query_heads",
                "qk_head_dim",
                "v_head_dim",
                "sliding_window_size",
                "k_eq_v",
                "kv_source_layer",
            ):
                object.__setattr__(self, attr, None)
        if self.num_kv_heads is not None and self.num_query_heads is not None:
            if self.num_query_heads <= 0:
                raise ValueError(f"num_query_heads must be positive, got {self.num_query_heads}")
            if self.num_query_heads % self.num_kv_heads != 0:
                raise ValueError(
                    f"num_query_heads ({self.num_query_heads}) must be a multiple of "
                    f"num_kv_heads ({self.num_kv_heads})"
                )
        if self.sliding_window_size is not None:
            if self.sliding_window_size != "full" and (
                not isinstance(self.sliding_window_size, int)
                or self.sliding_window_size <= 0
            ):
                raise ValueError(
                    "sliding_window_size must be a positive integer, 'full', or None"
                )
        if self.kv_source_layer is not None and self.kv_source_layer < 0:
            raise ValueError(
                f"kv_source_layer must be non-negative, got {self.kv_source_layer}"
            )

    def to_blockconfig(self) -> "BlockConfig":
        return BlockConfig(subblock_configs=(self,))


@dataclass(frozen=True, kw_only=True)
class MLAConfig(SubblockConfig):
    """Per-layer multi-head latent-attention geometry and compression ranks.

    MLA decodes one non-RoPE key/value pair for every query head while sharing
    the rotary key component. ``num_heads`` is consequently one coupled axis,
    not separate GQA query-head and KV-group axes.
    """

    kind: str = "mla"
    name: str = "mla"
    num_heads: int | None = None
    q_lora_rank: int | None = None
    kv_lora_rank: int | None = None

    def __post_init__(self) -> None:
        if self.no_op:
            object.__setattr__(self, "num_heads", None)
            object.__setattr__(self, "q_lora_rank", None)
            object.__setattr__(self, "kv_lora_rank", None)
            return
        for field in ("num_heads", "q_lora_rank", "kv_lora_rank"):
            value = getattr(self, field)
            if value is not None and value <= 0:
                raise ValueError(f"{field} must be positive, got {value}")


@dataclass(frozen=True, kw_only=True)
class FFNConfig(SubblockConfig):
    kind: str = "ffn"
    name: str = "ffn"
    intermediate_size: int | None = None

    def __post_init__(self) -> None:
        if self.no_op:
            object.__setattr__(self, "intermediate_size", None)

    def to_blockconfig(self) -> "BlockConfig":
        return BlockConfig(subblock_configs=(self,))


SUBBLOCK_CLS_DICT = {
    "attention": AttentionConfig,
    "mla": MLAConfig,
    "ffn": FFNConfig,
    "moe": MoEConfig,
    "mamba": MambaConfig,
}


def _coerce_subblock_config(value: SubblockConfig | dict[str, Any]) -> SubblockConfig:
    if isinstance(value, SubblockConfig):
        return value
    if not isinstance(value, dict):
        raise TypeError(f"Expected SubblockConfig or dict, got {type(value)!r}")
    kind = value.get("kind")
    if kind not in SUBBLOCK_CLS_DICT:
        raise ValueError(f"Unknown subblock kind {kind!r}; expected one of {sorted(SUBBLOCK_CLS_DICT)}")
    return SUBBLOCK_CLS_DICT[kind](**value)


@dataclass(frozen=True, kw_only=True)
class BlockConfig(BaseDataclass):
    subblock_configs: tuple[SubblockConfig, ...]

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "subblock_configs",
            tuple(_coerce_subblock_config(subblock) for subblock in self.subblock_configs),
        )

    def subblocks(self) -> list[SubblockRef]:
        return [
            SubblockRef(index=index, kind=config.kind, name=config.name, config=config)
            for index, config in enumerate(self.subblock_configs)
        ]

    def get_subblock(self, kind: str, name: str | None = None) -> SubblockConfig | None:
        for subblock in self.subblock_configs:
            if subblock.kind == kind and (name is None or subblock.name == name):
                return subblock
        return None

    def require_subblock(self, kind: str, name: str | None = None) -> SubblockConfig:
        subblock = self.get_subblock(kind, name)
        if subblock is None:
            suffix = f" named {name!r}" if name is not None else ""
            raise KeyError(f"BlockConfig has no {kind!r} subblock{suffix}")
        return subblock

    def without_subblocks(self, *kinds: str) -> "BlockConfig":
        return BlockConfig(
            subblock_configs=tuple(
                subblock for subblock in self.subblock_configs if subblock.kind not in kinds
            )
        )

    def with_subblock(self, subblock: SubblockConfig, *, replace_kinds: Iterable[str] = ()) -> "BlockConfig":
        replace = {subblock.kind, *replace_kinds}
        kept = tuple(existing for existing in self.subblock_configs if existing.kind not in replace)
        return BlockConfig(subblock_configs=(*kept, subblock))

    def prunable_axes(self) -> list[PrunableAxis]:
        axes: list[PrunableAxis] = []
        for ref in self.subblocks():
            cfg = ref.config
            if isinstance(cfg, AttentionConfig):
                fields = ("num_kv_heads", "num_query_heads", "qk_head_dim", "v_head_dim")
            elif isinstance(cfg, MLAConfig):
                fields = ("q_lora_rank", "kv_lora_rank")
            elif isinstance(cfg, FFNConfig):
                fields = ("intermediate_size",)
            elif isinstance(cfg, MoEConfig):
                fields = (
                    "num_experts",
                    "expert_intermediate_size",
                    "shared_expert_intermediate_size",
                    "latent_dim",
                )
            elif isinstance(cfg, MambaConfig):
                fields = ("num_heads", "head_dim", "state_dim")
            else:
                fields = ()
            for field in fields:
                if getattr(cfg, field, None) is not None:
                    axes.append(
                        PrunableAxis(
                            axis_id=f"{ref.name}.{field}",
                            layer_idx=None,
                            subblock=ref.name,
                            field=field,
                        )
                    )
        return axes

    def variant_axes(self) -> list[VariantAxis]:
        axes: list[VariantAxis] = []
        for ref in self.subblocks():
            cfg = ref.config
            if isinstance(cfg, AttentionConfig) and cfg.sliding_window_size is not None:
                axes.append(
                    VariantAxis(
                        axis_id=f"{ref.name}.sliding_window_size",
                        layer_idx=None,
                        subblock=ref.name,
                        field="sliding_window_size",
                    )
                )
            if isinstance(cfg, MoEConfig) and cfg.top_k is not None:
                axes.append(
                    VariantAxis(
                        axis_id=f"{ref.name}.top_k",
                        layer_idx=None,
                        subblock=ref.name,
                        field="top_k",
                    )
                )
        return axes

    def to_dict(self) -> dict[str, Any]:
        return {"subblock_configs": [_asdict(subblock) for subblock in self.subblock_configs]}


def maybe_cast_block_configs(
    block_configs: list[BlockConfig | dict[str, Any]] | None,
) -> list[BlockConfig] | None:
    if not block_configs:
        return block_configs
    return [BlockConfig(**conf) if isinstance(conf, dict) else conf for conf in block_configs]


def iter_subblocks(block_configs: Iterable[BlockConfig]) -> Iterable[SubblockRef]:
    for block_config in block_configs:
        yield from block_config.subblocks()
