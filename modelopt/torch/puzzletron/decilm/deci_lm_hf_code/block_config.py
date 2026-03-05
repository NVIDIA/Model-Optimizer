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
import dataclasses
import inspect
import warnings
from abc import abstractmethod
from dataclasses import dataclass
from typing import Any, List, Optional, Type, Union, get_args, get_origin


@dataclass(frozen=True, kw_only=True)
class BaseDataclass:
    """
    A dataclass base class with several utilities:
    1. Comparison via string representation.
    2. Initialization of dataclasses fields from dicts.
    3. Setting attributes even though it's frozen (but only inside __post_init__!)
    """

    def __eq__(self, other: "BaseDataclass") -> bool:
        return str(self) == str(other)

    def __hash__(self) -> int:
        return hash(str(self))

    def __lt__(self, other: "BaseDataclass") -> bool:
        return str(self) < str(other)

    def _force_setattr(self, name: str, value: Any) -> None:
        """
        Set an attribute even in frozen dataclasses.
        Use only inside __post_init__!
        """
        assert _is_called_from_post_init(), (
            "_force_setattr should only be called from __post_init__, "
            "if you need to change an attribute use dataclasses.replace "
            "or create a new instance :)"
        )
        object.__setattr__(self, name, value)

    def __post_init__(self):
        """
        Init dataclass fields from dicts
        """
        for field in dataclasses.fields(self):
            field_dict = getattr(self, field.name)
            if isinstance(field_dict, dict) and _is_dataclass_type(field.type):
                dataclass_cls = _get_dataclass_type(field.type)
                sub_fields = [field.name for field in dataclasses.fields(dataclass_cls)]
                unsupported_fields = [
                    field_name for field_name in field_dict.keys() if field_name not in sub_fields
                ]
                if len(unsupported_fields) > 0:
                    warnings.warn(
                        f"Removed unsupported fields {unsupported_fields} from {dataclass_cls}"
                    )

                field_dict = {k: v for k, v in field_dict.items() if k not in unsupported_fields}
                self._force_setattr(field.name, dataclass_cls(**field_dict))


def _is_called_from_post_init() -> bool:
    frame = inspect.currentframe()
    while frame:
        if frame.f_code.co_name == "__post_init__":
            return True
        frame = frame.f_back
    return False


def _is_dataclass_type(tp: Type) -> bool:
    """
    Like dataclasses.is_dataclass but also works for Optional[] and Union[] of a dataclass type
    """
    try:
        _get_dataclass_type(tp)
        return True
    except:
        return False


def _get_dataclass_type(tp: Type) -> dataclass:
    """
    If the given type is a dataclass, the function returns it.
    If it is a Union[] or Optional[], the function extracts the first dataclass type.
    If no dataclass type is found, the function raises a ValueError.
    """
    origin = get_origin(tp)
    if origin is Union:
        for type_in_union in get_args(tp):
            if dataclasses.is_dataclass(type_in_union):
                return type_in_union
    if dataclasses.is_dataclass(tp):
        return tp
    raise ValueError("Not a dataclass")


@dataclass(frozen=True, kw_only=True)
class SubblockConfig(BaseDataclass):
    no_op: bool = False
    replace_with_linear: bool = False
    sparsify: Optional[list[str]] = None
    weights_precision: Optional[str] = "bf16"

    def __post_init__(self):
        super().__post_init__()
        assert not (self.no_op and self.replace_with_linear)
        if self.no_op:
            self._force_setattr("sparsify", None)

    @abstractmethod
    def to_blockconfig(self) -> "BlockConfig":
        """ "
        Convert to a block including this subblock only.
        """
        ...


@dataclass(frozen=True, kw_only=True)
class MoEConfig(BaseDataclass):
    """
    Configuration class for Mixture of Experts parameters.
    """

    num_local_experts: int = 8
    num_experts_per_tok: int = 1
    expert_intermediate_dim: int = 8192
    shared_expert_intermediate_dim: int = 8192
    # router_aux_loss_coef: float = 0.01
    # router_z_loss_coef: float = 0.0  # Optional z-loss coefficient

    def __post_init__(self):
        # Validate the configuration
        if self.num_local_experts <= 0:
            raise ValueError(f"num_local_experts must be positive, got {self.num_local_experts}")
        if self.num_experts_per_tok <= 0:
            raise ValueError(f"top_k must be positive, got {self.top_k}")
        if self.num_experts_per_tok > self.num_local_experts:
            raise ValueError(
                f"top_k ({self.top_k}) cannot be greater than num_local_experts ({self.num_local_experts})"
            )
        # if self.router_aux_loss_coef < 0:
        #     raise ValueError(f"router_aux_loss_coef must be non-negative, got {self.router_aux_loss_coef}")


@dataclass(frozen=True, kw_only=True)
class MambaConfig(BaseDataclass):
    state_dim: int
    num_heads: int
    head_dim: int
    num_groups: int


@dataclass(frozen=True, kw_only=True)
class Llama4AttentionConfig(BaseDataclass):
    attention_chunk_size: Optional[int] = None
    use_rope: Optional[bool] = None
    use_qk_norm: Optional[bool] = None
    attn_scale: Optional[float] = None
    floor_scale: Optional[float] = None
    attn_temperature_tuning: Optional[bool] = None
    attention_dropout: Optional[float] = None


@dataclass(frozen=True, kw_only=True)
class AttentionConfig(SubblockConfig):
    num_key_value_heads: Optional[int] = None
    llama4: Optional[Llama4AttentionConfig] = None
    mamba: Optional[MambaConfig] = None

    def __post_init__(self):
        super().__post_init__()

        if self.no_op:
            assert not self.is_mamba
            assert not self.is_llama4

        if self.no_op or self.is_mamba:
            for irrelevant_att in [
                "num_key_value_heads",
            ]:
                self._force_setattr(irrelevant_att, None)
        else:
            assert self.num_key_value_heads is not None

    def to_blockconfig(self) -> "BlockConfig":
        return BlockConfig(attention=self, ffn=FFNConfig(no_op=True))

    @property
    def is_llama4(self) -> bool:
        return self.llama4 is not None

    @property
    def is_mamba(self) -> bool:
        return self.mamba is not None


@dataclass(frozen=True, kw_only=True)
class FFNConfig(SubblockConfig):
    moe: Optional[MoEConfig] = None
    intermediate_size: Optional[int] = None

    def __post_init__(self):
        super().__post_init__()
        if self.no_op:
            self._force_setattr("moe", None)
            self._force_setattr("intermediate_size", None)
        elif self.is_moe:
            self._force_setattr("intermediate_size", None)
        else:
            assert self.intermediate_size is not None, "Intermediate size must be provided for an FFN block"

    def to_blockconfig(self) -> "BlockConfig":
        return BlockConfig(attention=AttentionConfig(no_op=True), ffn=self)

    @property
    def is_moe(self) -> bool:
        return self.moe is not None


SUBBLOCK_CLS_DICT = {
    "attention": AttentionConfig,
    "ffn": FFNConfig,
}


@dataclass(frozen=True, kw_only=True)
class BlockConfig(BaseDataclass):
    attention: Optional[AttentionConfig] = None
    ffn: Optional[FFNConfig] = None
    parallel_blocks: Optional[list["BlockConfig"]] = None

    def __post_init__(self):
        super().__post_init__()
        if (self.parallel_blocks is not None) and isinstance(self.parallel_blocks[0], dict):
            initialized_block_configs = [
                BlockConfig(**block_config) for block_config in self.parallel_blocks
            ]
            self._force_setattr("parallel_blocks", initialized_block_configs)

    def to_dict(self) -> dict:
        """Convert BlockConfig to a dictionary."""
        return dataclasses.asdict(self)


def maybe_cast_block_configs(
    block_configs: List[BlockConfig | dict] | None,
) -> List[BlockConfig] | None:
    """Cast a list of dicts to BlockConfig objects if needed.

    Args:
        block_configs: List of BlockConfig or dict objects, or None.

    Returns:
        List of BlockConfig objects, or None if input is None/empty.
    """
    if not block_configs:
        return block_configs
    if isinstance(block_configs[0], dict):
        return [BlockConfig(**conf) for conf in block_configs]
    return block_configs
