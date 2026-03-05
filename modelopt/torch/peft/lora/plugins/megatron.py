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

"""Megatron-Core specific PEFT/LoRA plugins."""

import torch
import torch.nn as nn
from megatron.core.dist_checkpointing import ShardedTensor
from megatron.core.tensor_parallel.layers import ColumnParallelLinear, RowParallelLinear
from megatron.core.transformer.module import MegatronModule
from megatron.core.transformer.moe.experts import SequentialMLP

from modelopt.torch.quantization.nn import QuantModuleRegistry
from modelopt.torch.quantization.plugins.megatron import (
    _MegatronColumnParallelLinear as QuantColumnParallelLinear,
)
from modelopt.torch.quantization.plugins.megatron import (
    _MegatronRowParallelLinear as QuantRowParallelLinear,
)

from ...config import PEFTAttributeConfig
from ...custom import CUSTOM_MODEL_PLUGINS
from ..layer import LoRAModule, LoRAModuleRegistry

DEFAULT_LORA_RANK = 64
DEFAULT_SCALE = 1.0

__all__ = []


def megatron_replace_lora_module_hook(model: torch.nn.Module):
    """Configure Megatron-Core model PEFT/LoRA support.

    This callback is called before the LoRAModule replacement to configure
    distributed checkpointing support. For each MegatronModule:
    1. We enable heterogeneous distributed checkpointing

    Note: LoRAModule already has built-in get_extra_state and set_extra_state methods,
    so we don't need to register callbacks for them.
    """
    for name, module in model.named_modules():
        if isinstance(module, MegatronModule):
            # Enable heterogeneous distributed checkpointing
            if hasattr(module, "config") and hasattr(
                module.config, "hetereogenous_dist_checkpoint"
            ):
                module.config.hetereogenous_dist_checkpoint = True


# Register the hook
CUSTOM_MODEL_PLUGINS.add(megatron_replace_lora_module_hook)


class _MegatronParallelLoRABase(LoRAModule):
    """Base class for Megatron tensor parallel LoRA implementations.

    This class provides common functionality for both ColumnParallel and RowParallel
    LoRA implementations, reducing code duplication.
    """

    def _register_adapter_with_device(
        self,
        adapter_name: str,
        lora_a: nn.Module,
        lora_b: nn.Module,
        rank: int,
        scale: float,
        enable: bool,
    ) -> None:
        """Register LoRA adapter modules and ensure correct device placement.

        Args:
            adapter_name: Name of the adapter
            lora_a: LoRA A module (down-projection)
            lora_b: LoRA B module (up-projection)
            rank: Rank of the LoRA decomposition
        """
        # Move LoRA modules to the same device and dtype as the parent module
        # Try to get device and dtype from parent module's parameters or buffers
        device = None
        dtype = None
        for p in self.parameters():
            device = p.device
            dtype = p.dtype
            break
        if device is None:
            for b in self.buffers():
                device = b.device
                dtype = b.dtype
                break

        # If we found a device and dtype, move LoRA modules to match
        if device is not None:
            lora_a = lora_a.to(device)
            lora_b = lora_b.to(device)
        if dtype is not None:
            lora_a = lora_a.to(dtype)
            lora_b = lora_b.to(dtype)

        super()._register_adapter(adapter_name, lora_a, lora_b, rank, scale, enable)


@LoRAModuleRegistry.register({ColumnParallelLinear: "megatron_ColumnParallelLinear"})
class _LoRAMegatronColumnParallelLinear(_MegatronParallelLoRABase):
    """LoRA implementation for Megatron ColumnParallelLinear layers.

    This implementation creates column-parallel LoRA adapters that match
    the parallelization scheme of the base layer.
    """

    def update_layer_lora(
        self,
        adapter_name: str,
        attr_config: PEFTAttributeConfig,
    ) -> None:
        """Create and register a new LoRA adapter for ColumnParallelLinear.

        Args:
            adapter_name: Name for the new adapter
            rank: Rank of the LoRA decomposition
        """
        lora_a = nn.Linear(
            in_features=self.input_size,
            out_features=attr_config.rank,
            bias=False,
        )
        with torch.no_grad():
            attr_config.lora_a_init(lora_a.weight)

        lora_b = ColumnParallelLinear(
            attr_config.rank,
            self.output_size,
            config=self.config,
            bias=False,
            gather_output=False,
            init_method=attr_config.lora_b_init,
        )

        self._register_adapter_with_device(
            adapter_name, lora_a, lora_b, attr_config.rank, attr_config.scale, attr_config.enable
        )

    def sharded_state_dict(self, prefix="", sharded_offsets=(), metadata=None):
        """Sharding along axis 0 for ColumnParallelLinear, bias not sharded.

        For ColumnParallelLinear:
        (lora_a is a regular nn.Linear and is not sharded)
        - lora_b weight: sharded at dim 0
        """
        sharded_state_dict = super().sharded_state_dict(prefix, sharded_offsets, metadata)

        if hasattr(self, "_lora_adapters"):
            for adapter_name in self._lora_adapters:
                lora_b_name = f"lora_b_{adapter_name}"
                lora_b = self._lora_adapters[adapter_name]["lora_b"]

                assert isinstance(lora_b, ColumnParallelLinear)
                lora_b_sharded = lora_b.sharded_state_dict(
                    prefix=f"{prefix}{lora_b_name}.",
                    sharded_offsets=sharded_offsets,
                    metadata=metadata,
                )
                sharded_state_dict.update(lora_b_sharded)

        return sharded_state_dict


@LoRAModuleRegistry.register({RowParallelLinear: "megatron_RowParallelLinear"})
class _LoRAMegatronRowParallelLinear(_MegatronParallelLoRABase):
    """LoRA implementation for Megatron RowParallelLinear layers.

    This implementation creates row-parallel LoRA adapters that match
    the parallelization scheme of the base layer.
    """

    def update_layer_lora(
        self,
        adapter_name: str,
        attr_config: PEFTAttributeConfig,
    ) -> None:
        """Create and register a new LoRA adapter for RowParallelLinear.

        Args:
            adapter_name: Name for the new adapter
            rank: Rank of the LoRA decomposition
        """
        lora_a = RowParallelLinear(
            self.input_size,
            attr_config.rank,
            config=self.config,
            input_is_parallel=True,
            skip_bias_add=True,
            bias=False,
            init_method=attr_config.lora_a_init,
        )

        lora_b = nn.Linear(
            in_features=attr_config.rank,
            out_features=self.output_size,
            bias=False,
        )
        with torch.no_grad():
            attr_config.lora_b_init(lora_b.weight)

        self._register_adapter_with_device(
            adapter_name, lora_a, lora_b, attr_config.rank, attr_config.scale, attr_config.enable
        )

    def sharded_state_dict(self, prefix="", sharded_offsets=(), metadata=None):
        """Sharding along axis 1 for RowParallelLinear, bias not sharded.

        For RowParallelLinear:
        - lora_a weight: sharded at dim 1 (RowParallelLinear)
        (lora_b is a regular nn.Linear and is not sharded)
        """
        sharded_state_dict = super().sharded_state_dict(prefix, sharded_offsets, metadata)

        if hasattr(self, "_lora_adapters"):
            for adapter_name in self._lora_adapters:
                lora_a_name = f"lora_a_{adapter_name}"
                lora_a = self._lora_adapters[adapter_name]["lora_a"]

                assert isinstance(lora_a, RowParallelLinear)
                lora_a_sharded = lora_a.sharded_state_dict(
                    prefix=f"{prefix}{lora_a_name}.",
                    sharded_offsets=sharded_offsets,
                    metadata=metadata,
                )
                sharded_state_dict.update(lora_a_sharded)

        return sharded_state_dict


@LoRAModuleRegistry.register({SequentialMLP: "megatron_SequentialMLP"})
class _LoRAMegatronSequentialMLP(_MegatronParallelLoRABase):
    """LoRA for Megatron SequentialMLP with one shared lora_down and per-expert lora_up.

    Adapter layout:
    - ``lora_a_{adapter_name}``: a single ``nn.Linear(hidden_size, rank)`` shared across all
      local experts (the "down-projection").
    - ``lora_b_{adapter_name}``: an ``nn.ModuleList`` of ``num_local_experts`` individual
      ``nn.Linear(rank, hidden_size)`` modules (the "up-projections").

    Forward contribution:
      shared_down = lora_a(x_all)            # [total_tokens, rank]
      per_expert  = lora_up_i(chunk_i)       # [tokens_i, hidden_size]  for each expert i
      output      += scale * cat(per_expert)
    """

    def update_layer_lora(
        self,
        adapter_name: str,
        attr_config: PEFTAttributeConfig,
    ) -> None:
        """Create shared lora_down and per-expert lora_up modules and register them."""
        hidden_size = self.config.hidden_size
        rank = attr_config.rank

        lora_down = nn.Linear(hidden_size, rank, bias=False)
        lora_up = nn.ModuleList(
            [nn.Linear(rank, hidden_size, bias=False) for _ in range(self.num_local_experts)]
        )

        with torch.no_grad():
            attr_config.lora_a_init(lora_down.weight)
            for up in lora_up:
                attr_config.lora_b_init(up.weight)

        # Reuse device/dtype placement from _MegatronParallelLoRABase
        self._register_adapter_with_device(
            adapter_name, lora_down, lora_up, rank, attr_config.scale, attr_config.enable
        )

    def forward(
        self,
        permuted_local_hidden_states: torch.Tensor,
        tokens_per_expert: torch.Tensor,
        permuted_probs: torch.Tensor,
    ):
        """SequentialMLP forward with shared-down / per-expert-up LoRA residual."""
        # Run the base SequentialMLP forward directly (bypass LoRAModule.forward since
        # lora_b is a ModuleList, not a single callable module).
        output, output_bias = SequentialMLP.forward(
            self, permuted_local_hidden_states, tokens_per_expert, permuted_probs
        )

        for adapter in self._lora_adapters.values():
            if not adapter["enable"]:
                continue

            lora_down = adapter["lora_a"]  # shared nn.Linear
            lora_up_list = adapter["lora_b"]  # nn.ModuleList

            # Match the input scaling applied inside SequentialMLP when
            # moe_apply_probs_on_input is set, so LoRA sees the same effective input.
            lora_input = permuted_local_hidden_states
            if self.config.moe_apply_probs_on_input:
                lora_input = (permuted_probs.unsqueeze(-1) * lora_input).to(lora_input.dtype)

            # Single shared down-projection over ALL tokens
            down_out = lora_down(lora_input)  # [total_tokens, rank]

            # Split by expert and apply individual up-projections
            counts = (
                tokens_per_expert.tolist()
                if hasattr(tokens_per_expert, "tolist")
                else list(tokens_per_expert)
            )
            chunks = torch.split(down_out, counts)
            lora_out = torch.cat([up(chunk) for up, chunk in zip(lora_up_list, chunks)], dim=0)

            output = output + adapter["scale"] * lora_out

        return output, output_bias

    def sharded_state_dict(self, prefix="", sharded_offsets=(), metadata=None):
        """Extend SequentialMLP's sharded state dict with LoRA weights.

        - ``lora_a_{adapter_name}`` (lora_down): replicated across all EP ranks; saved
          once per (TP, EP=0, DP=0) rank combination.
        - ``lora_b_{adapter_name}.{i}`` (lora_up[i]): EP-sharded in the same way as
          ``local_experts.{i}``, i.e., mapped to global expert index.
        """
        from megatron.core.transformer import ensure_metadata_has_dp_cp_group

        # Base SequentialMLP state dict (handles local_experts weights).
        sharded_state_dict = SequentialMLP.sharded_state_dict(
            self, prefix, sharded_offsets, metadata
        )

        if not hasattr(self, "_lora_adapters") or not self._lora_adapters:
            return sharded_state_dict

        metadata = ensure_metadata_has_dp_cp_group(metadata)
        singleton_local_shards = (metadata or {}).get("singleton_local_shards", False)

        num_global_experts = self.ep_group.size() * self.num_local_experts
        local_expert_indices_offset = self.ep_group.rank() * self.num_local_experts
        # combined replica index that accounts for EP replication of lora_down
        combined_ep_dp_replica = self.ep_group.rank() * self.dp_group.size() + self.dp_group.rank()

        for adapter_name, adapter in self._lora_adapters.items():
            lora_down: nn.Linear = adapter["lora_a"]
            lora_up_list: nn.ModuleList = adapter["lora_b"]

            # ----- lora_down (shared / replicated across EP) -----
            lora_down_key = f"{prefix}lora_a_{adapter_name}.weight"
            sharded_state_dict[lora_down_key] = ShardedTensor.from_rank_offsets(
                lora_down_key,
                lora_down.weight,
                *sharded_offsets,
                replica_id=(0, self.tp_group.rank(), combined_ep_dp_replica),
                prepend_axis_num=len(sharded_offsets),
            )

            # ----- lora_up[i] (per-expert, EP-sharded) -----
            for expert_local_idx in range(self.num_local_experts):
                expert_global_idx = local_expert_indices_offset + expert_local_idx
                up: nn.Linear = lora_up_list[expert_local_idx]

                if singleton_local_shards:
                    lora_up_key = f"{prefix}lora_b_{adapter_name}.{expert_global_idx}.weight"
                    up_offsets = sharded_offsets
                else:
                    lora_up_key = f"{prefix}lora_b_{adapter_name}.weight"
                    up_offsets = (
                        *sharded_offsets,
                        (len(sharded_offsets), expert_global_idx, num_global_experts),
                    )

                sharded_state_dict[lora_up_key] = ShardedTensor.from_rank_offsets(
                    lora_up_key,
                    up.weight,
                    *up_offsets,
                    replica_id=(0, self.tp_group.rank(), self.dp_group.rank()),
                    prepend_axis_num=len(up_offsets),
                )

        return sharded_state_dict


# Register quantized versions if available
LoRAModuleRegistry.register({QuantColumnParallelLinear: "quant_megatron_ColumnParallelLinear"})(
    _LoRAMegatronColumnParallelLinear
)
LoRAModuleRegistry.register({QuantRowParallelLinear: "quant_megatron_RowParallelLinear"})(
    _LoRAMegatronRowParallelLinear
)


class _QuantLoRAMegatronColumnParallelLinear(
    _LoRAMegatronColumnParallelLinear, QuantColumnParallelLinear
):
    """Quantized LoRA ColumnParallelLinear that combines LoRA and quantization.

    This class ensures that the base layer functionality is quantized while
    preserving LoRA adapter functionality.
    """

    def _setup(self):
        QuantColumnParallelLinear._setup(self)


class _QuantLoRAMegatronRowParallelLinear(_LoRAMegatronRowParallelLinear, QuantRowParallelLinear):
    """Quantized LoRA RowParallelLinear that combines LoRA and quantization.

    This class ensures that the base layer functionality is quantized while
    preserving LoRA adapter functionality.
    """

    def _setup(self):
        QuantRowParallelLinear._setup(self)


QuantModuleRegistry.register(
    {_LoRAMegatronColumnParallelLinear: "lora_megatron_ColumnParallelLinear"}
)(_QuantLoRAMegatronColumnParallelLinear)
QuantModuleRegistry.register({_LoRAMegatronRowParallelLinear: "lora_megatron_RowParallelLinear"})(
    _QuantLoRAMegatronRowParallelLinear
)
