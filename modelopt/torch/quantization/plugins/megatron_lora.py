# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

"""Tensor-parallel adapters and distributed checkpointing for Megatron LoRA-QAT/QAD."""

import torch
from megatron.core.extensions.transformer_engine import (
    TEColumnParallelLinear,
    TELayerNormColumnParallelLinear,
    TERowParallelLinear,
)
from megatron.core.tensor_parallel.layers import ColumnParallelLinear, RowParallelLinear
from megatron.core.tensor_parallel.mappings import copy_to_tensor_model_parallel_region
from megatron.core.tensor_parallel.random import get_cuda_rng_tracker
from megatron.core.transformer.utils import make_sharded_tensors_for_checkpoint

from ..lora import QuantLoRARegistry, _QuantLoRALinear
from ..nn import QuantModuleRegistry
from .megatron import _QuantMegatronMLP

__all__ = []


@QuantLoRARegistry.register(
    {
        QuantModuleRegistry[ColumnParallelLinear]: "ColumnParallelLinear",
        QuantModuleRegistry[RowParallelLinear]: "RowParallelLinear",
    }
)
class _MegatronQuantLoRALinear(_QuantLoRALinear):
    """Adapt local weight shards with synchronized replicated-factor gradients."""

    def _setup(self, config):
        """Initialize sharded factors and synchronize replicated factors."""
        super()._setup(config)
        base_weight = self._parameters["weight"]
        self.lora_tp_group = self.parallel_state.tensor_parallel_group.group
        for name, parameter in (("lora_A", self.lora_A), ("lora_B", self.lora_B)):
            is_sharded = (name == "lora_A" and self._is_row_parallel) or (
                name == "lora_B" and self._is_column_parallel
            )
            parameter.tensor_model_parallel = is_sharded
            parameter.partition_dim = (1 if name == "lora_A" else 0) if is_sharded else -1
            parameter.partition_stride = getattr(base_weight, "partition_stride", 1)
            parameter.allreduce = getattr(base_weight, "allreduce", True)
            parameter.sequence_parallel = False
        if torch.distributed.get_world_size(self.lora_tp_group) > 1:
            if self._is_row_parallel and self.lora_A.is_cuda:
                with get_cuda_rng_tracker().fork():
                    torch.nn.init.kaiming_uniform_(self.lora_A, a=5**0.5)
            if not self._is_row_parallel:
                torch.distributed.broadcast(
                    self.lora_A.data,
                    src=torch.distributed.get_global_rank(self.lora_tp_group, 0),
                    group=self.lora_tp_group,
                )

    @staticmethod
    def _effective_weight(module, weight):
        """Combine local weight shards with TP-correct adapter gradients."""
        a, b = module.lora_A, module.lora_B
        if torch.distributed.get_world_size(module.lora_tp_group) > 1:
            if module._is_column_parallel:
                a = copy_to_tensor_model_parallel_region(a, group=module.lora_tp_group)
            elif module._is_row_parallel:
                b = copy_to_tensor_model_parallel_region(b, group=module.lora_tp_group)
        return weight + (module.lora_scale * b @ a).to(weight.dtype)

    def sharded_state_dict(self, prefix="", sharded_offsets=(), metadata=None):
        """Include adapter factors with their tensor-parallel checkpoint axes."""
        state = super().sharded_state_dict(prefix, sharded_offsets, metadata)
        axes = {}
        if self._is_column_parallel:
            axes["lora_B"] = 0
        elif self._is_row_parallel:
            axes["lora_A"] = 1
        state.update(
            make_sharded_tensors_for_checkpoint(
                {"lora_A": self.lora_A, "lora_B": self.lora_B},
                prefix,
                axes,
                sharded_offsets,
                tp_group=self.lora_tp_group,
                dp_cp_group=self.parallel_state.data_parallel_group.group,
            )
        )
        return state

    def merge_lora(self):
        """Merge local factors and remove adapter-specific parallel state."""
        super().merge_lora()
        del self.lora_tp_group


_QuantMegatronMLP._modelopt_state_keys = [
    *_QuantMegatronMLP._modelopt_state_keys,
    r".*linear_fc1\.lora_B$",
]


QuantLoRARegistry.register(
    {
        QuantModuleRegistry[TEColumnParallelLinear]: "TEColumnParallelLinear",
        QuantModuleRegistry[TELayerNormColumnParallelLinear]: "TELayerNormColumnParallelLinear",
        QuantModuleRegistry[TERowParallelLinear]: "TERowParallelLinear",
    }
)(_MegatronQuantLoRALinear)
