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

"""Fake quantization of the sparse-attention indexer query and K cache in Megatron-Core.

Covers ``CSAIndexer`` (DeepSeek-V4); GLM-5.3-Flash's k-pool indexer is not in Megatron-Core yet.
``indexer_k_quantizer`` fake-quantizes the key the indexer scores against, in the basis serving
writes it into the indexer K cache, and ``indexer_q_quantizer`` the query in the basis serving
quantizes it. The names avoid the ``*[kv]_bmm_quantizer`` globs, so the KV-cache presets leave them
disabled. The ModelOpt extra-state callbacks come from
``megatron_replace_quant_module_hook`` in the Megatron plugin, which covers every registered
QuantModule.
"""

import megatron.core.parallel_state as mcore_parallel
import torch
from megatron.core.parallel_state import get_data_parallel_group

from modelopt.torch.utils.distributed import ParallelState

from ..nn import QuantModule, QuantModuleRegistry, TensorQuantizer

__all__ = []

try:
    from megatron.core.transformer.experimental_attention_variant.csa import CSAIndexer
    from megatron.core.transformer.experimental_attention_variant.dsa import rotate_activation
except ImportError:  # megatron-core without Compressed Sparse Attention
    CSAIndexer = rotate_activation = None


class _QuantMegatronIndexer(QuantModule):
    """DeepSeek-V4 CSA indexer with fake quantization of its query and key (the K cache entry).

    ``indexer_q_quantizer`` and ``indexer_k_quantizer`` are applied to the query and key returned by
    ``forward_before_topk``, which every scoring path consumes, in the layout vLLM quantizes them:
    after norm and RoPE, without the Hadamard rotation that the indexer applies. The rotation hits q
    and k alike, so it leaves the index scores unchanged; both are rotated back around the QDQ
    (``rotate_activation`` is orthonormal and symmetric, so it is its own inverse). The indexer
    projections are TP-duplicated, so the amax only needs the DP/CP sync of ``parallel_state``.
    """

    def _setup(self):
        self.indexer_q_quantizer = TensorQuantizer()
        self.indexer_k_quantizer = TensorQuantizer()
        try:
            data_parallel_group = get_data_parallel_group(with_context_parallel=True)
        except AssertionError:
            data_parallel_group = get_data_parallel_group()
        self.parallel_state = ParallelState(
            data_parallel_group, mcore_parallel.get_tensor_model_parallel_group()
        )
        # Like MCore ColumnParallelLinear: a state dict saved before the indexer was quantized has
        # no ``_extra_state``; default it so a strict load does not report the key missing.
        self._register_load_state_dict_pre_hook(
            lambda state_dict, prefix, *args, **kwargs: state_dict.setdefault(
                f"{prefix}_extra_state"
            )
        )

    def forward(self, *args, **kwargs):
        # The registry matches subclasses that share ``forward``; overriding it keeps the converted
        # class from matching again, so an indexer reachable from two parents is not converted a
        # second time by ``replace_quant_module`` (inconsistent MRO).
        return super().forward(*args, **kwargs)

    @staticmethod
    def _quantize_unrotated(x: torch.Tensor, quantizer: TensorQuantizer, rotated: bool):
        # The rotation has no config switch: rotate back to the unrotated basis vLLM quantizes in
        # before the QDQ and rotate again after.
        if rotated:  # rotate back (rotate_activation is its own inverse)
            x = rotate_activation(x)
        x = quantizer(x)
        if rotated:  # rotate again, back to the basis the scores use
            x = rotate_activation(x)
        return x

    def forward_before_topk(self, *args, **kwargs):
        q, k, weights = super().forward_before_topk(*args, **kwargs)
        if self.indexer_q_quantizer.is_enabled:  # the query is always rotated
            q = self._quantize_unrotated(q, self.indexer_q_quantizer, rotated=True)
        if self.indexer_k_quantizer.is_enabled:
            k = self._quantize_unrotated(k, self.indexer_k_quantizer, self.compressor.rotate)
        return q, k, weights

    # torch emits and loads ``_extra_state`` only when the class overrides these two. ModelOpt binds
    # its extra-state callbacks per instance, which suffices for TE and MCore linears but not for
    # this plain MegatronModule; without the stubs the quantizer state is dropped on save.
    def get_extra_state(self):
        return None

    def set_extra_state(self, state):
        pass

    def modelopt_post_restore(self, prefix: str = ""):
        # The base implementation takes the first state_dict entry as device reference, which here
        # is the CPU ``_extra_state`` byte tensor; the TP-duplicated query projection is the anchor.
        self.indexer_q_quantizer.to(self.linear_wq_b.weight.device)
        self.indexer_k_quantizer.to(self.linear_wq_b.weight.device)


if CSAIndexer is not None:
    QuantModuleRegistry.register({CSAIndexer: "megatron_CSAIndexer"})(_QuantMegatronIndexer)
