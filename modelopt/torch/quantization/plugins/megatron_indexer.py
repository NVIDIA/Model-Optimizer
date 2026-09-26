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

"""Fake quantization of the sparse-attention indexer K cache in Megatron-Core.

Covers ``DSAIndexer`` (DeepSeek-V3.2, GLM-5.x) and ``CSAIndexer`` (DeepSeek-V4).
``indexer_k_quantizer`` fake-quantizes the key the indexer scores against, in the basis serving
writes it into the indexer K cache. The name avoids the ``*[kv]_bmm_quantizer`` globs, so the
KV-cache presets leave it disabled. The ModelOpt extra-state callbacks come from
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
    from megatron.core.transformer.experimental_attention_variant.dsa import (
        DSAIndexer,
        rotate_activation,
    )
except ImportError:
    DSAIndexer = rotate_activation = None

try:
    from megatron.core.transformer.experimental_attention_variant.csa import CSAIndexer
except ImportError:
    CSAIndexer = None


class _QuantMegatronIndexer(QuantModule):
    """DSA / CSA lightning indexer with fake quantization of its key, the indexer K cache entry.

    ``indexer_k_quantizer`` is applied to the key returned by ``forward_before_topk``, which every
    scoring path consumes, in the layout vLLM caches it for DeepSeek-V3.2/V4 and GLM-5: after norm
    and RoPE, without Hadamard rotation, with interleaved RoPE pairs adjacent. The rotation hits q
    and k alike, so it leaves the index scores unchanged: ``DSAIndexer`` must run with
    ``dsa_indexer_rotate_activation`` off, and ``CSAIndexer``, which always rotates, has its key
    rotated back around the QDQ (``rotate_activation`` is orthonormal and symmetric, so it is its
    own inverse). ``DSAIndexer`` with ``dsa_indexer_rope_interleaved`` (GLM-5) returns the RoPE
    dims as ``[even | odd]``; block-scaled formats (e.g. NVFP4) see the order, so the QDQ runs on
    adjacent pairs. GLM-5.3-Flash, whose vLLM cache holds the rotated key, has no Megatron-Core
    indexer yet. The indexer projections are TP-duplicated, so the amax only needs the DP/CP sync
    of ``parallel_state``.
    """

    def _setup(self):
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

    def _quantize_key(self, k: torch.Tensor) -> torch.Tensor:
        compressor = getattr(self, "compressor", None)  # CSAIndexer only
        if compressor is None and self.config.dsa_indexer_rotate_activation:
            raise ValueError(
                "indexer_k_quantizer quantizes the DSA indexer key as vLLM caches it, without the "
                "Hadamard rotation. Set dsa_indexer_rotate_activation=False: the rotation applies "
                "to both q and k, so it does not change the index scores."
            )
        # CSAIndexer has no switch for its compressor's rotation: rotate the key back to the
        # unrotated basis before the QDQ and rotate it again after.
        rotated = compressor is not None and compressor.rotate
        # CSAIndexer restores the adjacent pair layout itself (``mla_output_remove_interleaving``).
        rope_split = compressor is None and getattr(
            self.config, "dsa_indexer_rope_interleaved", False
        )
        rope_dim = self.qk_pos_emb_head_dim
        if rotated:  # rotate back (rotate_activation is its own inverse)
            k = rotate_activation(k)
        if rope_split:  # [even | odd] -> adjacent pairs
            pe, nope = k.split([rope_dim, k.shape[-1] - rope_dim], dim=-1)
            k = torch.cat([torch.stack(pe.chunk(2, dim=-1), dim=-1).flatten(-2), nope], dim=-1)
        k = self.indexer_k_quantizer(k)
        if rope_split:  # adjacent pairs -> [even | odd]
            pe, nope = k.split([rope_dim, k.shape[-1] - rope_dim], dim=-1)
            k = torch.cat([pe[..., 0::2], pe[..., 1::2], nope], dim=-1)
        if rotated:  # rotate again, back to the basis the scores use
            k = rotate_activation(k)
        return k

    def forward_before_topk(self, *args, **kwargs):
        q, k, weights = super().forward_before_topk(*args, **kwargs)
        if self.indexer_k_quantizer.is_enabled:
            k = self._quantize_key(k)
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
        # is the CPU ``_extra_state`` byte tensor; the TP-duplicated key projection is the anchor.
        self.indexer_k_quantizer.to(self.linear_wq_b.weight.device)


_mcore_indexers = {
    cls: key
    for cls, key in ((DSAIndexer, "megatron_DSAIndexer"), (CSAIndexer, "megatron_CSAIndexer"))
    if cls is not None
}
if _mcore_indexers:
    QuantModuleRegistry.register(_mcore_indexers)(_QuantMegatronIndexer)
