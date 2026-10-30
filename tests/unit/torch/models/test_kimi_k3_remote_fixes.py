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

"""The fixes Kimi-K3's own remote modeling needs on current transformers and fla-core.

The checkpoint's modeling is not available here, so the tests patch a stand-in remote module -- the
classes and functions the fixes reach -- on tiny CPU tensors. Every process-wide patch is undone after
each test.
"""

import inspect
import sys
import types

import pytest
import torch
from _test_utils.torch.kimi_k3_remote import remote_module

transformers = pytest.importorskip("transformers")

from modelopt.torch.models.kimi_k3 import _remote_fixes

# ------------------------------------------------------------------- the linear-attention fixes


@pytest.mark.parametrize("by", ["keyword", "position"])
def test_kda_beta_is_squashed_in_torch_only_when_the_kernel_flag_asks_for_it(by):
    """fla-core ignores ``use_beta_sigmoid_in_kernel``, so both KDA entry points apply
    ``sigmoid(beta)`` themselves (in beta's dtype) and clear the flag, whether the remote code
    passes beta by name or by position; calls without the flag pass beta through untouched. Each
    entry point is wrapped once, and keeps its kernel's name and signature."""
    remote = remote_module()
    kernels = (remote.chunk_kda, remote.fused_recurrent_kda)
    _remote_fixes._patch_kda_beta_sigmoid(remote)
    wrapped = (remote.chunk_kda, remote.fused_recurrent_kda)
    _remote_fixes._patch_kda_beta_sigmoid(remote)
    assert (remote.chunk_kda, remote.fused_recurrent_kda) == wrapped
    for kernel, entry in zip(kernels, wrapped):
        assert entry is not kernel
        assert entry.__name__ == kernel.__name__
        assert inspect.signature(entry) == inspect.signature(kernel)
    beta = torch.tensor([-8.0, 0.0, 8.4], dtype=torch.bfloat16)

    def call(kda, **flag):
        return (
            kda(1, 2, 3, 4, beta=beta, **flag) if by == "keyword" else kda(1, 2, 3, 4, beta, **flag)
        )

    assert call(remote.chunk_kda, use_beta_sigmoid_in_kernel=True) == "chunk_kda"
    call(remote.fused_recurrent_kda, use_beta_sigmoid_in_kernel=True)

    squashed = torch.sigmoid(beta.float()).to(torch.bfloat16)
    assert [name for name, *_ in remote.calls] == ["chunk_kda", "fused_recurrent_kda"]
    for _, seen, flag in remote.calls:
        assert flag is False
        assert seen.dtype == torch.bfloat16
        assert torch.equal(seen, squashed)

    remote.calls.clear()
    call(remote.chunk_kda)
    call(remote.fused_recurrent_kda, use_beta_sigmoid_in_kernel=False)
    assert [(seen is beta, flag) for _, seen, flag in remote.calls] == [(True, False)] * 2


def test_a_kernel_call_whose_beta_the_patch_cannot_find_is_refused():
    """A kernel whose signature hides beta (behind a decorator's ``*args``), asked for the sigmoid
    fla-core ignores, would run on the raw gate: the call is refused instead."""
    remote = remote_module()
    remote.chunk_kda = lambda *args, **kwargs: pytest.fail("ran on the raw gate")
    _remote_fixes._patch_kda_beta_sigmoid(remote)

    with pytest.raises(RuntimeError, match="cannot find the beta gate"):
        remote.chunk_kda(1, 2, 3, 4, torch.zeros(2), use_beta_sigmoid_in_kernel=True)


def test_kda_layers_whose_kernels_the_beta_patch_cannot_find_are_refused():
    """Left unwrapped, the raw gate would show up only as a NaN amax after a whole calibration, so
    KDA layers without the kernels to wrap stop the load; a model without KDA layers loads."""
    remote = remote_module()
    del remote.chunk_kda, remote.fused_recurrent_kda

    with pytest.raises(RuntimeError, match="KimiDeltaAttention"):
        _remote_fixes._patch_kda_beta_sigmoid(remote)

    del remote.KimiDeltaAttention
    _remote_fixes._patch_kda_beta_sigmoid(remote)


# ------------------------------------------------------------- transformers-version compatibility


def test_create_causal_mask_shim_adapts_the_remote_call_once():
    """The remote code calls ``create_causal_mask(input_embeds=..., cache_position=...)``; current
    transformers takes ``inputs_embeds`` and no ``cache_position``. The shim renames the one, drops
    the other, and wraps only once."""
    remote = remote_module()
    _remote_fixes._shim_create_causal_mask(remote)
    shim = remote.create_causal_mask
    _remote_fixes._shim_create_causal_mask(remote)
    assert remote.create_causal_mask is shim

    mask = remote.create_causal_mask(
        config="config",
        input_embeds="embeds",
        attention_mask="mask_2d",
        cache_position="positions",
        past_key_values="cache",
        position_ids="ids",
    )

    assert mask == "mask"
    assert remote.calls == [("create_causal_mask", "config", "embeds", "mask_2d", "cache", "ids")]


def test_create_causal_mask_shim_keeps_what_a_kwargs_signature_takes():
    """A ``create_causal_mask`` that takes ``**kwargs`` accepts every keyword: none is dropped."""
    remote = types.ModuleType("kimi_k3_kwargs_stub")
    remote.create_causal_mask = lambda config, inputs_embeds, **kwargs: kwargs
    _remote_fixes._shim_create_causal_mask(remote)

    kept = remote.create_causal_mask(config="config", input_embeds="embeds", cache_position=0)

    assert kept == {"cache_position": 0}


def test_create_causal_mask_shim_lets_a_rejected_mask_input_raise():
    """Only ``cache_position`` is dropped. A release that renamed ``attention_mask`` would otherwise
    get a mask built without the padding, silently, instead of the remote call's ``TypeError``."""
    remote = types.ModuleType("kimi_k3_renamed_stub")
    remote.create_causal_mask = lambda config, inputs_embeds, padding_mask=None: padding_mask
    _remote_fixes._shim_create_causal_mask(remote)

    with pytest.raises(TypeError):
        remote.create_causal_mask(
            config="config", input_embeds="embeds", attention_mask="mask_2d", cache_position=0
        )


def test_kimi_cache_mask_sizes_and_query_offset_count_the_cached_tokens():
    """transformers >= 5.13 passes the query length as an int; the remote cache expects a
    ``cache_position`` tensor. Either way kv_length is the query length plus the cached tokens.
    transformers >= 5.15 also asks for the query offset, which the remote cache lacks: the cached
    tokens, as transformers' own caches answer. A cache with its own offset keeps it."""
    remote = remote_module()
    _remote_fixes._patch_kimi_cache_get_mask_sizes(remote)
    cache = remote.KimiDynamicCache(seen_tokens=5)

    assert cache.get_mask_sizes(3, 2) == (8, 0)
    assert cache.get_mask_sizes(torch.arange(4), 1) == (9, 0)
    assert cache.get_query_offset(2) == 5
    assert remote.calls == [("get_seq_length", 2), ("get_seq_length", 1), ("get_seq_length", 2)]

    remote = remote_module()
    remote.KimiDynamicCache.get_query_offset = own = lambda self, layer_idx=0: 7
    _remote_fixes._patch_kimi_cache_get_mask_sizes(remote)
    assert remote.KimiDynamicCache.get_query_offset is own


def test_tie_weights_shim_leaves_an_inherited_tie_weights_alone(monkeypatch):
    """Only an override defined on the VLM class has the old signature; an inherited
    ``tie_weights`` already takes the kwargs and must keep receiving them."""

    class PreTrainedStub:
        def tie_weights(self, **kwargs):
            return kwargs

    remote = types.ModuleType("transformers_modules.inheriting_stub.modeling_kimi_k3")
    vlm = type("KimiK3ForConditionalGeneration", (PreTrainedStub,), {"__module__": remote.__name__})
    remote.KimiK3ForConditionalGeneration = vlm
    monkeypatch.setitem(sys.modules, remote.__name__, remote)

    _remote_fixes._shim_tie_weights_kwargs("transformers_modules.inheriting_stub")

    assert vlm().tie_weights(recompute_mapping=False) == {"recompute_mapping": False}
