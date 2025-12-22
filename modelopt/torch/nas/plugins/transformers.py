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

"""Plugin to add NAS support for HuggingFace Transformers models."""

from collections.abc import Callable, Sequence

import torch
from torch import nn
from transformers.models.bert.modeling_bert import BertAttention, BertSelfAttention
from transformers.models.gptj.modeling_gptj import GPTJAttention

from modelopt.torch.opt.dynamic import DynamicModule
from modelopt.torch.opt.hparam import HPType
from modelopt.torch.trace import Symbol
from modelopt.torch.trace.plugins.transformers import SymAttentionHead
from modelopt.torch.utils import make_divisible

from ..autonas import AutoNASConfig
from ..registry import DMRegistry
from ..traced_hp import TracedHp, TracedHpRegistry

__all__ = ["_DynamicAttention", "_DynamicBertAttention", "_DynamicGPTJAttention"]


@TracedHpRegistry.register(SymAttentionHead)
class AttentionHeadTracedHp(TracedHp):
    _hp_hidden_dim: TracedHp  # QKV out_features
    _hidden_dim_per_head: int

    @TracedHp.active.setter  # type: ignore[attr-defined]
    def active(self, val: HPType | None):
        super(AttentionHeadTracedHp, type(self)).active.fset(self, val)  # type: ignore[attr-defined]
        with self._hp_hidden_dim._force_configurable():
            self._hp_hidden_dim.active = self.active * self._hidden_dim_per_head

    @TracedHp.choices.setter  # type: ignore[attr-defined]
    def choices(self, val: Sequence[HPType]):
        super(AttentionHeadTracedHp, type(self)).choices.fset(self, val)  # type: ignore[attr-defined]
        with self._hp_hidden_dim._force_configurable():
            self._hp_hidden_dim.choices = [c * self._hidden_dim_per_head for c in self.choices]

    def _get_importance(self) -> TracedHp.Importance:
        imp_hidden_dim = self._hp_hidden_dim._get_importance()
        assert imp_hidden_dim is not None
        imp_heads = imp_hidden_dim.reshape(self.max, -1).sum(dim=1)
        return imp_heads

    def _enforce_order(self, order: torch.Tensor | None = None) -> None:
        """Convert order of heads to order of hidden_dim."""
        if order is None:
            order_hidden_dim = None
        else:
            h_to_d_slice = torch.arange(self._hp_hidden_dim.max, device=order.device).reshape(
                self.max, -1
            )
            order_hidden_dim = h_to_d_slice[order].flatten()

        super()._enforce_order(order)
        self._hp_hidden_dim._enforce_order(order_hidden_dim)

    def _resolve_dependencies(
        self, sym: Symbol, get_hp: Callable[[Symbol], TracedHp]
    ) -> dict[Symbol, TracedHp]:
        # get first dependency which is the hidden dim hparam and make it not configurable
        sym_hidden_dim = sym._dependencies.pop(0)
        self._hp_hidden_dim = get_hp(sym_hidden_dim)
        self._hp_hidden_dim._is_configurable = False

        # set up constrained choices hidden dim hp
        self._hidden_dim_per_head = self._hp_hidden_dim.original // self.original
        with self._hp_hidden_dim._force_configurable():
            self._hp_hidden_dim.choices = [c * self._hidden_dim_per_head for c in self.choices]

        # handle remaining dependencies to regular way
        mapping = super()._resolve_dependencies(sym, get_hp)
        mapping[sym_hidden_dim] = self._hp_hidden_dim

        return mapping


class _DynamicAttention(DynamicModule):
    """A generic attention layer with dynamic hyperparams."""

    def _setup(self):
        # register num_attention_heads hyperparameter
        hp_num_heads = TracedHp(list(range(1, self.num_attention_heads + 1)))
        self._register_hparam("num_attention_heads", hp_num_heads)

    def configure_qkv_out(self, q_name: str, k_name: str, v_name: str, out_name: str) -> None:
        """Utility to configure qkv and output projection matrices and their hparams."""
        # convert pytorch modules to dynamic modules in-place
        q = DMRegistry.convert(self.get_submodule(q_name))
        k = DMRegistry.convert(self.get_submodule(k_name))
        v = DMRegistry.convert(self.get_submodule(v_name))
        out = DMRegistry.convert(self.get_submodule(out_name))

        # link up embedding dim hparam manually and register with the attention module
        hp_embed_dim = q.get_hparam("in_features")
        self._register_hparam("embed_dim", hp_embed_dim)
        k.in_features = hp_embed_dim
        v.in_features = hp_embed_dim
        out.out_features = hp_embed_dim

        # link up hidden dim hparam manually and register with the attention module
        hp_hidden_dim = q.get_hparam("out_features")
        self._register_hparam("hidden_dim", hp_hidden_dim)
        k.out_features = hp_hidden_dim
        v.out_features = hp_hidden_dim
        out.in_features = hp_hidden_dim

        assert isinstance(out, nn.Linear)
        hp_hidden_dim.register_importance(
            lambda: torch.linalg.vector_norm(out.weight.detach(), dim=0)
        )

    def modify(
        self, *, n_heads_ratio: tuple[float, ...] | None = None, n_heads_divisor: int = 1
    ) -> None:
        # modify num_attention_heads
        hp = self.get_hparam("num_attention_heads")
        choices = (
            {r * hp.original for r in n_heads_ratio}
            if n_heads_ratio is not None
            else set(hp.choices)
        )
        choices = {int(make_divisible(c, n_heads_divisor)) for c in choices}
        hp.choices = list(set(hp.choices) & choices | {hp.original})


# Provide BertSelfAttention as a parent class since we do not register to DMRegistry
class _DynamicBertSelfAttention(_DynamicAttention, BertSelfAttention):
    """A ``BertSelfAttention`` layer with dynamic hyperparams."""

    def _setup(self):
        super()._setup()
        self._register_dynamic_attribute(
            "all_head_size", lambda mod, val: mod.num_attention_heads * mod.attention_head_size
        )

    def export(self) -> nn.Module:
        self.query.export()
        self.key.export()
        self.value.export()
        super().export()
        return self


# TODO: can we just register BertSelfAttention and infer BertSelfOutput from tracing?
@DMRegistry.register({BertAttention: "hf.BertAttention"})
class _DynamicBertAttention(_DynamicAttention):
    """A ``BertAttention`` layer with dynamic hyperparams."""

    def _setup(self):
        # convert pytorch modules to dynamic modules
        self.self = _DynamicBertSelfAttention.convert(self.self)
        self._register_hparam("num_attention_heads", self.self.get_hparam("num_attention_heads"))
        self.configure_qkv_out("self.query", "self.key", "self.value", "output.dense")

        # convert and link layer norm hparam to embed_dim
        self.output.LayerNorm = DMRegistry.convert(self.output.LayerNorm)
        self.output.LayerNorm.num_features = self.get_hparam("embed_dim")

    def export(self) -> nn.Module:
        self.self.export()
        self.output.dense.export()
        self.output.LayerNorm.export()
        super().export()
        return self


@DMRegistry.register({GPTJAttention: "hf.GPTJAttention"})
class _DynamicGPTJAttention(_DynamicAttention):
    """A ``GPTJAttention`` layer with dynamic hyperparams."""

    def _setup(self):
        super()._setup()
        self.configure_qkv_out("q_proj", "k_proj", "v_proj", "out_proj")

    def export(self) -> nn.Module:
        self.q_proj.export()
        self.k_proj.export()
        self.v_proj.export()
        self.out_proj.export()
        super().export()
        return self


def _n_heads_config():
    return {"n_heads_ratio": None, "n_heads_divisor": 1}


AutoNASConfig.register_default(
    {
        "hf.BertAttention": _n_heads_config(),
        "hf.GPTJAttention": _n_heads_config(),
    }
)
