# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Projection-aware activation scoring for Qwen GatedDeltaNet axes.

The additive key-group/value-lane orders use the exact output-contribution Gram.
Value-coordinate scores use exact post-normalization projected singleton energy.
Key-coordinate scores are a recurrence-aware proxy from convolved Q/K projection
energy, weighted by the owned value lanes' output projection energy. The artifact
labels each objective explicitly so diagnostics can compare it honestly.
"""

from __future__ import annotations

import torch
import torch.distributed as c10d

from ....pruning.gated_delta_net import GDNShape
from ..reduction import (
    full_weight,
    gather_scored_axis,
    reduce_token_sum,
    to_local_with_feature_group,
)
from .base import ScoringHook

__all__ = ["GatedDeltaNetActivationScorer"]


def _conditional_removal_order(gram: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Return least-to-most-important greedy deletion order and prospective loss."""
    gram = gram.double().cpu()
    size = gram.shape[0]
    removed: list[int] = []
    remaining = list(range(size))
    order: list[int] = []
    damage: list[float] = []
    while remaining:
        candidates = []
        for idx in remaining:
            loss = gram[idx, idx]
            if removed:
                loss = loss + 2.0 * gram[idx, removed].sum()
                prior = gram[removed][:, removed].sum()
                loss = loss + prior
            candidates.append((float(loss), idx))
        _, selected = min(candidates, key=lambda item: (item[0], item[1]))
        removed.append(selected)
        remaining.remove(selected)
        order.append(selected)
        damage.append(float(gram[removed][:, removed].sum()))
    return torch.tensor(order, dtype=torch.long), torch.tensor(damage, dtype=torch.float64)


def _ordinal_scores(order_least_first: torch.Tensor) -> torch.Tensor:
    scores = torch.empty(order_least_first.numel(), dtype=torch.float32)
    scores[order_least_first] = torch.arange(order_least_first.numel(), dtype=torch.float32)
    return scores


class GatedDeltaNetActivationScorer(ScoringHook):
    target_type = "gated_delta_net"
    method = "gdn_projection_aware_v1"

    def __init__(
        self,
        module,
        groups,
        *,
        token_chunk_size: int = 128,
        block_idx=None,
        name=None,
        **_kwargs,
    ):
        super().__init__(module, groups, block_idx=block_idx, name=name)
        required = (
            "in_proj_qkv",
            "out_proj",
            "conv1d",
            "num_k_heads",
            "num_v_heads",
            "head_k_dim",
            "head_v_dim",
        )
        missing = [attr for attr in required if not hasattr(module, attr)]
        if missing:
            raise ValueError(
                f"GatedDeltaNet scorer layer={block_idx} module={type(module).__name__} "
                f"is missing required attributes {missing}"
            )
        self.shape = GDNShape.from_module(module)
        self.token_chunk_size = max(1, int(token_chunk_size))
        self._lane_gram = None
        self._value_dim_energy = None
        self._key_dim_energy = None  # [local_key_heads, key_head_dim] — gathered at finalize
        self._qkv_feature_sharded = False  # True when in_proj_qkv output was TP-Shard(-1)
        self._token_count = 0
        self._qkv_hook_count = 0
        self._out_hook_count = 0
        self._out_weight = None
        self._conv_energy = None

    def __call__(self, module, args, output):
        raise RuntimeError("GatedDeltaNetActivationScorer uses child observation hooks")

    def register(self):
        if self._handles:
            return self._handles[0]
        self._register_handle(self.module.in_proj_qkv.register_forward_hook(self._observe_qkv))
        self._register_handle(self.module.out_proj.register_forward_pre_hook(self._observe_out_input))
        return self._handles[0]

    def _active(self) -> bool:
        return self.enabled and not ScoringHook._nested_disable_depth

    def _ensure_weights(self, device) -> None:
        if self._out_weight is None:
            weight = full_weight(self.module.out_proj.weight).detach().float().to(device)
            expected = self.shape.num_value_heads * self.shape.value_head_dim
            if tuple(weight.shape)[1] != expected:
                raise RuntimeError(
                    f"GDN layer={self.block_idx} out_proj width={weight.shape[1]} != {expected}"
                )
            self._out_weight = weight.view(
                weight.shape[0], self.shape.num_value_heads, self.shape.value_head_dim
            )
        if self._conv_energy is None:
            conv = full_weight(self.module.conv1d.weight).detach().float().to(device)
            conv = conv.reshape(conv.shape[0], -1).square().sum(dim=-1)
            key_width = self.shape.num_key_heads * self.shape.key_head_dim
            self._conv_energy = torch.stack(
                (
                    conv[:key_width].view(self.shape.num_key_heads, self.shape.key_head_dim),
                    conv[key_width : 2 * key_width].view(
                        self.shape.num_key_heads, self.shape.key_head_dim
                    ),
                )
            )

    def _observe_qkv(self, module, args, output):
        if not self._active():
            return None
        value = output[0] if isinstance(output, tuple) else output
        # Extract the rank-local shard without gathering: Q and K energy are additive
        # per token and can be accumulated locally, then gathered once at finalize.
        # (Contrast with _observe_out_input where the lane gram needs all V heads
        # simultaneously for cross-TP off-diagonal terms — that gather stays.)
        local, feature_group = to_local_with_feature_group(value, feature_dim=-1)
        local = local.detach().float()
        local = self._flatten_valid_tokens(
            local,
            trailing_dims=1,
            stream="qkv",
        )
        if local.shape[0] == 0:
            return None
        self._qkv_feature_sharded = feature_group is not None

        tp_size = c10d.get_world_size(feature_group) if feature_group is not None else 1
        tp_rank = c10d.get_rank(feature_group) if feature_group is not None else 0

        key_width = self.shape.num_key_heads * self.shape.key_head_dim
        full_qkv_width = 2 * key_width + self.shape.num_value_heads * self.shape.value_head_dim
        if self.shape.num_key_heads % tp_size != 0:
            raise RuntimeError(
                f"GDN layer={self.block_idx} num_key_heads={self.shape.num_key_heads} "
                f"is not divisible by tp_size={tp_size}"
            )
        expected_local_width = full_qkv_width // tp_size
        if local.shape[-1] != expected_local_width:
            raise RuntimeError(
                f"GDN layer={self.block_idx} local qkv width={local.shape[-1]} != "
                f"{expected_local_width} (full={full_qkv_width}, tp_size={tp_size})"
            )

        self._ensure_weights(local.device)
        local_key_heads = self.shape.num_key_heads // tp_size
        local_key_start = tp_rank * local_key_heads
        local_key_width = local_key_heads * self.shape.key_head_dim

        q = local[..., :local_key_width].reshape(-1, local_key_heads, self.shape.key_head_dim)
        k = local[..., local_key_width : 2 * local_key_width].reshape(
            -1, local_key_heads, self.shape.key_head_dim
        )
        local_conv = self._conv_energy[:, local_key_start : local_key_start + local_key_heads]
        q_energy = q.double().square().sum(dim=0) * local_conv[0].double()
        k_energy = k.double().square().sum(dim=0) * local_conv[1].double()

        out_lane_energy = self._out_weight.double().square().sum(dim=(0, 2))
        group_weight_full = out_lane_energy.view(
            self.shape.num_key_heads, self.shape.value_heads_per_group
        ).sum(dim=1)
        group_weight_local = group_weight_full[local_key_start : local_key_start + local_key_heads]

        contribution = (q_energy + k_energy) * group_weight_local[:, None]
        self._key_dim_energy = (
            contribution if self._key_dim_energy is None else self._key_dim_energy + contribution
        )
        self._qkv_hook_count += 1
        return None

    def _observe_out_input(self, module, args):
        if not self._active():
            return None
        value = args[0]
        local, feature_group = to_local_with_feature_group(value, feature_dim=-1)
        value = gather_scored_axis(local, feature_group, dim=-1).detach().float()
        value = self._flatten_valid_tokens(
            value,
            trailing_dims=1,
            stream="out",
        )
        if value.shape[0] == 0:
            return None
        expected = self.shape.num_value_heads * self.shape.value_head_dim
        if value.shape[-1] != expected:
            raise RuntimeError(
                f"GDN layer={self.block_idx} out_proj input width={value.shape[-1]} != {expected}"
            )
        self._ensure_weights(value.device)
        lanes = value.reshape(-1, self.shape.num_value_heads, self.shape.value_head_dim)
        if self._lane_gram is None:
            self._lane_gram = torch.zeros(
                self.shape.num_value_heads,
                self.shape.num_value_heads,
                dtype=torch.float64,
                device=value.device,
            )
            self._value_dim_energy = torch.zeros(
                self.shape.value_head_dim, dtype=torch.float64, device=value.device
            )
        for start in range(0, lanes.shape[0], self.token_chunk_size):
            chunk = lanes[start : start + self.token_chunk_size]
            lane_contrib = torch.einsum("nhd,ohd->nho", chunk, self._out_weight)
            self._lane_gram.add_(torch.einsum("nho,nko->hk", lane_contrib.double(), lane_contrib.double()))
            dim_contrib = torch.einsum("nhd,ohd->ndo", chunk, self._out_weight)
            self._value_dim_energy.add_(dim_contrib.double().square().sum(dim=(0, 2)))
        self._token_count += lanes.shape[0]
        self._out_hook_count += 1
        return None

    def finalize(self) -> dict:
        if self._lane_gram is None or self._value_dim_energy is None or self._key_dim_energy is None:
            raise RuntimeError(
                f"GDN layer={self.block_idx} did not observe all required hooks: "
                f"qkv={self._qkv_hook_count}, out={self._out_hook_count}"
            )
        if self._qkv_hook_count != self._out_hook_count:
            raise RuntimeError(
                f"GDN layer={self.block_idx} hook mismatch: "
                f"qkv={self._qkv_hook_count}, out={self._out_hook_count}"
            )
        count = torch.tensor(float(self._token_count), dtype=torch.float64, device=self._lane_gram.device)
        for tensor in (self._lane_gram, self._value_dim_energy, self._key_dim_energy, count):
            reduce_token_sum(tensor, self.groups.token_group)
        # Gather the key-dim accumulator from all TP ranks.  _lane_gram and
        # _value_dim_energy are already full-size (the out_proj hook gathers the full
        # value-lane tensor on every forward because the gram needs cross-rank terms).
        # _key_dim_energy was accumulated in local-shard space to avoid the unnecessary
        # gather of the V portion of in_proj_qkv; reassemble it here.
        qkv_feature_group = self.groups.tp_group if self._qkv_feature_sharded else None
        key_dim_energy_full = gather_scored_axis(self._key_dim_energy, qkv_feature_group, dim=0)
        if not torch.isfinite(self._lane_gram).all() or count.item() <= 0:
            raise RuntimeError(f"GDN layer={self.block_idx} produced invalid statistics")
        lane_gram = (self._lane_gram / count).cpu()
        value_energy = (self._value_dim_energy / count).cpu()
        key_energy = (key_dim_energy_full / count).cpu()
        lane_orders = []
        lane_damage = []
        r = self.shape.value_heads_per_group
        for group in range(self.shape.num_key_heads):
            ids = torch.arange(group * r, (group + 1) * r)
            order, damage = _conditional_removal_order(lane_gram[ids][:, ids])
            lane_orders.append(torch.flip(order, dims=(0,)))
            lane_damage.append(damage)
        group_gram = torch.empty(
            self.shape.num_key_heads, self.shape.num_key_heads, dtype=torch.float64
        )
        for left in range(self.shape.num_key_heads):
            li = torch.arange(left * r, (left + 1) * r)
            for right in range(self.shape.num_key_heads):
                ri = torch.arange(right * r, (right + 1) * r)
                group_gram[left, right] = lane_gram[li][:, ri].sum()
        group_remove, group_damage = _conditional_removal_order(group_gram)
        key_order = torch.argsort(key_energy, dim=-1, descending=True, stable=True)
        value_order = torch.argsort(value_energy, descending=True, stable=True)
        lane_scores = torch.stack(
            [_ordinal_scores(torch.flip(order, dims=(0,))) for order in lane_orders]
        )
        return {
            "format_version": 1,
            "head_ranking_method": "projection_aware_conditional_lane_gram_v1",
            "key_dim_ranking_method": "qk_activation_conv_output_weight_proxy_v1",
            "value_dim_ranking_method": "post_norm_projected_singleton_energy_v1",
            "shape": {
                "num_key_heads": self.shape.num_key_heads,
                "num_value_heads": self.shape.num_value_heads,
                "key_head_dim": self.shape.key_head_dim,
                "value_head_dim": self.shape.value_head_dim,
            },
            "lane_gram": lane_gram.float(),
            "key_group_damage_at_removal": group_damage.float(),
            "key_group_order_most_important_first": torch.flip(group_remove, dims=(0,)),
            "key_group_scores": _ordinal_scores(group_remove),
            "value_lane_damage_at_removal": torch.stack(lane_damage).float(),
            "value_lane_order_most_important_first": torch.stack(lane_orders),
            "value_lane_scores": lane_scores,
            "key_dim_scores": key_energy.float(),
            "key_dim_order_most_important_first": key_order,
            "value_dim_scores": value_energy.float(),
            "value_dim_order_most_important_first": value_order,
            "token_count": int(count.item()),
            "hook_fire_count": self._out_hook_count,
        }

    def checkpoint_state(self) -> dict:
        def cpu(value):
            return None if value is None else value.detach().cpu()

        return {
            "lane_gram": cpu(self._lane_gram),
            "value_dim_energy": cpu(self._value_dim_energy),
            # key_dim_energy is stored as [local_key_heads, key_head_dim] — the gather
            # over TP ranks happens at finalize, not here.
            "key_dim_energy": cpu(self._key_dim_energy),
            "qkv_feature_sharded": self._qkv_feature_sharded,
            "token_count": self._token_count,
            "qkv_hook_count": self._qkv_hook_count,
            "out_hook_count": self._out_hook_count,
        }

    def load_checkpoint_state(self, state: dict) -> None:
        device = next(self.module.parameters()).device

        def restore(value):
            return None if value is None else value.to(device)

        self._lane_gram = restore(state.get("lane_gram"))
        self._value_dim_energy = restore(state.get("value_dim_energy"))
        self._key_dim_energy = restore(state.get("key_dim_energy"))
        # Older checkpoints (pre-bug-fix) stored key_dim_energy in gathered full-size
        # form and have no "qkv_feature_sharded" key.  Detect that case by checking
        # the loaded shape against the known full width and mark as not sharded so the
        # no-op gather path is taken at finalize (correct for already-gathered data).
        if "qkv_feature_sharded" in state:
            self._qkv_feature_sharded = bool(state["qkv_feature_sharded"])
        elif self._key_dim_energy is not None:
            full_key_heads = self.shape.num_key_heads
            self._qkv_feature_sharded = self._key_dim_energy.shape[0] != full_key_heads
        else:
            self._qkv_feature_sharded = False
        self._token_count = int(state.get("token_count", 0))
        self._qkv_hook_count = int(state.get("qkv_hook_count", 0))
        self._out_hook_count = int(state.get("out_hook_count", 0))
