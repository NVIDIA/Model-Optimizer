# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Global per-layer-embedding (PLE) channel ranking and tensor surgery."""

from __future__ import annotations

from dataclasses import dataclass

import torch

__all__ = ["PLEPruningSpec"]


@dataclass(frozen=True)
class PLEPruningSpec:
    """Own every tensor coupled to one global PLE channel dimension.

    Gemma-style PLE uses one shared RMSNorm across all layer chunks. Therefore
    every layer must use the same permutation and retained width. Per-layer
    contribution scores are summed before sorting to preserve this invariant.
    """

    language_prefix: str
    layer_template: str
    num_layers: int
    width: int
    layer_gate_name: str = "per_layer_input_gate"
    layer_projection_name: str = "per_layer_projection"
    model_embedding_name: str = "embed_tokens_per_layer"
    model_projection_name: str = "per_layer_model_projection"
    model_norm_name: str = "per_layer_projection_norm"

    def __post_init__(self) -> None:
        if int(self.num_layers) < 1 or int(self.width) < 1:
            raise ValueError("PLE num_layers and width must be positive")
        if "{layer_idx}" not in self.layer_template:
            raise ValueError("PLE layer_template must contain {layer_idx}")

    def layer_prefix(self, layer_idx: int) -> str:
        return self.layer_template.format(layer_idx=int(layer_idx))

    def layer_score_key(self, layer_idx: int) -> str:
        return f"{self.layer_prefix(layer_idx)}.{self.layer_projection_name}"

    def order_from_score_logs(self, score_logs: dict[str, dict]) -> torch.Tensor:
        scores = []
        missing = []
        for layer_idx in range(self.num_layers):
            key = self.layer_score_key(layer_idx)
            log = score_logs.get(key)
            score = log.get("score") if isinstance(log, dict) else None
            if not torch.is_tensor(score):
                missing.append(key)
                continue
            score = score.reshape(-1).float()
            if score.numel() != self.width:
                raise ValueError(
                    f"PLE score {key!r} has {score.numel()} channels, expected {self.width}"
                )
            scores.append(score)
        if missing:
            raise ValueError(
                f"PLE ranking requires every layer score; missing {len(missing)}: {missing[:4]}"
            )
        return torch.argsort(torch.stack(scores).sum(dim=0), descending=True)

    def _validate_order(self, order: torch.Tensor) -> torch.Tensor:
        order = order.reshape(-1).to(dtype=torch.long)
        if order.numel() != self.width:
            raise ValueError(
                f"PLE order has {order.numel()} channels, expected {self.width}"
            )
        if not torch.equal(torch.sort(order.cpu()).values, torch.arange(self.width)):
            raise ValueError("PLE order is not a permutation")
        return order

    def _packed_embedding_key(self) -> str:
        return f"{self.language_prefix}.{self.model_embedding_name}.weight"

    def _packed_projection_key(self) -> str:
        return f"{self.language_prefix}.{self.model_projection_name}.weight"

    def _packed_projection_bias_key(self) -> str:
        return f"{self.language_prefix}.{self.model_projection_name}.bias"

    def _norm_keys(self) -> tuple[str, str]:
        prefix = f"{self.language_prefix}.{self.model_norm_name}"
        return f"{prefix}.weight", f"{prefix}.bias"

    def permute_state_dict(
        self,
        state_dict: dict[str, torch.Tensor],
        order: torch.Tensor,
    ) -> tuple[dict[str, torch.Tensor], set[str]]:
        order = self._validate_order(order)
        output = dict(state_dict)
        handled: set[str] = set()

        embedding_key = self._packed_embedding_key()
        if embedding_key in output:
            tensor = output[embedding_key]
            if tensor.shape[-1] != self.num_layers * self.width:
                raise ValueError(
                    f"{embedding_key} packed width={tensor.shape[-1]}, expected "
                    f"{self.num_layers * self.width}"
                )
            output[embedding_key] = tensor.view(
                *tensor.shape[:-1], self.num_layers, self.width
            ).index_select(-1, order.to(tensor.device)).reshape_as(tensor)
            handled.add(embedding_key)

        for key in (self._packed_projection_key(), self._packed_projection_bias_key()):
            if key not in output:
                continue
            tensor = output[key]
            if tensor.shape[0] != self.num_layers * self.width:
                raise ValueError(
                    f"{key} packed width={tensor.shape[0]}, expected "
                    f"{self.num_layers * self.width}"
                )
            output[key] = tensor.view(
                self.num_layers, self.width, *tensor.shape[1:]
            ).index_select(1, order.to(tensor.device)).reshape_as(tensor)
            handled.add(key)

        for key in self._norm_keys():
            if key in output:
                tensor = output[key]
                if tensor.shape[0] != self.width:
                    raise ValueError(f"{key} width={tensor.shape[0]}, expected {self.width}")
                output[key] = tensor.index_select(0, order.to(tensor.device))
                handled.add(key)

        for layer_idx in range(self.num_layers):
            prefix = self.layer_prefix(layer_idx)
            gate_weight = f"{prefix}.{self.layer_gate_name}.weight"
            gate_bias = f"{prefix}.{self.layer_gate_name}.bias"
            projection_weight = f"{prefix}.{self.layer_projection_name}.weight"
            for key in (gate_weight, gate_bias):
                if key in output:
                    tensor = output[key]
                    if tensor.shape[0] != self.width:
                        raise ValueError(f"{key} width={tensor.shape[0]}, expected {self.width}")
                    output[key] = tensor.index_select(0, order.to(tensor.device))
                    handled.add(key)
            if projection_weight in output:
                tensor = output[projection_weight]
                if tensor.shape[1] != self.width:
                    raise ValueError(
                        f"{projection_weight} width={tensor.shape[1]}, expected {self.width}"
                    )
                output[projection_weight] = tensor.index_select(1, order.to(tensor.device))
                handled.add(projection_weight)
        return output, handled

    def slice_state_dict(
        self,
        state_dict: dict[str, torch.Tensor],
        target_width: int,
    ) -> dict[str, torch.Tensor]:
        target_width = int(target_width)
        if not 0 < target_width <= self.width:
            raise ValueError(
                f"PLE target width must be in [1, {self.width}], got {target_width}"
            )
        output = dict(state_dict)
        embedding_key = self._packed_embedding_key()
        if embedding_key in output:
            tensor = output[embedding_key]
            output[embedding_key] = tensor.view(
                *tensor.shape[:-1], self.num_layers, self.width
            )[..., :target_width].reshape(*tensor.shape[:-1], self.num_layers * target_width)

        for key in (self._packed_projection_key(), self._packed_projection_bias_key()):
            if key in output:
                tensor = output[key]
                output[key] = tensor.view(
                    self.num_layers, self.width, *tensor.shape[1:]
                )[:, :target_width].reshape(
                    self.num_layers * target_width, *tensor.shape[1:]
                )

        for key in self._norm_keys():
            if key in output:
                output[key] = output[key][:target_width]

        for layer_idx in range(self.num_layers):
            prefix = self.layer_prefix(layer_idx)
            for suffix in (
                f"{self.layer_gate_name}.weight",
                f"{self.layer_gate_name}.bias",
            ):
                key = f"{prefix}.{suffix}"
                if key in output:
                    output[key] = output[key][:target_width]
            projection_weight = f"{prefix}.{self.layer_projection_name}.weight"
            if projection_weight in output:
                output[projection_weight] = output[projection_weight][:, :target_width]
        return output
