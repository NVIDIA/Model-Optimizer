# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Explicit distributed topology for vLLM runtime measurements."""

from __future__ import annotations

import subprocess
from dataclasses import asdict, dataclass
from functools import lru_cache
from typing import Any

from ..orchestration.mesh import normalize_vllm_topology

__all__ = ["RuntimeTopology"]


@dataclass(frozen=True)
class RuntimeTopology:
    tensor_parallel_size: int = 1
    pipeline_parallel_size: int = 1
    data_parallel_size: int = 1
    prefill_context_parallel_size: int = 1
    decode_context_parallel_size: int = 1
    enable_expert_parallel: bool = False
    distributed_executor_backend: str = "mp"
    gpu_group_size: int = 1

    def __post_init__(self) -> None:
        for field_name in (
            "tensor_parallel_size",
            "pipeline_parallel_size",
            "data_parallel_size",
            "prefill_context_parallel_size",
            "decode_context_parallel_size",
            "gpu_group_size",
        ):
            if int(getattr(self, field_name)) < 1:
                raise ValueError(f"runtime topology {field_name} must be >= 1")
        if self.gpu_group_size != self.world_size:
            raise ValueError(
                f"gpu_group_size={self.gpu_group_size} does not match vLLM "
                f"world_size={self.world_size}"
            )
        if self.decode_context_parallel_size > self.tensor_parallel_size:
            raise ValueError(
                "decode_context_parallel_size cannot exceed tensor_parallel_size"
            )
        if self.tensor_parallel_size % self.decode_context_parallel_size:
            raise ValueError(
                "decode_context_parallel_size must divide tensor_parallel_size"
            )
        if self.distributed_executor_backend not in {"mp", "external_launcher", "ray"}:
            raise ValueError(
                "distributed_executor_backend must be mp, external_launcher, or ray"
            )

    @property
    def world_size(self) -> int:
        return (
            self.tensor_parallel_size
            * self.pipeline_parallel_size
            * self.data_parallel_size
            * self.prefill_context_parallel_size
        )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def validate_model_dimensions(
        self,
        *,
        num_attention_heads: int,
        num_key_value_heads: int,
    ) -> None:
        if int(num_attention_heads) % self.tensor_parallel_size:
            raise ValueError(
                f"num_attention_heads={num_attention_heads} must divide TP={self.tensor_parallel_size}"
            )
        if int(num_key_value_heads) % self.tensor_parallel_size:
            raise ValueError(
                f"num_key_value_heads={num_key_value_heads} must divide TP={self.tensor_parallel_size}"
            )
        _validate_vllm_cli_support(
            self.prefill_context_parallel_size,
            self.decode_context_parallel_size,
            self.data_parallel_size,
            self.enable_expert_parallel,
        )

    @classmethod
    def from_config(cls, config: Any) -> "RuntimeTopology":
        if config is None:
            return cls()
        canonical = normalize_vllm_topology(config)
        return cls(
            tensor_parallel_size=canonical["tp"],
            pipeline_parallel_size=canonical["pp"],
            data_parallel_size=canonical["dp"],
            prefill_context_parallel_size=canonical["prefill_cp"],
            decode_context_parallel_size=canonical["decode_cp"],
            enable_expert_parallel=canonical["enable_expert_parallel"],
            distributed_executor_backend=canonical["distributed_executor_backend"],
            gpu_group_size=canonical["gpu_count"],
        )


@lru_cache(maxsize=4)
def _validate_vllm_cli_support(
    prefill_cp: int,
    decode_cp: int,
    data_parallel_size: int,
    enable_expert_parallel: bool,
) -> None:
    """Fail before model creation when installed vLLM lacks topology flags."""
    if (
        prefill_cp == 1
        and decode_cp == 1
        and data_parallel_size == 1
        and not enable_expert_parallel
    ):
        return
    completed = subprocess.run(
        # Recent vLLM releases intentionally omit advanced ParallelConfig
        # options from the short help and expose them through --help=all.
        ["vllm", "bench", "latency", "--help=all"],
        check=True,
        capture_output=True,
        text=True,
        timeout=60,
    )
    help_text = completed.stdout + completed.stderr
    required = []
    if prefill_cp > 1:
        required.append("--prefill-context-parallel-size")
    if decode_cp > 1:
        required.append("--decode-context-parallel-size")
    if data_parallel_size > 1:
        required.extend(("--data-parallel-size", "--data-parallel-size-local"))
    if enable_expert_parallel:
        required.append("--enable-expert-parallel")
    missing = [flag for flag in required if flag not in help_text]
    if missing:
        raise RuntimeError(
            "installed vLLM does not support requested context-parallel flags: "
            + ", ".join(missing)
        )
