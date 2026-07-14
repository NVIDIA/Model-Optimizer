# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Small point-to-point object transport independent of a model runtime."""

from __future__ import annotations

import inspect
from typing import Any

import torch

__all__ = ["distributed_isend_obj", "distributed_recv_obj", "distributed_send_obj"]


def _serialization_kwargs() -> dict:
    parameters = inspect.signature(
        torch.distributed.distributed_c10d._object_to_tensor
    ).parameters
    return {"group": None} if "group" in parameters else {}


def _communication_device(group=None) -> torch.device:
    if not torch.distributed.is_initialized():
        return torch.device("cpu")
    if torch.distributed.get_backend(group) == "nccl":
        return torch.device("cuda", torch.cuda.current_device())
    return torch.device("cpu")


def distributed_isend_obj(obj: Any, dst: int = 0, group=None):
    device = _communication_device(group)
    payload, size = torch.distributed.distributed_c10d._object_to_tensor(
        obj,
        device=device,
        **_serialization_kwargs(),
    )
    return [
        torch.distributed.isend(size, dst, group),
        torch.distributed.isend(payload, dst, group),
    ]


def distributed_send_obj(obj: Any, dst: int = 0, group=None) -> None:
    for work in distributed_isend_obj(obj, dst=dst, group=group):
        if work is not None:
            work.wait()


def distributed_recv_obj(src: int | None = None, group=None) -> Any:
    device = _communication_device(group)
    size = torch.empty(1, dtype=torch.long, device=device)
    torch.distributed.recv(size, src=src, group=group)
    payload = torch.empty(int(size.item()), dtype=torch.uint8, device=device)
    torch.distributed.recv(payload, src=src, group=group)
    return torch.distributed.distributed_c10d._tensor_to_object(
        payload,
        int(size.item()),
        **_serialization_kwargs(),
    )
