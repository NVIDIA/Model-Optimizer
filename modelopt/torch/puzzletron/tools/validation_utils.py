# SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Stable validation-artifact writer shared by AutoModel scoring paths."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from omegaconf import DictConfig, OmegaConf

from modelopt.torch.utils import json_dump

__all__ = ["write_results"]


def write_results(
    output_dir: str | Path,
    result_name: str,
    args: DictConfig,
    payload: dict[str, Any],
) -> None:
    output_path = Path(output_dir) / f"{result_name}.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    results = {
        **payload,
        "args": OmegaConf.to_container(args, resolve=True)
        if isinstance(args, DictConfig)
        else args.__dict__,
    }
    json_dump(results, output_path)
