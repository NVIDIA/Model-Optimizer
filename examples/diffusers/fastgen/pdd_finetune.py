# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Build the released-AutoModel Qwen-Image PDD setup owned by ModelOpt."""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import yaml

sys.dont_write_bytecode = True

_THIS_DIR = Path(__file__).resolve().parent
_REPO_ROOT = _THIS_DIR.parents[2]
for path in (_REPO_ROOT, _THIS_DIR):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=_THIS_DIR / "configs" / "pdd_qwen_image.yaml",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    from pdd_recipe import build_pdd_setup, initialize_pdd_distributed, resolve_pdd_recipe_config

    raw = yaml.safe_load(args.config.read_text())
    config = resolve_pdd_recipe_config(raw)
    initialize_pdd_distributed(
        backend="nccl" if config.device.type == "cuda" else "gloo",
        timeout_minutes=60,
    )
    setup = build_pdd_setup(config)
    logging.info(
        "PDD setup complete: lifecycle=%s student_keys=%d AutoModel=%s",
        setup.lifecycle,
        len(setup.checkpoint_keys),
        setup.automodel_snapshot["version"],
    )


if __name__ == "__main__":
    main()
