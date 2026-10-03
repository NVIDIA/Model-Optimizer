#!/usr/bin/env python3
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

"""Entry point for distillation training.

Usage:
    # Direct
    python run.py --config configs/wan_distillation.yaml

    # With accelerate (FSDP)
    accelerate launch --config_file configs/accelerate/fsdp_wan.yaml \\
        run.py --config configs/wan_distillation.yaml

    # With CLI overrides
    accelerate launch --config_file configs/accelerate/fsdp_wan.yaml \\
        run.py --config configs/wan_distillation.yaml \\
        optimization.learning_rate=1e-5 \\
        distillation.distillation_alpha=0.5
"""

from __future__ import annotations

import argparse
import logging
import sys

from omegaconf import OmegaConf

from .config import TrainerConfig
from .models import get_model_backend
from .trainer import DistillationTrainer

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Unified Distillation Trainer",
        allow_abbrev=False,
    )
    parser.add_argument("--config", type=str, required=True, help="Path to YAML config file.")
    args, overrides = parser.parse_known_args()
    return args, overrides


def _load_config_with_defaults(config_path: str) -> OmegaConf:
    """Load a YAML config, resolving a ``defaults`` base config if present.

    If the YAML contains a top-level ``defaults`` key (a path relative to the
    config file's directory), that base config is loaded first and the current
    config is merged on top.  The ``defaults`` key is removed before merging.

    Example::

        # configs/wan_distillation.yaml
        defaults: default.yaml      # resolved relative to configs/

        model:
          model_name: "wan"
          ...
    """
    from pathlib import Path

    raw = OmegaConf.load(config_path)
    if "defaults" not in raw:
        return raw

    defaults_path = Path(config_path).parent / raw.defaults
    if not defaults_path.exists():
        raise FileNotFoundError(
            f"defaults config not found: {defaults_path} (referenced from {config_path})"
        )
    base = OmegaConf.load(str(defaults_path))

    # Remove the 'defaults' key so it doesn't leak into TrainerConfig
    raw_dict = OmegaConf.to_container(raw)
    raw_dict.pop("defaults")
    override = OmegaConf.create(raw_dict)

    merged = OmegaConf.merge(base, override)
    logger.info(f"Loaded defaults from {defaults_path}")
    return merged


def main():
    args, cli_overrides = parse_args()

    # Load YAML config (with optional defaults inheritance)
    base_config = _load_config_with_defaults(args.config)

    # Apply CLI overrides (dot-notation, e.g. optimization.learning_rate=1e-5)
    if cli_overrides:
        cleaned = [o.lstrip("+") for o in cli_overrides if "=" in o]
        if cleaned:
            cli_config = OmegaConf.from_dotlist(cleaned)
            config = OmegaConf.merge(base_config, cli_config)
            logger.info(f"Applied {len(cleaned)} CLI overrides: {cleaned}")
        else:
            config = base_config
    else:
        config = base_config

    config_dict = OmegaConf.to_container(config, resolve=True)
    trainer_config = TrainerConfig(**config_dict)

    # Resolve model backend
    model_name = trainer_config.model.model_name
    model_variant = trainer_config.model.model_variant
    logger.info(
        f"Using model backend: {model_name}"
        + (f" (variant={model_variant})" if model_variant else "")
    )
    loader, create_adapter, pipeline_cls = get_model_backend(model_name, variant=model_variant)

    adapter = create_adapter()
    pipeline = pipeline_cls() if pipeline_cls is not None else None

    # Create trainer and run
    trainer = DistillationTrainer(
        config=trainer_config,
        model_loader=loader,
        training_adapter=adapter,
        inference_pipeline=pipeline,
    )

    stats = trainer.train()
    logger.info(f"Training stats: {stats}")


if __name__ == "__main__":
    main()
