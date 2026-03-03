#!/usr/bin/env python3
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


def main():
    args, cli_overrides = parse_args()

    # Load YAML config
    base_config = OmegaConf.load(args.config)

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
    logger.info(f"Using model backend: {model_name}" + (f" (variant={model_variant})" if model_variant else ""))
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
