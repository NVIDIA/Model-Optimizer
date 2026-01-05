#!/usr/bin/env python

"""
Train LTXV models using configuration from YAML files.

This script provides a command-line interface for training LTXV models using
either LoRA fine-tuning or full model fine-tuning. It loads configuration from
a YAML file and passes it to the trainer.

Basic usage:
    python scripts/train.py CONFIG_PATH [--disable-progress-bars]

For multi-GPU/FSDP training, configure and launch via Accelerate:
    accelerate config
    accelerate launch scripts/train.py CONFIG_PATH
"""

from pathlib import Path

import typer
import yaml
from rich.console import Console

from ltxv_trainer.config import LtxvTrainerConfig
from ltxv_trainer.trainer import LtxvTrainer

console = Console()
app = typer.Typer(
    pretty_exceptions_enable=False,
    no_args_is_help=True,
    help="Train LTXV models using configuration from YAML files.",
)


@app.command()
def main(
    config_path: str = typer.Argument(..., help="Path to YAML configuration file"),
    disable_progress_bars: bool = typer.Option(
        False,
        "--disable-progress-bars",
        help="Disable progress bars (useful for multi-process runs)",
    ),
) -> None:
    """Train the model using the provided configuration file."""
    # Load the configuration from the YAML file
    config_path = Path(config_path)
    if not config_path.exists():
        typer.echo(f"Error: Configuration file {config_path} does not exist.")
        raise typer.Exit(code=1)

    with open(config_path, "r") as file:
        config_data = yaml.safe_load(file)

    # Convert the loaded data to the LtxvTrainerConfig object
    try:
        trainer_config = LtxvTrainerConfig(**config_data)
    except Exception as e:
        typer.echo(f"Error: Invalid configuration data: {e}")
        raise typer.Exit(code=1) from e

    # Initialize the training process
    trainer = LtxvTrainer(trainer_config)
    trainer.train(disable_progress_bars=disable_progress_bars)


if __name__ == "__main__":
    app()
