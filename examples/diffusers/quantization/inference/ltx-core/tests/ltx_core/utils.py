"""Shared helpers for model-based tests."""

import os
from pathlib import Path

import pytest

DEFAULT_MODEL_PATH = Path(
    "/models/comfyui_models/checkpoints/ltx-av-step-1932500-interleaved-new-vae.safetensors",
)


def resolve_model_path() -> str:
    """Return the checkpoint path, preferring $MODEL_PATH when provided."""
    model_path = os.getenv("MODEL_PATH")
    if model_path:
        env_path = Path(model_path)
        if not env_path.is_file():
            raise FileNotFoundError(f"MODEL_PATH points to a missing file: {model_path}")
        return str(env_path)

    if DEFAULT_MODEL_PATH.is_file():
        return str(DEFAULT_MODEL_PATH)

    pytest.skip(
        "MODEL_PATH is not set and the default checkpoint is unavailable; skipping test.",
    )
