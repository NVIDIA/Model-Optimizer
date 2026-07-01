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

"""Shared utilities for the distillation example scripts (distill.py, export_to_hf.py).

Provides:
  - MODEL_REGISTRY          — maps short keys to (hf_model_id, anymodel_converter_name)
  - SCRIPT_DIR / DEFAULT_CONFIG_FILE — path constants
  - configure_logging()     — sets up root + per-library log levels
  - _load_hf_config()       — loads HF config.json without weights
  - _get_block_configs()    — retrieves or generates per-layer block_configs
  - _get_model_descriptor() — returns AnyModel ModelDescriptor (or None)
  - _load_bridge()          — loads an HF checkpoint into a Megatron Bridge object
  - _build_provider()       — converts a Bridge to a Megatron model provider
  - add_student_args()      — adds --student / --trust-remote-code to an ArgumentParser
  - run_entrypoint()        — standard __main__ teardown wrapper
"""

from __future__ import annotations

import argparse
import logging
import sys
from collections.abc import Callable
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from megatron.bridge import AutoBridge
    from megatron.bridge.models.model_provider import ModelProviderMixin

# ---------------------------------------------------------------------------
# Model registry
# ---------------------------------------------------------------------------

# Maps --student / --teacher key → (hf_model_id, anymodel_converter_name)
#
#   hf_model_id:        Default load path when --{student,teacher}-checkpoint is omitted.
#   anymodel_converter: Key for ConverterFactory and ModelDescriptorFactory.
#                       Also determines block_configs generation fallback.
MODEL_REGISTRY: dict[str, tuple[str, str]] = {
    "gptoss": ("openai/gpt-oss-20b", "gpt_oss"),
    "nemo2": ("nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16", "nemotron_h_v2"),
    "llama": ("meta-llama/Llama-3.2-3B-Instruct", "llama"),
    "qwen": ("Qwen/Qwen3-8B", "qwen3"),
}

SCRIPT_DIR: Path = Path(__file__).parent.resolve()
DEFAULT_CONFIG_FILE: Path = SCRIPT_DIR / "kd-container-default.yaml"

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------


def configure_logging() -> None:
    """Configure root logger and silence noisy third-party libraries."""
    logging.basicConfig(
        level=logging.DEBUG,
        format="%(asctime)s %(name)s %(levelname)s — %(message)s",
        handlers=[logging.StreamHandler(sys.stdout)],
    )
    for name in ("transformer_engine", "multistorageclient"):
        logging.getLogger(name).setLevel(logging.WARNING)
    for name in ("megatron.bridge", "megatron.core"):
        logging.getLogger(name).setLevel(logging.DEBUG)


_logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Model loading helpers
# ---------------------------------------------------------------------------


def _load_hf_config(load_path: str, trust_remote_code: bool):
    """Load only the HuggingFace config (no weights) from a path or Hub model ID."""
    from transformers import AutoConfig

    _logger.info("Loading HF config from %r", load_path)
    return AutoConfig.from_pretrained(load_path, trust_remote_code=trust_remote_code)


def _get_block_configs(hf_config, converter_name: str) -> list | None:
    """Load or generate block_configs for a model.

    Priority:
        1. ``hf_config.block_configs`` — set by AnyModel when saving (canonical source).
        2. ``ConverterFactory`` — generated from global model config (fallback).
        3. ``None`` — homogeneous model (no per-layer overrides).
    """
    from modelopt.torch.puzzletron.plugins.mbridge import load_block_configs

    return load_block_configs(hf_config, converter_name)


def _get_model_descriptor(converter_name: str):
    """Return the AnyModel ModelDescriptor for the given converter name, or None."""
    try:
        from modelopt.torch.puzzletron.anymodel.model_descriptor import ModelDescriptorFactory

        descriptor = ModelDescriptorFactory.get(converter_name)
        if descriptor is None:
            _logger.warning("No AnyModel descriptor found for converter '%s'", converter_name)
        return descriptor
    except ImportError:
        _logger.warning("ModelOpt AnyModel not installed; cannot obtain model descriptor")
        return None


def _load_bridge(load_path: str, trust_remote_code: bool, descriptor) -> AutoBridge:
    """Load an HF model into a Megatron Bridge object.

    If an AnyModel descriptor is available, the model is loaded inside ``deci_x_patcher``,
    which patches the HF model's ``from_pretrained`` path to correctly construct
    heterogeneous layers (different sub-layer types per slot).  For standard homogeneous
    models the patcher is a no-op, so using it unconditionally is safe.

    For models that rely on ``trust_remote_code`` (e.g. NemotronH), the descriptor's
    ``decoder_layer_cls()`` resolves classes from ``transformers_modules`` at patcher
    entry. We must therefore force-cache those dynamic modules before opening the
    patcher context.
    """
    from megatron.bridge import AutoBridge

    if descriptor is not None:
        from modelopt.torch.puzzletron.anymodel.puzzformer import deci_x_patcher
        from modelopt.torch.puzzletron.tools.checkpoint_utils_hf import force_cache_dynamic_modules

        _logger.info(
            "Loading HF model via deci_x_patcher (descriptor=%s)", type(descriptor).__name__
        )
        config = _load_hf_config(load_path, trust_remote_code)
        force_cache_dynamic_modules(config, load_path, trust_remote_code=trust_remote_code)
        with deci_x_patcher(model_descriptor=descriptor):
            return AutoBridge.from_hf_pretrained(load_path, trust_remote_code=trust_remote_code)
    else:
        _logger.info("Loading HF model without deci_x_patcher (AnyModel not available)")
        return AutoBridge.from_hf_pretrained(load_path, trust_remote_code=trust_remote_code)


def _build_provider(bridge: AutoBridge) -> ModelProviderMixin:
    """Convert a Bridge to a Megatron model provider with weight loading registered."""
    return bridge.to_megatron_provider(load_weights=True)


# ---------------------------------------------------------------------------
# Entrypoint helper
# ---------------------------------------------------------------------------


def run_entrypoint(
    main_fn: Callable[[argparse.Namespace], None],
    parse_fn: Callable[[], argparse.Namespace],
) -> None:
    """Parse args, run *main_fn*, and ensure the process group is torn down."""
    import torch

    args = parse_fn()
    try:
        main_fn(args)
    finally:
        if torch.distributed.is_initialized():
            torch.distributed.destroy_process_group()
