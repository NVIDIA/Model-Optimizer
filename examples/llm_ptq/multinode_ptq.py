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

"""Multi-node PTQ (Post-Training Quantization) with FSDP2 support."""

import argparse
import copy
import json
import os
import random
import time
import warnings
from collections.abc import Iterator, Mapping
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
from accelerate import Accelerator
from example_utils import build_quant_cfg, get_tokenizer
from tqdm import tqdm
from transformers import AutoModelForCausalLM, PreTrainedTokenizer, PreTrainedTokenizerFast

import modelopt.torch.opt as mto
import modelopt.torch.quantization as mtq
from modelopt.recipe import load_config
from modelopt.torch.export import get_model_type
from modelopt.torch.export.convert_hf_config import convert_hf_quant_config_format
from modelopt.torch.export.unified_export_hf import _export_transformers_checkpoint
from modelopt.torch.opt.config_loader import BUILTIN_CONFIG_ROOT
from modelopt.torch.quantization.config import QuantizeConfig, need_calibration
from modelopt.torch.quantization.utils import patch_fsdp_mp_dtypes
from modelopt.torch.utils.dataset_utils import get_dataset_dataloader, get_supported_datasets

# Constants
RAND_SEED = 1234

# Preset directories under modelopt_recipes/ that back the --qformat and
# --kv_cache_qformat CLI vocabularies. Each ``*.yaml`` file in these directories is
# automatically discovered and exposed as a valid CLI value via _PresetCfgChoices,
# so no code change in this script is required when a YAML is added or removed.
# This is deliberate: every preset YAML is CLI-exposed, there is no separate
# allow-list — the directory listing is the policy.
#
# That said, prefer NOT to add new YAMLs to these preset directories either. The
# long-term direction is to retire --qformat / --kv_cache_qformat entirely in favour
# of --recipe, which accepts a full PTQ recipe (see modelopt_recipes/general/ptq/
# and modelopt/recipe/). New quantization configurations should be authored as
# recipes, not as preset entries.
_QFORMAT_PRESET_DIR = "configs/ptq/presets/model"
_KV_QFORMAT_PRESET_DIR = "configs/ptq/presets/kv"

# Backward-compat short names → canonical preset basename. These aliases predate the
# YAML-driven discovery below and remain accepted so existing scripts keep working.
#
# DO NOT add new entries here. New quantization formats must be exposed via their YAML
# basename under modelopt_recipes/configs/ptq/presets/model/ — the directory listing is
# the canonical CLI vocabulary. This table exists solely to keep pre-existing short
# names (and the scripts/docs that hardcode them) working through deprecation, and
# should only ever shrink.
_QFORMAT_ALIASES: dict[str, str] = {
    "nvfp4_awq": "nvfp4_awq_lite",
}

# Sentinel value for ``--kv_cache_qformat`` meaning "no KV cache quantization".
_KV_NONE = "none"


class _PresetCfgChoices(Mapping[str, dict[str, Any]]):
    """Lazy mapping of qformat names → quant_cfg dicts loaded from preset YAMLs.

    Iterates the YAML files in ``modelopt_recipes/<subdir>/`` to populate the set
    of available qformat names; the supplied ``aliases`` table maps additional
    short names onto canonical preset basenames. Loading happens on first access
    and is memoised so repeated lookups are cheap.
    """

    def __init__(self, subdir: str, aliases: Mapping[str, str] | None = None):
        self._subdir = subdir
        self._aliases: dict[str, str] = dict(aliases or {})
        self._presets: set[str] = set()
        for entry in BUILTIN_CONFIG_ROOT.joinpath(subdir).iterdir():
            name = entry.name
            if name.endswith((".yaml", ".yml")):
                self._presets.add(name.rsplit(".", 1)[0])
        # Aliases that point at non-existent presets would silently fail at access
        # time; surface this at import instead.
        for alias, target in self._aliases.items():
            if target not in self._presets:
                raise ValueError(
                    f"Alias {alias!r} points at preset {target!r} which is not present "
                    f"under modelopt_recipes/{subdir}/."
                )
        self._cache: dict[str, dict[str, Any]] = {}

    def _canonical(self, key: str) -> str | None:
        if key in self._presets:
            return key
        return self._aliases.get(key)

    def __contains__(self, key: object) -> bool:
        return isinstance(key, str) and self._canonical(key) is not None

    def __getitem__(self, key: str) -> dict[str, Any]:
        canon = self._canonical(key)
        if canon is None:
            raise KeyError(key)
        if canon not in self._cache:
            self._cache[canon] = load_config(
                f"{self._subdir}/{canon}", schema_type=QuantizeConfig
            ).model_dump(exclude_unset=True)
        # Deepcopy on retrieval so callers can freely mutate the returned config
        # (append per-model overrides, etc.) without poisoning the cached entry.
        return copy.deepcopy(self._cache[canon])

    def __iter__(self) -> Iterator[str]:
        yield from sorted(self._presets | set(self._aliases))

    def __len__(self) -> int:
        return len(self._presets) + len(self._aliases)


QUANT_CFG_CHOICES: Mapping[str, dict[str, Any]] = _PresetCfgChoices(
    _QFORMAT_PRESET_DIR, _QFORMAT_ALIASES
)
KV_QUANT_CFG_CHOICES: Mapping[str, dict[str, Any]] = _PresetCfgChoices(_KV_QFORMAT_PRESET_DIR)

# Guard against a future ``none.yaml`` (or alias) colliding with the disable sentinel:
# argparse would silently allow both, but the runtime branch on ``!= _KV_NONE`` would
# become ambiguous and the user couldn't reach the real preset.
assert _KV_NONE not in KV_QUANT_CFG_CHOICES, (
    f"_KV_NONE sentinel {_KV_NONE!r} collides with a KV preset; rename the preset."
)


# Enable HuggingFace checkpointing
mto.enable_huggingface_checkpointing()


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Multi-node post-training quantization with FSDP2")

    parser.add_argument(
        "--pyt_ckpt_path",
        required=True,
        help="Path to PyTorch checkpoint",
    )
    parser.add_argument(
        "--qformat",
        default="fp8",
        choices=list(QUANT_CFG_CHOICES),
        help="Quantization format",
    )
    parser.add_argument(
        "--kv_cache_qformat",
        default="fp8",
        choices=[_KV_NONE, *KV_QUANT_CFG_CHOICES],
        help="KV cache quantization format",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=1,
        help="Batch size for calibration",
    )
    parser.add_argument(
        "--calib_size",
        type=str,
        default="512",
        help="Comma-separated list of calibration sizes per dataset",
    )
    parser.add_argument(
        "--dataset",
        help=(
            f"name of a dataset, or a comma separated list of datasets. "
            f"dataset choices are {get_supported_datasets()}"
        ),
        type=str,
        default=None,
    )
    parser.add_argument(
        "--export_path",
        default="exported_model",
        help="Directory to export the quantized model",
    )
    parser.add_argument(
        "--trust_remote_code",
        action="store_true",
        help="Trust remote code for HuggingFace models",
    )
    parser.add_argument("--awq_block_size", default=0, type=int)

    args = parser.parse_args()

    # Parse comma-separated lists
    args.dataset = args.dataset.split(",") if args.dataset else None
    args.calib_size = [int(x) for x in args.calib_size.split(",")]

    return args


def load_and_prepare_model(
    model_path: str,
    calib_dataloader: torch.utils.data.DataLoader,
    accelerator: Accelerator,
    trust_remote_code: bool = False,
) -> tuple[nn.Module, str, list[str], torch.utils.data.DataLoader]:
    """Load model and prepare it for FSDP2 distributed execution.

    Args:
        model_path: Path to the HuggingFace model
        calibration_dataloader: Calibration dataloader to be sharded for calibration
        accelerator: Accelerate's Accelerator instance
        trust_remote_code: Whether to trust remote code

    Returns:
        Tuple of (prepared_model, model_type, original_architectures, calibration_dataloader)
    """
    model = AutoModelForCausalLM.from_pretrained(
        model_path, dtype="auto", trust_remote_code=trust_remote_code
    )
    model.eval()
    model_type = get_model_type(model)
    # Need the original architectures for export
    # FSDP prefix is added to the architectures for FSDP2 wrapped models
    original_architectures = model.config.architectures

    # FSDP2 requires an optimizer to be prepared together with the model
    dummy_optimizer = torch.optim.SGD(model.parameters(), lr=0.0)
    model, _, calibration_dataloader = accelerator.prepare(model, dummy_optimizer, calib_dataloader)

    return model, model_type, original_architectures, calibration_dataloader


def create_calibration_dataloader(
    tokenizer: PreTrainedTokenizer | PreTrainedTokenizerFast,
    dataset_names: list[str],
    calib_sizes: list[int],
    batch_size: int,
) -> torch.utils.data.DataLoader:
    """Create calibration dataloader from dataset.

    Args:
        tokenizer: HuggingFace tokenizer
        dataset_names: List of dataset names (defaults to cnn_dailymail)
        calib_sizes: Number of samples for each dataset
        batch_size: Batch size for calibration

    Returns:
        DataLoader for calibration
    """

    return get_dataset_dataloader(
        dataset_name=dataset_names,
        tokenizer=tokenizer,
        batch_size=batch_size,
        num_samples=calib_sizes,
        device=None,  # Keep data on CPU, calibration loop handles device transfer
        include_labels=False,
    )


def create_fsdp2_calibration_loop(
    model: nn.Module,
    dataloader: torch.utils.data.DataLoader,
    accelerator: Accelerator,
):
    """Create calibration loop compatible with FSDP2.

    For FSDP2, we need to use the outer FSDP-wrapped model instead of
    the parameter passed by mtq.quantize to properly handle DTensor.

    Args:
        model: FSDP2-wrapped model
        dataloader: Calibration dataloader
        accelerator: Accelerator instance for device management

    Returns:
        Calibration function compatible with mtq.quantize
    """

    def calibrate(unwrapped_model):
        """Calibration loop that uses the FSDP-wrapped model."""
        for batch in tqdm(dataloader, desc="Calibrating"):
            if isinstance(batch, dict):
                batch = {
                    k: v.to(accelerator.device) if isinstance(v, torch.Tensor) else v
                    for k, v in batch.items()
                }
            # Use outer model (FSDP-wrapped), not the parameter
            # Important: We should forward pass using the unwrapped model
            # mtq.quantize will unwrap the model & pass to the forward_loop
            model(**batch)

    return calibrate


def export_model(
    model: nn.Module,
    accelerator: Accelerator,
    export_path: str | Path,
    architectures: list[str],
):
    """Export quantized model to HuggingFace format.

    Args:
        model: Quantized model
        accelerator: Accelerator instance for state dict gathering
        export_path: Directory to export model to
    """
    export_dir = Path(export_path)
    export_dir.mkdir(parents=True, exist_ok=True)

    post_state_dict, hf_quant_config = _export_transformers_checkpoint(
        model, torch.bfloat16, accelerator=accelerator
    )

    if accelerator.is_main_process:
        # Save hf_quant_config.json for backward compatibility
        with open(f"{export_dir}/hf_quant_config.json", "w") as file:
            json.dump(hf_quant_config, file, indent=4)

        hf_quant_config = convert_hf_quant_config_format(hf_quant_config)

        # Save model
        model.save_pretrained(export_dir, state_dict=post_state_dict, save_modelopt_state=False)

        original_config = f"{export_dir}/config.json"
        config_data = {}

        with open(original_config) as file:
            config_data = json.load(file)

        config_data["quantization_config"] = hf_quant_config
        # Update config architectures to use original architectures that does not have FSDP prefix
        config_data["architectures"] = architectures

        with open(original_config, "w") as file:
            json.dump(config_data, file, indent=4)


def main(args):
    """Main quantization workflow."""
    # Validate GPU availability
    if not torch.cuda.is_available():
        raise OSError("GPU is required for quantization.")

    # Validate quantization format
    if args.qformat not in QUANT_CFG_CHOICES:
        raise ValueError(
            f"Quantization format {args.qformat} not supported. Choose from: {list(QUANT_CFG_CHOICES)}"
        )

    # Set random seeds
    random.seed(RAND_SEED)
    np.random.seed(RAND_SEED)
    torch.manual_seed(RAND_SEED)

    # Initialize accelerator
    accelerator = Accelerator()

    print(f"Rank: {os.environ.get('RANK', 'Not set')}")
    print(f"World Size: {os.environ.get('WORLD_SIZE', 'Not set')}")
    print(f"Local Rank: {os.environ.get('LOCAL_RANK', 'Not set')}")

    # Load tokenizer
    tokenizer = get_tokenizer(args.pyt_ckpt_path, trust_remote_code=args.trust_remote_code)
    default_padding_side = tokenizer.padding_side
    tokenizer.padding_side = "left"  # Left padding for better calibration

    # Set default dataset if not provided
    if args.dataset is None:
        args.dataset = ["cnn_dailymail", "nemotron-post-training-dataset-v2"]
        warnings.warn(
            "No dataset specified. Defaulting to cnn_dailymail and nemotron-post-training-dataset-v2."
        )
        # Adjust calib_size to match dataset length by extending or truncating as needed
        args.calib_size = (args.calib_size + [args.calib_size[-1]] * len(args.dataset))[
            : len(args.dataset)
        ]

    # Create calibration dataloader with max batch size
    calib_dataloader = create_calibration_dataloader(
        tokenizer=tokenizer,
        dataset_names=args.dataset,
        calib_sizes=args.calib_size,
        batch_size=args.batch_size,
    )

    # Load and prepare model
    model, model_type, original_architectures, calib_dataloader = load_and_prepare_model(
        model_path=args.pyt_ckpt_path,
        calib_dataloader=calib_dataloader,
        accelerator=accelerator,
        trust_remote_code=args.trust_remote_code,
    )

    quant_cfg = QUANT_CFG_CHOICES[args.qformat]

    quant_cfg = build_quant_cfg(
        quant_cfg,
        args.awq_block_size,
    )

    enable_quant_kv_cache = args.kv_cache_qformat != _KV_NONE
    print(f"{'Enable' if enable_quant_kv_cache else 'Disable'} KV cache quantization")

    # Check if any bmm_quantizer is in the quant_cfg. If so, we need to enable the bmm_quantizer.
    if enable_quant_kv_cache:
        quant_cfg = mtq.update_quant_cfg_with_kv_cache_quant(
            quant_cfg,
            KV_QUANT_CFG_CHOICES[args.kv_cache_qformat]["quant_cfg"],
        )

    # Quantize the model
    if accelerator.is_main_process:
        print("Starting quantization...")

    start_time = time.time()

    if need_calibration(quant_cfg):
        calibrate_fn = create_fsdp2_calibration_loop(model, calib_dataloader, accelerator)
    else:
        calibrate_fn = None
        warnings.warn("Dynamic quantization. Calibration skipped.")

    with torch.no_grad():
        model = mtq.quantize(model, quant_cfg, forward_loop=calibrate_fn)

    elapsed = time.time() - start_time

    if accelerator.is_main_process:
        print(f"Quantization completed in {elapsed:.2f}s")
        mtq.print_quant_summary(model)

    start_time = time.time()
    export_model(model, accelerator, args.export_path, original_architectures)
    elapsed = time.time() - start_time

    if accelerator.is_main_process:
        # Restore default padding and export the tokenizer as well.
        if tokenizer is not None:
            tokenizer.padding_side = default_padding_side
            tokenizer.save_pretrained(args.export_path)
        # Export the model
        print(f"Export completed in {elapsed:.2f}s")
        print(f"Model exported to {args.export_path}")

    print("Unpatching FSDP2 MP dtypes")


if __name__ == "__main__":
    args = parse_args()
    # This context manager can be removed once the update to FSDP2 function is reflected in torch
    with patch_fsdp_mp_dtypes():
        main(args)
