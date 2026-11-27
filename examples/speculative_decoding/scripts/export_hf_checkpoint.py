# SPDX-FileCopyrightText: Copyright (c) 2023-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

"""Export a HF checkpoint (with ModelOpt state) for deployment."""

import argparse

import torch
from transformers import PretrainedConfig

import modelopt.torch.opt as mto
from modelopt.torch.export import export_speculative_decoding
from modelopt.torch.speculative.utils import (
    CONFIG_OVERRIDES_HELP,
    load_vlm_or_llm,
    parse_config_overrides,
)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Export a HF checkpoint (with ModelOpt state) for deployment."
    )
    parser.add_argument("--model_path", type=str, default="Path of the trained checkpoint.")
    parser.add_argument("--trust_remote_code", action="store_true", help="Trust remote code")
    parser.add_argument(
        "--export_path", type=str, default="Destination directory for exported files."
    )
    parser.add_argument(
        "--config_overrides",
        type=str,
        default=None,
        help=CONFIG_OVERRIDES_HELP,
    )
    parser.add_argument(
        "--base_model", help="Original local native checkpoint, required for MTP export."
    )
    parser.add_argument("--device", help="Device for native MTP requantization (default: cpu).")
    return parser.parse_args()


def main():
    """Export a trained draft, dispatching native MTP checkpoints to their adapter."""
    args = parse_args()
    config, _ = PretrainedConfig.get_config_dict(args.model_path)
    if "mtp_boost" in config:
        if not args.base_model:
            raise ValueError(
                "Native MTP export requires --base_model with the original checkpoint."
            )
        if args.config_overrides:
            raise ValueError(
                "Native MTP export uses the original configuration; omit --config_overrides."
            )
        export_speculative_decoding(
            args.model_path,
            export_dir=args.export_path,
            base_model=args.base_model,
            device=args.device or "cpu",
        )
    else:
        if args.base_model or args.device:
            raise ValueError("--base_model and --device apply only to native MTP checkpoints.")
        mto.enable_huggingface_checkpointing()
        model = load_vlm_or_llm(
            args.model_path,
            dtype="auto",
            trust_remote_code=args.trust_remote_code,
            config_overrides=parse_config_overrides(args.config_overrides),
        )
        model.eval()
        with torch.inference_mode():
            export_speculative_decoding(model, export_dir=args.export_path)
    print(f"Exported checkpoint to {args.export_path}")


if __name__ == "__main__":
    main()
