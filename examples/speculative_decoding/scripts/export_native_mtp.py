# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

"""Export a BF16 native-MTP training checkpoint into a DSV4 checkpoint."""

from __future__ import annotations

import argparse

import torch

from modelopt.torch.speculative.mtp import (
    create_native_mtp_boost_model,
    export_native_mtp_checkpoint,
    load_native_mtp_boost_checkpoint,
)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Export a trained native MTP into a DSV4-compatible checkpoint directory."
    )
    parser.add_argument("--base-model", required=True, help="Original native DSV4 checkpoint.")
    parser.add_argument(
        "--training-checkpoint",
        required=True,
        help="Trainer checkpoint directory or mtp_boost.pt written during MTP boost training.",
    )
    parser.add_argument("--output-dir", required=True, help="New native DSV4 checkpoint directory.")
    parser.add_argument("--adapter", default="auto", help="Native MTP adapter (default: auto).")
    parser.add_argument(
        "--device",
        default="cpu",
        help="Device used to materialize the BF16 MTP masters (default: cpu).",
    )
    return parser.parse_args()


def main() -> None:
    """Load saved masters and export them in the original native tensor layouts."""
    args = _parse_args()
    model = create_native_mtp_boost_model(
        args.base_model,
        dtype=torch.bfloat16,
        device=args.device,
        adapter=args.adapter,
    )
    load_native_mtp_boost_checkpoint(model, args.training_checkpoint)
    output = export_native_mtp_checkpoint(
        model,
        args.base_model,
        args.output_dir,
        adapter=args.adapter,
    )
    print(f"Exported native MTP checkpoint to {output}")


if __name__ == "__main__":
    main()
