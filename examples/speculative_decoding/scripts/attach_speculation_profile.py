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

"""Attach a measured ``speculation_profile.json`` to an already-exported checkpoint.

Acceptance can only be measured against a servable checkpoint, so the measurement
necessarily comes *after* export -- that ordering is causal, not a design choice.
What is avoidable is re-exporting to deliver the result: attaching a measured profile
is pure file I/O, so reloading multi-GB weights and rewriting every shard to place one
small JSON beside them is waste.

    export_hf_checkpoint.py --export_path ckpt/     # stub profile written
    specdec_bench/run.py    --save_dir  run/        # measures acceptance
    attach_speculation_profile.py \\
        --export_path ckpt/ --speculation_profile run/speculation_profile.json

The file is validated before it replaces the existing one, so a malformed profile
leaves the checkpoint's current profile intact.
"""

import argparse
import json
import shutil
from pathlib import Path

from modelopt.torch.export.plugins.hf_spec_export import read_speculation_profile


def parse_args():
    parser = argparse.ArgumentParser(
        description="Attach a measured speculation profile to an exported checkpoint."
    )
    parser.add_argument(
        "--export_path",
        type=str,
        required=True,
        help="Exported checkpoint directory to attach the profile to.",
    )
    parser.add_argument(
        "--speculation_profile",
        type=str,
        required=True,
        help="speculation_profile.json produced by examples/specdec_bench.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    export_dir = Path(args.export_path)
    if not export_dir.is_dir():
        raise NotADirectoryError(f"--export_path is not a directory: {export_dir}")

    # Validate first: a malformed profile must not clobber a good one.
    profile = read_speculation_profile(args.speculation_profile)

    target = export_dir / "speculation_profile.json"
    if target.exists():
        backup = target.with_suffix(".json.bak")
        shutil.copy2(target, backup)
        print(f"Existing profile backed up to {backup}")

    with open(target, "w") as f:
        json.dump(profile, f, indent=2)
    measured = profile.get("measured")
    print(f"Wrote {target} (measured={measured})")
    if measured is False:
        print("WARNING: the attached profile is an unmeasured stub.")


if __name__ == "__main__":
    main()
