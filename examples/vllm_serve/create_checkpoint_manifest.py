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

"""Create a deterministic, no-clobber checkpoint manifest for calibration."""

from __future__ import annotations

import argparse
from pathlib import Path

from modelopt.torch.sparsity.attention_sparsity.calibration.checkpoint_manifest import (
    create_checkpoint_manifest,
)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(allow_abbrev=False)
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("--model-id", required=True)
    args = parser.parse_args(argv)
    try:
        manifest = create_checkpoint_manifest(args.checkpoint, model=args.model_id)
    except (OSError, ValueError) as error:
        parser.error(str(error))
    print(f"[ModelOpt] Wrote {manifest.manifest_path}")
    print(f"CHECKPOINT_MANIFEST_SHA256={manifest.sha256}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
