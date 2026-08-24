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

"""Create an exact FlashAttention source archive and its runtime witness."""

from __future__ import annotations

import argparse

from modelopt.torch.sparsity.attention_sparsity.calibration.source_manifest import (
    create_source_manifest_from_git_archive,
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Create a Git-free runtime witness for one exact FlashAttention commit"
    )
    parser.add_argument("checkout", help="Clean FlashAttention Git checkout")
    parser.add_argument("--expected-commit", required=True, help="Required 40-hex HEAD commit")
    parser.add_argument("--archive-output", required=True, help="New exact git-archive tar path")
    parser.add_argument(
        "--manifest-output", required=True, help="New canonical source-witness JSON path"
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    generated = create_source_manifest_from_git_archive(
        args.checkout,
        expected_commit=args.expected_commit,
        source_kind="flash-attention-4",
        archive_output=args.archive_output,
        manifest_output=args.manifest_output,
    )
    print(f"git_commit={generated.git_commit}")
    print(f"git_tree={generated.git_tree}")
    print(f"git_archive_sha256={generated.git_archive_sha256}")
    print(f"source_manifest_sha256={generated.manifest_sha256}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
