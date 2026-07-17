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

"""Generate Puzzletron's cached, self-contained campaign report."""

import argparse
import json
from pathlib import Path

from modelopt.torch.puzzletron.diagnostics.campaign_progress_report import (
    generate_campaign_progress_report,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--puzzle-dir", type=Path, required=True)
    parser.add_argument("--model-name", default="Puzzletron model")
    parser.add_argument("--no-cache", action="store_true")
    parser.add_argument("--rebuild-section", action="append", default=[])
    return parser


def main() -> None:
    args = build_parser().parse_args()
    result = generate_campaign_progress_report(
        args.puzzle_dir,
        model_name=args.model_name,
        use_cache=not args.no_cache,
        rebuild_sections=args.rebuild_section,
    )
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
