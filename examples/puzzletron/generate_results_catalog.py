#!/usr/bin/env python3
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

"""Generate the central Puzzletron results catalog from structured result leaves."""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from puzzletron_orchestrator.result_catalog import (  # noqa: E402
    LEGACY_WRAPPER_SCHEMA,
    CatalogSource,
    build_results_catalog,
    render_catalog_yaml,
)

REPORTS_ROOT = REPOSITORY_ROOT / "examples" / "puzzletron" / "reports"


def _repository_path(path: Path) -> str:
    return path.relative_to(REPOSITORY_ROOT).as_posix()


def _summary_path(path: Path, record: dict) -> str | None:
    sibling = path.with_name("summary.md")
    if sibling.is_file():
        return _repository_path(sibling)
    if record.get("schema") == LEGACY_WRAPPER_SCHEMA:
        artifacts = record.get("artifacts")
        if isinstance(artifacts, list):
            for artifact in artifacts:
                if isinstance(artifact, dict) and artifact.get("role") == "legacy_summary_html":
                    value = artifact.get("path")
                    return value if isinstance(value, str) and value else None
    return None


def load_sources(reports_root: Path = REPORTS_ROOT) -> list[CatalogSource]:
    """Load every structured result leaf beneath the reports tree."""

    sources = []
    paths = sorted(reports_root.rglob("result.json")) + sorted(
        reports_root.rglob("result_record.json")
    )
    for path in paths:
        record = json.loads(path.read_text())
        if not isinstance(record, dict):
            raise ValueError(f"result must be a mapping: {path}")
        sources.append(
            CatalogSource(
                _repository_path(path),
                record,
                summary_path=_summary_path(path, record),
            )
        )
    return sources


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=REPORTS_ROOT / "catalog.yaml")
    parser.add_argument(
        "--generated-at",
        default=datetime.now(timezone.utc).isoformat(),
        help="UTC catalog generation time; pass an explicit value for reproducible output.",
    )
    parser.add_argument("--generator-revision", default="modelopt.puzzletron.catalog/v1")
    args = parser.parse_args(argv)
    catalog = build_results_catalog(
        load_sources(),
        generated_at=args.generated_at,
        generator_revision=args.generator_revision,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(render_catalog_yaml(catalog))
    print(args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
