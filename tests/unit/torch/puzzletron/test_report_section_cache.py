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

from pathlib import Path

import pytest

from modelopt.torch.puzzletron.diagnostics.report_section_cache import (
    ReportSectionCache,
    fingerprint_paths,
    publish_report_transaction,
    stable_digest,
)


def _cache_kwargs(tmp_path: Path, source: Path, builder, **overrides):
    kwargs = {
        "section_id": "replacement",
        "schema_version": 1,
        "extractor_version": 1,
        "sources": fingerprint_paths(tmp_path, (source,), hash_contents=True),
        "config_identity": stable_digest({"granularity": "subblock"}),
        "dependency_identities": {},
        "builder": builder,
    }
    kwargs.update(overrides)
    return kwargs


def test_unchanged_snapshot_is_reused_without_builder_call(tmp_path: Path):
    source = tmp_path / "input.json"
    source.write_text('{"value": 1}\n', encoding="utf-8")
    cache = ReportSectionCache(tmp_path / "report", campaign_identity="campaign-a")
    calls = []

    def build():
        calls.append("build")
        return {"value": 1}, "<p>one</p>", {"count": 1}

    kwargs = _cache_kwargs(tmp_path, source, build)
    first = cache.load_or_build(**kwargs)
    second = cache.load_or_build(**kwargs)

    assert calls == ["build"]
    assert first.cache_hit is False
    assert second.cache_hit is True
    assert second.snapshot.data == {"value": 1}


def test_changed_source_rebuilds_only_requested_section(tmp_path: Path):
    replacement_source = tmp_path / "replacement.json"
    mip_source = tmp_path / "mip.json"
    replacement_source.write_text('{"value": 1}\n', encoding="utf-8")
    mip_source.write_text('{"value": 2}\n', encoding="utf-8")
    cache = ReportSectionCache(tmp_path / "report", campaign_identity="campaign-a")
    calls = {"replacement": 0, "mip": 0}

    def replacement_builder():
        calls["replacement"] += 1
        return {"value": calls["replacement"]}, "<p>replacement</p>", {}

    def mip_builder():
        calls["mip"] += 1
        return {"value": calls["mip"]}, "<p>mip</p>", {}

    replacement_kwargs = _cache_kwargs(tmp_path, replacement_source, replacement_builder)
    mip_kwargs = _cache_kwargs(
        tmp_path,
        mip_source,
        mip_builder,
        section_id="mip",
    )
    cache.load_or_build(**replacement_kwargs)
    cache.load_or_build(**mip_kwargs)

    replacement_source.write_text('{"value": 3}\n', encoding="utf-8")
    replacement_kwargs["sources"] = fingerprint_paths(
        tmp_path, (replacement_source,), hash_contents=True
    )
    replacement = cache.load_or_build(**replacement_kwargs)
    mip = cache.load_or_build(**mip_kwargs)

    assert calls == {"replacement": 2, "mip": 1}
    assert replacement.cache_hit is False
    assert mip.cache_hit is True


def test_extractor_version_invalidates_snapshot(tmp_path: Path):
    source = tmp_path / "input.json"
    source.write_text("{}\n", encoding="utf-8")
    cache = ReportSectionCache(tmp_path / "report", campaign_identity="campaign-a")
    calls = []

    def build():
        calls.append("build")
        return {}, "", {}

    kwargs = _cache_kwargs(tmp_path, source, build)
    first = cache.load_or_build(**kwargs)
    second = cache.load_or_build(**{**kwargs, "extractor_version": 2})

    assert calls == ["build", "build"]
    assert first.snapshot.input_digest != second.snapshot.input_digest
    assert second.cache_hit is False


def test_corrupt_snapshot_is_rebuilt(tmp_path: Path):
    source = tmp_path / "input.json"
    source.write_text("{}\n", encoding="utf-8")
    cache = ReportSectionCache(tmp_path / "report", campaign_identity="campaign-a")
    calls = []

    def build():
        calls.append("build")
        return {"generation": len(calls)}, "<p>body</p>", {}

    kwargs = _cache_kwargs(tmp_path, source, build)
    first = cache.load_or_build(**kwargs)
    first.snapshot_path.write_text("not-json", encoding="utf-8")
    second = cache.load_or_build(**kwargs)

    assert calls == ["build", "build"]
    assert second.cache_hit is False
    assert second.snapshot.data == {"generation": 2}


def test_failed_builder_preserves_prior_snapshot_and_manifest(tmp_path: Path):
    source = tmp_path / "input.json"
    source.write_text('{"value": 1}\n', encoding="utf-8")
    report_dir = tmp_path / "report"
    cache = ReportSectionCache(report_dir, campaign_identity="campaign-a")

    first = cache.load_or_build(
        **_cache_kwargs(tmp_path, source, lambda: ({"value": 1}, "<p>one</p>", {}))
    )
    cache.publish_manifest({"selected": first.snapshot.input_digest})
    html_path = report_dir / "campaign_report.html"
    html_path.write_text("old-html", encoding="utf-8")
    manifest_path = report_dir / "report_manifest.json"
    old_snapshot = first.snapshot_path.read_bytes()
    old_html = html_path.read_bytes()
    old_manifest = manifest_path.read_bytes()

    source.write_text('{"value": 2}\n', encoding="utf-8")

    def fail():
        raise RuntimeError("builder failed")

    with pytest.raises(RuntimeError, match="builder failed"):
        cache.load_or_build(**_cache_kwargs(tmp_path, source, fail))

    def reject(_path: Path) -> None:
        raise RuntimeError("verification failed")

    with pytest.raises(RuntimeError, match="verification failed"):
        publish_report_transaction(
            html_path=html_path,
            html="new-html",
            manifest_path=manifest_path,
            manifest={"selected": "new"},
            verifier=reject,
        )

    assert first.snapshot_path.read_bytes() == old_snapshot
    assert html_path.read_bytes() == old_html
    assert manifest_path.read_bytes() == old_manifest


def test_deleted_partial_source_changes_ledger_digest(tmp_path: Path):
    first = tmp_path / "raw/first.json"
    second = tmp_path / "raw/second.json"
    first.parent.mkdir()
    first.write_text("{}\n", encoding="utf-8")
    second.write_text("{}\n", encoding="utf-8")

    before = fingerprint_paths(tmp_path, (first, second), hash_contents=False)
    second.unlink()
    after = fingerprint_paths(tmp_path, (first,), hash_contents=False)

    assert stable_digest(before) != stable_digest(after)
