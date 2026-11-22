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

"""Provider identity for legacy FA4 and accuracy-reference capture manifests."""

from collections.abc import Mapping

__all__ = [
    "capture_deployment_geometry",
    "capture_source_fields",
    "capture_source_provenance",
    "validate_capture_source",
]

_SOURCE_SUFFIXES = (
    "",
    "_commit",
    "_git_tree",
    "_git_archive_sha256",
    "_manifest_path",
    "_manifest_sha256",
    "_directory_count",
    "_file_count",
    "_total_size_bytes",
)


def capture_source_fields(*, reference: bool) -> frozenset[str]:
    """Return the exact source fields for one capture-manifest provider."""
    prefix = "provider_source" if reference else "fa4_source"
    fields = {prefix + suffix for suffix in _SOURCE_SUFFIXES}
    if reference:
        fields.update({"provider_kind", "evidence_level"})
    return frozenset(fields)


def validate_capture_source(raw: Mapping[str, object], *, reference: bool) -> None:
    """Validate source evidence without interpreting a reference run as FA4 parity."""
    prefix = "provider_source" if reference else "fa4_source"
    if reference:
        if raw.get("provider_kind") != "triton-reference":
            raise ValueError("capture manifest provider_kind must be triton-reference")
        if raw.get("evidence_level") != "accuracy-reference-only":
            raise ValueError("reference capture evidence_level must be accuracy-reference-only")
    extended_page = raw.get("capture_manifest_schema_version") in (8, 9, 10, 11)
    if extended_page:
        page = raw.get("observed_kv_page_tokens")
        if not reference or type(page) is not int or page not in (16, 4160):
            raise ValueError("reference capture observed_kv_page_tokens must be 16 or 4160")
    elif "observed_kv_page_tokens" in raw:
        raise ValueError("observed KV pages require reference capture schema 8 or 9")
    for suffix, length in {
        "_commit": 40,
        "_git_tree": 40,
        "_git_archive_sha256": 64,
        "_manifest_sha256": 64,
    }.items():
        field = prefix + suffix
        value = raw.get(field)
        if (
            not isinstance(value, str)
            or len(value) != length
            or any(character not in "0123456789abcdef" for character in value)
        ):
            raise ValueError(f"capture manifest {field} is not canonical hexadecimal evidence")
    for suffix in ("", "_manifest_path"):
        field = prefix + suffix
        if not isinstance(raw.get(field), str) or not raw[field]:
            raise ValueError(f"capture manifest {field} must be a non-empty path")
    for suffix in ("_directory_count", "_file_count", "_total_size_bytes"):
        field = prefix + suffix
        value = raw.get(field)
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise ValueError(f"capture manifest {field} must be an integer >= 0")
    if raw[prefix + "_file_count"] == 0:
        raise ValueError("capture manifest must bind at least one provider source file")


def capture_source_provenance(raw: Mapping[str, object]) -> dict[str, object]:
    """Carry provider identity into a candidate without changing the capture payload."""
    reference = raw.get("capture_manifest_schema_version") in (6, 7, 8, 9, 10, 11)
    validate_capture_source(raw, reference=reference)
    result = {field: raw[field] for field in sorted(capture_source_fields(reference=reference))}
    if raw.get("capture_manifest_schema_version") in (8, 9, 10, 11):
        result["observed_kv_page_tokens"] = raw["observed_kv_page_tokens"]
    if not reference:
        result["provider_kind"] = "fa4"
        result["evidence_level"] = "fa4-calibration-observations"
    return result


def capture_deployment_geometry(
    geometry: Mapping[str, object], source_provenance: Mapping[str, object] | None
) -> Mapping[str, object]:
    """Bind a reference candidate to observed cache pages without changing logical tiles."""
    provider = (source_provenance or {}).get("capture_provider", {})
    if not isinstance(provider, Mapping) or "observed_kv_page_tokens" not in provider:
        return geometry
    page = provider["observed_kv_page_tokens"]
    if (
        provider.get("provider_kind") != "triton-reference"
        or provider.get("evidence_level") != "accuracy-reference-only"
        or type(page) is not int
        or page not in (16, 4160)
    ):
        raise ValueError("nondefault KV pages require explicit reference capture provenance")
    contract = geometry["contract"]
    if not isinstance(contract, Mapping):
        raise ValueError("deployment geometry contract must be an object")
    return {**geometry, "contract": {**contract, "kv_page_tokens": page}}
