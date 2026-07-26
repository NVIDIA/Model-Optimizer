# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from types import SimpleNamespace

import pytest

from puzzletron_setup import SetupError
from puzzletron_setup.v2.hf_datasets import (
    HfSubsetCatalog,
    discover_hf_subset_catalog,
    format_subset_choice,
    proportional_subset_weights,
)


class _FakeApi:
    def __init__(self, *, paths=(), card_data=None):
        self.paths = tuple(paths)
        self.card_data = {} if card_data is None else card_data
        self.calls = []

    def dataset_info(self, source, revision=None, files_metadata=False):
        self.calls.append((source, revision, files_metadata))
        return SimpleNamespace(
            sha="immutable-sha",
            card_data=self.card_data,
            siblings=[
                SimpleNamespace(rfilename=path, size=index + 1)
                for index, path in enumerate(self.paths)
            ],
        )


def _sizes(*entries):
    return {
        "size": {
            "configs": [
                {
                    "config": name,
                    "num_rows": rows,
                    "num_bytes_original_files": num_bytes,
                }
                for name, rows, num_bytes in entries
            ]
        }
    }


def test_catalog_merges_dynamic_names_sizes_and_hosted_media():
    api = _FakeApi(
        paths=(
            "small/media/part-000.tar",
            "small/metadata.jsonl",
            "external/metadata.jsonl",
        ),
        card_data={
            "configs": [
                {"config_name": "small", "default": True},
                {"config_name": "external"},
            ]
        },
    )

    catalog = discover_hf_subset_catalog(
        "nvidia/example",
        config_names_loader=lambda source, revision=None: ["small", "external"],
        size_payload_loader=lambda source: _sizes(
            ("small", 10, 1024),
            ("external", 30, 4096),
        ),
        api=api,
        require_hosted_media=True,
    )

    assert catalog.source == "nvidia/example"
    assert catalog.revision == "immutable-sha"
    assert catalog.default_subset == "small"
    assert [item.name for item in catalog.subsets] == ["small", "external"]
    assert catalog.subsets[0].selectable
    assert catalog.subsets[0].disabled_reason is None
    assert not catalog.subsets[1].selectable
    assert catalog.subsets[1].disabled_reason == "external media required"
    assert api.calls == [("nvidia/example", None, True)]


def test_catalog_preserves_all_46_dynamic_configurations():
    names = [f"subset_{index:02d}" for index in range(46)]

    catalog = discover_hf_subset_catalog(
        "nvidia/forty-six",
        config_names_loader=lambda source, revision=None: names,
        size_payload_loader=lambda source: _sizes(
            *((name, index + 1, (index + 1) * 1000) for index, name in enumerate(names))
        ),
        api=_FakeApi(),
    )

    assert len(catalog.subsets) == 46
    assert [item.name for item in catalog.subsets] == names


def test_generic_catalog_does_not_require_nemotron_media_layout():
    catalog = discover_hf_subset_catalog(
        "owner/generic",
        config_names_loader=lambda source, revision=None: ["trainable"],
        size_payload_loader=lambda source: _sizes(("trainable", 5, 2048)),
        api=_FakeApi(paths=("unrelated/data.parquet",)),
        require_hosted_media=False,
    )

    assert catalog.subsets[0].selectable
    assert format_subset_choice(catalog.subsets[0]) == (
        "trainable — 5 rows — 2.00 KiB"
    )


def test_missing_or_zero_size_metadata_disables_subset():
    catalog = discover_hf_subset_catalog(
        "owner/incomplete",
        config_names_loader=lambda source, revision=None: ["missing", "empty"],
        size_payload_loader=lambda source: _sizes(("empty", 0, 64)),
        api=_FakeApi(),
    )

    assert [item.disabled_reason for item in catalog.subsets] == [
        "row count unavailable",
        "subset has no rows",
    ]


def test_catalog_serialization_preserves_immutable_capabilities():
    catalog = discover_hf_subset_catalog(
        "owner/serial",
        revision="requested",
        config_names_loader=lambda source, revision=None: ["hosted", "external"],
        size_payload_loader=lambda source: _sizes(
            ("hosted", 20, 200),
            ("external", 10, 100),
        ),
        api=_FakeApi(paths=("hosted/media/part.tar",)),
        require_hosted_media=True,
    )

    restored = HfSubsetCatalog.from_dict(catalog.to_dict())

    assert restored == catalog
    assert restored.revision == "immutable-sha"
    assert restored.subsets[1].disabled_reason == "external media required"


def test_proportional_weights_follow_rows_and_sum_exactly():
    catalog = discover_hf_subset_catalog(
        "owner/weighted",
        config_names_loader=lambda source, revision=None: ["small", "large"],
        size_payload_loader=lambda source: _sizes(
            ("small", 10, 100),
            ("large", 30, 300),
        ),
        api=_FakeApi(),
    )

    weights = proportional_subset_weights(catalog, ["small", "large"])

    assert weights == {"small": 0.25, "large": 0.75}
    assert sum(weights.values()) == 1.0


@pytest.mark.parametrize(
    ("selected", "message"),
    [
        ([], "at least one"),
        (["small", "small"], "unique"),
        (["unknown"], "unknown"),
        (["external"], "unavailable"),
    ],
)
def test_proportional_weights_reject_invalid_selections(selected, message):
    catalog = discover_hf_subset_catalog(
        "owner/invalid",
        config_names_loader=lambda source, revision=None: ["small", "external"],
        size_payload_loader=lambda source: _sizes(
            ("small", 10, 100),
            ("external", 20, 200),
        ),
        api=_FakeApi(paths=("small/media/part.tar",)),
        require_hosted_media=True,
    )

    with pytest.raises(SetupError, match=message):
        proportional_subset_weights(catalog, selected)
