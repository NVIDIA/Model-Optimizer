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

# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import json

import pytest
from PIL import Image

from modelopt.torch.puzzletron.dataset.acquisition import (
    NEMOTRON_VLM_DATASET,
    PUZZLE_KD_DATASET,
    VLM_HEADER_SUBSETS,
    TextAcquisitionSpec,
    VlmAcquisitionSpec,
    largest_remainder_quotas,
    materialize_nemotron_vlm_dataset,
    materialize_puzzle_kd_dataset,
)
from modelopt.torch.puzzletron.dataset.multimodal import (
    load_materialized_conversation_dataset,
    normalize_nemotron_vlm_sample,
)


def _messages(image_name: str, *, answer: str = "answer"):
    return [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image_name},
                {"type": "text", "text": "question"},
            ],
        },
        {"role": "assistant", "content": [{"type": "text", "text": answer}]},
    ]


def test_vlm_acquisition_rejects_invalid_bounds_and_duplicate_subsets(tmp_path):
    with pytest.raises(ValueError, match="num_samples"):
        VlmAcquisitionSpec(output_dir=tmp_path, num_samples=0)
    with pytest.raises(ValueError, match="max_shards_per_subset"):
        VlmAcquisitionSpec(output_dir=tmp_path, max_shards_per_subset=0)
    with pytest.raises(ValueError, match="unique"):
        VlmAcquisitionSpec(output_dir=tmp_path, subsets=("wiki_en", "wiki_en"))


def test_largest_remainder_quotas_are_proportional_exact_and_stable():
    assert largest_remainder_quotas({"small": 10, "large": 30}, 7) == {
        "small": 2,
        "large": 5,
    }
    assert largest_remainder_quotas({"a": 1, "b": 1, "c": 1}, 2) == {
        "a": 1,
        "b": 1,
        "c": 0,
    }


def test_nemotron_normalizer_preserves_full_conversation_and_replaces_image():
    image = Image.new("RGB", (3, 2), color="red")

    normalized = normalize_nemotron_vlm_sample(
        {"id": "row-1", "messages": _messages("asset.png"), "image": image},
        subset="wiki_en",
        revision="abc123",
    )

    assert normalized["source"] == {
        "dataset": NEMOTRON_VLM_DATASET,
        "revision": "abc123",
        "subset": "wiki_en",
        "row_id": "row-1",
    }
    assert normalized["image_count"] == 1
    assert normalized["conversation"][0]["content"][0]["image"] is image
    assert normalized["conversation"][1]["content"][0]["text"] == "answer"


def test_nemotron_materializer_is_bounded_balanced_and_manifested(tmp_path):
    calls = []

    def sample_loader(*, subset, num_samples, seed, max_shards, revision):
        calls.append((subset, num_samples, seed, max_shards, revision))
        for index in range(3):
            yield {
                "id": f"{subset}-{index}",
                "messages": _messages(f"{subset}-{index}.png"),
                "image": Image.new("RGB", (2, 2), color=(index, 0, 0)),
            }

    spec = VlmAcquisitionSpec(
        output_dir=tmp_path,
        subsets=("sparsetables", "plotqa_cot"),
        num_samples=3,
        seed=17,
        max_shards_per_subset=2,
        revision="resolved-sha",
    )
    manifest = materialize_nemotron_vlm_dataset(spec, sample_loader=sample_loader)
    dataset = load_materialized_conversation_dataset(tmp_path)

    assert manifest["sample_count"] == 3
    assert manifest["acquisition"]["source"] == NEMOTRON_VLM_DATASET
    assert manifest["acquisition"]["subsets"] == ["sparsetables", "plotqa_cot"]
    assert [dataset[index]["source"]["row_id"] for index in range(3)] == [
        "sparsetables-0",
        "plotqa_cot-0",
        "sparsetables-1",
    ]
    assert calls == [
        ("sparsetables", 3, 17, 2, "resolved-sha"),
        ("plotqa_cot", 3, 18, 2, "resolved-sha"),
    ]


def test_nemotron_materializer_redistributes_exhausted_subset_quota(tmp_path):
    def sample_loader(*, subset, **kwargs):
        del kwargs
        count = 1 if subset == "small" else 8
        for index in range(count):
            yield {
                "id": f"{subset}-{index}",
                "messages": _messages(f"{subset}-{index}.png"),
                "image": Image.new("RGB", (2, 2), color=(index, 0, 0)),
            }

    manifest = materialize_nemotron_vlm_dataset(
        VlmAcquisitionSpec(
            output_dir=tmp_path,
            subsets=("small", "large"),
            subset_rows=(("small", 10), ("large", 30)),
            num_samples=6,
            revision="sha",
        ),
        sample_loader=sample_loader,
    )
    dataset = load_materialized_conversation_dataset(tmp_path)

    assert [dataset[index]["source"]["subset"] for index in range(6)] == [
        "small",
        "large",
        "large",
        "large",
        "large",
        "large",
    ]
    assert manifest["diagnostics"]["requested_quotas"] == {
        "small": 2,
        "large": 4,
    }
    assert manifest["diagnostics"]["materialized_rows"] == {
        "small": 1,
        "large": 5,
    }
    assert manifest["diagnostics"]["redistributed_rows"] == 1


def test_nemotron_materializer_rejects_shortfall_and_cache_mismatch(tmp_path):
    def one_row(**kwargs):
        del kwargs
        yield {
            "id": "only",
            "messages": _messages("only.png"),
            "image": Image.new("RGB", (2, 2)),
        }

    with pytest.raises(RuntimeError, match="only found 1/2"):
        materialize_nemotron_vlm_dataset(
            VlmAcquisitionSpec(
                output_dir=tmp_path,
                subsets=("sparsetables",),
                num_samples=2,
                revision="sha",
            ),
            sample_loader=one_row,
        )

    materialize_nemotron_vlm_dataset(
        VlmAcquisitionSpec(
            output_dir=tmp_path,
            subsets=("sparsetables",),
            num_samples=1,
            revision="sha",
        ),
        sample_loader=one_row,
    )
    with pytest.raises(ValueError, match="does not match"):
        materialize_nemotron_vlm_dataset(
            VlmAcquisitionSpec(
                output_dir=tmp_path,
                subsets=("plotqa_cot",),
                num_samples=1,
                revision="sha",
            ),
            sample_loader=one_row,
        )


def test_existing_vlm_materialization_reuses_pinned_revision_offline(tmp_path):
    def one_row(**kwargs):
        del kwargs
        yield {
            "id": "only",
            "messages": _messages("only.png"),
            "image": Image.new("RGB", (2, 2)),
        }

    materialize_nemotron_vlm_dataset(
        VlmAcquisitionSpec(
            output_dir=tmp_path,
            subsets=("sparsetables",),
            num_samples=1,
            revision="pinned-sha",
        ),
        sample_loader=one_row,
    )

    manifest = materialize_nemotron_vlm_dataset(
        VlmAcquisitionSpec(
            output_dir=tmp_path,
            subsets=("sparsetables",),
            num_samples=1,
        ),
        sample_loader=lambda **kwargs: pytest.fail(f"unexpected download: {kwargs}"),
        revision_resolver=lambda *_args: pytest.fail("unexpected revision lookup"),
    )

    assert manifest["acquisition"]["revision"] == "pinned-sha"


def test_puzzle_kd_materializer_bounds_both_splits(tmp_path):
    calls = []
    rows = {
        "train": [
            {"messages": [{"role": "user", "content": f"train-{index}"}]} for index in range(4)
        ],
        "validation": [
            {"messages": [{"role": "user", "content": f"validation-{index}"}]} for index in range(3)
        ],
    }

    def loader(source, *, split, revision, streaming):
        calls.append((source, split, revision, streaming))
        return rows[split]

    spec = TextAcquisitionSpec(
        output_dir=tmp_path,
        train_samples=3,
        validation_samples=2,
        seed=9,
        revision="text-sha",
    )
    manifest = materialize_puzzle_kd_dataset(spec, dataset_loader=loader)

    from datasets import load_from_disk

    dataset = load_from_disk(tmp_path)
    assert len(dataset["train"]) == 3
    assert len(dataset["validation"]) == 2
    assert manifest["acquisition"]["source"] == PUZZLE_KD_DATASET
    assert calls == [
        (PUZZLE_KD_DATASET, "train", "text-sha", True),
        (PUZZLE_KD_DATASET, "validation", "text-sha", True),
    ]
    assert json.loads((tmp_path / "puzzletron_acquisition.json").read_text()) == manifest


def test_first_class_defaults_are_stable():
    assert VLM_HEADER_SUBSETS == ("sparsetables", "plotqa_cot", "wiki_en")
