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

from __future__ import annotations

from types import SimpleNamespace

from puzzletron_setup.v2.bundle import _bundle_readme
from puzzletron_setup.v2.defaults import DefaultsResolver
from puzzletron_setup.v2.hf_datasets import HfSubsetCatalog, HfSubsetInfo
from puzzletron_setup.v2.prompts import InteractiveBackend, PromptChoice, ScriptedBackend
from puzzletron_setup.v2.session import WizardSession
from puzzletron_setup.v2.state import WizardState
from puzzletron_setup.v2.wizard import (
    _CUSTOM_DATA_SOURCE,
    _NEMOTRON_VLM_DATA_SOURCE,
    _PUZZLE_KD_DATA_SOURCE,
    _data_source_choices,
    data_section,
)


class _CapturingBackend(ScriptedBackend):
    def __init__(self, answers):
        super().__init__(answers)
        self.checkbox_calls = []

    def checkbox(self, message, choices, defaults):
        self.checkbox_calls.append((message, tuple(choices), tuple(defaults)))
        return super().checkbox(message, choices, defaults)


def _context(*, multimodal: bool):
    return {
        "model": SimpleNamespace(
            inventory=SimpleNamespace(multimodal=multimodal),
        )
    }


def _catalog(source, entries, *, default=None):
    return HfSubsetCatalog(
        source=source,
        revision="immutable-sha",
        default_subset=default,
        subsets=tuple(
            HfSubsetInfo(
                name=name,
                num_rows=rows,
                num_bytes_original_files=num_bytes,
                selectable=disabled is None,
                disabled_reason=disabled,
            )
            for name, rows, num_bytes, disabled in entries
        ),
    )


def _puzzle_kd_catalog():
    return _catalog(
        _PUZZLE_KD_DATA_SOURCE,
        [("default", 1000, 4096, None)],
        default="default",
    )


def _nemotron_catalog():
    entries = [
        ("sparsetables", 100, 1024, None),
        ("plotqa_cot", 300, 2048, None),
        ("wiki_en", 200, 4096, None),
        ("external", 400, 8192, "external media required"),
    ]
    entries.extend(
        (f"subset_{index:02d}", index + 1, (index + 1) * 1000, None)
        for index in range(42)
    )
    return _catalog(_NEMOTRON_VLM_DATA_SOURCE, entries)


def test_data_choices_include_first_class_sources_and_deduplicate_default():
    resolver = DefaultsResolver(file_defaults={"data": {"source": _PUZZLE_KD_DATA_SOURCE}})

    choices = _data_source_choices(resolver)

    assert [choice.title for choice in choices] == [
        f"Default — {_PUZZLE_KD_DATA_SOURCE}",
        "NVIDIA Nemotron-VLM v2 (image-text)",
        "Custom local path or Hugging Face dataset",
    ]


def test_puzzle_kd_always_asks_bounded_acquisition_questions(tmp_path):
    destination = tmp_path / "already-downloaded"
    destination.mkdir()
    state = WizardState.start(tmp_path / "campaign", defaults_path=None)
    backend = ScriptedBackend(
        [
            _PUZZLE_KD_DATA_SOURCE,
            str(destination),
            11,
            100,
            20,
            "fixed",
            2048,
        ]
    )

    assert data_section(
        WizardSession(state, backend),
        DefaultsResolver(),
        _context(multimodal=False),
        catalog_loader=lambda source, **kwargs: _puzzle_kd_catalog(),
    )

    assert backend.remaining == 0
    assert state.get_field("data.source") == str(destination.resolve())
    assert state.collection("data_acquisition") == {
        "adapter": "puzzle_kd_v2",
        "source": _PUZZLE_KD_DATA_SOURCE,
        "output": str(destination.resolve()),
        "seed": 11,
        "train_samples": 100,
        "validation_samples": 20,
    }


def test_wizard_emits_canonical_padded_varlen_layout(tmp_path):
    state = WizardState.start(tmp_path / "campaign", defaults_path=None)
    backend = ScriptedBackend(
        [
            _PUZZLE_KD_DATA_SOURCE,
            str(tmp_path / "text"),
            11,
            100,
            20,
            "padded_varlen",
            2048,
        ]
    )

    assert data_section(
        WizardSession(state, backend),
        DefaultsResolver(),
        _context(multimodal=False),
        catalog_loader=lambda source, **kwargs: _puzzle_kd_catalog(),
    )

    assert state.get_field("data.layout") == "padded_varlen"


def test_nemotron_vlm_records_subsets_size_seed_and_shard_bound(tmp_path):
    state = WizardState.start(tmp_path / "campaign", defaults_path=None)
    destination = tmp_path / "vlm"
    backend = _CapturingBackend(
        [
            _NEMOTRON_VLM_DATA_SOURCE,
            str(destination),
            19,
            ["sparsetables", "plotqa_cot"],
            64,
            2,
            "packed_varlen",
            4096,
        ]
    )

    assert data_section(
        WizardSession(state, backend),
        DefaultsResolver(),
        _context(multimodal=True),
        catalog_loader=lambda source, **kwargs: _nemotron_catalog(),
    )

    assert state.get_field("data.modality") == "multimodal"
    assert state.collection("data_acquisition")["subsets"] == [
        "sparsetables",
        "plotqa_cot",
    ]
    assert state.collection("data_acquisition")["num_samples"] == 64
    assert state.collection("data_acquisition")["max_shards_per_subset"] == 2
    assert state.collection("data_acquisition")["subset_rows"] == {
        "sparsetables": 100,
        "plotqa_cot": 300,
    }
    assert state.collection("data_acquisition")["subset_weights"] == {
        "sparsetables": 0.25,
        "plotqa_cot": 0.75,
    }
    selection = state.collection("data_subset_selection")
    assert selection["revision"] == "immutable-sha"
    assert selection["subsets"] == [
        {
            "name": "sparsetables",
            "num_rows": 100,
            "num_bytes_original_files": 1024,
            "weight": 0.25,
        },
        {
            "name": "plotqa_cot",
            "num_rows": 300,
            "num_bytes_original_files": 2048,
            "weight": 0.75,
        },
    ]
    assert len(backend.checkbox_calls) == 1
    message, choices, defaults = backend.checkbox_calls[0]
    assert message == "Dataset subsets:"
    assert len(choices) == 46
    assert choices[0].title == "sparsetables — 100 rows — 1.00 KiB"
    assert choices[3].disabled == "external media required"
    assert defaults == ("sparsetables", "plotqa_cot", "wiki_en")
    catalogs = state.collection("hf_dataset_catalogs")
    assert list(catalogs) == [
        f"{_NEMOTRON_VLM_DATA_SOURCE}@immutable-sha",
    ]


def test_generic_hugging_face_dataset_uses_dynamic_subset_checkbox(
    tmp_path,
    monkeypatch,
):
    state = WizardState.start(tmp_path / "campaign", defaults_path=None)
    backend = _CapturingBackend(
        [
            _CUSTOM_DATA_SOURCE,
            "owner/generic",
            "text",
            ["small", "large"],
            "padded_varlen",
            1024,
        ]
    )
    catalog = _catalog(
        "owner/generic",
        [
            ("small", 10, 100, None),
            ("large", 90, 900, None),
        ],
        default="small",
    )
    monkeypatch.setattr(
        "puzzletron_setup.v2.wizard.infer_dataset_modality",
        lambda source: SimpleNamespace(modality="text", evidence="test catalog"),
    )

    assert data_section(
        WizardSession(state, backend),
        DefaultsResolver(),
        _context(multimodal=False),
        catalog_loader=lambda source, **kwargs: catalog,
    )

    assert state.get_field("data.source") == "owner/generic"
    assert state.collection("data_subset_selection")["subsets"] == [
        {
            "name": "small",
            "num_rows": 10,
            "num_bytes_original_files": 100,
            "weight": 0.1,
        },
        {
            "name": "large",
            "num_rows": 90,
            "num_bytes_original_files": 900,
            "weight": 0.9,
        },
    ]
    assert backend.checkbox_calls[0][2] == ("small",)


def test_resume_reuses_the_revision_locked_subset_catalog(tmp_path):
    state = WizardState.start(tmp_path / "campaign", defaults_path=None)
    answers = [
        _NEMOTRON_VLM_DATA_SOURCE,
        str(tmp_path / "vlm"),
        42,
        ["sparsetables"],
        8,
        1,
        "fixed",
        1024,
    ]
    assert data_section(
        WizardSession(state, ScriptedBackend(answers)),
        DefaultsResolver(),
        _context(multimodal=True),
        catalog_loader=lambda source, **kwargs: _nemotron_catalog(),
    )

    resumed = ScriptedBackend([])
    assert data_section(
        WizardSession(state, resumed),
        DefaultsResolver(),
        _context(multimodal=True),
        catalog_loader=lambda source, **kwargs: (_ for _ in ()).throw(
            AssertionError("unexpected Hugging Face lookup")
        ),
    )

    assert resumed.remaining == 0


def test_bundle_readme_emits_bounded_vlm_materialization_command(tmp_path):
    document = _bundle_readme(
        tmp_path / "campaign",
        "/repo",
        {
            "adapter": "nemotron_vlm_v2",
            "source": _NEMOTRON_VLM_DATA_SOURCE,
            "output": "/datasets/vlm",
            "seed": 42,
            "subsets": ["sparsetables", "plotqa_cot"],
            "subset_rows": {"sparsetables": 100, "plotqa_cot": 300},
            "revision": "sha",
            "num_samples": 64,
            "max_shards_per_subset": 2,
        },
    )

    assert "materialize_dataset.py nemotron_vlm_v2" in document
    assert "--subsets sparsetables plotqa_cot" in document
    assert "--subset-rows sparsetables=100 plotqa_cot=300" in document
    assert "--revision sha" in document
    assert "--num-samples 64" in document
    assert "--max-shards-per-subset 2" in document


def test_checkbox_rejects_a_disabled_scripted_selection(tmp_path):
    state = WizardState.start(tmp_path / "campaign", defaults_path=None)
    backend = ScriptedBackend([["hosted", "external"], ["hosted"]])

    selected = WizardSession(state, backend).checkbox(
        "data.subsets",
        "Subsets:",
        [
            PromptChoice("hosted", "hosted"),
            PromptChoice(
                "external",
                "external",
                disabled="external media required",
            ),
        ],
    )

    assert selected == ["hosted"]
    assert backend.remaining == 0


def test_interactive_checkbox_passes_disabled_reason_to_questionary(monkeypatch):
    rendered = []

    class _Question:
        @staticmethod
        def ask():
            return ["hosted"]

    class _Questionary:
        @staticmethod
        def Choice(**kwargs):  # noqa: N802 - mirrors questionary's public constructor
            rendered.append(kwargs)
            return kwargs

        @staticmethod
        def checkbox(message, choices):
            assert message == "Subsets:"
            assert choices == rendered
            return _Question()

    monkeypatch.setattr(
        "puzzletron_setup.v2.prompts._questionary",
        lambda: _Questionary(),
    )

    selected = InteractiveBackend().checkbox(
        "Subsets:",
        [
            PromptChoice("hosted", "hosted"),
            PromptChoice(
                "external",
                "external",
                disabled="external media required",
            ),
        ],
        defaults=("hosted",),
    )

    assert selected == ["hosted"]
    assert rendered[0]["checked"]
    assert rendered[0]["disabled"] is None
    assert rendered[1]["disabled"] == "external media required"
