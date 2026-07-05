# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

"""Unit tests for modelopt.torch.opt.config_loader.

These tests exercise the config loading layer directly (path resolution, raw
YAML parsing, ``modelopt-schema`` comments, ``$import`` resolution, and schema
validation) using synthetic schema types injected under the ``modelopt.``
namespace.  Recipe-level integration behaviour is covered separately in
``tests/unit/recipe/test_loader.py``.
"""

import copy
import sys
import types
from pathlib import Path
from typing import Any

import pytest
import yaml
from pydantic import BaseModel, ConfigDict
from typing_extensions import NotRequired, TypedDict

from modelopt.torch.opt.config_loader import (
    _canonical_key,
    _child_schema,
    _find_import_marker,
    _list_element_schema,
    _ListSnippet,
    _load_raw_config,
    _load_raw_config_with_schema,
    _parse_exmy,
    _parse_exmy_num_bits,
    _parse_modelopt_schema,
    _resolve_config_path,
    _resolve_imports,
    _schema_equal,
    _schema_label,
    _schema_type,
    _unwrap_schema_type,
    _validate_modelopt_schema,
    load_config,
)

# ---------------------------------------------------------------------------
# Synthetic schema types, injected under the ``modelopt.`` namespace
# ---------------------------------------------------------------------------

# ``_schema_type`` only resolves import paths under ``modelopt.``, so the test
# schemas are registered in ``sys.modules`` under this synthetic module name.
SCHEMA_MODULE_NAME = "modelopt._test_config_loader_schemas"


class Entry(TypedDict):
    """A list-element schema (mirrors QuantizerCfgEntry-style TypedDicts)."""

    name: str
    enable: NotRequired[bool]
    cfg: NotRequired[dict[str, Any]]


class Numerics(BaseModel):
    """A dict-shaped snippet schema (mirrors QuantizerAttributeConfig)."""

    model_config = ConfigDict(extra="forbid")

    num_bits: int | tuple[int, int] = 8
    axis: int | None = None


class Top(BaseModel):
    """A top-level config schema with a typed list field."""

    model_config = ConfigDict(extra="forbid")

    algorithm: str | None = None
    items: list[Entry] = []


class RawTop(BaseModel):
    """A top-level config schema with an untyped-dict list field."""

    raw_items: list[dict[str, Any]] = []


NUMERICS_SCHEMA = f"# modelopt-schema: {SCHEMA_MODULE_NAME}.Numerics\n"
ENTRY_SCHEMA = f"# modelopt-schema: {SCHEMA_MODULE_NAME}.Entry\n"
ENTRY_LIST_SCHEMA = f"# modelopt-schema: {SCHEMA_MODULE_NAME}.EntryList\n"
TOP_SCHEMA = f"# modelopt-schema: {SCHEMA_MODULE_NAME}.Top\n"
RAW_TOP_SCHEMA = f"# modelopt-schema: {SCHEMA_MODULE_NAME}.RawTop\n"


@pytest.fixture(autouse=True)
def _schema_module():
    """Register the synthetic schema module for ``modelopt-schema`` resolution."""
    module = types.ModuleType(SCHEMA_MODULE_NAME)
    module.Entry = Entry
    module.EntryList = list[Entry]
    module.Numerics = Numerics
    module.Top = Top
    module.RawTop = RawTop
    sys.modules[SCHEMA_MODULE_NAME] = module
    yield
    sys.modules.pop(SCHEMA_MODULE_NAME, None)


def _write(path: Path, text: str) -> Path:
    path.write_text(text, encoding="utf-8")
    return path


def _write_numerics(path: Path, body: str) -> Path:
    return _write(path, NUMERICS_SCHEMA + body)


def _write_entry(path: Path, body: str) -> Path:
    return _write(path, ENTRY_SCHEMA + body)


def _write_entry_list(path: Path, body: str) -> Path:
    return _write(path, ENTRY_LIST_SCHEMA + body)


# ---------------------------------------------------------------------------
# ExMy parsing
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("e4m3", (4, 3)),
        ("E4M3", (4, 3)),
        ("e2m1", (2, 1)),
        ("E10M2", (10, 2)),
    ],
)
def test_parse_exmy_matches(raw, expected):
    """ExMy strings are parsed case-insensitively into (exponent, mantissa) tuples."""
    assert _parse_exmy(raw) == expected


@pytest.mark.parametrize("raw", ["fp8", "e4", "m3", "e4m", "e4m3x", "xe4m3", ""])
def test_parse_exmy_non_matching_returns_string(raw):
    """Strings that are not exactly ExMy are returned unchanged."""
    assert _parse_exmy(raw) == raw


def test_parse_exmy_num_bits_only_in_known_keys():
    """Only ``num_bits`` / ``scale_bits`` string values are converted."""
    obj = {"num_bits": "e4m3", "scale_bits": "e8m0", "name": "e4m3", "other": "e2m1"}
    assert _parse_exmy_num_bits(obj) == {
        "num_bits": (4, 3),
        "scale_bits": (8, 0),
        "name": "e4m3",
        "other": "e2m1",
    }


def test_parse_exmy_num_bits_nested_dicts_and_lists():
    """Conversion recurses through nested dicts and lists."""
    obj = [{"cfg": {"num_bits": "e2m1"}}, {"entries": [{"scale_bits": "e4m3"}]}]
    assert _parse_exmy_num_bits(obj) == [
        {"cfg": {"num_bits": (2, 1)}},
        {"entries": [{"scale_bits": (4, 3)}]},
    ]


def test_parse_exmy_num_bits_non_string_values_untouched():
    """Non-string ``num_bits`` values (ints, None) pass through unchanged."""
    obj = {"num_bits": 8, "scale_bits": None}
    assert _parse_exmy_num_bits(obj) == {"num_bits": 8, "scale_bits": None}


@pytest.mark.parametrize("scalar", [42, None, "e4m3", 1.5])
def test_parse_exmy_num_bits_scalar_passthrough(scalar):
    """Scalars outside any dict/list are returned unchanged (even ExMy strings)."""
    assert _parse_exmy_num_bits(scalar) == scalar


# ---------------------------------------------------------------------------
# Config path resolution
# ---------------------------------------------------------------------------


def test_resolve_config_path_string_with_suffix(tmp_path):
    """A string ending in .yml resolves directly to the filesystem path."""
    cfg = _write(tmp_path / "cfg.yml", "a: 1\n")
    assert _resolve_config_path(str(cfg)) == cfg


def test_resolve_config_path_string_yaml_suffix(tmp_path):
    """A string ending in .yaml resolves directly to the filesystem path."""
    cfg = _write(tmp_path / "cfg.yaml", "a: 1\n")
    assert _resolve_config_path(str(cfg)) == cfg


def test_resolve_config_path_string_without_suffix_probes_yml_then_yaml(tmp_path):
    """Suffix-less strings probe .yml before .yaml."""
    _write(tmp_path / "cfg.yml", "a: 1\n")
    _write(tmp_path / "cfg.yaml", "a: 2\n")
    resolved = _resolve_config_path(str(tmp_path / "cfg"))
    assert isinstance(resolved, Path)
    assert resolved.suffix == ".yml"


def test_resolve_config_path_string_without_suffix_falls_back_to_yaml(tmp_path):
    """Suffix-less strings find a .yaml file when no .yml exists."""
    cfg = _write(tmp_path / "cfg.yaml", "a: 1\n")
    assert _resolve_config_path(str(tmp_path / "cfg")) == cfg


def test_resolve_config_path_path_object(tmp_path):
    """Path inputs with and without suffix resolve to the same file."""
    cfg = _write(tmp_path / "cfg.yml", "a: 1\n")
    assert _resolve_config_path(cfg) == cfg
    assert _resolve_config_path(tmp_path / "cfg") == cfg


def test_resolve_config_path_missing_raises(tmp_path):
    """Missing files raise ValueError listing all probed candidates."""
    with pytest.raises(ValueError, match=r"Cannot find config file .* paths checked"):
        _resolve_config_path(str(tmp_path / "nonexistent"))


def test_resolve_config_path_invalid_type_raises():
    """Non-str/Path/Traversable inputs are rejected."""
    with pytest.raises(ValueError, match="Invalid config file"):
        _resolve_config_path(12345)  # type: ignore[arg-type]


def test_resolve_config_path_builtin_library(tmp_path, monkeypatch):
    """A relative name missing locally resolves into the built-in config library."""
    monkeypatch.chdir(tmp_path)  # ensure no accidental local hit
    resolved = _resolve_config_path("configs/numerics/fp8")
    assert resolved.is_file()
    assert str(resolved).replace("\\", "/").endswith("configs/numerics/fp8.yaml")


def test_resolve_config_path_local_overrides_builtin(tmp_path, monkeypatch):
    """A local file with the same relative path shadows the built-in config."""
    monkeypatch.chdir(tmp_path)
    local = tmp_path / "configs" / "numerics"
    local.mkdir(parents=True)
    _write(local / "fp8.yml", "num_bits: 8\n")
    resolved = _resolve_config_path("configs/numerics/fp8")
    assert _canonical_key(resolved) == str((local / "fp8.yml").resolve())


def test_canonical_key_resolves_path_aliases(tmp_path, monkeypatch):
    """Relative and absolute spellings of the same file share one cycle key."""
    cfg = _write(tmp_path / "cfg.yml", "a: 1\n")
    monkeypatch.chdir(tmp_path)
    assert _canonical_key(Path("cfg.yml")) == _canonical_key(cfg)


# ---------------------------------------------------------------------------
# modelopt-schema comment parsing
# ---------------------------------------------------------------------------


def test_parse_modelopt_schema_absent():
    """Files without a schema comment yield None."""
    assert _parse_modelopt_schema("a: 1\n", "test.yml") is None


def test_parse_modelopt_schema_present():
    """A schema comment in the leading comment block is parsed."""
    text = "# A plain comment.\n# modelopt-schema: modelopt.foo.Bar\na: 1\n"
    assert _parse_modelopt_schema(text, "test.yml") == "modelopt.foo.Bar"


def test_parse_modelopt_schema_blank_lines_tolerated():
    """Blank lines before/between leading comments do not stop the scan."""
    text = "\n\n# comment\n\n# modelopt-schema: modelopt.foo.Bar\na: 1\n"
    assert _parse_modelopt_schema(text, "test.yml") == "modelopt.foo.Bar"


def test_parse_modelopt_schema_indented_comment():
    """Leading whitespace before the comment marker is accepted."""
    text = "  # modelopt-schema: modelopt.foo.Bar\na: 1\n"
    assert _parse_modelopt_schema(text, "test.yml") == "modelopt.foo.Bar"


def test_parse_modelopt_schema_after_content_ignored():
    """Schema comments after the first non-comment line are ignored."""
    text = "a: 1\n# modelopt-schema: modelopt.foo.Bar\n"
    assert _parse_modelopt_schema(text, "test.yml") is None


def test_parse_modelopt_schema_duplicate_raises():
    """Two schema comments in the preamble are rejected."""
    text = "# modelopt-schema: modelopt.a.B\n# modelopt-schema: modelopt.c.D\na: 1\n"
    with pytest.raises(ValueError, match="multiple modelopt-schema comments"):
        _parse_modelopt_schema(text, "test.yml")


# ---------------------------------------------------------------------------
# Raw YAML loading — single and multi-document
# ---------------------------------------------------------------------------


def test_load_raw_config_empty_file(tmp_path):
    """An empty file loads as an empty dict."""
    cfg = _write(tmp_path / "cfg.yml", "")
    assert _load_raw_config(cfg) == {}


def test_load_raw_config_comments_only(tmp_path):
    """A comments-only file loads as an empty dict."""
    cfg = _write(tmp_path / "cfg.yml", "# just a comment\n")
    assert _load_raw_config(cfg) == {}


def test_load_raw_config_dict_with_exmy(tmp_path):
    """A mapping document is returned with ExMy strings converted."""
    cfg = _write(tmp_path / "cfg.yml", "num_bits: e4m3\naxis: 0\n")
    assert _load_raw_config(cfg) == {"num_bits": (4, 3), "axis": 0}


def test_load_raw_config_list_root(tmp_path):
    """A list document is returned as a list (with ExMy conversion applied)."""
    cfg = _write(tmp_path / "cfg.yml", "- name: a\n  num_bits: e2m1\n- name: b\n")
    assert _load_raw_config(cfg) == [{"name": "a", "num_bits": (2, 1)}, {"name": "b"}]


def test_load_raw_config_scalar_root_raises(tmp_path):
    """A scalar root document is rejected."""
    cfg = _write(tmp_path / "cfg.yml", "42\n")
    with pytest.raises(ValueError, match="must contain a YAML mapping or list"):
        _load_raw_config(cfg)


def test_load_raw_config_multi_doc_dict_dict_merges_content_wins(tmp_path):
    """Two mapping documents merge; the second (content) wins on key conflicts."""
    cfg = _write(tmp_path / "cfg.yml", "a: 1\nb: 1\n---\nb: 2\nc: 3\n")
    assert _load_raw_config(cfg) == {"a": 1, "b": 2, "c": 3}


def test_load_raw_config_multi_doc_null_content(tmp_path):
    """A null second document leaves only the header mapping."""
    cfg = _write(tmp_path / "cfg.yml", "a: 1\n---\n")
    assert _load_raw_config(cfg) == {"a": 1}


def test_load_raw_config_multi_doc_list_body_returns_list_snippet(tmp_path):
    """A header + list body produces a _ListSnippet carrying only ``imports``."""
    cfg = _write(
        tmp_path / "cfg.yml",
        "imports:\n  fp8: some/path\nother_header_key: dropped\n---\n- num_bits: e4m3\n",
    )
    raw = _load_raw_config(cfg)
    assert isinstance(raw, _ListSnippet)
    assert raw.imports == {"fp8": "some/path"}
    # Non-imports header keys are dropped; ExMy conversion applies to the body.
    assert raw.content == [{"num_bits": (4, 3)}]


def test_load_raw_config_multi_doc_list_body_without_imports(tmp_path):
    """A header without ``imports`` yields a _ListSnippet with empty imports."""
    cfg = _write(tmp_path / "cfg.yml", "{}\n---\n- name: a\n")
    raw = _load_raw_config(cfg)
    assert isinstance(raw, _ListSnippet)
    assert raw.imports == {}
    assert raw.content == [{"name": "a"}]


def test_load_raw_config_multi_doc_header_not_dict_raises(tmp_path):
    """A non-mapping first document is rejected with a clear message."""
    cfg = _write(tmp_path / "cfg.yml", "- x\n---\n- y\n")
    with pytest.raises(ValueError, match="first YAML document must be a mapping, got list"):
        _load_raw_config(cfg)


def test_load_raw_config_multi_doc_scalar_content_raises(tmp_path):
    """A scalar second document is rejected with a clear message."""
    cfg = _write(tmp_path / "cfg.yml", "a: 1\n---\n42\n")
    with pytest.raises(ValueError, match="second YAML document must be a mapping or list"):
        _load_raw_config(cfg)


def test_load_raw_config_three_documents_raises(tmp_path):
    """More than two YAML documents are rejected."""
    cfg = _write(tmp_path / "cfg.yml", "a: 1\n---\nb: 2\n---\nc: 3\n")
    with pytest.raises(ValueError, match="expected 1 or 2 YAML documents, got 3"):
        _load_raw_config(cfg)


def test_load_raw_config_null_header_silently_drops_list_body(tmp_path):
    """An explicit ``null`` first document silently discards the second document.

    NOTE: This looks like a genuine sharp edge in ``_load_raw_config_with_schema``:
    the ``docs[0] is None`` early return fires before the two-document branch, so
    ``null\\n---\\n<list>`` loads as ``{}`` and the list body is lost without any
    error.  Asserting current behavior; a fix would likely treat a null header
    as an empty mapping (or reject it) instead of dropping the body.
    """
    cfg = _write(tmp_path / "cfg.yml", "null\n---\n- name: a\n")
    assert _load_raw_config(cfg) == {}


def test_load_raw_config_with_schema_captures_schema_and_path(tmp_path):
    """The schema comment and resolved path are surfaced on the raw config."""
    cfg = _write_numerics(tmp_path / "cfg.yml", "num_bits: 8\n")
    raw = _load_raw_config_with_schema(cfg)
    assert raw.schema == f"{SCHEMA_MODULE_NAME}.Numerics"
    assert raw.path == cfg
    assert raw.data == {"num_bits": 8}


def test_load_raw_config_builtin_traversable(tmp_path, monkeypatch):
    """Built-in library configs load through the Traversable path."""
    monkeypatch.chdir(tmp_path)
    raw = _load_raw_config("configs/numerics/fp8")
    assert raw == {"num_bits": (4, 3), "axis": None}


# ---------------------------------------------------------------------------
# Schema path resolution and labelling
# ---------------------------------------------------------------------------


def test_schema_type_resolves_injected_module():
    """Schema paths under ``modelopt.`` resolve through sys.modules."""
    assert _schema_type(f"{SCHEMA_MODULE_NAME}.Numerics") is Numerics
    assert _schema_type(f"{SCHEMA_MODULE_NAME}.EntryList") == list[Entry]


def test_schema_type_outside_modelopt_raises():
    """Schema paths outside ``modelopt.`` are rejected (no arbitrary imports)."""
    with pytest.raises(ValueError, match="schemas must live under 'modelopt\\.'"):
        _schema_type("pydantic.BaseModel")


def test_schema_type_missing_attr_component_raises():
    """A bare ``modelopt.`` prefix without an attribute name is invalid."""
    with pytest.raises(ValueError, match="Invalid modelopt-schema path"):
        _schema_type("modelopt.")


def test_schema_type_unresolvable_attr_raises():
    """A missing attribute on a loaded module reports the schema path."""
    with pytest.raises(ValueError, match="Cannot resolve modelopt-schema"):
        _schema_type(f"{SCHEMA_MODULE_NAME}.DoesNotExist")


def test_schema_type_initializing_module_reports_circular_import(monkeypatch):
    """A module still initializing produces the circular-import diagnostic."""
    name = "modelopt._test_initializing_module"
    module = types.ModuleType(name)
    module.__spec__ = types.SimpleNamespace(_initializing=True)  # type: ignore[assignment]
    monkeypatch.setitem(sys.modules, name, module)
    with pytest.raises(ValueError, match="still being initialized"):
        _schema_type(f"{name}.NotThereYet")


def test_schema_label_variants():
    """Labels prefer the schema path, then qualname, then a placeholder."""
    assert _schema_label(Numerics, "modelopt.x.Y") == "modelopt.x.Y"
    assert _schema_label(Numerics) == "Numerics"
    assert _schema_label(None) == "<untyped>"


# ---------------------------------------------------------------------------
# Typing helpers
# ---------------------------------------------------------------------------


def test_unwrap_schema_type():
    """Optional/NotRequired wrappers are stripped; real unions are preserved."""
    assert _unwrap_schema_type(NotRequired[int]) is int
    assert _unwrap_schema_type(int | None) is int
    assert _unwrap_schema_type(int) is int
    assert _unwrap_schema_type(None) is None
    # A union with more than one non-None member is not collapsed.
    assert _unwrap_schema_type(int | str | None) == (int | str | None)


def test_schema_equal():
    """Structural schema comparison tolerates Optional/NotRequired wrappers."""
    assert _schema_equal(Numerics, Numerics)
    assert _schema_equal(Numerics | None, Numerics)
    assert _schema_equal(list[Entry], list[Entry])
    assert _schema_equal(NotRequired[list[int]], list[int] | None)
    assert _schema_equal(list[int | None], list[int])  # element unwraps to int
    assert not _schema_equal(list[int], list[str])
    assert not _schema_equal(int, list[int])
    assert not _schema_equal(Numerics, Top)


def test_list_element_schema():
    """Element schemas are extracted only from unambiguous typed lists."""
    assert _list_element_schema(list[Entry]) is Entry
    assert _list_element_schema(list[int] | None) is int
    assert _list_element_schema(list) is None
    assert _list_element_schema(list[Any]) is None
    assert _list_element_schema(dict[str, int]) is None
    assert _list_element_schema(None) is None
    # Ambiguous union of two differently-typed lists yields no element schema.
    assert _list_element_schema(list[int] | list[str]) is None


def test_child_schema():
    """Child schemas resolve through BaseModel fields, TypedDicts, and dicts."""
    assert _child_schema(Top, "items") == list[Entry]
    assert _child_schema(Top, "algorithm") is str  # Optional unwrapped
    assert _child_schema(Top, "missing") is None
    assert _child_schema(Entry, "name") is str
    assert _child_schema(Entry, "enable") is bool  # NotRequired unwrapped
    assert _child_schema(dict[str, int], "anything") is int
    assert _child_schema(None, "key") is None
    assert _child_schema(Top, 3) is None  # non-string keys never match fields


# ---------------------------------------------------------------------------
# _validate_modelopt_schema
# ---------------------------------------------------------------------------


def test_validate_modelopt_schema_no_schema_is_noop():
    """Without a schema path or type, validation never fires."""
    _validate_modelopt_schema(None, {"anything": object()}, "test.yml")


def test_validate_modelopt_schema_valid_data_passes():
    """Valid payloads pass against a schema referenced by path."""
    _validate_modelopt_schema(f"{SCHEMA_MODULE_NAME}.Numerics", {"num_bits": 8}, "test.yml")
    _validate_modelopt_schema(f"{SCHEMA_MODULE_NAME}.EntryList", [{"name": "a"}], "test.yml")


def test_validate_modelopt_schema_invalid_data_raises():
    """Invalid payloads raise ValueError naming the file and the schema."""
    with pytest.raises(
        ValueError,
        match=r"Config file bad\.yml does not match modelopt-schema .*Numerics",
    ):
        _validate_modelopt_schema(f"{SCHEMA_MODULE_NAME}.Numerics", {"bogus": 1}, "bad.yml")


def test_validate_modelopt_schema_with_explicit_type():
    """A pre-resolved schema type is honored without re-resolving the path."""
    _validate_modelopt_schema(None, {"num_bits": 8}, "test.yml", schema_type=Numerics)
    with pytest.raises(ValueError, match="does not match modelopt-schema"):
        _validate_modelopt_schema(None, {"bogus": 1}, "test.yml", schema_type=Numerics)


# ---------------------------------------------------------------------------
# load_config — schema selection and validation
# ---------------------------------------------------------------------------


def test_load_config_plain_dict_no_schema(tmp_path):
    """Without any schema, the resolved payload is returned as a raw dict."""
    cfg = _write(tmp_path / "cfg.yml", "a: 1\nnested:\n  num_bits: e4m3\n")
    assert load_config(cfg) == {"a": 1, "nested": {"num_bits": (4, 3)}}


def test_load_config_list_root_no_schema(tmp_path):
    """A list-valued config without schema is returned as a raw list."""
    cfg = _write(tmp_path / "cfg.yml", "- name: a\n- name: b\n")
    assert load_config(cfg) == [{"name": "a"}, {"name": "b"}]


def test_load_config_declared_schema_returns_instance(tmp_path):
    """The ``modelopt-schema`` comment turns the payload into a model instance."""
    cfg = _write_numerics(tmp_path / "cfg.yml", "num_bits: e4m3\naxis: 0\n")
    result = load_config(cfg)
    assert isinstance(result, Numerics)
    assert result.num_bits == (4, 3)
    assert result.axis == 0


def test_load_config_declared_list_schema_validates_list(tmp_path):
    """A declared list[TypedDict] schema validates a list-rooted file."""
    cfg = _write_entry_list(tmp_path / "cfg.yml", "- name: a\n- name: b\n  enable: false\n")
    result = load_config(cfg)
    assert result == [{"name": "a"}, {"name": "b", "enable": False}]


def test_load_config_declared_schema_validation_error_names_file_and_schema(tmp_path):
    """Validation failures name the offending file and the schema path."""
    cfg = _write_numerics(tmp_path / "cfg.yml", "unknown_field: 1\n")
    with pytest.raises(
        ValueError,
        match=rf"Config file .*cfg\.yml does not match modelopt-schema '{SCHEMA_MODULE_NAME}\.Numerics'",
    ):
        load_config(cfg)


def test_load_config_schema_type_param_without_comment(tmp_path):
    """An explicit ``schema_type`` validates files with no schema comment."""
    cfg = _write(tmp_path / "cfg.yml", "num_bits: 8\n")
    result = load_config(cfg, schema_type=Numerics)
    assert isinstance(result, Numerics)
    assert result.num_bits == 8


def test_load_config_schema_type_param_overrides_declared_comment(tmp_path):
    """The ``schema_type`` argument takes precedence over the file's comment."""
    cfg = _write_numerics(tmp_path / "cfg.yml", "num_bits: 8\n")
    # dict[str, Any] wins over the declared Numerics schema: raw dict comes back.
    result = load_config(cfg, schema_type=dict[str, Any])
    assert result == {"num_bits": 8}
    assert not isinstance(result, Numerics)


def test_load_config_wrong_type_rejected(tmp_path):
    """Type errors inside fields surface as schema validation failures."""
    cfg = _write_numerics(tmp_path / "cfg.yml", "axis: not_an_int\n")
    with pytest.raises(ValueError, match="does not match modelopt-schema"):
        load_config(cfg)


def test_load_config_empty_file_with_schema_uses_defaults(tmp_path):
    """An empty file validates to a schema instance with all defaults."""
    cfg = _write(tmp_path / "cfg.yml", "")
    result = load_config(cfg, schema_type=Top)
    assert isinstance(result, Top)
    assert result.algorithm is None
    assert result.items == []


def test_load_config_missing_file_raises(tmp_path):
    """Missing config files raise a helpful ValueError."""
    with pytest.raises(ValueError, match="Cannot find config file"):
        load_config(str(tmp_path / "nope"))


# ---------------------------------------------------------------------------
# Import resolution — dict merging and precedence
# ---------------------------------------------------------------------------


def test_import_replaces_dict_value(tmp_path):
    """A sole ``$import`` key in a dict value is replaced by the snippet."""
    snippet = _write_numerics(tmp_path / "num.yml", "num_bits: e4m3\naxis: 0\n")
    cfg = _write(tmp_path / "cfg.yml", f"imports:\n  num: '{snippet}'\nslot:\n  $import: num\n")
    assert load_config(cfg) == {"slot": {"num_bits": (4, 3), "axis": 0}}


def test_import_inline_keys_extend_snippet(tmp_path):
    """Inline keys not present in the snippet extend the merged dict."""
    snippet = _write_numerics(tmp_path / "num.yml", "num_bits: 8\n")
    cfg = _write(
        tmp_path / "cfg.yml",
        f"imports:\n  num: '{snippet}'\nslot:\n  $import: num\n  axis: 1\n",
    )
    assert load_config(cfg) == {"slot": {"num_bits": 8, "axis": 1}}


def test_import_inline_keys_override_snippet(tmp_path):
    """Inline keys take precedence over imported values."""
    snippet = _write_numerics(tmp_path / "num.yml", "num_bits: 8\naxis: 0\n")
    cfg = _write(
        tmp_path / "cfg.yml",
        f"imports:\n  num: '{snippet}'\nslot:\n  $import: num\n  num_bits: 4\n",
    )
    assert load_config(cfg) == {"slot": {"num_bits": 4, "axis": 0}}


def test_import_nested_override_is_shallow_replace(tmp_path):
    """An inline nested dict replaces the imported one wholesale (no deep merge)."""
    snippet = _write_entry(tmp_path / "e.yml", "name: base\ncfg:\n  a: 1\n  b: 2\n")
    cfg = _write(
        tmp_path / "cfg.yml",
        f"imports:\n  e: '{snippet}'\nslot:\n  $import: e\n  cfg:\n    a: 9\n",
    )
    result = load_config(cfg)
    assert result["slot"]["cfg"] == {"a": 9}  # imported b:2 was not deep-merged


def test_import_multi_import_merges_and_later_wins(tmp_path):
    """``$import: [a, b]`` merges snippets; later entries override earlier ones."""
    first = _write_numerics(tmp_path / "a.yml", "num_bits: 8\naxis: 0\n")
    second = _write_numerics(tmp_path / "b.yml", "num_bits: 4\n")
    cfg = _write(
        tmp_path / "cfg.yml",
        f"imports:\n  a: '{first}'\n  b: '{second}'\nslot:\n  $import: [a, b]\n",
    )
    assert load_config(cfg) == {"slot": {"num_bits": 4, "axis": 0}}


def test_import_same_snippet_used_twice(tmp_path):
    """The same import name can be referenced from multiple dict values."""
    snippet = _write_numerics(tmp_path / "num.yml", "num_bits: 8\n")
    cfg = _write(
        tmp_path / "cfg.yml",
        f"imports:\n  num: '{snippet}'\nfirst:\n  $import: num\nsecond:\n  $import: num\n",
    )
    result = load_config(cfg)
    assert result["first"] == result["second"] == {"num_bits": 8}


def test_import_section_is_consumed(tmp_path):
    """The top-level ``imports`` section never appears in the resolved payload."""
    snippet = _write_numerics(tmp_path / "num.yml", "num_bits: 8\n")
    cfg = _write(tmp_path / "cfg.yml", f"imports:\n  num: '{snippet}'\nslot:\n  $import: num\n")
    assert "imports" not in load_config(cfg)


def test_import_null_imports_section_treated_as_absent(tmp_path):
    """An empty/null ``imports:`` key is dropped and the body loads normally."""
    cfg = _write(tmp_path / "cfg.yml", "imports:\nfoo: 1\n")
    assert load_config(cfg) == {"foo": 1}


def test_import_recursive_snippet(tmp_path):
    """Snippets may declare their own imports, resolved before validation."""
    inner = _write_numerics(tmp_path / "inner.yml", "num_bits: e4m3\naxis: 0\n")
    entry = _write_entry(
        tmp_path / "entry.yml",
        f"imports:\n  num: '{inner}'\nname: inner\ncfg:\n  $import: num\n",
    )
    cfg = _write(tmp_path / "cfg.yml", f"imports:\n  e: '{entry}'\nslot:\n  $import: e\n")
    assert load_config(cfg) == {"slot": {"name": "inner", "cfg": {"num_bits": (4, 3), "axis": 0}}}


def test_import_multi_document_list_snippet_loads_directly(tmp_path):
    """A multi-document list snippet resolves its own header imports."""
    inner = _write_numerics(tmp_path / "inner.yml", "num_bits: e4m3\n")
    kv = _write_entry_list(
        tmp_path / "kv.yml",
        f"imports:\n  num: '{inner}'\n---\n- name: kv\n  cfg:\n    $import: num\n",
    )
    assert load_config(kv) == [{"name": "kv", "cfg": {"num_bits": (4, 3)}}]


def test_resolve_imports_does_not_mutate_input(tmp_path):
    """_resolve_imports is pure: the raw input tree is left untouched."""
    snippet = _write_numerics(tmp_path / "num.yml", "num_bits: 8\n")
    data = {"imports": {"num": str(snippet)}, "slot": {"$import": "num", "axis": 1}}
    snapshot = copy.deepcopy(data)
    result = _resolve_imports(data)
    assert result == {"slot": {"num_bits": 8, "axis": 1}}
    assert data == snapshot


# ---------------------------------------------------------------------------
# Import resolution — typed list splicing and appending
# ---------------------------------------------------------------------------


def test_import_list_snippet_splices_into_typed_list(tmp_path):
    """A list-schema snippet is spliced (flattened) into a typed list field."""
    snippet = _write_entry_list(
        tmp_path / "extra.yml",
        "- name: lm_head\n  enable: false\n- name: router\n  enable: false\n",
    )
    cfg = _write(
        tmp_path / "cfg.yml",
        TOP_SCHEMA
        + f"imports:\n  extra: '{snippet}'\n"
        + "algorithm: max\nitems:\n  - name: base\n  - $import: extra\n",
    )
    result = load_config(cfg)
    assert isinstance(result, Top)
    assert result.algorithm == "max"
    assert result.items == [
        {"name": "base"},
        {"name": "lm_head", "enable": False},
        {"name": "router", "enable": False},
    ]


def test_import_element_snippet_appends_to_typed_list(tmp_path):
    """An element-schema snippet is appended as a single list entry."""
    snippet = _write_entry(tmp_path / "one.yml", "name: lm_head\nenable: false\n")
    cfg = _write(tmp_path / "cfg.yml", f"imports:\n  one: '{snippet}'\nitems:\n  - $import: one\n")
    result = load_config(cfg, schema_type=Top)  # schema via param, not comment
    assert result.items == [{"name": "lm_head", "enable": False}]


def test_import_dict_snippet_appends_to_dict_typed_list(tmp_path):
    """A dict snippet appends into ``list[dict[str, Any]]`` fields."""
    snippet = _write_numerics(tmp_path / "num.yml", "num_bits: e4m3\n")
    cfg = _write(
        tmp_path / "cfg.yml",
        RAW_TOP_SCHEMA + f"imports:\n  num: '{snippet}'\nraw_items:\n  - $import: num\n",
    )
    result = load_config(cfg)
    assert isinstance(result, RawTop)
    assert result.raw_items == [{"num_bits": (4, 3)}]


def test_import_wrong_schema_in_typed_list_raises(tmp_path):
    """A snippet matching neither the list nor the element schema is rejected."""
    snippet = _write_numerics(tmp_path / "num.yml", "num_bits: 8\n")
    cfg = _write(
        tmp_path / "cfg.yml",
        TOP_SCHEMA + f"imports:\n  num: '{snippet}'\nitems:\n  - $import: num\n",
    )
    with pytest.raises(ValueError, match=r"expected either .*for splicing or .*for appending"):
        load_config(cfg)


def test_import_bare_list_entry_in_untyped_list_raises(tmp_path):
    """A bare ``$import`` list entry requires a typed list schema."""
    snippet = _write_entry_list(tmp_path / "extra.yml", "- name: a\n")
    cfg = _write(
        tmp_path / "cfg.yml",
        f"imports:\n  extra: '{snippet}'\nthings:\n  - $import: extra\n",
    )
    with pytest.raises(ValueError, match="requires a typed list schema"):
        load_config(cfg)


def test_import_list_snippet_in_dict_value_raises(tmp_path):
    """A list-valued snippet cannot stand in for a dict value."""
    snippet = _write_entry_list(tmp_path / "extra.yml", "- name: a\n")
    cfg = _write(tmp_path / "cfg.yml", f"imports:\n  extra: '{snippet}'\nfoo:\n  $import: extra\n")
    with pytest.raises(ValueError, match="must resolve to a dict, got list"):
        load_config(cfg)


# ---------------------------------------------------------------------------
# Import resolution — error handling and cycles
# ---------------------------------------------------------------------------


def test_import_unknown_reference_lists_available_imports(tmp_path):
    """Unknown ``$import`` names report the declared import names."""
    snippet = _write_numerics(tmp_path / "num.yml", "num_bits: 8\n")
    cfg = _write(tmp_path / "cfg.yml", f"imports:\n  num: '{snippet}'\nslot:\n  $import: nope\n")
    with pytest.raises(
        ValueError, match=r"Unknown \$import reference 'nope' .*Available imports: \['num'\]"
    ):
        load_config(cfg)


def test_import_marker_without_imports_section_raises_with_context(tmp_path):
    """``$import`` without any declared imports reports the tree location."""
    cfg = _write(tmp_path / "cfg.yml", "outer:\n  inner:\n    $import: missing\n")
    with pytest.raises(
        ValueError,
        match=r"Unknown \$import reference 'missing' in root\.outer\.inner\. No imports are declared",
    ):
        load_config(cfg)


def test_import_marker_in_list_without_imports_reports_index(tmp_path):
    """Unresolved list-entry markers report their list index in the context."""
    cfg = _write(tmp_path / "cfg.yml", "items:\n  - $import: missing\n")
    with pytest.raises(ValueError, match=r"root\.items\[0\].*No imports are declared"):
        load_config(cfg)


def test_import_empty_imports_dict_treated_as_no_imports(tmp_path):
    """An explicit empty ``imports: {}`` behaves like no imports at all."""
    cfg = _write(tmp_path / "cfg.yml", "imports: {}\nslot:\n  $import: x\n")
    with pytest.raises(ValueError, match="No imports are declared"):
        load_config(cfg)


def test_import_section_not_a_dict_raises(tmp_path):
    """A list-valued ``imports`` section is rejected."""
    cfg = _write(tmp_path / "cfg.yml", "imports:\n  - some/path\nslot: 1\n")
    with pytest.raises(ValueError, match=r"'imports' must be a dict .*got: list"):
        load_config(cfg)


def test_import_empty_path_raises(tmp_path):
    """A declared import with an empty path is rejected by name."""
    cfg = _write(tmp_path / "cfg.yml", "imports:\n  num:\nslot: 1\n")
    with pytest.raises(ValueError, match="Import 'num' has an empty config path"):
        load_config(cfg)


def test_import_missing_file_raises(tmp_path):
    """A declared import pointing at a nonexistent file fails resolution."""
    cfg = _write(tmp_path / "cfg.yml", f"imports:\n  num: '{tmp_path / 'nope.yml'}'\nslot: 1\n")
    with pytest.raises(ValueError, match="Cannot find config file"):
        load_config(cfg)


def test_import_snippet_without_schema_comment_raises(tmp_path):
    """Imported snippets must declare a modelopt-schema comment."""
    snippet = _write(tmp_path / "num.yml", "num_bits: 8\n")  # no schema comment
    cfg = _write(tmp_path / "cfg.yml", f"imports:\n  num: '{snippet}'\nslot:\n  $import: num\n")
    with pytest.raises(
        ValueError, match=r"Import 'num' .*must reference a snippet with\s+a modelopt-schema"
    ):
        load_config(cfg)


def test_import_snippet_failing_own_schema_raises(tmp_path):
    """Imported snippets are validated against their declared schema."""
    snippet = _write_numerics(tmp_path / "num.yml", "bogus_field: 1\n")
    cfg = _write(tmp_path / "cfg.yml", f"imports:\n  num: '{snippet}'\nslot:\n  $import: num\n")
    with pytest.raises(ValueError, match=r"num\.yml does not match modelopt-schema"):
        load_config(cfg)


def test_import_snippets_are_loaded_eagerly(tmp_path):
    """Declared imports are loaded and validated even when never referenced."""
    snippet = _write_numerics(tmp_path / "num.yml", "bogus_field: 1\n")
    cfg = _write(tmp_path / "cfg.yml", f"imports:\n  num: '{snippet}'\nslot: 1\n")
    with pytest.raises(ValueError, match="does not match modelopt-schema"):
        load_config(cfg)


def test_import_circular_two_files_raises(tmp_path):
    """Mutually-importing snippets are detected as a cycle."""
    a_path = tmp_path / "a.yml"
    b_path = tmp_path / "b.yml"
    _write_numerics(a_path, f"imports:\n  other: '{b_path}'\naxis: 0\n")
    _write_numerics(b_path, f"imports:\n  other: '{a_path}'\naxis: 1\n")
    with pytest.raises(ValueError, match="Circular import detected"):
        load_config(a_path)


def test_import_self_import_raises(tmp_path):
    """A snippet importing itself is detected as a cycle."""
    a_path = tmp_path / "a.yml"
    _write_numerics(a_path, f"imports:\n  me: '{a_path}'\naxis: 0\n")
    with pytest.raises(ValueError, match="Circular import detected"):
        load_config(a_path)


# ---------------------------------------------------------------------------
# _find_import_marker
# ---------------------------------------------------------------------------


def test_find_import_marker_nested():
    """The first unresolved marker is located with a readable context path."""
    obj = {"a": {"b": [{"c": {"$import": "x"}}]}}
    assert _find_import_marker(obj) == ("x", "root.a.b[0].c")


def test_find_import_marker_none_when_clean():
    """Trees without markers return None."""
    assert _find_import_marker({"a": [1, {"b": 2}], "c": "text"}) is None


# ---------------------------------------------------------------------------
# Round-tripping
# ---------------------------------------------------------------------------


def test_resolved_config_round_trips_through_yaml(tmp_path):
    """A resolved config can be dumped and reloaded to the identical payload."""
    snippet = _write_numerics(tmp_path / "num.yml", "num_bits: 8\naxis: 0\n")
    cfg = _write(
        tmp_path / "cfg.yml",
        f"imports:\n  num: '{snippet}'\nalgorithm: max\nslot:\n  $import: num\n",
    )
    resolved = load_config(cfg)
    round_trip = _write(tmp_path / "resolved.yml", yaml.safe_dump(resolved))
    assert load_config(round_trip) == resolved


def test_schema_validated_round_trip(tmp_path):
    """model_dump of a schema-validated config reloads to an equal instance."""
    cfg = _write_numerics(tmp_path / "cfg.yml", "num_bits: e4m3\naxis: 0\n")
    first = load_config(cfg)
    dumped = first.model_dump()
    # ExMy tuples serialize as lists in YAML; the loader accepts both forms.
    dumped["num_bits"] = list(dumped["num_bits"])
    round_trip = _write(tmp_path / "resolved.yml", NUMERICS_SCHEMA + yaml.safe_dump(dumped))
    assert load_config(round_trip) == first
