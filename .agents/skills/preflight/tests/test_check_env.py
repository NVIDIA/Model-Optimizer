"""Tests for the preflight environment presence checker."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

SCRIPT = Path(__file__).parents[1] / "scripts" / "check_env.py"
SPEC = importlib.util.spec_from_file_location("check_env", SCRIPT)
assert SPEC and SPEC.loader
check_env = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(check_env)


def test_check_reports_names_without_values(monkeypatch, capsys):
    secret = "do-not-print-this-value"
    monkeypatch.setenv("PREFLIGHT_SET", secret)
    monkeypatch.delenv("PREFLIGHT_MISSING", raising=False)

    assert check_env.main(["PREFLIGHT_SET", "PREFLIGHT_MISSING"]) == 1

    output = capsys.readouterr().out
    assert output == "PREFLIGHT_SET: set\nPREFLIGHT_MISSING: missing\n"
    assert secret not in output


def test_check_deduplicates_names(monkeypatch, capsys):
    monkeypatch.setenv("PREFLIGHT_SET", "value")

    assert check_env.main(["PREFLIGHT_SET", "PREFLIGHT_SET"]) == 0
    assert capsys.readouterr().out == "PREFLIGHT_SET: set\n"


@pytest.mark.parametrize("name", ["BAD-NAME", "1BAD", "A=B", ""])
def test_check_rejects_invalid_names(name, capsys):
    assert check_env.main([name]) == 2
    assert "Invalid environment variable name" in capsys.readouterr().out
