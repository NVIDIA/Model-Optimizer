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

"""Unit tests for the sensitivity CLI helpers in ``sensitivity.__main__``.

Covers ``_validate_calibration_dir``, ``_render_ranked_table``, ``_default_output_json``, and
``main()`` -- all fast, no GPU, no real ONNX quantization (``score`` is monkeypatched).
"""

from __future__ import annotations

import importlib
import json
import os

import pytest

cli = importlib.import_module("modelopt.onnx.quantization.sensitivity.__main__")


class TestDefaultOutputJson:
    """``_default_output_json`` derives ``<stem>.sensitivity.json`` next to the ONNX file."""

    def test_absolute_path_input(self, tmp_path):
        onnx_path = str(tmp_path / "model.onnx")
        assert cli._default_output_json(onnx_path) == str(tmp_path / "model.sensitivity.json")

    def test_relative_path_input_uses_absolute_dirname(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        result = cli._default_output_json("nested/model.onnx")
        assert result == str(tmp_path / "nested" / "model.sensitivity.json")


class TestValidateCalibrationDir:
    """``_validate_calibration_dir`` gates the shard-directory loader against runaway sizes."""

    def test_empty_dir_raises(self, tmp_path):
        with pytest.raises(FileNotFoundError, match=r"No \.npz files"):
            cli._validate_calibration_dir(str(tmp_path))

    def test_directory_with_shards_under_limit_passes(self, tmp_path):
        for i in range(3):
            (tmp_path / f"shard_{i}.npz").write_bytes(b"\x00" * 128)
        cli._validate_calibration_dir(str(tmp_path))

    def test_single_shard_over_per_file_cap_raises(self, tmp_path, monkeypatch):
        monkeypatch.setattr(cli, "_CALIB_MAX_SIZE_BYTES", 64)
        (tmp_path / "big.npz").write_bytes(b"\x00" * 128)
        with pytest.raises(ValueError, match="File size validation failed"):
            cli._validate_calibration_dir(str(tmp_path))

    def test_aggregate_over_cap_raises(self, tmp_path, monkeypatch):
        monkeypatch.setattr(cli, "_CALIB_MAX_SIZE_BYTES", 1024)
        monkeypatch.setattr(cli, "_CALIB_DIR_MAX_TOTAL_BYTES", 200)
        for i in range(3):
            (tmp_path / f"shard_{i}.npz").write_bytes(b"\x00" * 128)
        with pytest.raises(ValueError, match="Aggregate calibration directory size"):
            cli._validate_calibration_dir(str(tmp_path))


class TestRenderRankedTable:
    """``_render_ranked_table`` covers header, hidden-count footer, empty scores, and failed."""

    @staticmethod
    def _base_result(**overrides):
        return {
            "target_precision": "int8",
            "metric": "kl_div",
            "granularity": "op_type",
            "scores": {},
            "failed": [],
            **overrides,
        }

    def test_no_scores_no_failed_reports_none_found(self):
        out = cli._render_ranked_table(self._base_result())
        assert "no quantizable targets found" in out

    def test_no_scores_but_all_failed(self):
        out = cli._render_ranked_table(self._base_result(failed=["Conv", "MatMul"]))
        assert "2 target(s) failed to probe" in out

    def test_normal_ranking_with_hidden_zeros(self):
        scores = {"Add": 2.5, "Conv": 0.3, "Softmax": 0.0, "Gemm": 0.0}
        out = cli._render_ranked_table(self._base_result(scores=scores))
        assert "Add" in out and "2.500" in out
        assert "<-- highest impact" in out and "<-- lowest impact" in out
        assert "2 target(s) with score 0.0 hidden" in out
        # By default zero-score rows are NOT rendered.
        assert "Softmax" not in out and "Gemm" not in out

    def test_show_zero_scores_renders_zero_rows_and_no_hidden_footer(self):
        scores = {"Add": 2.5, "Softmax": 0.0}
        out = cli._render_ranked_table(self._base_result(scores=scores), show_zero_scores=True)
        assert "Softmax" in out
        assert "hidden" not in out

    def test_all_zero_scores_reports_special_footer(self):
        scores = {"a": 0.0, "b": 0.0}
        out = cli._render_ranked_table(self._base_result(scores=scores))
        assert "all 2 target(s) scored 0.0" in out

    def test_failed_footer_appended_after_ranking(self):
        scores = {"Add": 1.0}
        out = cli._render_ranked_table(self._base_result(scores=scores, failed=["X", "Y", "Z"]))
        assert "Add" in out
        assert "3 target(s) failed to probe" in out


class TestMain:
    """``main()`` glues arg parsing, path validation, ``score``, JSON emit, and stderr render."""

    @staticmethod
    def _stub_result():
        return {
            "target_precision": "int8",
            "metric": "kl_div",
            "granularity": "op_type",
            "calibration_source": "synthetic",
            "num_calibration_samples": 8,
            "scores": {"Conv": 0.1, "MatMul": 0.4},
            "failed": [],
        }

    def test_synthetic_calibration_happy_path(self, tmp_path, monkeypatch, capsys):
        onnx_path = tmp_path / "m.onnx"
        onnx_path.write_bytes(b"\x00" * 32)
        output_json = tmp_path / "out.json"
        recorded: dict = {}

        def fake_score(**kwargs):
            recorded.update(kwargs)
            return self._stub_result()

        monkeypatch.setattr(cli, "score", fake_score)
        rc = cli.main(
            [
                "--onnx_path",
                str(onnx_path),
                "--num_calib_samples",
                "8",
                "--output_json",
                str(output_json),
            ]
        )
        assert rc == 0
        assert recorded["calibration_data"] is None
        assert recorded["num_synthetic_samples"] == 8
        payload = json.loads(output_json.read_text())
        assert payload["onnx_path"] == os.path.abspath(onnx_path)
        assert payload["scores"] == {"Conv": 0.1, "MatMul": 0.4}
        # Ranked table rendered to stderr.
        assert "MatMul" in capsys.readouterr().err

    def test_default_output_json_used_when_not_provided(self, tmp_path, monkeypatch):
        onnx_path = tmp_path / "m.onnx"
        onnx_path.write_bytes(b"\x00")
        monkeypatch.setattr(cli, "score", lambda **_: self._stub_result())

        rc = cli.main(["--onnx_path", str(onnx_path)])
        assert rc == 0
        assert (tmp_path / "m.sensitivity.json").is_file()

    def test_oversize_onnx_raises_before_score(self, tmp_path, monkeypatch):
        # Cap ONNX at 64 bytes to trip validation without needing GB-sized inputs.
        monkeypatch.setattr(cli, "_ONNX_MAX_SIZE_BYTES", 64)
        onnx_path = tmp_path / "big.onnx"
        onnx_path.write_bytes(b"\x00" * 128)
        called = {"n": 0}

        def fake_score(**_):
            called["n"] += 1
            return self._stub_result()

        monkeypatch.setattr(cli, "score", fake_score)
        with pytest.raises(ValueError, match="File size validation failed"):
            cli.main(["--onnx_path", str(onnx_path)])
        assert called["n"] == 0, "score() must not be called after size validation fails"

    def test_calibration_dir_path_is_validated(self, tmp_path, monkeypatch):
        onnx_path = tmp_path / "m.onnx"
        onnx_path.write_bytes(b"\x00")
        calib_dir = tmp_path / "calib"
        calib_dir.mkdir()
        # Empty dir -> _validate_calibration_dir raises before score() runs.
        monkeypatch.setattr(cli, "score", lambda **_: self._stub_result())
        with pytest.raises(FileNotFoundError, match=r"No \.npz files"):
            cli.main(
                [
                    "--onnx_path",
                    str(onnx_path),
                    "--calibration_data_path",
                    str(calib_dir),
                ]
            )
