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

"""MLflow tracking for ``examples/megatron_bridge/quantize.py``.

``mlflow_utils`` deliberately imports no Megatron, so the whole flag-to-artifact path is
exercised here without a GPU, a ``torchrun`` launch, or the mlflow client. ``quantize.py``
itself does need Megatron to import, so the last test guards the seam between the two files
as text.
"""

import argparse
import getpass
import importlib.util
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml

_EXAMPLE_DIR = Path(__file__).resolve().parents[3] / "examples" / "megatron_bridge"
_SCRIPT = _EXAMPLE_DIR / "quantize.py"
_SPEC = importlib.util.spec_from_file_location(
    "megatron_bridge_mlflow_utils", _EXAMPLE_DIR / "mlflow_utils.py"
)
assert _SPEC is not None and _SPEC.loader is not None
mlflow_utils = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(mlflow_utils)

URI = "https://mlflow.example.com"
RECIPE = "general/ptq/nvfp4_default-kv_fp8_cast"


class FakeMlflow:
    """Stand-in for the mlflow module, so these tests need no server and no dependency."""

    def __init__(self):
        self.status = None
        self.texts = {}
        self.artifacts = {}

    def set_tracking_uri(self, uri):
        self.tracking_uri = uri

    def set_experiment(self, name):
        self.experiment = name

    def start_run(self, run_name=None):
        self.run_name = run_name
        return SimpleNamespace(info=SimpleNamespace(experiment_id="7", run_id="deadbeef"))

    def log_params(self, params):
        pass

    def set_tags(self, tags):
        pass

    def log_text(self, text, artifact_file):
        self.texts[artifact_file] = text

    def log_artifact(self, local_path, artifact_path=None):
        self.artifacts[Path(local_path).name] = artifact_path

    def log_metrics(self, metrics):
        pass

    def end_run(self, status=None):
        self.status = status


@pytest.fixture
def fake_mlflow(monkeypatch):
    fake = FakeMlflow()
    monkeypatch.setitem(sys.modules, "mlflow", fake)
    return fake


def _parse(monkeypatch, *argv):
    """The parser side of ``quantize.py``, limited to what the tracking reads off args."""
    monkeypatch.setattr(getpass, "getuser", lambda: "tester")
    parser = argparse.ArgumentParser()
    parser.add_argument("--hf_model_name_or_path", default="/models/Qwen3-0.6B")
    parser.add_argument("--export_megatron_path", default="/tmp/out")
    parser.add_argument("--recipe", default=None)
    parser.add_argument("--quant_cfg", default=None)
    parser.add_argument("--tp_size", type=int, default=1)
    mlflow_utils.add_mlflow_args(parser)

    args = parser.parse_args(list(argv))
    mlflow_utils.resolve_mlflow_args(args, parser)
    args.checkpoint_exported = False
    return args


def _tracked(monkeypatch, export_path, *extra):
    return _parse(monkeypatch, "--export_megatron_path", str(export_path), "--mlflow", URI, *extra)


# --- flags ----------------------------------------------------------------------------


def test_tracking_is_off_by_default(monkeypatch):
    args = _parse(monkeypatch)

    assert args.mlflow is None
    assert args.mlflow_experiment is None
    assert not args.mlflow_required


def test_the_flag_defaults_the_experiment_name(monkeypatch):
    args = _parse(monkeypatch, "--recipe", RECIPE, "--mlflow", f"{URI}/")

    assert args.mlflow == URI  # trailing slash stripped
    assert args.mlflow_experiment == (
        "tester/megatron_bridge_quantize/Qwen3-0.6B-nvfp4_default-kv_fp8_cast"
    )
    assert args.mlflow_run_name is None


def test_the_experiment_falls_back_to_quant_cfg_without_a_recipe(monkeypatch):
    args = _parse(monkeypatch, "--quant_cfg", "nvfp4", "--mlflow", URI)

    assert args.mlflow_experiment == "tester/megatron_bridge_quantize/Qwen3-0.6B-nvfp4"


@pytest.mark.parametrize("sep", ["-", "_"])
def test_multiword_flags_accept_both_spellings(monkeypatch, sep):
    args = _parse(
        monkeypatch,
        "--mlflow",
        URI,
        f"--mlflow{sep}experiment",
        "team/sweep",
        f"--mlflow{sep}run{sep}name",
        "calib-512",
    )

    assert args.mlflow_experiment == "team/sweep"
    assert args.mlflow_run_name == "calib-512"


def test_the_environment_alone_enables_tracking(monkeypatch):
    """MLFLOW_TRACKING_URI is MLflow's own variable, so exporting it opts in on its own."""
    monkeypatch.setenv("MLFLOW_TRACKING_URI", f"{URI}/")

    args = _parse(monkeypatch, "--quant_cfg", "nvfp4")

    assert args.mlflow == URI
    assert not args.mlflow_required  # ... but it was not an explicit request


def test_a_bad_uri_is_fatal_only_when_it_was_asked_for(monkeypatch):
    """The variable is commonly exported for other tooling, so it must not fail a run that
    never asked to be tracked -- unlike an explicit --mlflow."""
    monkeypatch.setenv("MLFLOW_TRACKING_URI", "file:///local/mlruns")

    with pytest.warns(UserWarning, match="continuing untracked"):
        assert _parse(monkeypatch).mlflow is None

    with pytest.raises(SystemExit):
        _parse(monkeypatch, "--mlflow", "file:///local/mlruns")


# --- what a run records ---------------------------------------------------------------


def test_params_track_every_cli_argument(monkeypatch):
    """Params are derived from the parsed args, so a new flag needs no bookkeeping here."""
    monkeypatch.setattr(mlflow_utils.dist, "size", lambda: 8)
    args = _parse(monkeypatch, "--quant_cfg", "nvfp4", "--mlflow", URI, "--tp_size", "2")

    params, _ = mlflow_utils._run_inputs(args)

    assert params["hf_model_name_or_path"] == "/models/Qwen3-0.6B"
    assert params["tp_size"] == 2
    # Data parallelism is implicit, so the parallelism flags alone do not say how many GPUs ran.
    assert params["world_size"] == 8
    # The tracking settings describe the destination, not the quantization.
    assert not {"mlflow", "mlflow_experiment", "mlflow_run_name", "checkpoint_exported"} & set(
        params
    )
    args.some_future_flag = "future"
    assert mlflow_utils._run_inputs(args)[0]["some_future_flag"] == "future"


def test_run_inputs_carry_the_resolved_recipe(monkeypatch):
    args = _parse(monkeypatch, "--recipe", RECIPE, "--mlflow", URI)

    _, texts = mlflow_utils._run_inputs(args)

    # $imports are expanded, so the artifact stands alone.
    recipe = yaml.safe_load(texts["recipe/resolved_recipe.yaml"])
    assert recipe["metadata"]["recipe_type"] == "ptq"
    assert recipe["quantize"]["quant_cfg"]


def test_run_inputs_omit_the_recipe_when_unused(monkeypatch):
    args = _parse(monkeypatch, "--quant_cfg", "nvfp4", "--mlflow", URI)

    assert mlflow_utils._run_inputs(args)[1] == {}


def test_run_tags_identify_the_produced_checkpoint(monkeypatch, tmp_path):
    """checkpoint_path must name what the run *writes*: the export and any QAD run are
    pointed at the Megatron checkpoint, so tagging the input would never join the two."""
    export = tmp_path / "Qwen3-0.6B-nvfp4-megatron"
    args = _tracked(monkeypatch, export)

    assert mlflow_utils._run_tags(args) == {
        "model": "Qwen3-0.6B",
        "checkpoint_path": str(export),
        "source_checkpoint_path": "/models/Qwen3-0.6B",
    }


def test_the_checkpoint_tag_is_absolute(monkeypatch):
    """A relative --export_megatron_path is useless as a join key."""
    args = _tracked(monkeypatch, "megatron_ckpt")

    assert Path(mlflow_utils._run_tags(args)["checkpoint_path"]).is_absolute()


def test_run_outputs_name_the_summary(monkeypatch):
    args = _tracked(monkeypatch, "/tmp/megatron_ckpt")

    files = mlflow_utils._run_outputs(args)

    assert files["summary/quant_summary.txt"] == Path("/tmp/megatron_ckpt/.quant_summary.txt")


# --- opening and closing the run ------------------------------------------------------


def test_non_master_ranks_do_not_open_a_run(monkeypatch, tmp_path):
    """Under torchrun only the master rank uploads, so the others must not touch the server."""
    monkeypatch.setattr(mlflow_utils.dist, "is_master", lambda: False)
    args = _tracked(monkeypatch, tmp_path)
    calls = []
    monkeypatch.setattr(mlflow_utils, "_run_inputs", lambda a: calls.append(a) or ({}, {}))

    with mlflow_utils.mlflow_run(args):
        args.checkpoint_exported = True

    assert calls == []
    assert not (tmp_path / ".experiment.json").exists()


def test_untracked_runs_do_not_gather_inputs(monkeypatch, tmp_path):
    """Without --mlflow the recipe must not be re-read: it is parsed again in
    get_quant_config, and the extra load prints a second '[load_recipe] loading:' line."""
    args = _parse(monkeypatch, "--recipe", RECIPE, "--export_megatron_path", str(tmp_path))
    calls = []
    monkeypatch.setattr(mlflow_utils, "_run_inputs", lambda a: calls.append(a) or ({}, {}))

    with mlflow_utils.mlflow_run(args):
        pass

    assert calls == []


def test_experiment_json_lands_in_the_checkpoint_and_on_the_server(
    monkeypatch, fake_mlflow, tmp_path
):
    """The tags point run -> checkpoint; this file points checkpoint -> run."""
    args = _tracked(monkeypatch, tmp_path, "--quant_cfg", "nvfp4")

    with mlflow_utils.mlflow_run(args):
        args.checkpoint_exported = True  # stand in for bridge.save_megatron_model

    written = json.loads((tmp_path / ".experiment.json").read_text())
    assert written["experiment_name"] == "tester/megatron_bridge_quantize/Qwen3-0.6B-nvfp4"
    assert written["run_id"] == "deadbeef"
    # Uploaded without the leading dot, and while the run is still open.
    assert json.loads(fake_mlflow.texts["experiment.json"]) == written
    assert fake_mlflow.status == "FINISHED"


def test_a_failed_save_writes_no_pointer_but_still_records_the_run(
    monkeypatch, fake_mlflow, tmp_path
):
    """The checkpoint was never written, so nothing on disk may claim this run produced it."""
    args = _tracked(monkeypatch, tmp_path, "--quant_cfg", "nvfp4")

    with pytest.raises(RuntimeError), mlflow_utils.mlflow_run(args):
        raise RuntimeError("calibration blew up")

    assert not (tmp_path / ".experiment.json").exists()
    assert json.loads(fake_mlflow.texts["experiment.json"])["run_id"] == "deadbeef"
    assert fake_mlflow.status == "FAILED"


def test_an_untracked_export_drops_an_inherited_pointer(monkeypatch, tmp_path):
    """A reused --export_megatron_path would otherwise keep the previous run's pointer,
    which would name a run that did not produce these weights."""
    inherited = tmp_path / ".experiment.json"
    inherited.write_text('{"run_id": "stale"}')
    args = _parse(monkeypatch, "--export_megatron_path", str(tmp_path))

    with mlflow_utils.mlflow_run(args):
        args.checkpoint_exported = True

    assert not inherited.exists()


def test_a_failed_untracked_run_leaves_the_directory_alone(monkeypatch, tmp_path):
    """Nothing was exported, so whatever checkpoint is already there keeps its pointer."""
    inherited = tmp_path / ".experiment.json"
    inherited.write_text('{"run_id": "stale"}')
    args = _parse(monkeypatch, "--export_megatron_path", str(tmp_path))

    with pytest.raises(RuntimeError), mlflow_utils.mlflow_run(args):
        raise RuntimeError("calibration blew up")

    assert inherited.exists()


# --- the seam with quantize.py --------------------------------------------------------


def test_quantize_script_wires_the_tracking():
    """``quantize.py`` needs Megatron to import, so the wiring is checked as text: a renamed
    flag or a dropped call would otherwise only surface in the Megatron example lane."""
    source = _SCRIPT.read_text()

    assert "from mlflow_utils import add_mlflow_args, mlflow_run, resolve_mlflow_args" in source
    assert "add_mlflow_args(parser)" in source
    assert "resolve_mlflow_args(args, parser)" in source
    assert "with mlflow_run(args):" in source
    # The provenance pointer is gated on the save having happened.
    assert "args.checkpoint_exported = False" in source
    assert "args.checkpoint_exported = True" in source
    # Every args attribute the tracking reads is a flag quantize.py registers.
    for flag in ("--hf_model_name_or_path", "--export_megatron_path", "--recipe", "--quant_cfg"):
        assert f'"{flag}"' in source
