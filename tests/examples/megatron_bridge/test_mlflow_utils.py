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
import importlib.util
import json
import sys
from pathlib import Path

import pytest
import yaml
from _test_utils.mlflow import clean_env  # noqa: F401

from modelopt.torch.utils import mlflow as mlflow_lib
from modelopt.torch.utils.mlflow import masked_args

_EXAMPLE_DIR = Path(__file__).resolve().parents[3] / "examples" / "megatron_bridge"
_SCRIPT = _EXAMPLE_DIR / "quantize.py"


def _load(name: str):
    """Import one file from ``examples/megatron_bridge`` as a module.

    The scripts need Megatron to import, which this lane has: each declares its own ``Tool``
    beside the flags that Tool reads, so the two cannot drift.
    """
    # The scripts import each other by bare name, the way they do when run directly.
    if str(_EXAMPLE_DIR) not in sys.path:
        sys.path.insert(0, str(_EXAMPLE_DIR))
    spec = importlib.util.spec_from_file_location(
        f"megatron_bridge_{name}", _EXAMPLE_DIR / f"{name}.py"
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


mlflow_utils = _load("mlflow_utils")
QUANTIZE = _load("quantize").QUANTIZE

URI = "https://mlflow.example.com"
RECIPE = "general/ptq/nvfp4_default-kv_fp8_cast"


def _parse(*argv):
    """Parse *argv* the way ``quantize.py`` would, limited to what the tracking reads."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--hf_model_name_or_path", default="/models/Qwen3-0.6B")
    parser.add_argument("--export_megatron_path", default="/tmp/out")
    parser.add_argument("--recipe")
    parser.add_argument("--quant_cfg")
    parser.add_argument("--tp_size", type=int, default=1)
    mlflow_lib.add_mlflow_args(parser, QUANTIZE)

    args = parser.parse_args(list(argv))
    mlflow_lib.resolve_mlflow_args(args, parser, QUANTIZE)
    args.checkpoint_exported = False
    return args


def _tracked(export_path, *extra):
    return _parse("--export_megatron_path", str(export_path), "--mlflow", URI, *extra)


# --- flags ----------------------------------------------------------------------------


def test_tracking_is_off_by_default():
    args = _parse()

    assert args.mlflow is None
    assert args.mlflow_experiment is None
    assert not args.mlflow_required


def test_the_environment_alone_enables_tracking(monkeypatch):
    """MLFLOW_TRACKING_URI is MLflow's own variable, so exporting it opts in on its own."""
    monkeypatch.setenv("MLFLOW_TRACKING_URI", f"{URI}/")

    args = _parse("--quant_cfg", "nvfp4")

    assert args.mlflow == URI
    assert not args.mlflow_required  # ... but it was not an explicit request


def test_a_bad_uri_is_fatal_only_when_it_was_asked_for(monkeypatch):
    """The variable is commonly exported for other tooling, so it must not fail a run that
    never asked to be tracked -- unlike an explicit --mlflow."""
    monkeypatch.setenv("MLFLOW_TRACKING_URI", "file:///local/mlruns")

    with pytest.warns(UserWarning, match="continuing untracked"):
        assert _parse().mlflow is None

    with pytest.raises(SystemExit):
        _parse("--mlflow", "file:///local/mlruns")


def test_the_printed_arguments_mask_tracking_credentials(monkeypatch):
    """quantize.py hands the namespace to print_args, which dumps it verbatim; a torchrun
    job log is routinely archived, so the URI's credentials must not reach it."""
    # Fake credentials: TruffleHog flags any scheme://user:pass@host, and this test exists
    # precisely to prove they are masked.
    creds = "https://svc:s3cret@mlflow.example.com"  # trufflehog:ignore
    args = _parse("--mlflow", creds, "--quant_cfg", "nvfp4")

    printed = masked_args(args)

    assert args.mlflow == creds  # the live namespace still reaches the client
    assert printed.mlflow == "https://***@mlflow.example.com"
    assert printed.quant_cfg == "nvfp4"  # every other argument survives the copy


# --- what a run records ---------------------------------------------------------------


def test_params_track_every_cli_argument(monkeypatch):
    """Params are derived from the parsed args, so a new flag needs no bookkeeping here."""
    monkeypatch.setattr(mlflow_utils.dist, "size", lambda: 8)
    args = _parse("--quant_cfg", "nvfp4", "--mlflow", URI, "--tp_size", "2")

    params = mlflow_lib.describe_run(args, QUANTIZE, 8)["params"]

    assert params["hf_model_name_or_path"] == "/models/Qwen3-0.6B"
    assert params["tp_size"] == 2
    # Data parallelism is implicit, so the parallelism flags alone do not say how many GPUs ran.
    assert params["world_size"] == 8
    # The tracking settings describe the destination, not the quantization.
    assert not {"mlflow", "mlflow_experiment", "mlflow_run_name", "checkpoint_exported"} & set(
        params
    )
    args.some_future_flag = "future"
    assert mlflow_lib.describe_run(args, QUANTIZE)["params"]["some_future_flag"] == "future"


def test_run_inputs_carry_the_resolved_recipe(monkeypatch):
    args = _parse("--recipe", RECIPE, "--mlflow", URI)

    texts = mlflow_lib.describe_run(args, QUANTIZE)["texts"]

    # $imports are expanded, so the artifact stands alone.
    recipe = yaml.safe_load(texts["recipe/resolved_recipe.yaml"])
    assert recipe["metadata"]["recipe_type"] == "ptq"
    assert recipe["quantize"]["quant_cfg"]


def test_run_inputs_omit_the_recipe_when_unused(monkeypatch):
    args = _parse("--quant_cfg", "nvfp4", "--mlflow", URI)

    assert mlflow_lib.describe_run(args, QUANTIZE)["texts"] == {}


def test_the_checkpoint_tag_is_absolute(monkeypatch):
    """A relative --export_megatron_path is useless as a join key."""
    args = _tracked("megatron_ckpt")

    assert Path(mlflow_lib.run_tags(args, QUANTIZE)["checkpoint_path"]).is_absolute()


def test_run_outputs_name_the_summary(monkeypatch):
    args = _tracked("/tmp/megatron_ckpt")

    files = QUANTIZE.outputs(args)

    assert files["summary/quant_summary.txt"] == Path("/tmp/megatron_ckpt/.quant_summary.txt")


# --- opening and closing the run ------------------------------------------------------


def test_non_master_ranks_do_not_open_a_run(monkeypatch, tmp_path):
    """Under torchrun only the master rank uploads, so the others must not touch the server."""
    monkeypatch.setattr(mlflow_utils.dist, "is_master", lambda: False)
    args = _tracked(tmp_path)
    calls = []
    # Patched on the library, which is where tracked_run resolves it.
    monkeypatch.setattr(mlflow_lib, "describe_run", lambda a, t, w=1: calls.append(a) or {})

    with mlflow_utils.mlflow_run(args, QUANTIZE):
        args.checkpoint_exported = True

    assert calls == []
    assert not (tmp_path / ".experiment.json").exists()


def test_untracked_runs_do_not_gather_inputs(monkeypatch, tmp_path):
    """Without --mlflow the recipe must not be re-read: it is parsed again in
    get_quant_config, and the extra load prints a second '[load_recipe] loading:' line."""
    args = _parse("--recipe", RECIPE, "--export_megatron_path", str(tmp_path))
    calls = []
    # Patched on the library, which is where tracked_run resolves it.
    monkeypatch.setattr(mlflow_lib, "describe_run", lambda a, t, w=1: calls.append(a) or {})

    with mlflow_utils.mlflow_run(args, QUANTIZE):
        pass

    assert calls == []


def test_experiment_json_lands_in_the_checkpoint_and_on_the_server(
    monkeypatch, fake_mlflow, tmp_path
):
    """The tags point run -> checkpoint; this file points checkpoint -> run."""
    args = _tracked(tmp_path, "--quant_cfg", "nvfp4")

    with mlflow_utils.mlflow_run(args, QUANTIZE):
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
    args = _tracked(tmp_path, "--quant_cfg", "nvfp4")

    with pytest.raises(RuntimeError), mlflow_utils.mlflow_run(args, QUANTIZE):
        raise RuntimeError("calibration blew up")

    assert not (tmp_path / ".experiment.json").exists()
    assert json.loads(fake_mlflow.texts["experiment.json"])["run_id"] == "deadbeef"
    assert fake_mlflow.status == "FAILED"


def test_an_untracked_export_drops_an_inherited_pointer(monkeypatch, tmp_path):
    """A reused --export_megatron_path would otherwise keep the previous run's pointer,
    which would name a run that did not produce these weights."""
    inherited = tmp_path / ".experiment.json"
    inherited.write_text('{"run_id": "stale"}')
    args = _parse("--export_megatron_path", str(tmp_path))

    with mlflow_utils.mlflow_run(args, QUANTIZE):
        args.checkpoint_exported = True

    assert not inherited.exists()


def test_a_completed_export_clears_the_pointer_when_optional_tracking_fails(
    monkeypatch, fake_mlflow, tmp_path
):
    """$MLFLOW_TRACKING_URI is best-effort, so an unreachable server disables tracking from
    inside the block -- after the tracked path already skipped the untracked cleanup. The
    fresh checkpoint must still not keep the previous run's pointer."""
    monkeypatch.setenv("MLFLOW_TRACKING_URI", URI)

    def explode(name):
        raise ConnectionError("no route to host")

    fake_mlflow.set_experiment = explode
    inherited = tmp_path / ".experiment.json"
    inherited.write_text('{"run_id": "from-an-earlier-run"}')
    args = _parse("--export_megatron_path", str(tmp_path), "--quant_cfg", "nvfp4")

    with mlflow_utils.mlflow_run(args, QUANTIZE):
        args.checkpoint_exported = True

    assert args.mlflow_required is False
    assert not inherited.exists()


def test_a_failed_untracked_run_leaves_the_directory_alone(monkeypatch, tmp_path):
    """Nothing was exported, so whatever checkpoint is already there keeps its pointer."""
    inherited = tmp_path / ".experiment.json"
    inherited.write_text('{"run_id": "stale"}')
    args = _parse("--export_megatron_path", str(tmp_path))

    with pytest.raises(RuntimeError), mlflow_utils.mlflow_run(args, QUANTIZE):
        raise RuntimeError("calibration blew up")

    assert inherited.exists()


# --- the seam with quantize.py --------------------------------------------------------


# The script's seam, as text: it needs Megatron to import, so a renamed flag or a dropped
# call would otherwise only surface in this lane. The other four join in [2/2].
_WIRING = {
    "QUANTIZE": (
        _SCRIPT,
        "with mlflow_run(args, QUANTIZE):",
        ("--hf_model_name_or_path", "--export_megatron_path", "--recipe", "--quant_cfg"),
    ),
}


@pytest.mark.parametrize("tool", list(_WIRING))
def test_every_script_wires_the_tracking(tool):
    script, opens_the_run, flags = _WIRING[tool]
    source = script.read_text()

    # Registered before parsing, or the flags would not exist on the command line.
    assert source.index(f"add_mlflow_args(parser, {tool}") < source.index("parser.parse_args()")
    assert f"resolve_mlflow_args(args, parser, {tool})" in source
    assert opens_the_run in source
    # The namespace reaches print_args masked, so a user:token@ URI stays out of the job log.
    assert "print_args(masked_args(args))" in source
    # Every args attribute the tracking reads is a flag the script registers.
    for flag in flags:
        assert f'"{flag}"' in source
