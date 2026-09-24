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
import os
import sys
from datetime import datetime
from pathlib import Path

import pytest
import yaml
from _test_utils.mlflow import clean_env  # noqa: F401

from modelopt.torch.utils import mlflow as mlflow_lib
from modelopt.torch.utils.mlflow import masked_args

_EXAMPLE_DIR = Path(__file__).resolve().parents[3] / "examples" / "megatron_bridge"
_SCRIPT = _EXAMPLE_DIR / "quantize.py"
_DISTILL_SCRIPT = _EXAMPLE_DIR / "distill.py"
_PRUNE_SCRIPT = _EXAMPLE_DIR / "prune_minitron.py"
_DISTILL_EXPORT_SCRIPT = _EXAMPLE_DIR / "export_distilled_megatron_to_hf.py"
_EXPORT_SCRIPT = _EXAMPLE_DIR / "export_quantized_megatron_to_hf.py"


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
PRUNE = _load("prune_minitron").PRUNE
EXPORT = _load("export_quantized_megatron_to_hf").EXPORT
DISTILL = _load("distill").DISTILL
DISTILL_EXPORT = _load("export_distilled_megatron_to_hf").DISTILL_EXPORT
_TOOLS = {
    t.name.removeprefix("megatron_bridge_").upper(): t
    for t in (QUANTIZE, PRUNE, EXPORT, DISTILL, DISTILL_EXPORT)
}

URI = "https://mlflow.example.com"
RECIPE = "general/ptq/nvfp4_default-kv_fp8_cast"


# What each script's parser offers, limited to what the tracking reads off args.
_SCRIPT_ARGS = {
    "QUANTIZE": {
        "--hf_model_name_or_path": "/models/Qwen3-0.6B",
        "--export_megatron_path": "/tmp/out",
        "--recipe": None,
        "--quant_cfg": None,
        "--tp_size": 1,
    },
    "PRUNE": {
        "--hf_model_name_or_path": "/models/Qwen3-0.6B",
        "--output_megatron_path": None,
        "--output_hf_path": None,
        "--prune_target_params": None,
        "--prune_target_active_params": None,
        "--prune_target_memory_mb": None,
        "--prune_export_config": None,
    },
    "EXPORT": {
        "--hf_model_name_or_path": "/models/Qwen3-0.6B",
        "--megatron_path": "/ckpts/Qwen3-0.6B-nvfp4-megatron/",
        "--export_unified_hf_path": "/tmp/hf_out",
    },
    "DISTILL": {
        "--validate_only": False,
        "--student_hf_path": "/models/Qwen3-0.6B",
        "--student_megatron_path": "/ckpts/Qwen3-0.6B-nvfp4-megatron",
        "--teacher_hf_path": "/models/Qwen3-0.6B",
        "--output_dir": "/runs/qad",
    },
    "DISTILL_EXPORT": {
        "--student_hf_path": "/models/Qwen3-0.6B",
        "--megatron_path": "/runs/qad/checkpoints",
        "--hf_export_path": "/tmp/hf_out",
        "--export_iterations": None,
    },
}

_FLOAT_FLAGS = {"--prune_target_params", "--prune_target_active_params", "--prune_target_memory_mb"}


def _parse_for(tool_name, *argv):
    """Parse *argv* the way the script owning *tool_name* would."""
    tool = _TOOLS[tool_name]
    parser = argparse.ArgumentParser()
    for flag, default in _SCRIPT_ARGS[tool_name].items():
        if flag == "--export_iterations":
            parser.add_argument(flag, nargs="*", default=default)
        elif flag in _FLOAT_FLAGS:
            parser.add_argument(flag, type=float, default=default)
        elif isinstance(default, bool):
            parser.add_argument(flag, action="store_true")
        elif isinstance(default, int):
            parser.add_argument(flag, type=int, default=default)
        else:
            parser.add_argument(flag, default=default)
    mlflow_utils.add_mlflow_args(parser, tool, log_checkpoints=tool is DISTILL)

    args = parser.parse_args(list(argv))
    mlflow_utils.resolve_mlflow_args(args, parser, tool)
    args.checkpoint_exported = False
    return args


def _parse(*argv):
    return _parse_for("QUANTIZE", *argv)


def _parse_prune(*argv):
    return _parse_for("PRUNE", *argv)


def _parse_export(*argv):
    return _parse_for("EXPORT", *argv)


def _parse_distill(*argv):
    return _parse_for("DISTILL", *argv)


def _parse_distill_export(*argv):
    return _parse_for("DISTILL_EXPORT", *argv)


def _tracked(export_path, *extra):
    return _parse("--export_megatron_path", str(export_path), "--mlflow", URI, *extra)


# --- flags ----------------------------------------------------------------------------


def test_tracking_is_off_by_default():
    args = _parse()

    assert args.mlflow is None
    assert args.mlflow_experiment is None
    assert not args.mlflow_required


@pytest.mark.parametrize(
    ("tool", "argv", "experiment"),
    [
        (
            "QUANTIZE",
            ["--recipe", RECIPE],
            "megatron_bridge_quantize/Qwen3-0.6B-nvfp4_default-kv_fp8_cast",
        ),
        ("QUANTIZE", ["--quant_cfg", "nvfp4"], "megatron_bridge_quantize/Qwen3-0.6B-nvfp4"),
        (
            "PRUNE",
            ["--prune_target_params", "4e9"],
            "megatron_bridge_prune/Qwen3-0.6B-params-4000000000.0",
        ),
        (
            "PRUNE",
            ["--prune_target_memory_mb", "8000"],
            "megatron_bridge_prune/Qwen3-0.6B-memory_mb-8000.0",
        ),
        (
            # Combining targets is supported, so one of them names the run and the params
            # carry the rest.
            "PRUNE",
            ["--prune_target_params", "4e9", "--prune_target_memory_mb", "8000"],
            "megatron_bridge_prune/Qwen3-0.6B-params-4000000000.0",
        ),
        (
            "PRUNE",
            ["--prune_export_config", '{"hidden_size": 1024}'],
            "megatron_bridge_prune/Qwen3-0.6B-export_config",
        ),
        (
            "EXPORT",
            [],
            "megatron_bridge_export/Qwen3-0.6B-Qwen3-0.6B-nvfp4-megatron",
        ),
        (
            "DISTILL",
            [],
            "megatron_bridge_distill/Qwen3-0.6B-Qwen3-0.6B-nvfp4-megatron",
        ),
        ("DISTILL", ["--student_megatron_path", ""], "megatron_bridge_distill/Qwen3-0.6B-bf16"),
        (
            "DISTILL_EXPORT",
            [],
            "megatron_bridge_distill_export/Qwen3-0.6B-checkpoints",
        ),
    ],
)
def test_each_tool_names_its_experiment(tool, argv, experiment):
    """One convention across the scripts: $USER/<script>/<model>-<what this run did>. A
    trailing slash must not make the variant empty, and an export config names itself rather
    than its contents."""
    args = _parse_for(tool, "--mlflow", URI, *argv)

    assert args.mlflow_experiment == f"tester/{experiment}"


@pytest.mark.parametrize("sep", ["-", "_"])
def test_multiword_flags_accept_both_spellings(monkeypatch, sep):
    args = _parse(
        "--mlflow",
        URI,
        f"--mlflow{sep}experiment",
        "team/sweep",
        f"--mlflow{sep}run{sep}name",
        "calib-512",
    )
    distill = _parse_distill("--mlflow", URI, f"--mlflow{sep}log{sep}checkpoints")

    assert args.mlflow_experiment == "team/sweep"
    assert args.mlflow_run_name == "calib-512"
    assert distill.mlflow_log_checkpoints


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


def test_run_tags_identify_the_produced_checkpoint(monkeypatch, tmp_path):
    """checkpoint_path must name what the run *writes*: the export and any distillation are
    pointed at the Megatron checkpoint, so tagging the input would never join the two."""
    export = tmp_path / "Qwen3-0.6B-nvfp4-megatron"
    args = _tracked(export)

    assert mlflow_lib.run_tags(args, QUANTIZE) == {
        "model": "Qwen3-0.6B",
        "checkpoint_path": str(export),
        "source_checkpoint_path": "/models/Qwen3-0.6B",
    }


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


# Every script's seam, as text: the scripts need Megatron to import, so a renamed flag or a
# dropped call would otherwise only surface in the Megatron example lane.
_WIRING = {
    "QUANTIZE": (
        _SCRIPT,
        "with mlflow_run(args, QUANTIZE):",
        ("--hf_model_name_or_path", "--export_megatron_path", "--recipe", "--quant_cfg"),
    ),
    "PRUNE": (
        _PRUNE_SCRIPT,
        "with mlflow_run(args, PRUNE):",
        ("--hf_model_name_or_path", "--output_megatron_path", "--output_hf_path"),
    ),
    "EXPORT": (
        _EXPORT_SCRIPT,
        "with mlflow_run(args, EXPORT):",
        ("--hf_model_name_or_path", "--megatron_path", "--export_unified_hf_path"),
    ),
    "DISTILL_EXPORT": (_DISTILL_EXPORT_SCRIPT, "with mlflow_run(args, DISTILL_EXPORT):", ()),
    "DISTILL": (
        _DISTILL_SCRIPT,
        "with distill_run(args, DISTILL) as owns_the_run:",
        ("--student_hf_path", "--student_megatron_path", "--output_dir"),
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


@pytest.mark.parametrize(
    ("script", "saves"),
    [(_SCRIPT, 1), (_EXPORT_SCRIPT, 1), (_PRUNE_SCRIPT, 2)],
    ids=["quantize", "export", "prune-writes-either-or-both"],
)
def test_the_pointer_is_gated_on_the_save_having_happened(script, saves):
    """Not on the output path existing: the directory is usually there before the weights."""
    source = script.read_text()

    assert "args.checkpoint_exported = False" in source
    assert source.count("args.checkpoint_exported = True") == saves


def test_the_distill_script_defers_to_megatron_bridge():
    """Its run is opened here and joined there, and Megatron-Bridge calls sys.exit() from
    inside train(), so anything placed after distill() never runs."""
    source = _DISTILL_SCRIPT.read_text()

    assert "**logger_kwargs(args, DISTILL)," in source
    assert "try:\n        distill(config)\n    finally:" in source
    # The marker is read before training and handed back, so a resumed run that saved
    # nothing cannot stamp the checkpoint it resumed from.
    assert "saved_before = checkpoint_marker(args, DISTILL)" in source
    assert "record_checkpoint_provenance(args, DISTILL, saved_before=saved_before" in source
    # The Tool reports no checkpoint for --validate_only, so the helper needs no guard here.
    assert "if not args.validate_only:" not in source
    assert "mlflow_run(" not in source  # the single-pass wrapper is not what this uses


def test_each_distilled_export_is_pointed_at_the_run_as_it_is_written():
    """The VLM branch still has its process group; the LLM branch uses the rank it captured
    before dist.cleanup()."""
    source = _DISTILL_EXPORT_SCRIPT.read_text()

    assert "record_exported_checkpoint(args, hf_export_path, dist.is_master())" in source
    assert "record_exported_checkpoint(args, hf_export_path, is_rank_0)" in source


# --- the export tool: same shape as quantize, different arguments -----------------------


# --- the distill tool: Megatron-Bridge owns the run -------------------------------------


def test_distill_hands_its_settings_to_megatron_bridge():
    """The training loop logs the metrics, so the run is configured rather than opened."""
    args = _parse_distill("--mlflow", URI, "--mlflow_run_name", "qad-1")

    kwargs = mlflow_utils.logger_kwargs(args, DISTILL)

    assert kwargs["mlflow_tracking_uri"] == URI
    assert kwargs["mlflow_experiment"] == (
        "tester/megatron_bridge_distill/Qwen3-0.6B-Qwen3-0.6B-nvfp4-megatron"
    )
    assert kwargs["mlflow_run_name"] == "qad-1"
    # The join keys: what this run consumed (the PTQ checkpoint, not the model it came
    # from) and what it produces.
    assert kwargs["mlflow_tags"]["checkpoint_path"] == "/runs/qad/checkpoints"
    assert kwargs["mlflow_tags"]["source_checkpoint_path"] == "/ckpts/Qwen3-0.6B-nvfp4-megatron"
    assert kwargs["mlflow_tags"]["model"] == "Qwen3-0.6B"  # still the model, not the ckpt dir


def test_an_untracked_distill_hands_megatron_bridge_nothing():
    """The mlflow_* LoggerConfig fields only exist in Megatron-Bridge 0.6+, so an untracked
    run must not pass them at all -- it would break on an older one for no reason."""
    assert mlflow_utils.logger_kwargs(_parse_distill(), DISTILL) == {}


@pytest.mark.parametrize(("argv", "expected"), [([], False), (["--mlflow_log_checkpoints"], True)])
def test_checkpoint_artifacts_are_off_unless_asked_for(argv, expected):
    """Megatron-Bridge defaults this on, which pushes every saved checkpoint over HTTP."""
    args = _parse_distill("--mlflow", URI, *argv)

    assert mlflow_utils.logger_kwargs(args, DISTILL)["mlflow_log_artifacts"] is expected


def _saved(tmp_path, iteration):
    """Stand in for Megatron-Bridge having written a checkpoint at *iteration*."""
    checkpoints = tmp_path / "checkpoints"
    checkpoints.mkdir(exist_ok=True)
    (checkpoints / "latest_checkpointed_iteration.txt").write_text(f"{iteration}\n")


@pytest.mark.parametrize("tracked", [True, False], ids=["tracked", "untracked"])
@pytest.mark.parametrize(
    ("saved_before", "saved_now", "is_last", "writes"),
    [
        (None, None, True, False),
        ("10", 10, True, False),
        ("10", 20, True, True),
        ("10", 20, False, False),
    ],
    ids=["nothing-saved", "resumed-then-died", "saved-its-own", "not-the-owning-rank"],
)
def test_only_a_run_that_saved_settles_the_pointer(
    monkeypatch, tmp_path, saved_before, saved_now, is_last, writes, tracked
):
    """A resumed run starts with the previous run's checkpoint in place -- ``load`` points at
    the same directory -- so only a moved marker shows this run saved one of its own.
    Megatron-Bridge owns the run on the last rank, and an untracked re-run reaches this too,
    to clear the pointer it inherited."""
    calls = []
    monkeypatch.setattr(mlflow_utils, "log_active_run_experiment_json", calls.append)
    if saved_now is not None:
        _saved(tmp_path, saved_now)
    args = _parse_distill(*(["--mlflow", URI] if tracked else []), "--output_dir", str(tmp_path))

    mlflow_utils.record_checkpoint_provenance(
        args, DISTILL, saved_before=saved_before, is_main=is_last
    )

    assert calls == ([str(tmp_path / "checkpoints")] if writes else [])


# --- the seams with the two other scripts -----------------------------------------------


def test_only_the_training_script_offers_the_checkpoint_upload_flag():
    """Declared once beside the Tool records, not decided inside add_mlflow_args."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--hf_model_name_or_path", default="/models/m")
    parser.add_argument("--megatron_path", default="/ckpts/m")
    parser.add_argument("--export_unified_hf_path", default="/tmp/out")
    mlflow_utils.add_mlflow_args(parser, EXPORT)

    with pytest.raises(SystemExit):
        parser.parse_args(["--mlflow_log_checkpoints"])


def test_the_three_stages_chain_on_the_server(monkeypatch, tmp_path):
    """Each stage's source is the previous stage's output, so a tag query walks the chain
    without reading any checkpoint from disk."""
    ptq, qad, hf = tmp_path / "ptq", tmp_path / "qad", tmp_path / "hf"

    quantize = _parse("--export_megatron_path", str(ptq), "--mlflow", URI, "--quant_cfg", "nvfp4")
    distill = _parse_distill(
        "--mlflow",
        URI,
        "--student_megatron_path",
        str(ptq),
        "--output_dir",
        str(qad),
    )
    export = _parse_export(
        "--mlflow",
        URI,
        "--megatron_path",
        str(qad / "checkpoints"),
        "--export_unified_hf_path",
        str(hf),
    )

    # The distilled export is normally pointed at one iteration, which has to join the
    # checkpoints directory the distillation tagged.
    distilled = _parse_distill_export(
        "--mlflow", URI, "--megatron_path", str(qad / "checkpoints" / "iter_0000500")
    )

    quantize_tags = mlflow_lib.run_tags(quantize, QUANTIZE)
    distill_tags = mlflow_utils.logger_kwargs(distill, DISTILL)["mlflow_tags"]
    export_tags = mlflow_lib.run_tags(export, EXPORT)
    distilled_tags = mlflow_lib.run_tags(distilled, DISTILL_EXPORT)

    assert distill_tags["source_checkpoint_path"] == quantize_tags["checkpoint_path"]
    assert export_tags["source_checkpoint_path"] == distill_tags["checkpoint_path"]
    assert distilled_tags["source_checkpoint_path"] == distill_tags["checkpoint_path"]


def test_the_distill_run_name_defaults_to_the_documented_timestamp(fake_mlflow):
    """Megatron-Bridge would name it randomly; the flag promises the UTC start time. Settled
    in distill_run, so the run it joins and the config it records carry the same one."""
    args = _parse_distill("--mlflow", URI)

    with mlflow_utils.distill_run(args, DISTILL):
        run_name = mlflow_utils.logger_kwargs(args, DISTILL)["mlflow_run_name"]

    assert datetime.strptime(run_name, "%Y%m%d-%H%M%S")
    assert fake_mlflow.run_name == run_name


@pytest.mark.parametrize("is_master", [True, False], ids=["rank-0", "another-rank"])
def test_an_unmovable_credential_is_not_recorded_at_all(monkeypatch, capsys, is_master):
    """Megatron-Bridge persists what it is given, so a credential its variables cannot hold
    buys nothing by being passed through. Every rank has to decline, or LoggerConfig differs
    by rank and the last one enters Megatron-Bridge's MLflow path with no URI."""
    monkeypatch.setattr(mlflow_utils.dist, "is_master", lambda: is_master)
    # Fake credential; this test exists to prove it reaches nothing that records it.
    args = _parse_distill("--mlflow", "https://sekrit@mlflow.example.com")  # trufflehog:ignore

    assert mlflow_utils.logger_kwargs(args, DISTILL) == {}

    out = capsys.readouterr().out
    assert "sekrit" not in out  # the warning names the URI, so it has to mask it
    assert ("MLFLOW_TRACKING_USERNAME" in out) is is_master  # warned once, not per rank


def test_megatron_bridge_never_receives_the_credentials(monkeypatch):
    """Its config becomes MLflow params and run_config.yaml inside the checkpoint, so a
    credential handed to it would be durable in both."""
    # Fake credentials; this test exists to prove they do not reach the config.
    creds = "https://svc:s3cret@mlflow.example.com"  # trufflehog:ignore

    kwargs = mlflow_utils.logger_kwargs(_parse_distill("--mlflow", creds), DISTILL)

    assert kwargs["mlflow_tracking_uri"] == "https://mlflow.example.com"
    assert "s3cret" not in str(kwargs)
    assert os.environ["MLFLOW_TRACKING_PASSWORD"] == "s3cret"  # where MLflow reads it


@pytest.mark.parametrize("tracked", [True, False], ids=["tracked-warns", "untracked-quiet"])
def test_finding_no_run_is_reported_only_when_tracking_was_asked_for(
    monkeypatch, tmp_path, capsys, tracked
):
    """A tracked run has just had its inherited pointer cleared, so silence would look like
    success; an untracked one finding no run is the normal state."""
    monkeypatch.setattr(mlflow_utils, "log_active_run_experiment_json", lambda path: False)
    _saved(tmp_path, 20)
    args = _parse_distill(*(["--mlflow", URI] if tracked else []), "--output_dir", str(tmp_path))

    mlflow_utils.record_checkpoint_provenance(args, DISTILL, saved_before="10")

    assert ("no run was found" in capsys.readouterr().out) is tracked


def test_the_distill_run_carries_what_the_other_scripts_carry(monkeypatch, fake_mlflow, tmp_path):
    """Megatron-Bridge cannot log the invocation or distill.py's own arguments, so the run
    is opened here -- and Megatron-Bridge then logs its metrics into this same run."""
    monkeypatch.setattr(mlflow_utils.dist, "is_last_process", lambda: True)
    args = _parse_distill("--mlflow", URI, "--output_dir", str(tmp_path))

    with mlflow_utils.distill_run(args, DISTILL):
        pass

    # Its own arguments, which Megatron-Bridge's flattened config does not carry.
    assert fake_mlflow.params["teacher_hf_path"] == "/models/Qwen3-0.6B"
    assert fake_mlflow.params["student_megatron_path"] == "/ckpts/Qwen3-0.6B-nvfp4-megatron"
    assert "command.txt" in fake_mlflow.texts
    assert fake_mlflow.status == "FINISHED"


@pytest.mark.parametrize(("code", "status"), [(0, "FINISHED"), (2, "FAILED")])
def test_megatron_bridges_exit_is_a_failure_only_when_it_failed(
    monkeypatch, fake_mlflow, tmp_path, code, status
):
    """--exit_interval leaves train() through sys.exit(0), which is a finished run."""
    monkeypatch.setattr(mlflow_utils.dist, "is_last_process", lambda: True)
    args = _parse_distill("--mlflow", URI, "--output_dir", str(tmp_path))

    with pytest.raises(SystemExit), mlflow_utils.distill_run(args, DISTILL):
        raise SystemExit(code)

    assert fake_mlflow.status == status


@pytest.mark.parametrize("is_last", [True, False])
def test_the_run_opens_on_the_rank_megatron_bridge_looks_at(
    monkeypatch, fake_mlflow, tmp_path, is_last
):
    """Opened anywhere else, the last rank finds nothing active and starts a second run."""
    monkeypatch.setattr(mlflow_utils.dist, "is_last_process", lambda: is_last)
    args = _parse_distill("--mlflow", URI, "--output_dir", str(tmp_path))

    with mlflow_utils.distill_run(args, DISTILL):
        pass

    assert (fake_mlflow.run_name is not None) is is_last  # inert on every other rank


def test_only_a_tool_that_settles_its_pointer_is_asked_whether_it_exported(tmp_path):
    """So a script whose checkpoints are pointed at the run as they are written -- and which
    therefore never defines the flag -- is not one reordered condition away from failing."""
    args = _parse_distill_export("--hf_export_path", str(tmp_path))
    del args.checkpoint_exported  # as export_distilled_megatron_to_hf.py leaves it

    with mlflow_utils.mlflow_run(args, DISTILL_EXPORT):
        pass


def test_a_logger_that_disabled_itself_does_not_also_report_a_missing_pointer(
    monkeypatch, fake_mlflow, tmp_path, capsys
):
    """It already said tracking was off; a second warning about provenance reads as a
    separate fault rather than the degradation the user was just told about."""
    monkeypatch.setenv("MLFLOW_TRACKING_URI", URI)

    def explode(name):
        raise ConnectionError("no route to host")

    fake_mlflow.set_experiment = explode
    args = _parse_distill_export("--hf_export_path", str(tmp_path))

    with mlflow_utils.mlflow_run(args, DISTILL_EXPORT):
        mlflow_utils.record_exported_checkpoint(args, tmp_path, is_main=True)

    assert args.mlflow is None
    assert "carries no provenance pointer" not in capsys.readouterr().out


def test_an_unreachable_server_is_not_handed_to_megatron_bridge(monkeypatch, fake_mlflow, tmp_path):
    """A URI inherited from $MLFLOW_TRACKING_URI is best-effort, so an unreachable server
    disables tracking rather than failing the run. Megatron-Bridge opens its run from inside
    the training loop, where the same dead URI would abort the training instead."""
    monkeypatch.setenv("MLFLOW_TRACKING_URI", URI)

    def explode(name):
        raise ConnectionError("no route to host")

    fake_mlflow.set_experiment = explode
    args = _parse_distill("--output_dir", str(tmp_path))

    with mlflow_utils.distill_run(args, DISTILL):
        assert mlflow_utils.logger_kwargs(args, DISTILL) == {}


# --- the pruning tool: same shape, but its output is optional ---------------------------


@pytest.mark.parametrize(
    "flag", ["--output_megatron_path", "--output_hf_path"], ids=["megatron", "hf"]
)
def test_a_pruning_run_points_whichever_checkpoint_it_saved(fake_mlflow, tmp_path, flag):
    """``prune_minitron.py`` makes the two output flags a required exclusive group, so
    exactly one is set and it is the one a chain joins on."""
    args = _parse_prune("--mlflow", URI, flag, str(tmp_path), "--prune_target_params", "4e9")

    assert PRUNE.checkpoint(args) == str(tmp_path)

    with mlflow_utils.mlflow_run(args, PRUNE):
        args.checkpoint_exported = True

    assert json.loads((tmp_path / ".experiment.json").read_text())["run_id"] == "deadbeef"
    assert fake_mlflow.status == "FINISHED"


@pytest.mark.parametrize(
    ("argv", "score"),
    [
        (["--prune_target_params", "4e9"], 0.87),
        (["--prune_export_config", '{"num_layers": 2}'], None),
    ],
    ids=["searched", "manual-export-config"],
)
def test_a_pruning_run_points_its_checkpoint_at_itself_and_reports_its_score(
    fake_mlflow, tmp_path, argv, score
):
    """The score is known only after the search, so it is read on the way out, and it sorts
    across runs in the UI as a metric rather than a binary artifact. ``--prune_export_config``
    scores nothing, so there is no number to report."""
    args = _parse_prune("--mlflow", URI, "--output_megatron_path", str(tmp_path), *argv)

    with mlflow_utils.mlflow_run(args, PRUNE):
        args.prune_score = score  # what main() stashes once mtp.prune has scored
        args.checkpoint_exported = True

    assert json.loads((tmp_path / ".experiment.json").read_text())["run_id"] == "deadbeef"
    assert fake_mlflow.metrics.get("prune_score") == score
    assert "prune_score" not in fake_mlflow.params  # a result, not an input


# --- the distilled-export tool: several checkpoints, one run ----------------------------


def test_the_parent_of_several_exports_gets_no_pointer(fake_mlflow, tmp_path):
    """With --export_iterations the named path holds one directory per iteration, so it is
    not itself a checkpoint; each export points itself at the run instead."""
    args = _parse_distill_export(
        "--mlflow", URI, "--hf_export_path", str(tmp_path), "--export_iterations", "1", "2"
    )

    with mlflow_utils.mlflow_run(args, DISTILL_EXPORT):
        args.checkpoint_exported = True

    assert not (tmp_path / ".experiment.json").exists()
    assert not DISTILL_EXPORT.settles_pointer


@pytest.mark.parametrize("is_main", [True, False], ids=["owning-rank", "other-rank"])
def test_each_exported_checkpoint_points_at_the_run(monkeypatch, tmp_path, is_main):
    """The LLM branch destroys the process group first, so every rank then looks like rank 0
    and they would all write; the caller passes the rank it captured beforehand."""
    written = []
    monkeypatch.setattr(
        mlflow_utils, "log_active_run_experiment_json", lambda p: written.append(p) or True
    )
    args = _parse_distill_export("--mlflow", URI)

    for iteration in ("iter_0000001", "iter_0000002"):
        mlflow_utils.record_exported_checkpoint(args, tmp_path / iteration, is_main=is_main)

    expected = [tmp_path / "iter_0000001", tmp_path / "iter_0000002"]
    assert written == (expected if is_main else [])


def test_prune_script_records_the_score():
    """Stashed unconditionally after the search, not only when the gate flag is passed."""
    source = _PRUNE_SCRIPT.read_text()

    assert 'args.prune_score = pruning_scores.get("best", {}).get("score")' in source
    assert source.index("args.prune_score =") < source.index(
        "if args.score_lower_bound is not None:"
    )
