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

"""MLflow tracking for the vLLM fake-quant server.

``vllm_mlflow_utils`` deliberately imports no vLLM, so the whole launcher-to-worker
handover is exercised here without a GPU, a running server, or the mlflow client.
"""

import argparse
import getpass
import importlib
import os
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml

_EXAMPLES_DIR = Path(__file__).resolve().parents[3] / "examples" / "vllm_serve"

URI = "https://mlflow.example.com"
# Fake credentials for the redaction test. TruffleHog's URI detector flags any
# scheme://user:pass@host, so the marker sits on the definition; this test exists
# precisely to prove such credentials are masked.
CREDS_URI = "https://alice:s3cret@mlflow.example.com"  # trufflehog:ignore

# What the launcher publishes plus what it reads, cleared between tests so one case cannot
# leak tracking configuration into the next.
_TRACKED_ENV = (
    "MLFLOW_TRACKING_URI",
    "MLFLOW_EXPERIMENT_NAME",
    "MODELOPT_MLFLOW_RUN_NAME",
    "MODELOPT_MLFLOW_REQUIRED",
    "MODELOPT_MLFLOW_COMMAND",
    "RECIPE_PATH",
    "QUANT_CFG",
    "KV_QUANT_CFG",
    "MODELOPT_STATE_PATH",
    "QUANT_FILE_PATH",
)

QUANT_CONFIG = {
    "dataset": "cnn_dailymail",
    "calib_size": 512,
    "quant_cfg": "NVFP4_DEFAULT_CFG",
    "kv_quant_cfg": None,
    "quant_file_path": None,
    "modelopt_state_path": None,
    "calib_batch_size": 1,
    "recipe_path": None,
}


class FakeMlflow:
    """Stand-in for the mlflow module, so these tests need no server and no dependency."""

    def __init__(self):
        self.tracking_uri = None
        self.experiment = None
        self.run_name = None
        self.status = None
        self.params = {}
        self.tags = {}
        self.texts = {}
        self.metrics = {}
        self.artifacts = {}

    def set_tracking_uri(self, uri):
        self.tracking_uri = uri

    def set_experiment(self, name):
        self.experiment = name

    def start_run(self, run_name=None):
        self.run_name = run_name
        return SimpleNamespace(info=SimpleNamespace(experiment_id="7", run_id="deadbeef"))

    def log_params(self, params):
        self.params.update(params)

    def set_tags(self, tags):
        self.tags.update(tags)

    def log_text(self, text, artifact_file):
        self.texts[artifact_file] = text

    def log_artifact(self, local_path, artifact_path=None):
        self.artifacts[Path(local_path).name] = (artifact_path, Path(local_path).read_text())

    def log_metrics(self, metrics):
        self.metrics.update(metrics)

    def end_run(self, status=None):
        self.status = status


@pytest.fixture(autouse=True)
def clean_env(monkeypatch):
    monkeypatch.setattr(getpass, "getuser", lambda: "tester")
    for name in _TRACKED_ENV:
        monkeypatch.delenv(name, raising=False)


@pytest.fixture
def mlflow_utils(monkeypatch):
    monkeypatch.syspath_prepend(str(_EXAMPLES_DIR))
    return importlib.import_module("vllm_mlflow_utils")


@pytest.fixture
def fake_mlflow(monkeypatch):
    fake = FakeMlflow()
    monkeypatch.setitem(sys.modules, "mlflow", fake)
    return fake


def _resolve(mlflow_utils, monkeypatch, model="/ckpts/Qwen3-0.6B", **flags):
    """Run the launcher's side of the handover, the way vllm_serve_fakequant.py does."""
    parser = argparse.ArgumentParser()
    parser.add_argument("model")
    mlflow_utils.add_mlflow_args(parser)
    argv = [model, *(token for k, v in flags.items() for token in (f"--{k}", v))]
    monkeypatch.setattr(sys, "argv", ["vllm_serve_fakequant.py", *argv])
    args = parser.parse_args(argv)
    mlflow_utils.resolve_mlflow_args(args, parser)
    return args


def _worker(rank=0, model="/ckpts/Qwen3-0.6B", **model_config):
    """A stand-in for the vLLM worker, holding only what the tracker reads off it."""
    return SimpleNamespace(
        rank=rank,
        vllm_config=SimpleNamespace(
            model_config=SimpleNamespace(model=model, **model_config),
            parallel_config=SimpleNamespace(tensor_parallel_size=8, world_size=8),
            cache_config=SimpleNamespace(cache_dtype="auto"),
        ),
    )


# --- launcher side -------------------------------------------------------------------


def test_flag_publishes_tracking_config_to_the_environment(mlflow_utils, monkeypatch):
    _resolve(mlflow_utils, monkeypatch, mlflow=f"{URI}/")

    assert os.environ["MLFLOW_TRACKING_URI"] == URI  # trailing slash stripped
    assert os.environ["MODELOPT_MLFLOW_REQUIRED"] == "1"
    assert (
        os.environ["MLFLOW_EXPERIMENT_NAME"] == "tester/vllm_serve_fakequant/Qwen3-0.6B-unquantized"
    )
    # The command a user typed, not the worker subprocess's own argv.
    assert "vllm_serve_fakequant.py" in os.environ["MODELOPT_MLFLOW_COMMAND"]


def test_experiment_name_follows_the_quantization_settings(mlflow_utils, monkeypatch):
    monkeypatch.setenv("QUANT_CFG", "NVFP4_DEFAULT_CFG")
    monkeypatch.setenv("KV_QUANT_CFG", "FP8_KV_CFG")
    _resolve(mlflow_utils, monkeypatch, mlflow=URI)

    assert (
        os.environ["MLFLOW_EXPERIMENT_NAME"]
        == "tester/vllm_serve_fakequant/Qwen3-0.6B-NVFP4_DEFAULT_CFG-FP8_KV_CFG"
    )


def test_recipe_names_the_experiment_when_set(mlflow_utils, monkeypatch):
    monkeypatch.setenv("RECIPE_PATH", "/recipes/nvfp4_default-kv_fp8_cast.yaml")
    monkeypatch.setenv("QUANT_CFG", "NVFP4_DEFAULT_CFG")  # the recipe is authoritative
    _resolve(mlflow_utils, monkeypatch, mlflow=URI)

    assert (
        os.environ["MLFLOW_EXPERIMENT_NAME"]
        == "tester/vllm_serve_fakequant/Qwen3-0.6B-nvfp4_default-kv_fp8_cast"
    )


def test_explicit_experiment_and_run_name_win(mlflow_utils, monkeypatch):
    _resolve(
        mlflow_utils,
        monkeypatch,
        mlflow=URI,
        mlflow_experiment="team/sweep",
        mlflow_run_name="calib-512",
    )

    assert os.environ["MLFLOW_EXPERIMENT_NAME"] == "team/sweep"
    assert os.environ["MODELOPT_MLFLOW_RUN_NAME"] == "calib-512"


@pytest.mark.parametrize("sep", ["-", "_"])
def test_multiword_flags_accept_both_spellings(mlflow_utils, monkeypatch, sep):
    """vLLM's FlexibleArgumentParser rewrites --foo_bar to --foo-bar before matching, so a
    flag registered only under the underscored spelling is unreachable from its CLI."""
    parser = argparse.ArgumentParser()
    parser.add_argument("model")
    mlflow_utils.add_mlflow_args(parser)

    args = parser.parse_args(
        [
            "/ckpts/m",
            "--mlflow",
            URI,
            f"--mlflow{sep}experiment",
            "team/sweep",
            f"--mlflow{sep}run{sep}name",
            "calib-512",
        ]
    )

    assert args.mlflow_experiment == "team/sweep"
    assert args.mlflow_run_name == "calib-512"


@pytest.mark.parametrize(
    ("env", "expected"),
    [
        ({"RECIPE_PATH": "/r/w4a8.yaml"}, "w4a8"),
        ({"QUANT_CFG": "NVFP4_DEFAULT_CFG"}, "NVFP4_DEFAULT_CFG"),
        ({"KV_QUANT_CFG": "FP8_KV_CFG"}, "FP8_KV_CFG"),
        ({"MODELOPT_STATE_PATH": "/x/vllm_fq_modelopt_state.pth"}, "modelopt_state"),
        ({"QUANT_FILE_PATH": "/x/quantizer_state.pth"}, "quantizer_state"),
        ({}, "unquantized"),
    ],
)
def test_quant_variant_reads_the_examples_own_settings(mlflow_utils, monkeypatch, env, expected):
    for name, value in env.items():
        monkeypatch.setenv(name, value)
    assert mlflow_utils.quant_variant() == expected


def test_explicit_flag_with_a_bad_uri_fails_at_launch(mlflow_utils, monkeypatch):
    with pytest.raises(SystemExit):
        _resolve(mlflow_utils, monkeypatch, mlflow="mlflow.example.com")


def test_bad_uri_from_the_environment_serves_untracked(mlflow_utils, monkeypatch):
    monkeypatch.setenv("MLFLOW_TRACKING_URI", "sqlite:///mlflow.db")
    with pytest.warns(UserWarning, match="continuing untracked"):
        _resolve(mlflow_utils, monkeypatch)

    # Cleared, so a worker does not retry the URI the launcher already rejected.
    assert "MLFLOW_TRACKING_URI" not in os.environ


def test_printed_uri_carries_no_credentials(mlflow_utils, monkeypatch, capsys):
    """This line lands in the worker log that the run uploads, so it has to be masked the
    same way MlflowRunLogger masks every URI it prints."""
    _resolve(mlflow_utils, monkeypatch, mlflow=CREDS_URI)

    printed = capsys.readouterr().out
    assert "s3cret" not in printed and "alice" not in printed
    assert "https://mlflow.example.com" in printed
    # Only the display is masked; the workers still need the credentials to authenticate.
    assert os.environ["MLFLOW_TRACKING_URI"] == CREDS_URI


def test_environment_uri_is_best_effort_not_required(mlflow_utils, monkeypatch):
    monkeypatch.setenv("MLFLOW_TRACKING_URI", URI)
    _resolve(mlflow_utils, monkeypatch)

    assert os.environ["MODELOPT_MLFLOW_REQUIRED"] == "0"


def test_no_uri_publishes_nothing(mlflow_utils, monkeypatch):
    _resolve(mlflow_utils, monkeypatch)

    assert "MLFLOW_TRACKING_URI" not in os.environ
    assert "MODELOPT_MLFLOW_COMMAND" not in os.environ


def test_ray_copy_list_covers_what_the_launcher_sets(mlflow_utils):
    """Ray copies only the names on this list, so a name the worker needs must be on it."""
    assert {
        mlflow_utils.TRACKING_URI_ENV,
        mlflow_utils.EXPERIMENT_ENV,
        mlflow_utils.RUN_NAME_ENV,
        mlflow_utils.REQUIRED_ENV,
        mlflow_utils.COMMAND_ENV,
    } <= mlflow_utils.MLFLOW_ENV_VARS
    # Credentials are never set here, only forwarded; without them a worker authenticates
    # as nobody and the run fails to open.
    assert "MLFLOW_TRACKING_TOKEN" in mlflow_utils.MLFLOW_ENV_VARS


# --- worker side ---------------------------------------------------------------------


def test_tracker_is_inert_without_a_tracking_uri(mlflow_utils):
    tracker = mlflow_utils.FakeQuantMlflowTracker(_worker(), QUANT_CONFIG)
    assert not tracker.enabled
    # Every entry point is safe to call unconditionally, so the worker needs no branching.
    tracker.start()
    tracker.log_quant_config({"quant_cfg": {}})
    tracker.log_quant_summary(object())
    tracker.finish("FINISHED")


def test_only_rank_zero_records_the_run(mlflow_utils, monkeypatch):
    monkeypatch.setenv("MLFLOW_TRACKING_URI", URI)
    assert mlflow_utils.FakeQuantMlflowTracker(_worker(rank=0), QUANT_CONFIG).enabled
    assert not mlflow_utils.FakeQuantMlflowTracker(_worker(rank=3), QUANT_CONFIG).enabled


def test_worker_names_the_experiment_when_started_without_the_launcher(mlflow_utils, monkeypatch):
    """`vllm serve --worker-cls fakequant_worker.FakeQuantWorker` sets no experiment name."""
    monkeypatch.setenv("MLFLOW_TRACKING_URI", URI)
    monkeypatch.setenv("QUANT_CFG", "NVFP4_DEFAULT_CFG")
    tracker = mlflow_utils.FakeQuantMlflowTracker(_worker(), QUANT_CONFIG)

    assert (
        tracker._logger.experiment_name
        == "tester/vllm_serve_fakequant/Qwen3-0.6B-NVFP4_DEFAULT_CFG"
    )


def test_start_uploads_the_launchers_command_and_the_serving_settings(
    mlflow_utils, monkeypatch, fake_mlflow
):
    monkeypatch.setenv("MLFLOW_TRACKING_URI", URI)
    monkeypatch.setenv("MLFLOW_EXPERIMENT_NAME", "tester/vllm_serve_fakequant/Qwen3-0.6B-nvfp4")
    monkeypatch.setenv(
        "MODELOPT_MLFLOW_COMMAND", "python3 vllm_serve_fakequant.py /ckpts/x -tp 8\n"
    )
    tracker = mlflow_utils.FakeQuantMlflowTracker(
        _worker(served_model_name="qwen", max_model_len=4096), QUANT_CONFIG
    )
    tracker.start()
    tracker.finish("FINISHED")

    assert fake_mlflow.experiment == "tester/vllm_serve_fakequant/Qwen3-0.6B-nvfp4"
    assert fake_mlflow.texts["command.txt"].startswith("python3 vllm_serve_fakequant.py")
    # The quantization settings and the serving settings are both searchable.
    assert fake_mlflow.params["quant_cfg"] == "NVFP4_DEFAULT_CFG"
    assert fake_mlflow.params["calib_size"] == 512
    assert fake_mlflow.params["tensor_parallel_size"] == 8
    assert fake_mlflow.params["max_model_len"] == 4096
    # The join key with the hf_ptq run that produced the checkpoint being served.
    assert fake_mlflow.tags["checkpoint_path"] == "/ckpts/Qwen3-0.6B"
    assert fake_mlflow.tags["model"] == "Qwen3-0.6B"
    assert fake_mlflow.tags["tool"] == "vllm_serve_fakequant"
    assert fake_mlflow.tags["served_model_name"] == "qwen"
    assert fake_mlflow.status == "FINISHED"


def test_quant_config_is_uploaded_before_calibration(mlflow_utils, monkeypatch, fake_mlflow):
    monkeypatch.setenv("MLFLOW_TRACKING_URI", URI)
    tracker = mlflow_utils.FakeQuantMlflowTracker(_worker(), QUANT_CONFIG)
    tracker.start()
    tracker.log_quant_config({"quant_cfg": {"*weight_quantizer": {"num_bits": 4}}})
    uploaded = fake_mlflow.texts["recipe/quant_cfg.yaml"]
    tracker.finish("FINISHED")

    # Present while the run was still open, so a crash during calibration keeps it.
    assert yaml.safe_load(uploaded) == {"quant_cfg": {"*weight_quantizer": {"num_bits": 4}}}


def test_a_best_effort_start_failure_leaves_no_staging_directory(
    mlflow_utils, monkeypatch, fake_mlflow
):
    """start() reports an unusable server by disabling itself rather than raising, and every
    later method returns before the cleanup -- so the temp dir would outlive the process."""
    monkeypatch.setenv("MLFLOW_TRACKING_URI", URI)  # from the environment, so not required

    def explode(*args, **kwargs):
        raise ConnectionError("no route to host")

    fake_mlflow.set_experiment = explode
    tracker = mlflow_utils.FakeQuantMlflowTracker(_worker(), QUANT_CONFIG)

    tracker.start()

    assert not tracker.enabled  # downgraded to untracked, the serve carries on
    assert tracker._staging is None
    tracker.finish("FINISHED")  # still safe to call


def test_an_explicit_flag_start_failure_leaves_no_staging_directory(
    mlflow_utils, monkeypatch, fake_mlflow
):
    monkeypatch.setenv("MLFLOW_TRACKING_URI", URI)
    monkeypatch.setenv("MODELOPT_MLFLOW_REQUIRED", "1")  # --mlflow, so failure is fatal

    def explode(*args, **kwargs):
        raise ConnectionError("no route to host")

    fake_mlflow.set_experiment = explode
    tracker = mlflow_utils.FakeQuantMlflowTracker(_worker(), QUANT_CONFIG)

    with pytest.raises(ConnectionError):
        tracker.start()

    assert tracker._staging is None


def test_a_recipe_run_does_not_duplicate_the_config(mlflow_utils, monkeypatch, fake_mlflow):
    """get_quant_config returns the recipe's quantize section unchanged, and
    recipe/resolved_recipe.yaml already carries it -- properly serialized."""
    monkeypatch.setenv("MLFLOW_TRACKING_URI", URI)
    tracker = mlflow_utils.FakeQuantMlflowTracker(
        _worker(), {**QUANT_CONFIG, "recipe_path": "/r/nvfp4.yaml", "quant_cfg": None}
    )
    tracker._logger.start()  # bypass _start_texts, which would load the recipe from disk
    tracker.log_quant_config(object())
    tracker._logger.finish("FINISHED")

    assert "recipe/quant_cfg.yaml" not in fake_mlflow.texts


def test_a_pydantic_config_is_dumped_as_yaml_not_repr(mlflow_utils, monkeypatch, fake_mlflow):
    """yaml.safe_dump cannot represent a QuantizeConfig; a repr fallback would upload an
    unparseable blob under a .yaml name and look like it worked."""
    monkeypatch.setenv("MLFLOW_TRACKING_URI", URI)

    class FakeQuantizeConfig:
        def model_dump(self, mode=None):
            return {"quant_cfg": [{"quantizer_name": "*weight_quantizer", "enable": True}]}

    tracker = mlflow_utils.FakeQuantMlflowTracker(_worker(), QUANT_CONFIG)
    tracker.start()
    tracker.log_quant_config(FakeQuantizeConfig())
    uploaded = fake_mlflow.texts["recipe/quant_cfg.yaml"]
    tracker.finish("FINISHED")

    assert yaml.safe_load(uploaded) == {
        "quant_cfg": [{"quantizer_name": "*weight_quantizer", "enable": True}]
    }


def test_an_unserializable_config_warns_instead_of_killing_the_serve(
    mlflow_utils, monkeypatch, fake_mlflow, capsys
):
    monkeypatch.setenv("MLFLOW_TRACKING_URI", URI)
    tracker = mlflow_utils.FakeQuantMlflowTracker(_worker(), QUANT_CONFIG)
    tracker.start()
    tracker.log_quant_config({"cfg": object()})  # no representer, no model_dump
    tracker.finish("FINISHED")

    assert "could not serialize the quantization config" in capsys.readouterr().out
    assert "recipe/quant_cfg.yaml" not in fake_mlflow.texts


def test_quant_summary_is_uploaded_from_the_staging_directory(
    mlflow_utils, monkeypatch, fake_mlflow
):
    monkeypatch.setenv("MLFLOW_TRACKING_URI", URI)
    tracker = mlflow_utils.FakeQuantMlflowTracker(_worker(), QUANT_CONFIG)

    # Stand in for mtq.print_quant_summary(model, output_dir=...), which is what writes it.
    def write_summary(model, output_dir):
        Path(output_dir, ".quant_summary.txt").write_text("2 TensorQuantizers found in model\n")

    monkeypatch.setattr(
        importlib.import_module("modelopt.torch.quantization"),
        "print_quant_summary",
        write_summary,
    )
    tracker.start()
    tracker.log_quant_summary(object())
    tracker.finish("FINISHED")

    artifact_path, content = fake_mlflow.artifacts["quant_summary.txt"]
    assert artifact_path == "summary"
    assert "2 TensorQuantizers" in content


def test_a_run_with_no_summary_uploads_none(mlflow_utils, monkeypatch, fake_mlflow):
    """A reload from MODELOPT_STATE_PATH on a non-zero rank writes no summary."""
    monkeypatch.setenv("MLFLOW_TRACKING_URI", URI)
    tracker = mlflow_utils.FakeQuantMlflowTracker(_worker(), QUANT_CONFIG)
    tracker.start()
    tracker.finish("FINISHED")

    assert "quant_summary.txt" not in fake_mlflow.artifacts


def test_a_failed_step_closes_the_run_and_reraises(mlflow_utils, monkeypatch, fake_mlflow):
    monkeypatch.setenv("MLFLOW_TRACKING_URI", URI)
    tracker = mlflow_utils.FakeQuantMlflowTracker(_worker(), QUANT_CONFIG)
    tracker.start()

    with pytest.raises(RuntimeError, match="out of memory"), tracker.fail_on_error():
        raise RuntimeError("out of memory")

    assert fake_mlflow.status == "FAILED"
    # The log is attached even though the worker never reached the end of warm-up.
    assert any(name.endswith(".log") for name in fake_mlflow.artifacts)


def test_a_closed_run_is_not_reopened_by_a_later_step(mlflow_utils, monkeypatch, fake_mlflow):
    """vLLM drives the worker through several guarded steps after warm-up finishes."""
    monkeypatch.setenv("MLFLOW_TRACKING_URI", URI)
    tracker = mlflow_utils.FakeQuantMlflowTracker(_worker(), QUANT_CONFIG)
    tracker.start()
    tracker.finish("FINISHED")

    with pytest.raises(RuntimeError), tracker.fail_on_error():
        raise RuntimeError("the server died an hour later")

    assert fake_mlflow.status == "FINISHED"
