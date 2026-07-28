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

import getpass
import importlib
import io
import logging
import sys
from argparse import Namespace
from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml

_EXAMPLES_DIR = Path(__file__).resolve().parents[3] / "examples" / "hf_ptq"


@pytest.fixture
def mlflow_utils(monkeypatch):
    monkeypatch.syspath_prepend(str(_EXAMPLES_DIR))
    monkeypatch.setattr(getpass, "getuser", lambda: "tester")
    return importlib.import_module("mlflow_utils")


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
        self.artifacts = []

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
        self.artifacts.append((Path(local_path).name, artifact_path))

    def log_metric(self, key, value):
        self.metrics[key] = value

    def end_run(self, status=None):
        self.status = status


def _args(**overrides):
    args = Namespace(
        mlflow="https://mlflow.example.com",
        mlflow_experiment="tester/hf_ptq/model-nvfp4",
        mlflow_run_name=None,
        pyt_ckpt_path="/models/Qwen3-0.6B",
        recipe=None,
        qformat="nvfp4",
        kv_cache_qformat="fp8_cast",
        calib_size=[1024],
        calib_seq=512,
        batch_size=0,
        sparsity_fmt="dense",
        export_path="exported_model",
        dist_state=SimpleNamespace(is_main=True, world_size=1),
    )
    return Namespace(**{**vars(args), **overrides})


@pytest.fixture
def fake_mlflow(monkeypatch):
    fake = FakeMlflow()
    monkeypatch.setitem(sys.modules, "mlflow", fake)
    import requests

    monkeypatch.setattr(requests, "get", lambda *a, **kw: SimpleNamespace(status_code=200))
    return fake


@pytest.mark.parametrize(
    ("uri", "expected"),
    [
        ("https://mlflow.example.com/", "https://mlflow.example.com"),
        ("https://mlflow.example.com", "https://mlflow.example.com"),
        ("http://localhost:5000", "http://localhost:5000"),
        ("https://host/mlflow/", "https://host/mlflow"),
    ],
)
def test_validate_tracking_uri_accepts_http_servers(mlflow_utils, uri, expected):
    assert mlflow_utils.validate_tracking_uri(uri) == expected


@pytest.mark.parametrize(
    "uri",
    [
        "",  # --mlflow with no value and no MLFLOW_TRACKING_URI
        "mlflow.example.com",  # missing scheme
        "/local/mlruns",  # local path
        "file:///local/mlruns",  # unsupported backend
        "sqlite:///mlflow.db",
        "https://",  # no host
    ],
)
def test_validate_tracking_uri_rejects_non_servers(mlflow_utils, uri):
    with pytest.raises(ValueError):
        mlflow_utils.validate_tracking_uri(uri)


@pytest.mark.parametrize(
    ("ckpt", "recipe", "qformat", "expected"),
    [
        # Local directory, trailing slash, built-in recipe name.
        (
            "/models/Llama-3.3-70B-Instruct/",
            "general/ptq/nvfp4_default-kv_fp8_cast",
            "fp8",
            "tester/hf_ptq/Llama-3.3-70B-Instruct-nvfp4_default-kv_fp8_cast",
        ),
        # Hugging Face id collapses to its basename; recipe file path drops the suffix.
        (
            "nvidia/Llama-3.3-70B-Instruct",
            "./my/tuned.yaml",
            "fp8",
            "tester/hf_ptq/Llama-3.3-70B-Instruct-tuned",
        ),
        # No recipe: the quantization format names the variant.
        ("openai/gpt-oss-20b", None, "nvfp4", "tester/hf_ptq/gpt-oss-20b-nvfp4"),
        # Comma-separated formats and other unsafe characters are sanitized.
        ("/models/my model!", None, "nvfp4,fp8", "tester/hf_ptq/my_model-nvfp4_fp8"),
    ],
)
def test_default_experiment_name(mlflow_utils, ckpt, recipe, qformat, expected):
    args = _args(pyt_ckpt_path=ckpt, recipe=recipe, qformat=qformat)
    assert mlflow_utils.default_experiment_name(args) == expected


def test_default_experiment_name_survives_unusable_username(mlflow_utils, monkeypatch):
    """A container without a passwd entry for the uid must not break the run."""
    monkeypatch.setattr(getpass, "getuser", lambda: (_ for _ in ()).throw(OSError))
    args = _args(pyt_ckpt_path="/models/Qwen3-0.6B", recipe=None, qformat="nvfp4")
    assert mlflow_utils.default_experiment_name(args) == "unknown/hf_ptq/Qwen3-0.6B-nvfp4"


def test_tee_stream_writes_to_both_and_delegates(mlflow_utils):
    original, sink = io.StringIO(), io.StringIO()
    tee = mlflow_utils.TeeStream(original, sink)

    print("hello", file=tee)
    tee.flush()

    assert original.getvalue() == "hello\n"
    assert sink.getvalue() == "hello\n"
    # Progress bars check isatty(); it must report the real stream, not the tee.
    assert tee.isatty() == original.isatty()


def test_logger_is_inert_without_the_flag(mlflow_utils, monkeypatch):
    """Without --mlflow nothing is imported, captured or uploaded."""
    monkeypatch.setitem(sys.modules, "mlflow", None)
    logger = mlflow_utils.MlflowRunLogger(_args(mlflow=None))
    stdout = sys.stdout

    logger.start()
    assert sys.stdout is stdout
    logger.finish("FINISHED")


def test_logger_skips_non_main_ranks(mlflow_utils, monkeypatch):
    monkeypatch.setitem(sys.modules, "mlflow", None)
    args = _args(dist_state=SimpleNamespace(is_main=False, world_size=8))
    logger = mlflow_utils.MlflowRunLogger(args)

    logger.start()
    logger.finish("FINISHED")


def test_logger_logs_inputs_and_outputs(mlflow_utils, fake_mlflow, tmp_path, monkeypatch):
    monkeypatch.setattr(sys, "argv", ["hf_ptq.py", "--pyt_ckpt_path", "/models/Qwen3-0.6B"])
    (tmp_path / ".quant_summary.txt").write_text("448 TensorQuantizers found in model\n")
    (tmp_path / ".moe.html").write_text("<html>experts</html>")
    args = _args(
        recipe="general/ptq/nvfp4_default-kv_fp8_cast",
        export_path=str(tmp_path),
    )
    logger = mlflow_utils.MlflowRunLogger(args)

    logger.start()
    try:
        assert fake_mlflow.tracking_uri == "https://mlflow.example.com"
        assert fake_mlflow.experiment == "tester/hf_ptq/model-nvfp4"
        # The default run name is the UTC start time.
        assert len(fake_mlflow.run_name) == 15 and fake_mlflow.run_name[8] == "-"
    finally:
        logger.finish("FINISHED")

    assert fake_mlflow.params["model"] == "/models/Qwen3-0.6B"
    assert fake_mlflow.params["qformat"] == "nvfp4"
    assert fake_mlflow.params["kv_cache_qformat"] == "fp8_cast"
    assert fake_mlflow.tags["user"] == "tester"

    command = fake_mlflow.texts["command.txt"]
    assert "hf_ptq.py --pyt_ckpt_path /models/Qwen3-0.6B" in command
    assert "torchrun" not in command

    # The recipe is uploaded resolved, so $imports are expanded and it stands alone.
    recipe = yaml.safe_load(fake_mlflow.texts["recipe/resolved_recipe.yaml"])
    assert recipe["metadata"]["recipe_type"] == "ptq"
    assert recipe["quantize"]["quant_cfg"]

    assert "TensorQuantizers" in fake_mlflow.texts["summary/quant_summary.txt"]
    assert fake_mlflow.texts["summary/moe.html"] == "<html>experts</html>"
    assert ("hf_ptq.log", "logs") in fake_mlflow.artifacts
    assert "total_time_s" in fake_mlflow.metrics
    assert fake_mlflow.status == "FINISHED"


def test_command_flags_the_invisible_torchrun_wrapper(
    mlflow_utils, fake_mlflow, tmp_path, monkeypatch
):
    """Under torchrun, sys.argv is the worker's, so the launcher must be called out."""
    monkeypatch.setattr(sys, "argv", ["hf_ptq.py", "--use_fsdp2"])
    monkeypatch.setenv("WORLD_SIZE", "8")
    monkeypatch.setenv("LOCAL_WORLD_SIZE", "8")
    logger = mlflow_utils.MlflowRunLogger(_args(recipe=None, export_path=str(tmp_path)))

    logger.start()
    logger.finish("FINISHED")

    command = fake_mlflow.texts["command.txt"]
    assert "hf_ptq.py --use_fsdp2" in command
    assert "WORLD_SIZE=8" in command and "not part of sys.argv" in command


def test_capture_includes_preconfigured_library_logging(mlflow_utils, fake_mlflow, tmp_path):
    """transformers/huggingface_hub bind sys.stderr at import, long before capture starts."""
    library_logger = logging.getLogger("test_preconfigured_library")
    handler = logging.StreamHandler(sys.stderr)
    library_logger.addHandler(handler)
    logger = mlflow_utils.MlflowRunLogger(_args(recipe=None, export_path=str(tmp_path)))

    try:
        logger.start()
        log_path = logger._log_dir / "hf_ptq.log"
        library_logger.warning("Rate limited. Waiting 169.0s before retry")
        captured = log_path.read_text()
        logger.finish("FINISHED")
    finally:
        library_logger.removeHandler(handler)

    assert "Rate limited" in captured
    # The handler must be handed back its own stream, or later logging writes to a closed file.
    assert handler.stream is sys.stderr


def test_logger_omits_recipe_artifact_without_a_recipe(mlflow_utils, fake_mlflow, tmp_path):
    logger = mlflow_utils.MlflowRunLogger(_args(recipe=None, export_path=str(tmp_path)))

    logger.start()
    logger.finish("FINISHED")

    assert "recipe/resolved_recipe.yaml" not in fake_mlflow.texts
    assert fake_mlflow.params["recipe"] == ""


def test_logger_restores_streams_and_reports_failure(mlflow_utils, fake_mlflow, tmp_path):
    """A failed quantization is still recorded, with its log attached."""
    logger = mlflow_utils.MlflowRunLogger(_args(recipe=None, export_path=str(tmp_path)))
    stdout, stderr = sys.stdout, sys.stderr

    logger.start()
    print("calibrating")
    logger.finish("FAILED")

    assert sys.stdout is stdout and sys.stderr is stderr
    assert fake_mlflow.status == "FAILED"
    assert ("hf_ptq.log", "logs") in fake_mlflow.artifacts


def test_logger_never_raises_when_the_server_dies_mid_run(mlflow_utils, fake_mlflow, tmp_path):
    logger = mlflow_utils.MlflowRunLogger(_args(recipe=None, export_path=str(tmp_path)))
    logger.start()

    def explode(*args, **kwargs):
        raise RuntimeError("server gone")

    fake_mlflow.log_artifact = explode
    logger.finish("FINISHED")

    assert sys.stdout is not None and not isinstance(sys.stdout, mlflow_utils.TeeStream)


def test_unreachable_server_fails_before_the_model_loads(mlflow_utils, monkeypatch, tmp_path):
    import requests

    monkeypatch.setitem(sys.modules, "mlflow", FakeMlflow())
    monkeypatch.setattr(
        requests,
        "get",
        lambda *a, **kw: (_ for _ in ()).throw(requests.ConnectionError("no route to host")),
    )
    logger = mlflow_utils.MlflowRunLogger(_args(recipe=None, export_path=str(tmp_path)))
    stdout = sys.stdout

    with pytest.raises(ConnectionError, match="unreachable"):
        logger.start()

    # The capture must be torn down so the failure is readable on the console.
    assert sys.stdout is stdout


def test_parse_args_defaults_the_experiment_name(monkeypatch):
    monkeypatch.syspath_prepend(str(_EXAMPLES_DIR))
    monkeypatch.setattr(getpass, "getuser", lambda: "tester")
    hf_ptq = importlib.import_module("hf_ptq")
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "hf_ptq.py",
            "--pyt_ckpt_path",
            "/models/Qwen3-0.6B",
            "--recipe",
            "general/ptq/nvfp4_default-kv_fp8_cast",
            "--mlflow",
            "https://mlflow.example.com/",
        ],
    )

    args = hf_ptq.parse_args()

    assert args.mlflow == "https://mlflow.example.com"
    assert args.mlflow_experiment == "tester/hf_ptq/Qwen3-0.6B-nvfp4_default-kv_fp8_cast"
    assert args.mlflow_run_name is None


def test_parse_args_leaves_mlflow_off_by_default(monkeypatch):
    monkeypatch.syspath_prepend(str(_EXAMPLES_DIR))
    hf_ptq = importlib.import_module("hf_ptq")
    monkeypatch.setattr(sys, "argv", ["hf_ptq.py", "--pyt_ckpt_path", "/models/Qwen3-0.6B"])

    args = hf_ptq.parse_args()

    assert args.mlflow is None
    assert args.mlflow_experiment is None


def test_parse_args_rejects_a_bad_tracking_uri(monkeypatch):
    monkeypatch.syspath_prepend(str(_EXAMPLES_DIR))
    hf_ptq = importlib.import_module("hf_ptq")
    monkeypatch.setattr(
        sys,
        "argv",
        ["hf_ptq.py", "--pyt_ckpt_path", "/models/Qwen3-0.6B", "--mlflow", "not-a-url"],
    )

    with pytest.raises(SystemExit):
        hf_ptq.parse_args()


def test_mlflow_flag_falls_back_to_the_environment(monkeypatch):
    monkeypatch.setenv("MLFLOW_TRACKING_URI", "https://mlflow.example.com/")
    monkeypatch.syspath_prepend(str(_EXAMPLES_DIR))
    monkeypatch.setattr(getpass, "getuser", lambda: "tester")
    hf_ptq = importlib.import_module("hf_ptq")
    monkeypatch.setattr(
        sys, "argv", ["hf_ptq.py", "--pyt_ckpt_path", "/models/Qwen3-0.6B", "--mlflow"]
    )

    args = hf_ptq.parse_args()

    assert args.mlflow == "https://mlflow.example.com"
