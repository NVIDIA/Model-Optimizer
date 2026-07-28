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
import io
import logging
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

import modelopt
from modelopt.torch.utils.mlflow import (
    MlflowRunLogger,
    TeeStream,
    default_experiment_name,
    validate_tracking_uri,
)

URI = "https://mlflow.example.com"


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

    def log_metrics(self, metrics):
        self.metrics.update(metrics)

    def end_run(self, status=None):
        self.status = status


@pytest.fixture
def fake_mlflow(monkeypatch):
    fake = FakeMlflow()
    monkeypatch.setitem(sys.modules, "mlflow", fake)
    import requests

    monkeypatch.setattr(requests, "get", lambda *a, **kw: SimpleNamespace(status_code=200))
    return fake


@pytest.fixture(autouse=True)
def deterministic_user(monkeypatch):
    monkeypatch.setattr(getpass, "getuser", lambda: "tester")


def _logger(**kwargs):
    kwargs.setdefault("experiment_name", "tester/hf_ptq/model-nvfp4")
    return MlflowRunLogger(URI, **kwargs)


@pytest.mark.parametrize(
    ("uri", "expected"),
    [
        (f"{URI}/", URI),
        (URI, URI),
        ("http://localhost:5000", "http://localhost:5000"),
        ("https://host/mlflow/", "https://host/mlflow"),
    ],
)
def test_validate_tracking_uri_accepts_http_servers(uri, expected):
    assert validate_tracking_uri(uri) == expected


@pytest.mark.parametrize(
    "uri",
    [
        "",  # no URI given and no MLFLOW_TRACKING_URI
        "mlflow.example.com",  # missing scheme
        "/local/mlruns",  # local path
        "file:///local/mlruns",  # unsupported backend
        "sqlite:///mlflow.db",
        "https://",  # no host
    ],
)
def test_validate_tracking_uri_rejects_non_servers(uri):
    with pytest.raises(ValueError):
        validate_tracking_uri(uri)


def test_missing_scheme_is_the_only_case_that_suggests_https():
    """Suggesting https://sqlite:///... for a URI that already has a scheme is nonsense."""
    with pytest.raises(ValueError, match=r"Did you mean https://mlflow\.example\.com\?"):
        validate_tracking_uri("mlflow.example.com")
    with pytest.raises(ValueError) as excinfo:
        validate_tracking_uri("sqlite:///mlflow.db")
    assert "Did you mean" not in str(excinfo.value)


@pytest.mark.parametrize(
    ("model", "variant", "expected"),
    [
        # Local directory with a trailing slash.
        (
            "/models/Llama-3.3-70B-Instruct/",
            "nvfp4_default-kv_fp8_cast",
            "tester/hf_ptq/Llama-3.3-70B-Instruct-nvfp4_default-kv_fp8_cast",
        ),
        # A Hugging Face id collapses to its basename.
        ("nvidia/Llama-3.3-70B-Instruct", "tuned", "tester/hf_ptq/Llama-3.3-70B-Instruct-tuned"),
        ("openai/gpt-oss-20b", "nvfp4", "tester/hf_ptq/gpt-oss-20b-nvfp4"),
        # Version dots survive; other unsafe characters do not.
        ("/models/Qwen3.6-35B-A3B", "nvfp4", "tester/hf_ptq/Qwen3.6-35B-A3B-nvfp4"),
        ("/models/my model!", "nvfp4,fp8", "tester/hf_ptq/my_model-nvfp4_fp8"),
    ],
)
def test_default_experiment_name(model, variant, expected):
    assert default_experiment_name("hf_ptq", model, variant) == expected


def test_default_experiment_name_takes_an_explicit_user_and_tool():
    assert default_experiment_name("llm_eval", "/m/Qwen3-0.6B", "mmlu", user="alice") == (
        "alice/llm_eval/Qwen3-0.6B-mmlu"
    )


def test_default_experiment_name_survives_unusable_username(monkeypatch):
    """A container without a passwd entry for the uid must not break the run."""
    monkeypatch.setattr(getpass, "getuser", lambda: (_ for _ in ()).throw(OSError))
    assert default_experiment_name("hf_ptq", "/m/Qwen3-0.6B", "nvfp4") == (
        "unknown/hf_ptq/Qwen3-0.6B-nvfp4"
    )


def test_default_experiment_name_stays_storable():
    """SQL-backed stores keep experiment names in a VARCHAR(256) column."""
    name = default_experiment_name("t" * 150, "/m/" + "m" * 150, "v" * 150, user="u" * 150)

    assert len(name) <= 250


def test_tee_stream_writes_to_both_and_delegates():
    original, sink = io.StringIO(), io.StringIO()
    tee = TeeStream(original, sink)

    print("hello", file=tee)
    tee.flush()

    assert original.getvalue() == "hello\n"
    assert sink.getvalue() == "hello\n"
    # Progress bars check isatty(); it must report the real stream, not the tee.
    assert tee.isatty() == original.isatty()


def test_tee_stream_does_not_recurse_on_its_own_attributes():
    """Delegating _stream/_sink when they are unset would recurse until the stack blows."""
    tee = TeeStream.__new__(TeeStream)  # never ran __init__, so both are missing

    with pytest.raises(AttributeError):
        tee._stream


def test_tee_stream_tolerates_a_closed_sink():
    """Handlers registered after start keep the tee; writes must not raise once it is closed."""
    original, sink = io.StringIO(), io.StringIO()
    tee = TeeStream(original, sink)
    sink.close()

    tee.write("after close")
    tee.flush()

    assert original.getvalue() == "after close"


def test_logger_is_inert_when_disabled(monkeypatch):
    """Disabled means nothing is imported, captured or uploaded."""
    monkeypatch.setitem(sys.modules, "mlflow", None)
    logger = _logger(enabled=False)
    stdout = sys.stdout

    logger.start(params={"model": "x"})
    assert sys.stdout is stdout
    assert logger.run_url == ""
    logger.finish("FINISHED")


def test_logger_logs_inputs_and_outputs(fake_mlflow, tmp_path, monkeypatch):
    monkeypatch.setattr(sys, "argv", ["hf_ptq.py", "--pyt_ckpt_path", "/models/Qwen3-0.6B"])
    (tmp_path / ".quant_summary.txt").write_text("706 TensorQuantizers found in model\n")
    logger = _logger(run_name="unit-test")

    logger.start(
        params={"model": "/models/Qwen3-0.6B", "qformat": "nvfp4"},
        tags={"extra": "value"},
        texts={"recipe/resolved_recipe.yaml": "metadata:\n  recipe_type: ptq\n"},
    )
    try:
        assert fake_mlflow.tracking_uri == URI
        assert fake_mlflow.experiment == "tester/hf_ptq/model-nvfp4"
        assert fake_mlflow.run_name == "unit-test"
        assert logger.run_url == f"{URI}/#/experiments/7/runs/deadbeef"
    finally:
        logger.finish(
            "FINISHED",
            files={
                "summary/quant_summary.txt": tmp_path / ".quant_summary.txt",
                "summary/moe.html": tmp_path / ".moe.html",  # absent: must be skipped
            },
        )

    assert fake_mlflow.params == {"model": "/models/Qwen3-0.6B", "qformat": "nvfp4"}
    assert fake_mlflow.tags["extra"] == "value"
    assert fake_mlflow.tags["user"] == "tester"

    command = fake_mlflow.texts["command.txt"]
    assert "hf_ptq.py --pyt_ckpt_path /models/Qwen3-0.6B" in command
    assert "torchrun" not in command
    assert "recipe_type: ptq" in fake_mlflow.texts["recipe/resolved_recipe.yaml"]

    # The ModelOpt version travels with the run as an artifact, not only as a tag.
    version = fake_mlflow.texts["version.txt"]
    assert version.strip() == modelopt.__version__ and version.endswith("\n")
    assert fake_mlflow.tags["modelopt_version"] == modelopt.__version__

    # The log keeps its name; the summary is renamed out of its dotfile form.
    assert ("hf_ptq.log", "logs") in fake_mlflow.artifacts
    assert ("quant_summary.txt", "summary") in fake_mlflow.artifacts
    assert not any(name == "moe.html" for name, _ in fake_mlflow.artifacts)
    assert "total_time_s" in fake_mlflow.metrics
    assert fake_mlflow.status == "FINISHED"


def test_run_name_defaults_to_the_utc_timestamp(fake_mlflow):
    logger = _logger()

    logger.start()
    logger.finish("FINISHED")

    assert len(fake_mlflow.run_name) == 15 and fake_mlflow.run_name[8] == "-"


def test_command_flags_the_invisible_torchrun_wrapper(fake_mlflow, monkeypatch):
    """Under torchrun, sys.argv is the worker's, so the launcher must be called out."""
    monkeypatch.setattr(sys, "argv", ["hf_ptq.py", "--use_fsdp2"])
    monkeypatch.setenv("WORLD_SIZE", "8")
    monkeypatch.setenv("LOCAL_WORLD_SIZE", "8")
    logger = _logger()

    logger.start()
    logger.finish("FINISHED")

    command = fake_mlflow.texts["command.txt"]
    assert "hf_ptq.py --use_fsdp2" in command
    assert "WORLD_SIZE=8" in command and "not part of sys.argv" in command


def test_capture_includes_preconfigured_library_logging(fake_mlflow, monkeypatch):
    """transformers/huggingface_hub bind sys.stderr at import, long before capture starts."""
    monkeypatch.setattr(sys, "argv", ["hf_ptq.py"])
    library_logger = logging.getLogger("test_preconfigured_library")
    handler = logging.StreamHandler(sys.stderr)
    library_logger.addHandler(handler)
    logger = _logger()

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


def test_logger_restores_streams_and_reports_failure(fake_mlflow, monkeypatch):
    """A failed run is still recorded, with its log attached."""
    monkeypatch.setattr(sys, "argv", ["hf_ptq.py"])
    logger = _logger()
    stdout, stderr = sys.stdout, sys.stderr

    logger.start()
    print("calibrating")
    logger.finish("FAILED")

    assert sys.stdout is stdout and sys.stderr is stderr
    assert fake_mlflow.status == "FAILED"
    assert ("hf_ptq.log", "logs") in fake_mlflow.artifacts


def test_logger_never_raises_when_the_server_dies_mid_run(fake_mlflow, capsys):
    logger = _logger()
    logger.start()

    def explode(*args, **kwargs):
        raise RuntimeError("server gone")

    fake_mlflow.log_artifact = explode
    logger.finish("FINISHED")

    assert not isinstance(sys.stdout, TeeStream)
    # A swallowed upload failure must still be visible, or the run looks complete.
    assert "server gone" in capsys.readouterr().out


def test_start_is_idempotent(fake_mlflow):
    """A second start would install a second tee and orphan the first temp directory."""
    logger = _logger(run_name="first")
    logger.start()
    fake_mlflow.run_name = "clobbered"

    logger.start()

    assert fake_mlflow.run_name == "clobbered"  # no second start_run
    assert isinstance(sys.stdout, TeeStream) and not isinstance(sys.stdout._stream, TeeStream)
    logger.finish("FINISHED")


def test_unreachable_server_fails_before_the_work_starts(monkeypatch):
    import requests

    monkeypatch.setitem(sys.modules, "mlflow", FakeMlflow())
    monkeypatch.setattr(
        requests,
        "get",
        lambda *a, **kw: (_ for _ in ()).throw(requests.ConnectionError("no route to host")),
    )
    logger = _logger()
    stdout = sys.stdout

    with pytest.raises(ConnectionError, match="unreachable"):
        logger.start()

    # The capture must be torn down so the failure is readable on the console.
    assert sys.stdout is stdout
