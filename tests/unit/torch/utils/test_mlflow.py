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
from modelopt.torch.utils.logging import TeeStream
from modelopt.torch.utils.mlflow import (
    MlflowRunLogger,
    _git_sha,
    _redact_argv,
    command_text,
    default_experiment_name,
    validate_tracking_uri,
)

URI = "https://mlflow.example.com"
# Fake credentials for the redaction tests. TruffleHog's URI detector flags any
# scheme://user:pass@host, so the marker sits on the definitions; these tests exist
# precisely to prove such credentials are masked.
CREDS_URI = "https://user:tok@mlflow.example.com"  # trufflehog:ignore
SHORT_CREDS_URI = "https://u:tok@host"  # trufflehog:ignore


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
        self.artifact_text = {}

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
        self.artifact_text[Path(local_path).name] = Path(local_path).read_text()

    def log_metrics(self, metrics):
        self.metrics.update(metrics)

    def end_run(self, status=None):
        self.status = status


@pytest.fixture
def fake_mlflow(monkeypatch):
    fake = FakeMlflow()
    monkeypatch.setitem(sys.modules, "mlflow", fake)
    return fake


def _unreachable(fake):
    """Make MLflow's own first request fail, the way a dead server does."""

    def explode(*args, **kwargs):
        raise ConnectionError("no route to host")

    fake.set_experiment = explode
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


def test_command_can_record_another_processs_invocation(monkeypatch):
    """A worker's own sys.argv is spawn plumbing, so the caller can supply the real one."""
    monkeypatch.setattr(sys, "argv", ["-c", "from multiprocessing.spawn import spawn_main"])

    command = command_text(["vllm_serve_fakequant.py", "/ckpts/model", "--api-key", "sk-secret"])

    assert "vllm_serve_fakequant.py /ckpts/model" in command
    assert "sk-secret" not in command
    assert "spawn_main" not in command


def test_log_text_uploads_while_the_run_is_open(fake_mlflow):
    """For a value settled midway through, which a later crash would otherwise lose."""
    logger = _logger()
    logger.start()

    logger.log_text("recipe/quant_cfg.yaml", "quant_cfg: {}\n")
    uploaded_before_finish = fake_mlflow.texts.get("recipe/quant_cfg.yaml")

    logger.finish("FAILED")
    assert uploaded_before_finish == "quant_cfg: {}\n"


def test_log_text_is_inert_outside_an_open_run(fake_mlflow):
    logger = _logger(enabled=False)
    logger.log_text("recipe/quant_cfg.yaml", "quant_cfg: {}\n")

    logger = _logger()  # enabled, but never started
    logger.log_text("recipe/quant_cfg.yaml", "quant_cfg: {}\n")

    assert fake_mlflow.texts == {}


def test_log_text_never_raises_when_the_upload_fails(fake_mlflow, capsys):
    """Losing one artifact must not take down the quantization that produced it."""
    logger = _logger()
    logger.start()

    def explode(*args, **kwargs):
        raise ConnectionError("no route to host")

    fake_mlflow.log_text = explode
    logger.log_text("recipe/quant_cfg.yaml", "quant_cfg: {}\n")
    logger.finish("FINISHED")

    assert "could not upload recipe/quant_cfg.yaml" in capsys.readouterr().out


def test_capture_includes_preconfigured_library_logging(fake_mlflow, monkeypatch):
    """transformers/huggingface_hub bind sys.stderr at import, long before capture starts."""
    monkeypatch.setattr(sys, "argv", ["hf_ptq.py"])
    library_logger = logging.getLogger("test_preconfigured_library")
    handler = logging.StreamHandler(sys.stderr)
    library_logger.addHandler(handler)
    logger = _logger()

    try:
        logger.start()
        log_path = logger._log_path
        library_logger.warning("Rate limited. Waiting 169.0s before retry")
        captured = log_path.read_text()
        logger.finish("FINISHED")
    finally:
        library_logger.removeHandler(handler)

    assert "Rate limited" in captured
    # The handler must be handed back its own stream, or later logging writes to a closed file.
    assert handler.stream is sys.stderr


def test_capture_hands_back_handlers_bound_during_the_run(fake_mlflow, monkeypatch):
    """A library imported mid-run binds sys.stderr -- which is the tee at that point. Restoring
    only the handlers seen at start would leave it writing to a closed file forever."""
    monkeypatch.setattr(sys, "argv", ["hf_ptq.py"])
    logger = _logger()
    late = logging.getLogger("test_bound_during_run")

    logger.start()
    handler = logging.StreamHandler(sys.stderr)  # sys.stderr is the tee here
    late.addHandler(handler)
    try:
        logger.finish("FINISHED")
        assert handler.stream is sys.stderr
        assert not isinstance(handler.stream, TeeStream)
        late.warning("after the run")  # must not raise on a closed sink
    finally:
        late.removeHandler(handler)


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
    """Opening the run is the readiness check -- MLflow's own request, so it honours the
    client's TLS and retry configuration instead of a parallel probe that would not."""
    monkeypatch.setitem(sys.modules, "mlflow", _unreachable(FakeMlflow()))
    logger = _logger()
    stdout = sys.stdout

    with pytest.raises(ConnectionError, match="no route to host"):
        logger.start()

    # The capture must be torn down so the failure is readable on the console.
    assert sys.stdout is stdout


@pytest.mark.parametrize(
    ("argv", "expected"),
    [
        # A secret-looking option masks the value that follows it.
        (["--hf_token", "hf_abc123"], ["--hf_token", "***"]),
        (["--api-key=abc123"], ["--api-key=***"]),
        (["--password", "hunter2", "--verbose"], ["--password", "***", "--verbose"]),
        # Credentials embedded in a URI are masked wherever they appear.
        (
            ["--mlflow", CREDS_URI],
            ["--mlflow", "https://***@mlflow.example.com"],
        ),
        # Ordinary arguments are untouched, including values that merely contain the word.
        # A secret value can itself start with "-".
        (["--hf_token", "-secret"], ["--hf_token", "***"]),
        (["--dataset", "token_data"], ["--dataset", "token_data"]),
        (["--pyt_ckpt_path", "/models/Qwen3-0.6B"], ["--pyt_ckpt_path", "/models/Qwen3-0.6B"]),
    ],
)
def test_redact_argv_masks_credentials(argv, expected):
    assert _redact_argv(argv) == expected


def test_command_artifact_carries_no_secrets(fake_mlflow, monkeypatch):
    """argv reaches the server as command.txt, so secrets in it must not."""
    monkeypatch.setattr(
        sys, "argv", ["run.py", "--hf_token", "hf_abc123", "--mlflow", SHORT_CREDS_URI]
    )
    logger = _logger()

    logger.start()
    logger.finish("FINISHED")

    command = fake_mlflow.texts["command.txt"]
    assert "run.py" in command and "--hf_token" in command
    for secret in ("hf_abc123", "u:tok"):
        assert secret not in command


def test_params_and_run_url_mask_credentials(monkeypatch):
    """A credential-bearing tracking URI is usable, but must not be echoed or uploaded."""
    fake = FakeMlflow()
    monkeypatch.setitem(sys.modules, "mlflow", fake)
    monkeypatch.setattr(sys, "argv", ["run.py"])
    logger = MlflowRunLogger(CREDS_URI, "tester/hf_ptq/m-nvfp4", run_name="masked")

    logger.start(params={"hf_token": "secret", "endpoint": "https://u:p@host", "qformat": "nvfp4"})
    url = logger.run_url
    logger.finish("FINISHED")

    assert fake.params == {"hf_token": "***", "endpoint": "https://***@host", "qformat": "nvfp4"}
    assert "tok" not in url and "https://***@mlflow.example.com" in url
    # The real URI is still what talks to the server.
    assert fake.tracking_uri == CREDS_URI


def test_failure_after_start_run_does_not_orphan_the_run(monkeypatch):
    """If logging the inputs fails, the run would otherwise sit in RUNNING forever."""
    fake = FakeMlflow()
    monkeypatch.setitem(sys.modules, "mlflow", fake)

    def explode(*args, **kwargs):
        raise RuntimeError("network blip")

    fake.log_params = explode
    logger = _logger()
    stdout = sys.stdout

    with pytest.raises(RuntimeError, match="network blip"):
        logger.start(params={"model": "x"})

    assert fake.status == "FAILED"
    assert sys.stdout is stdout
    # finish() is now inert: the caller never got a logger to call it on anyway.
    logger.finish("FINISHED")
    assert fake.status == "FAILED"


def test_git_sha_is_read_without_a_subprocess():
    """Bandit forbids the subprocess call this used to make; it reads .git directly now."""
    sha = _git_sha()

    assert sha == "unknown" or (1 <= len(sha) <= 9 and all(c in "0123456789abcdef" for c in sha))
    assert "subprocess" not in dir(sys.modules["modelopt.torch.utils.mlflow"])


@pytest.mark.parametrize("in_worktree", [False, True])
def test_git_sha_resolves_in_a_checkout_and_a_worktree(tmp_path, monkeypatch, in_worktree):
    """A worktree's .git is a *file* pointing at the real git dir, and its refs live in the
    main checkout -- a directory-only reader silently reports "unknown" for every worktree."""
    main = tmp_path / "repo" / ".git"
    (main / "refs" / "heads").mkdir(parents=True)
    (main / "refs" / "heads" / "main").write_text("a" * 40 + "\n")

    if in_worktree:
        checkout = tmp_path / "wt"
        wt_git = main / "worktrees" / "wt"
        wt_git.mkdir(parents=True)
        (wt_git / "HEAD").write_text("ref: refs/heads/main\n")
        (wt_git / "commondir").write_text("../..\n")
        checkout.mkdir()
        (checkout / ".git").write_text(f"gitdir: {wt_git}\n")
    else:
        checkout = main.parent
        (main / "HEAD").write_text("ref: refs/heads/main\n")

    # _git_sha locates .git relative to the module file, three parents up.
    fake_module = checkout / "modelopt" / "torch" / "utils" / "mlflow.py"
    fake_module.parent.mkdir(parents=True)
    fake_module.touch()
    monkeypatch.setattr("modelopt.torch.utils.mlflow.__file__", str(fake_module))

    assert _git_sha() == "a" * 9


def test_git_sha_handles_a_detached_head(tmp_path, monkeypatch):
    git_dir = tmp_path / "repo" / ".git"
    git_dir.mkdir(parents=True)
    (git_dir / "HEAD").write_text("b" * 40 + "\n")
    fake_module = tmp_path / "repo" / "modelopt" / "torch" / "utils" / "mlflow.py"
    fake_module.parent.mkdir(parents=True)
    fake_module.touch()
    monkeypatch.setattr("modelopt.torch.utils.mlflow.__file__", str(fake_module))

    assert _git_sha() == "b" * 9


def test_track_closes_the_run_with_the_right_status(fake_mlflow, monkeypatch):
    """Mirrors mlflow.start_run(): the block's outcome decides the status."""
    monkeypatch.setattr(sys, "argv", ["hf_ptq.py"])

    with _logger().track(params={"qformat": "nvfp4"}):
        pass

    assert fake_mlflow.status == "FINISHED"
    assert fake_mlflow.params == {"qformat": "nvfp4"}


def test_track_marks_a_raising_block_failed(fake_mlflow, monkeypatch, tmp_path):
    monkeypatch.setattr(sys, "argv", ["hf_ptq.py"])
    stdout = sys.stdout

    summary = {"summary/quant_summary.txt": tmp_path / ".quant_summary.txt"}
    with (
        pytest.raises(RuntimeError, match="calibration exploded"),
        _logger().track(files=summary),
    ):
        # post_quantize writes the summary during the run, so the test must too.
        (tmp_path / ".quant_summary.txt").write_text("706 TensorQuantizers\n")
        raise RuntimeError("calibration exploded")

    assert fake_mlflow.status == "FAILED"
    assert sys.stdout is stdout
    # The outputs named upfront are still uploaded, and the traceback is in the log.
    assert ("quant_summary.txt", "summary") in fake_mlflow.artifacts
    assert "RuntimeError: calibration exploded" in fake_mlflow.artifact_text["hf_ptq.log"]


def test_failed_run_uploads_the_traceback(fake_mlflow, monkeypatch):
    """finish() runs from the caller's finally, before the interpreter prints the traceback,
    so without help the log stops at the last line the script printed."""
    monkeypatch.setattr(sys, "argv", ["hf_ptq.py"])
    logger = _logger()
    logger.start()
    print("calibrating")

    try:
        raise RuntimeError("calibration exploded")
    except RuntimeError:
        logger.finish("FAILED")

    log = fake_mlflow.artifact_text["hf_ptq.log"]
    assert "calibrating" in log
    assert "Traceback (most recent call last)" in log
    assert "RuntimeError: calibration exploded" in log


def test_successful_run_uploads_no_traceback(fake_mlflow, monkeypatch):
    monkeypatch.setattr(sys, "argv", ["hf_ptq.py"])
    logger = _logger()

    logger.start()
    logger.finish("FINISHED")

    assert "Traceback" not in fake_mlflow.artifact_text["hf_ptq.log"]


def test_only_files_this_run_produced_are_uploaded(fake_mlflow, tmp_path, monkeypatch):
    """An export directory is reused across attempts. A run that crashes before writing its
    summary must not upload the previous run's file as though it were its own."""
    monkeypatch.setattr(sys, "argv", ["hf_ptq.py"])
    stale = tmp_path / ".quant_summary.txt"
    stale.write_text("from a previous run\n")
    fresh = tmp_path / ".moe.html"
    logger = _logger()

    outputs = {"summary/quant_summary.txt": stale, "summary/moe.html": fresh}
    logger.start(files=outputs)
    fresh.write_text("<html>written by this run</html>")  # produced during the run
    logger.finish("FAILED", files=outputs)

    uploaded = [name for name, _ in fake_mlflow.artifacts]
    assert "moe.html" in uploaded
    assert "quant_summary.txt" not in uploaded


def test_stale_check_survives_unnormalized_string_paths(fake_mlflow, tmp_path, monkeypatch):
    """files accepts str as well as Path, and "./out/x" is the same file as "out/x" but not
    the same string -- keying the snapshot on the raw value would miss and upload it anyway."""
    monkeypatch.setattr(sys, "argv", ["hf_ptq.py"])
    monkeypatch.chdir(tmp_path)
    (tmp_path / "out").mkdir()
    stale = tmp_path / "out" / ".quant_summary.txt"
    stale.write_text("from a previous run\n")
    outputs = {"summary/quant_summary.txt": "./out/.quant_summary.txt"}
    logger = _logger()

    logger.start(files=outputs)
    logger.finish("FAILED", files=outputs)

    assert "quant_summary.txt" not in [name for name, _ in fake_mlflow.artifacts]


def test_optional_tracking_warns_and_continues_when_the_server_is_unreachable(monkeypatch, capsys):
    """A URI inferred from the environment must not be able to fail the job it is watching."""
    monkeypatch.setitem(sys.modules, "mlflow", _unreachable(FakeMlflow()))
    logger = _logger(required=False)
    stdout = sys.stdout

    logger.start(params={"model": "x"})  # must not raise
    logger.finish("FINISHED")

    assert logger.enabled is False
    assert sys.stdout is stdout
    assert "tracking disabled" in capsys.readouterr().out


def test_required_tracking_still_raises(monkeypatch):
    """An explicit request is different: a broken server should fail loudly."""
    monkeypatch.setitem(sys.modules, "mlflow", _unreachable(FakeMlflow()))

    with pytest.raises(ConnectionError, match="no route to host"):
        _logger(required=True).start()
