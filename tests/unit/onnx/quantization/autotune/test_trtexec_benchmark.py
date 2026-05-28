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

"""Tests for ``TrtExecBenchmark`` — the trtexec-CLI benchmarking pipeline.

These tests fully mock ``subprocess.run`` so the trtexec binary is never
invoked. They cover:

- Construction of the ``trtexec`` base command (args, plugin libraries).
- ``--remoteAutoTuningConfig`` URL parsing (both ``--key=value`` and
  ``--key value`` forms) and the validation errors it raises.
- Auto-injection of ``--safe`` and ``--skipInference`` for remote autotuning.
- The ``run()`` pipeline: standard local invocation; remote scp + ssh
  ``trtexec_safe`` invocation; fallback to ``trtexec --safe`` when
  ``trtexec_safe`` fails; ``sshpass`` prefix when a password is configured.
- Latency parsing from both ``_STD_PATTERN`` (GPU Compute Time) and
  ``_SAFE_PATTERN`` (Average over N runs - GPU latency).
- Error paths: non-zero trtexec returncode, scp failure, missing trtexec
  binary, unparseable stdout.
"""

from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

# Importing ``benchmark`` transitively triggers ``import tensorrt``. In
# environments where the package is locatable but its shared libs are missing
# (e.g. partial CUDA installs), collection would otherwise fail. Mirror the
# soft-skip pattern used by ``test_region_inspect.py``.
try:
    from modelopt.onnx.quantization.autotune import benchmark as bm
    from modelopt.onnx.quantization.autotune.benchmark import TrtExecBenchmark
except ImportError:  # pragma: no cover — exercised only in TRT-less envs
    pytest.skip("TrtExecBenchmark requires TensorRT", allow_module_level=True)


def _make_proc(returncode=0, stdout="", stderr=""):
    """Build a ``subprocess.run``-style result object."""
    return SimpleNamespace(returncode=returncode, stdout=stdout, stderr=stderr)


# ===========================================================================
# Standalone helper tests — these exercise the module-level functions in
# isolation, with no TrtExecBenchmark construction or filesystem state.
# ===========================================================================


# --- _redact_url_password ---


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("ssh://alice:s3cret@host", "ssh://alice:******@host"),
        ("ssh://alice:s3cret@host:22/path?q=1", "ssh://alice:******@host:22/path?q=1"),
        ("ssh://alice@host", "ssh://alice@host"),  # no password
        ("ssh://host", "ssh://host"),  # no userinfo
        ("--flag=ssh://u:p@h", "--flag=ssh://u:******@h"),
        ("https://user:secret@host", "https://user:******@host"),  # any scheme
    ],
)
def test_redact_url_password(raw, expected):
    assert bm._redact_url_password(raw) == expected


# --- _build_base_trtexec_cmd ---


def test_build_base_trtexec_cmd_includes_expected_flags(tmp_path):
    cmd = bm._build_base_trtexec_cmd(
        timing_runs=11,
        warmup_runs=3,
        engine_path=str(tmp_path / "engine.trt"),
        timing_cache_file=str(tmp_path / "cache.bin"),
    )
    assert "--avgRuns=11" in cmd
    assert "--iterations=11" in cmd
    assert "--warmUp=3" in cmd
    assert "--stronglyTyped" in cmd
    assert any(arg.startswith("--saveEngine=") for arg in cmd)
    assert any(arg.startswith("--timingCacheFile=") for arg in cmd)


def test_build_base_trtexec_cmd_skips_missing_plugin(tmp_path):
    cmd = bm._build_base_trtexec_cmd(
        timing_runs=1,
        warmup_runs=1,
        engine_path=str(tmp_path / "engine.trt"),
        timing_cache_file=str(tmp_path / "cache.bin"),
        plugin_libraries=[str(tmp_path / "absent.so")],
    )
    assert not any("--staticPlugins" in arg for arg in cmd)


def test_build_base_trtexec_cmd_adds_existing_plugin(tmp_path):
    plugin = tmp_path / "plugin.so"
    plugin.write_bytes(b"")
    cmd = bm._build_base_trtexec_cmd(
        timing_runs=1,
        warmup_runs=1,
        engine_path=str(tmp_path / "engine.trt"),
        timing_cache_file=str(tmp_path / "cache.bin"),
        plugin_libraries=[str(plugin)],
    )
    assert f"--staticPlugins={plugin.resolve()}" in cmd


def test_build_base_trtexec_cmd_warns_via_logger_on_missing_plugin(tmp_path):
    log = MagicMock()
    bm._build_base_trtexec_cmd(
        timing_runs=1,
        warmup_runs=1,
        engine_path=str(tmp_path / "engine.trt"),
        timing_cache_file=str(tmp_path / "cache.bin"),
        plugin_libraries=[str(tmp_path / "absent.so")],
        log=log,
    )
    assert log.warning.called
    assert "Plugin library not found" in log.warning.call_args.args[0]


# --- _extract_remote_config_value ---


def test_extract_remote_config_value_returns_none_when_absent():
    assert bm._extract_remote_config_value(["--fp16", "--workspace=4096"]) is None


def test_extract_remote_config_value_equals_form():
    args = ["--fp16", "--remoteAutoTuningConfig=ssh://a:p@h?x=1"]
    assert bm._extract_remote_config_value(args) == "ssh://a:p@h?x=1"


def test_extract_remote_config_value_space_form():
    args = ["--fp16", "--remoteAutoTuningConfig", "ssh://a:p@h?x=1"]
    assert bm._extract_remote_config_value(args) == "ssh://a:p@h?x=1"


def test_extract_remote_config_value_empty_value_returned_verbatim():
    """Empty value (``--remoteAutoTuningConfig=``) returned as ``""`` for the caller to flag."""
    assert bm._extract_remote_config_value(["--remoteAutoTuningConfig="]) == ""


def test_extract_remote_config_value_rejects_duplicates():
    args = [
        "--remoteAutoTuningConfig=ssh://a:p@h?x=1",
        "--remoteAutoTuningConfig=ssh://a:p@h?x=2",
    ]
    with pytest.raises(ValueError, match="Exactly one"):
        bm._extract_remote_config_value(args)


def test_extract_remote_config_value_missing_value_at_end_of_argv():
    with pytest.raises(ValueError, match="Missing value"):
        bm._extract_remote_config_value(["--remoteAutoTuningConfig"])


def test_extract_remote_config_value_malformed_redacts_password():
    secret = "SuperSecret-2026"
    malformed = f"--remoteAutoTuningConfigssh://alice:{secret}@10.0.0.5"
    with pytest.raises(ValueError, match="Malformed") as exc_info:
        bm._extract_remote_config_value([malformed])
    assert secret not in str(exc_info.value)
    assert "alice:******@" in str(exc_info.value)


def test_extract_remote_config_value_malformed_redacts_in_debug_log():
    secret = "SuperSecret-2026"
    malformed = f"--remoteAutoTuningConfigssh://alice:{secret}@10.0.0.5"
    log = MagicMock()
    with pytest.raises(ValueError):
        bm._extract_remote_config_value([malformed], log=log)
    debug_msgs = [c.args[0] for c in log.debug.call_args_list]
    assert all(secret not in m for m in debug_msgs)


# --- _parse_remote_autotuning_url ---


def test_parse_remote_autotuning_url_full():
    cfg = bm._parse_remote_autotuning_url(
        "ssh://alice:s3cret@10.0.0.5:2222?"
        "remote_exec_path=/opt/trt/bin/trtexec&remote_lib_path=/opt/trt/lib"
    )
    assert cfg.user == "alice"
    assert cfg.password == "s3cret"
    assert cfg.ip == "10.0.0.5"
    assert cfg.port == 2222
    assert cfg.bin_path == "/opt/trt/bin"
    assert cfg.lib_path == "/opt/trt/lib"
    assert cfg.options == {
        "remote_exec_path": "/opt/trt/bin/trtexec",
        "remote_lib_path": "/opt/trt/lib",
    }


def test_parse_remote_autotuning_url_defaults_port_to_22():
    cfg = bm._parse_remote_autotuning_url(
        "ssh://alice@host?remote_exec_path=/x/trtexec&remote_lib_path=/y"
    )
    assert cfg.port == 22


def test_parse_remote_autotuning_url_empty_password_becomes_empty_string():
    cfg = bm._parse_remote_autotuning_url(
        "ssh://alice@host?remote_exec_path=/x/trtexec&remote_lib_path=/y"
    )
    assert cfg.password == ""


@pytest.mark.parametrize(
    ("url", "match"),
    [
        ("http://alice@host?remote_exec_path=/x&remote_lib_path=/y", "Only 'ssh://'"),
        ("ssh://host?remote_exec_path=/x&remote_lib_path=/y", "remote user"),
        ("ssh://alice@host?remote_exec_path=/x", "Missing required query parameters"),
        (
            "ssh://alice@host?remote_exec_path=/a&remote_exec_path=/b&remote_lib_path=/y",
            "Duplicate query parameters",
        ),
    ],
)
def test_parse_remote_autotuning_url_validation_errors(url, match):
    with pytest.raises(ValueError, match=match):
        bm._parse_remote_autotuning_url(url)


# --- _parse_remote_autotuning_url argv-smuggling guards (CVE-2017-1000117 class) ---


@pytest.mark.parametrize(
    "evil_user",
    [
        "-oProxyCommand=evil",  # specific known argv-smuggling payload
        "-l",  # single dash + letter (short flag for ssh)
        "--debug",  # long flag form
    ],
)
def test_parse_remote_autotuning_url_rejects_user_starting_with_dash(evil_user):
    """A username beginning with ``-`` would be reinterpreted as an ssh/scp flag.

    Without this guard, ``ssh://-oProxyCommand=evil@host/...`` would expand to
    ``scp -oProxyCommand=evil@host:...`` and execute the attacker's command.
    """
    url = (
        f"ssh://{evil_user}@10.0.0.5?"
        "remote_exec_path=/opt/trt/bin/trtexec&remote_lib_path=/opt/trt/lib"
    )
    with pytest.raises(ValueError, match="Remote user.*must not start with '-'"):
        bm._parse_remote_autotuning_url(url)


def test_parse_remote_autotuning_url_rejects_host_starting_with_dash():
    """A hostname beginning with ``-`` is the same argv-smuggling vector via the host position."""
    # urlparse lowercases hostnames, so capitalization doesn't matter here.
    url = (
        "ssh://alice@-oproxycommand?"
        "remote_exec_path=/opt/trt/bin/trtexec&remote_lib_path=/opt/trt/lib"
    )
    with pytest.raises(ValueError, match="Remote host.*must not start with '-'"):
        bm._parse_remote_autotuning_url(url)


def test_parse_remote_autotuning_url_accepts_normal_user_and_host():
    """Regression guard: usernames and hosts not starting with ``-`` are still accepted."""
    cfg = bm._parse_remote_autotuning_url(
        "ssh://alice@10.0.0.5?remote_exec_path=/opt/trt/bin/trtexec&remote_lib_path=/opt/trt/lib"
    )
    assert cfg.user == "alice"
    assert cfg.ip == "10.0.0.5"


# --- _ensure_remote_autotuning_flags ---


def test_ensure_remote_autotuning_flags_appends_both_when_missing():
    result = bm._ensure_remote_autotuning_flags(["--fp16"])
    assert result == ["--fp16", "--safe", "--skipInference"]


def test_ensure_remote_autotuning_flags_preserves_user_supplied():
    result = bm._ensure_remote_autotuning_flags(["--safe", "--fp16"])
    assert result.count("--safe") == 1
    assert "--skipInference" in result


def test_ensure_remote_autotuning_flags_returns_new_list():
    original = ["--fp16"]
    result = bm._ensure_remote_autotuning_flags(original)
    assert result is not original
    assert original == ["--fp16"]  # input untouched


def test_ensure_remote_autotuning_flags_warns_per_injected_flag():
    log = MagicMock()
    bm._ensure_remote_autotuning_flags([], log=log)
    assert log.warning.call_count == 2


# ===========================================================================
# Integration tests — exercise TrtExecBenchmark.__init__ end-to-end.
# ===========================================================================


@pytest.fixture
def bench(tmp_path):
    """A plain ``TrtExecBenchmark`` instance with a temp timing cache."""
    return TrtExecBenchmark(
        timing_cache_file=str(tmp_path / "cache.bin"),
        warmup_runs=1,
        timing_runs=2,
    )


# --- __init__ command construction ---


def test_init_builds_base_cmd_with_expected_flags(tmp_path):
    """Base command contains the standard trtexec flags derived from ctor args."""
    cache = str(tmp_path / "cache.bin")
    b = TrtExecBenchmark(timing_cache_file=cache, warmup_runs=3, timing_runs=7)

    assert "--avgRuns=7" in b._base_cmd
    assert "--iterations=7" in b._base_cmd
    assert "--warmUp=3" in b._base_cmd
    assert "--stronglyTyped" in b._base_cmd
    assert f"--timingCacheFile={cache}" in b._base_cmd
    assert any(arg.startswith("--saveEngine=") for arg in b._base_cmd)


def test_init_extra_trtexec_args_appended(tmp_path):
    """User-supplied trtexec args are appended to the base command."""
    b = TrtExecBenchmark(
        timing_cache_file=str(tmp_path / "cache.bin"),
        trtexec_args=["--fp16", "--workspace=4096"],
    )
    assert b._base_cmd[-2:] == ["--fp16", "--workspace=4096"]


def test_init_missing_plugin_library_is_skipped(tmp_path):
    """Missing plugin .so paths produce a warning and don't appear in the base cmd."""
    b = TrtExecBenchmark(
        timing_cache_file=str(tmp_path / "cache.bin"),
        plugin_libraries=[str(tmp_path / "absent_plugin.so")],
    )
    assert not any("--staticPlugins" in arg for arg in b._base_cmd)


def test_init_existing_plugin_library_added(tmp_path):
    """Plugin .so paths that exist on disk are added as ``--staticPlugins``."""
    plugin = tmp_path / "fake_plugin.so"
    plugin.write_bytes(b"")
    b = TrtExecBenchmark(
        timing_cache_file=str(tmp_path / "cache.bin"),
        plugin_libraries=[str(plugin)],
    )
    assert any(arg == f"--staticPlugins={plugin.resolve()}" for arg in b._base_cmd)


def test_init_safe_flag_sets_is_safe(tmp_path):
    """``--safe`` in trtexec_args (without remote config) flips ``is_safe``."""
    b = TrtExecBenchmark(
        timing_cache_file=str(tmp_path / "cache.bin"),
        trtexec_args=["--safe"],
    )
    assert b.is_safe is True
    assert b.has_remote_config is False


# --- --remoteAutoTuningConfig parsing ---


_REMOTE_URL = (
    "ssh://alice:s3cret@10.0.0.5:2222?"
    "remote_exec_path=/opt/trt/bin/trtexec&remote_lib_path=/opt/trt/lib"
)


@pytest.fixture
def trtexec_version_ok():
    """Pretend trtexec >= 10.15 is available so --safe injection succeeds."""
    with patch(
        "modelopt.onnx.quantization.autotune.benchmark._check_for_trtexec",
        return_value="/usr/local/bin/trtexec",
    ) as m:
        yield m


@pytest.mark.usefixtures("trtexec_version_ok")
def test_remote_config_equals_form_parses(tmp_path):
    """``--remoteAutoTuningConfig=ssh://...`` (single arg) is parsed correctly."""
    b = TrtExecBenchmark(
        timing_cache_file=str(tmp_path / "cache.bin"),
        trtexec_args=[f"--remoteAutoTuningConfig={_REMOTE_URL}"],
    )
    assert b.has_remote_config is True
    assert b.remote_user == "alice"
    assert b.remote_password == "s3cret"
    assert b.remote_ip == "10.0.0.5"
    assert b.remote_port == 2222
    assert b.remote_bin_path == "/opt/trt/bin"
    assert b.remote_lib_path == "/opt/trt/lib"


@pytest.mark.usefixtures("trtexec_version_ok")
def test_remote_config_space_form_parses(tmp_path):
    """``--remoteAutoTuningConfig ssh://...`` (two args) is parsed correctly."""
    b = TrtExecBenchmark(
        timing_cache_file=str(tmp_path / "cache.bin"),
        trtexec_args=["--remoteAutoTuningConfig", _REMOTE_URL],
    )
    assert b.has_remote_config is True
    assert b.remote_ip == "10.0.0.5"


@pytest.mark.usefixtures("trtexec_version_ok")
def test_remote_config_injects_safe_and_skipinference(tmp_path):
    """When remote config is set, ``--safe`` and ``--skipInference`` are added if missing."""
    b = TrtExecBenchmark(
        timing_cache_file=str(tmp_path / "cache.bin"),
        trtexec_args=[f"--remoteAutoTuningConfig={_REMOTE_URL}"],
    )
    assert "--safe" in b.trtexec_args
    assert "--skipInference" in b.trtexec_args
    assert b.is_safe is True


@pytest.mark.usefixtures("trtexec_version_ok")
def test_remote_config_keeps_user_supplied_safe(tmp_path):
    """If user already passed ``--safe``, it's not duplicated."""
    b = TrtExecBenchmark(
        timing_cache_file=str(tmp_path / "cache.bin"),
        trtexec_args=[f"--remoteAutoTuningConfig={_REMOTE_URL}", "--safe"],
    )
    assert b.trtexec_args.count("--safe") == 1


@pytest.mark.usefixtures("trtexec_version_ok")
def test_remote_config_default_port_is_22(tmp_path):
    """A URL with no explicit port falls back to SSH port 22."""
    url = (
        "ssh://alice:s3cret@10.0.0.5?"
        "remote_exec_path=/opt/trt/bin/trtexec&remote_lib_path=/opt/trt/lib"
    )
    b = TrtExecBenchmark(
        timing_cache_file=str(tmp_path / "cache.bin"),
        trtexec_args=[f"--remoteAutoTuningConfig={url}"],
    )
    assert b.remote_port == 22


@pytest.mark.parametrize(
    ("bad_args", "match"),
    [
        (
            [
                f"--remoteAutoTuningConfig={_REMOTE_URL}",
                f"--remoteAutoTuningConfig={_REMOTE_URL}",
            ],
            "Exactly one",
        ),
        (["--remoteAutoTuningConfig"], "Missing value"),
        (
            ["--remoteAutoTuningConfig=http://10.0.0.5/?remote_exec_path=x&remote_lib_path=y"],
            "Only 'ssh://'",
        ),
        (
            ["--remoteAutoTuningConfig=ssh://10.0.0.5:22?remote_exec_path=/x&remote_lib_path=/y"],
            "remote user",
        ),
        (
            ["--remoteAutoTuningConfig=ssh://alice@10.0.0.5:22?remote_exec_path=/x"],
            "Missing required query parameters",
        ),
        (
            [
                "--remoteAutoTuningConfig=ssh://alice@10.0.0.5:22?"
                "remote_exec_path=/a&remote_exec_path=/b&remote_lib_path=/y"
            ],
            "Duplicate query parameters",
        ),
    ],
)
@pytest.mark.usefixtures("trtexec_version_ok")
def test_remote_config_validation_errors(tmp_path, bad_args, match):
    """Malformed ``--remoteAutoTuningConfig`` values raise ``ValueError``."""
    with pytest.raises(ValueError, match=match):
        TrtExecBenchmark(
            timing_cache_file=str(tmp_path / "cache.bin"),
            trtexec_args=bad_args,
        )


@pytest.mark.usefixtures("trtexec_version_ok")
def test_malformed_remote_config_redacts_password(tmp_path, caplog):
    """Malformed ``--remoteAutoTuningConfig`` ValueError + debug log must not leak the password.

    The else branch in ``__init__`` fires when the flag is mis-separated
    (e.g. ``--remoteAutoTuningConfig`` followed by ``ssh://...`` with no ``=``).
    Both the raised ``ValueError`` and any debug log line about the arg must
    mask the SSH password.
    """
    secret = "TopSecret-2026!"
    malformed = (
        f"--remoteAutoTuningConfigssh://alice:{secret}@10.0.0.5:22"
        "/remote_exec_path=/x&remote_lib_path=/y"
    )
    with (
        caplog.at_level("DEBUG", logger="modelopt.onnx"),
        pytest.raises(ValueError, match="Malformed --remoteAutoTuningConfig") as exc_info,
    ):
        TrtExecBenchmark(
            timing_cache_file=str(tmp_path / "cache.bin"),
            trtexec_args=[malformed],
        )

    # The ValueError message must NOT contain the password — leaking it would
    # surface secrets in stack traces and crash reports.
    assert secret not in str(exc_info.value)
    assert "alice:******@" in str(exc_info.value)

    # The debug log line for the arg must also not contain the password.
    for record in caplog.records:
        assert secret not in record.getMessage()


def test_remote_config_requires_trtexec_10_15(tmp_path):
    """When trtexec is too old, remote autotuning surfaces an ImportError."""
    with (
        patch(
            "modelopt.onnx.quantization.autotune.benchmark._check_for_trtexec",
            side_effect=ImportError("trtexec < 10.15"),
        ),
        pytest.raises(ImportError, match="trtexec < 10.15"),
    ):
        TrtExecBenchmark(
            timing_cache_file=str(tmp_path / "cache.bin"),
            trtexec_args=[f"--remoteAutoTuningConfig={_REMOTE_URL}"],
        )


# --- run() — local trtexec pipeline ---


def test_run_invokes_trtexec_with_onnx_path(bench, tmp_path):
    """``run(path)`` invokes trtexec with ``--onnx=<path>`` appended."""
    model = tmp_path / "model.onnx"
    model.write_bytes(b"")

    proc = _make_proc(stdout="[I] GPU Compute Time: min = 1.0 ms, max = 2.0 ms, median = 1.5 ms")
    with patch("subprocess.run", return_value=proc) as run_mock:
        latency = bench.run(str(model))

    assert latency == pytest.approx(1.5)
    cmd = run_mock.call_args.args[0]
    assert cmd[0] == "trtexec"
    assert f"--onnx={model}" in cmd


def test_run_writes_bytes_to_temp_file_before_invoking(bench):
    """``run(bytes)`` writes the bytes to disk and points trtexec at that file."""
    proc = _make_proc(stdout="[I] GPU Compute Time: min = 1.0 ms, max = 2.0 ms, median = 4.25 ms")
    with patch("subprocess.run", return_value=proc) as run_mock:
        latency = bench.run(b"\x08onnx-bytes")

    assert latency == pytest.approx(4.25)
    cmd = run_mock.call_args.args[0]
    assert f"--onnx={bench.temp_model_path}" in cmd


def test_run_writes_log_file_when_requested(bench, tmp_path):
    """``log_file`` receives stdout, stderr and the constructed command."""
    log_file = tmp_path / "logs" / "trtexec.log"
    proc = _make_proc(
        stdout="[I] GPU Compute Time: median = 2.0 ms",
        stderr="some warning",
    )
    with patch("subprocess.run", return_value=proc):
        bench.run(str(tmp_path / "model.onnx"), log_file=str(log_file))

    contents = log_file.read_text()
    assert "Command:" in contents
    assert "STDOUT:" in contents and "STDERR:" in contents
    assert "some warning" in contents


def test_run_returns_inf_on_nonzero_returncode(bench, tmp_path):
    """Non-zero exit from trtexec yields ``inf`` and short-circuits parsing."""
    proc = _make_proc(returncode=1, stderr="engine build failed", stdout="")
    with patch("subprocess.run", return_value=proc):
        assert bench.run(str(tmp_path / "m.onnx")) == float("inf")


def test_run_returns_inf_when_latency_not_parseable(bench, tmp_path):
    """Stdout that doesn't match either pattern yields ``inf``."""
    proc = _make_proc(stdout="all done, no latency line here")
    with patch("subprocess.run", return_value=proc):
        assert bench.run(str(tmp_path / "m.onnx")) == float("inf")


def test_run_returns_inf_when_trtexec_binary_missing(bench, tmp_path):
    """A ``FileNotFoundError`` from subprocess.run is mapped to ``inf``."""
    with patch("subprocess.run", side_effect=FileNotFoundError):
        assert bench.run(str(tmp_path / "m.onnx")) == float("inf")


def test_run_returns_inf_on_unexpected_exception(bench, tmp_path):
    """Any non-FileNotFoundError raised mid-pipeline still yields ``inf``."""
    with patch("subprocess.run", side_effect=OSError("disk full")):
        assert bench.run(str(tmp_path / "m.onnx")) == float("inf")


def test_call_dunder_forwards_to_run(bench, tmp_path):
    """Calling the benchmark instance directly invokes ``run`` and returns its result."""
    proc = _make_proc(stdout="[I] GPU Compute Time: median = 9.81 ms")
    with patch("subprocess.run", return_value=proc):
        latency = bench(str(tmp_path / "m.onnx"))
    assert latency == pytest.approx(9.81)


def test_del_swallows_cleanup_errors(tmp_path):
    """``__del__`` warns but does not raise when ``shutil.rmtree`` errors."""
    b = TrtExecBenchmark(timing_cache_file=str(tmp_path / "cache.bin"))
    with patch.object(bm.shutil, "rmtree", side_effect=PermissionError("denied")):
        b.__del__()  # Must not raise; logs a warning.


def test_run_parses_std_pattern(bench, tmp_path):
    """``_STD_PATTERN`` matches the real ``GPU Compute Time`` line."""
    stdout = (
        "[01/15/2026-12:00:00] [I] === Performance summary ===\n"
        "[I] GPU Compute Time: min = 0.8 ms, max = 1.2 ms, mean = 0.95 ms, "
        "median = 0.92 ms, percentile(99%) = 1.18 ms\n"
    )
    with patch("subprocess.run", return_value=_make_proc(stdout=stdout)):
        assert bench.run(str(tmp_path / "m.onnx")) == pytest.approx(0.92)


# --- run() — remote scp + ssh trtexec_safe pipeline ---


@pytest.fixture
def remote_bench(tmp_path, trtexec_version_ok):
    """A ``TrtExecBenchmark`` configured for remote autotuning.

    Requires ``trtexec_version_ok`` so ``_check_for_trtexec`` is patched during
    ``TrtExecBenchmark.__init__``.
    """
    del trtexec_version_ok  # consumed via pytest fixture injection
    return TrtExecBenchmark(
        timing_cache_file=str(tmp_path / "cache.bin"),
        trtexec_args=[f"--remoteAutoTuningConfig={_REMOTE_URL}"],
    )


def test_remote_run_scp_then_ssh_trtexec_safe(remote_bench, tmp_path):
    """The remote path runs trtexec → scp → ssh trtexec_safe, parsing _SAFE_PATTERN."""
    trtexec_proc = _make_proc(stdout="")  # build only; --skipInference
    scp_proc = _make_proc()
    safe_stdout = "[01/15/2026-12:00:00] [I] Average over 10 runs - GPU latency: 3.42 ms\n"
    ssh_proc = _make_proc(stdout=safe_stdout)

    with patch("subprocess.run", side_effect=[trtexec_proc, scp_proc, ssh_proc]) as run_mock:
        latency = remote_bench.run(str(tmp_path / "m.onnx"))

    assert latency == pytest.approx(3.42)
    assert run_mock.call_count == 4
    trtexec_cmd, scp_cmd, ssh_cmd, cleanup_cmd = (c.args[0] for c in run_mock.call_args_list)
    assert trtexec_cmd[0] == "trtexec"
    # The remote URL in this test carries a password, so scp/ssh are prefixed with sshpass.
    assert "scp" in scp_cmd
    assert "alice@10.0.0.5:" in scp_cmd[-1]
    assert "ssh" in ssh_cmd
    assert "alice@10.0.0.5" in ssh_cmd
    # The remote command string runs trtexec_safe with the engine path.
    remote_cmd_str = ssh_cmd[-1]
    assert "trtexec_safe" in remote_cmd_str
    assert "--loadEngine=" in remote_cmd_str
    print(cleanup_cmd)
    assert "rm -f trtexec_benchmark_model.trt" in cleanup_cmd


def test_remote_run_uses_sshpass_when_password_set(remote_bench, tmp_path):
    """When the URL carries a password, both scp and ssh are prefixed with ``sshpass``."""
    trtexec_proc = _make_proc(stdout="")
    scp_proc = _make_proc()
    ssh_proc = _make_proc(stdout="[I] Average over 5 runs - GPU latency: 2.0 ms")

    with patch("subprocess.run", side_effect=[trtexec_proc, scp_proc, ssh_proc]) as run_mock:
        remote_bench.run(str(tmp_path / "m.onnx"))

    _, scp_cmd, ssh_cmd, cleanup_cmd = (c.args[0] for c in run_mock.call_args_list)
    assert scp_cmd[:3] == ["sshpass", "-p", "s3cret"]
    assert ssh_cmd[:3] == ["sshpass", "-p", "s3cret"]
    assert "rm -f trtexec_benchmark_model.trt" in cleanup_cmd


def test_remote_run_scp_failure_returns_inf(remote_bench, tmp_path):
    """If scp fails, the pipeline short-circuits before ssh and returns ``inf``."""
    trtexec_proc = _make_proc(stdout="")
    scp_proc = _make_proc(returncode=1, stderr="permission denied")

    with patch("subprocess.run", side_effect=[trtexec_proc, scp_proc]) as run_mock:
        latency = remote_bench.run(str(tmp_path / "m.onnx"))

    assert latency == float("inf")
    assert run_mock.call_count == 2  # no ssh call


def test_remote_run_falls_back_to_trtexec_safe_flag(remote_bench, tmp_path):
    """If ``trtexec_safe`` errors, fall back to ``trtexec --safe`` and parse _STD_PATTERN."""
    trtexec_proc = _make_proc(stdout="")
    scp_proc = _make_proc()
    safe_bin_fail = _make_proc(returncode=127, stderr="trtexec_safe: not found")
    fallback_stdout = "[I] GPU Compute Time: median = 5.55 ms"
    fallback_proc = _make_proc(stdout=fallback_stdout)

    with patch(
        "subprocess.run",
        side_effect=[trtexec_proc, scp_proc, safe_bin_fail, fallback_proc],
    ) as run_mock:
        latency = remote_bench.run(str(tmp_path / "m.onnx"))

    assert latency == pytest.approx(5.55)
    fallback_cmd = run_mock.call_args_list[-2].args[0]
    remote_cmd_str = fallback_cmd[-1]
    assert "trtexec --safe" in remote_cmd_str
    assert "trtexec_safe" not in remote_cmd_str


def test_remote_run_both_safe_paths_fail_returns_inf(remote_bench, tmp_path):
    """If both ``trtexec_safe`` and the ``trtexec --safe`` fallback fail, return ``inf``."""
    trtexec_proc = _make_proc(stdout="")
    scp_proc = _make_proc()
    safe_bin_fail = _make_proc(returncode=127, stderr="not found")
    fallback_fail = _make_proc(returncode=1, stderr="also failed")

    with patch(
        "subprocess.run",
        side_effect=[trtexec_proc, scp_proc, safe_bin_fail, fallback_fail],
    ):
        assert remote_bench.run(str(tmp_path / "m.onnx")) == float("inf")


# --- network_timeout_seconds ---


def test_network_timeout_default_is_five_minutes(tmp_path):
    """Default network timeout is 5 minutes (300s)."""
    b = TrtExecBenchmark(timing_cache_file=str(tmp_path / "cache.bin"))
    assert b.network_timeout_seconds == 300


def test_network_timeout_custom_value_stored(tmp_path):
    """User-supplied ``network_timeout_seconds`` is stored on the instance."""
    b = TrtExecBenchmark(
        timing_cache_file=str(tmp_path / "cache.bin"),
        network_timeout_seconds=12.5,
    )
    assert b.network_timeout_seconds == 12.5


def test_local_trtexec_call_uses_no_timeout(bench, tmp_path):
    """The local engine build path passes ``timeout=None`` (engine builds can be long)."""
    proc = _make_proc(stdout="[I] GPU Compute Time: median = 1.0 ms")
    with patch("subprocess.run", return_value=proc) as run_mock:
        bench.run(str(tmp_path / "m.onnx"))

    # Exactly one subprocess call for the local pipeline; timeout must be None.
    assert run_mock.call_count == 1
    assert run_mock.call_args.kwargs.get("timeout") is None


@pytest.mark.usefixtures("trtexec_version_ok")
def test_remote_pipeline_passes_timeout_to_scp_and_ssh(tmp_path):
    """scp, ssh trtexec_safe, and the ssh fallback all receive ``network_timeout_seconds``."""
    timeout = 7.0
    b = TrtExecBenchmark(
        timing_cache_file=str(tmp_path / "cache.bin"),
        trtexec_args=[f"--remoteAutoTuningConfig={_REMOTE_URL}"],
        network_timeout_seconds=timeout,
    )

    trtexec_proc = _make_proc(stdout="")
    scp_proc = _make_proc()
    safe_fail = _make_proc(returncode=1, stderr="trtexec_safe not found")
    fallback_proc = _make_proc(stdout="[I] GPU Compute Time: median = 4.0 ms")

    with patch(
        "subprocess.run",
        side_effect=[trtexec_proc, scp_proc, safe_fail, fallback_proc],
    ) as run_mock:
        b.run(str(tmp_path / "m.onnx"))

    # Engine build (call 0) has no timeout; the three remote calls all use it.
    assert run_mock.call_args_list[0].kwargs.get("timeout") is None
    for idx in (1, 2, 3):  # scp, ssh trtexec_safe, ssh fallback
        assert run_mock.call_args_list[idx].kwargs.get("timeout") == timeout, (
            f"call {idx} did not receive timeout={timeout}"
        )


@pytest.mark.usefixtures("trtexec_version_ok")
def test_scp_timeout_returns_inf_and_logs(tmp_path, caplog):
    """A ``subprocess.TimeoutExpired`` during the scp step returns ``inf`` and is logged."""
    import subprocess

    b = TrtExecBenchmark(
        timing_cache_file=str(tmp_path / "cache.bin"),
        trtexec_args=[f"--remoteAutoTuningConfig={_REMOTE_URL}"],
        network_timeout_seconds=1.0,
    )
    trtexec_proc = _make_proc(stdout="")
    timeout_exc = subprocess.TimeoutExpired(cmd=["scp"], timeout=1.0)

    with (
        caplog.at_level("ERROR", logger="modelopt.onnx"),
        patch("subprocess.run", side_effect=[trtexec_proc, timeout_exc]),
    ):
        assert b.run(str(tmp_path / "m.onnx")) == float("inf")

    assert any("timed out" in r.getMessage().lower() for r in caplog.records)


@pytest.mark.usefixtures("trtexec_version_ok")
def test_ssh_trtexec_safe_timeout_returns_inf(tmp_path):
    """A timeout on the ssh ``trtexec_safe`` call also returns ``inf``."""
    import subprocess

    b = TrtExecBenchmark(
        timing_cache_file=str(tmp_path / "cache.bin"),
        trtexec_args=[f"--remoteAutoTuningConfig={_REMOTE_URL}"],
        network_timeout_seconds=1.0,
    )
    trtexec_proc = _make_proc(stdout="")
    scp_proc = _make_proc()
    timeout_exc = subprocess.TimeoutExpired(cmd=["ssh"], timeout=1.0)

    with patch(
        "subprocess.run",
        side_effect=[trtexec_proc, scp_proc, timeout_exc],
    ):
        assert b.run(str(tmp_path / "m.onnx")) == float("inf")


@pytest.mark.usefixtures("trtexec_version_ok")
def test_ssh_fallback_timeout_returns_inf(tmp_path):
    """A timeout on the ``trtexec --safe`` fallback ssh call returns ``inf``."""
    import subprocess

    b = TrtExecBenchmark(
        timing_cache_file=str(tmp_path / "cache.bin"),
        trtexec_args=[f"--remoteAutoTuningConfig={_REMOTE_URL}"],
        network_timeout_seconds=1.0,
    )
    trtexec_proc = _make_proc(stdout="")
    scp_proc = _make_proc()
    safe_fail = _make_proc(returncode=1, stderr="trtexec_safe failed")
    timeout_exc = subprocess.TimeoutExpired(cmd=["ssh"], timeout=1.0)

    with patch(
        "subprocess.run",
        side_effect=[trtexec_proc, scp_proc, safe_fail, timeout_exc],
    ):
        assert b.run(str(tmp_path / "m.onnx")) == float("inf")


# --- pattern constants ---


def test_std_pattern_matches_gpu_compute_time_line():
    """The std pattern matches a typical ``[I] GPU Compute Time: … median = X ms`` line."""
    import re

    text = "[I] GPU Compute Time: min = 1 ms, max = 2 ms, median = 1.42 ms"
    match = re.search(bm._STD_PATTERN, text, re.IGNORECASE)
    assert match and match.group(1) == "1.42"


def test_safe_pattern_matches_average_over_runs_line():
    """The safe pattern matches the trtexec_safe ``Average over N runs - GPU latency`` line."""
    import re

    text = "[01/15/2026-12:00:00] [I] Average over 10 runs - GPU latency: 7.89 ms"
    match = re.search(bm._SAFE_PATTERN, text, re.IGNORECASE)
    assert match and match.group(1) == "7.89"
