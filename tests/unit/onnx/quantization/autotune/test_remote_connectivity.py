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

"""Tests for remote board connectivity checking and workflow error handling.

Covers:
- URI flag parsing: --remoteAutoTuningConfig= and space-separated forms.
- Hostname/port extraction defaults and fallback (warn-and-skip on unparseable URIs).
- TCP connection success, retry exhaustion, and back-off timing.
- _benchmark_failed() post-failure re-check: genuine failure returns inf,
  board-lost raises RemoteConnectionError.
- Workflow-level _benchmark_or_save(): state saved and error propagated for
  baseline, per-scheme, and final benchmark failures.
- Recovery boundary: committed regions survive restart; the active region
  may be re-profiled from scratch (candidates/measurements are not serialized).
"""

import logging
from unittest.mock import MagicMock, call, patch

import pytest
import yaml

# benchmark.py transitively loads TensorRT native libraries at import time.
# Skip the entire module when the native libs are absent (same limitation as
# neighboring autotune tests like test_autotuner.py).
try:
    import modelopt.onnx.quantization.autotune.benchmark as bm
    from modelopt.onnx.quantization.autotune.benchmark import (
        _check_remote_connectivity,
        _try_connect,
    )
except ImportError as _imp_err:
    pytest.skip(f"TensorRT native libraries not available: {_imp_err}", allow_module_level=True)

from _test_utils.onnx.quantization.autotune.models import (
    _create_branch_merge_onnx_model,
    _create_simple_conv_onnx_model,
)

import modelopt.onnx.quantization.autotune.workflows as wf
from modelopt.onnx.quantization.autotune import QDQAutotuner
from modelopt.onnx.quantization.autotune.common import Config, RemoteConnectionError
from modelopt.onnx.quantization.autotune.workflows import (
    _benchmark_or_save,
    region_pattern_autotuning_workflow,
)

# ============================================================================
# TrtExecBenchmark — remote_connection_retries validation
# ============================================================================


class TestRetriesValidation:
    """remote_connection_retries must be an int in [1, 10]."""

    @pytest.mark.parametrize("value", [0, -1, -100])
    def test_rejects_non_positive(self, value, tmp_path):
        with pytest.raises(ValueError, match="between 1 and 10"):
            bm.TrtExecBenchmark(
                timing_cache_file=str(tmp_path / "c.bin"),
                remote_connection_retries=value,
            )

    def test_rejects_above_upper_bound(self, tmp_path):
        with pytest.raises(ValueError, match="between 1 and 10"):
            bm.TrtExecBenchmark(
                timing_cache_file=str(tmp_path / "c.bin"),
                remote_connection_retries=11,
            )

    @pytest.mark.parametrize("value", [1.0, "3", None])
    def test_rejects_non_int_types(self, value, tmp_path):
        with pytest.raises(ValueError, match="between 1 and 10"):
            bm.TrtExecBenchmark(
                timing_cache_file=str(tmp_path / "c.bin"),
                remote_connection_retries=value,
            )

    @pytest.mark.parametrize("value", [1, 3, 10])
    def test_accepts_valid_range(self, value, tmp_path):
        bench = bm.TrtExecBenchmark(
            timing_cache_file=str(tmp_path / "c.bin"),
            remote_connection_retries=value,
        )
        assert bench._remote_connection_retries == value


# ============================================================================
# _check_remote_connectivity — flag parsing
# ============================================================================


class TestFlagParsing:
    """--remoteAutoTuningConfig extraction from trtexec arg lists."""

    def test_equals_form(self):
        """--remoteAutoTuningConfig=ssh://host:22 is found."""
        args = ["--fp16", "--remoteAutoTuningConfig=ssh://board:22", "--safe"]
        with patch.object(bm, "_try_connect", return_value=None) as mock_conn:
            _check_remote_connectivity(args)
        mock_conn.assert_called_once_with("board", 22, 3)

    def test_space_separated_form(self):
        """--remoteAutoTuningConfig <value> (two tokens) is found."""
        args = ["--remoteAutoTuningConfig", "ssh://board:22", "--safe"]
        with patch.object(bm, "_try_connect", return_value=None) as mock_conn:
            _check_remote_connectivity(args)
        mock_conn.assert_called_once_with("board", 22, 3)

    def test_no_flag_is_noop(self):
        """Without --remoteAutoTuningConfig, the function returns immediately."""
        with patch.object(bm, "_try_connect") as mock_conn:
            _check_remote_connectivity(["--fp16", "--safe"])
        mock_conn.assert_not_called()

    def test_strips_surrounding_quotes(self):
        """Embedded quotes around the value are stripped before parsing."""
        args = ['--remoteAutoTuningConfig="ssh://board:22"']
        with patch.object(bm, "_try_connect", return_value=None) as mock_conn:
            _check_remote_connectivity(args)
        mock_conn.assert_called_once_with("board", 22, 3)

    def test_custom_retries_forwarded(self):
        """The retries parameter is forwarded to _try_connect."""
        args = ["--remoteAutoTuningConfig=ssh://board:22"]
        with patch.object(bm, "_try_connect", return_value=None) as mock_conn:
            _check_remote_connectivity(args, retries=7)
        mock_conn.assert_called_once_with("board", 22, 7)

    def test_trailing_flag_without_value_raises(self):
        """--remoteAutoTuningConfig as last arg (no value) raises."""
        args = ["--fp16", "--remoteAutoTuningConfig"]
        with pytest.raises(RemoteConnectionError, match="has no value"):
            _check_remote_connectivity(args)

    def test_equals_empty_value_raises(self):
        """--remoteAutoTuningConfig= (empty after =) raises."""
        args = ["--remoteAutoTuningConfig="]
        with pytest.raises(RemoteConnectionError, match="has no value"):
            _check_remote_connectivity(args)

    def test_equals_only_quotes_raises(self):
        """--remoteAutoTuningConfig='\"\"' (quotes only, empty after strip) raises."""
        args = ['--remoteAutoTuningConfig=""']
        with pytest.raises(RemoteConnectionError, match="has no value"):
            _check_remote_connectivity(args)


# ============================================================================
# _check_remote_connectivity — hostname / port / scheme handling
# ============================================================================


class TestURIParsing:
    """Hostname, port, and scheme extraction; warn-and-skip on unparseable URIs."""

    def test_malformed_uri_skips_with_warning(self):
        """ssh://[board (unclosed bracket) -> ValueError from urlparse -> warn and skip."""
        args = ["--remoteAutoTuningConfig=ssh://[board"]
        with patch.object(bm, "_try_connect") as mock_conn:
            _check_remote_connectivity(args)
        mock_conn.assert_not_called()

    def test_bare_ip_port_skips_with_warning(self):
        """192.168.1.100:22 (no scheme) -> hostname=None -> warn and skip."""
        args = ["--remoteAutoTuningConfig=192.168.1.100:22"]
        with patch.object(bm, "_try_connect") as mock_conn:
            _check_remote_connectivity(args)
        mock_conn.assert_not_called()

    def test_invalid_port_skips_with_warning(self):
        """ssh://board:notaport -> ValueError from urlparse -> warn and skip."""
        args = ["--remoteAutoTuningConfig=ssh://board:notaport"]
        with patch.object(bm, "_try_connect") as mock_conn:
            _check_remote_connectivity(args)
        mock_conn.assert_not_called()

    def test_unknown_scheme_skips_with_warning(self):
        """myproto://board -> no default port for 'myproto' -> warn and skip."""
        args = ["--remoteAutoTuningConfig=myproto://board"]
        with patch.object(bm, "_try_connect") as mock_conn:
            _check_remote_connectivity(args)
        mock_conn.assert_not_called()

    def test_ssh_scheme_defaults_to_port_22(self):
        """ssh://admin@board (no explicit port) -> port 22."""
        args = ["--remoteAutoTuningConfig=ssh://admin@board"]
        with patch.object(bm, "_try_connect", return_value=None) as mock_conn:
            _check_remote_connectivity(args)
        mock_conn.assert_called_once_with("board", 22, 3)

    def test_http_scheme_defaults_to_port_80(self):
        """http://board -> port 80."""
        args = ["--remoteAutoTuningConfig=http://board"]
        with patch.object(bm, "_try_connect", return_value=None) as mock_conn:
            _check_remote_connectivity(args)
        mock_conn.assert_called_once_with("board", 80, 3)

    def test_https_scheme_defaults_to_port_443(self):
        """https://board -> port 443."""
        args = ["--remoteAutoTuningConfig=https://board"]
        with patch.object(bm, "_try_connect", return_value=None) as mock_conn:
            _check_remote_connectivity(args)
        mock_conn.assert_called_once_with("board", 443, 3)

    def test_explicit_port_overrides_scheme_default(self):
        """ssh://board:2222 -> port 2222, not 22."""
        args = ["--remoteAutoTuningConfig=ssh://board:2222"]
        with patch.object(bm, "_try_connect", return_value=None) as mock_conn:
            _check_remote_connectivity(args)
        mock_conn.assert_called_once_with("board", 2222, 3)


# ============================================================================
# Credential redaction — warnings must never leak userinfo
# ============================================================================


class TestCredentialRedaction:
    """Warnings from unparseable URIs must never contain passwords or userinfo."""

    _SECRET = "s3cret-Pa$$w0rd"

    @pytest.mark.parametrize(
        "uri",
        [
            # Missing hostname (bare ip:port, no scheme) — hits hostname=None branch
            f"user:{_SECRET}@192.168.1.100:22",
            # Invalid port — hits ValueError branch
            f"ssh://admin:{_SECRET}@board:badport",
            # Unknown scheme — hits scheme-not-in-defaults branch
            f"myproto://admin:{_SECRET}@board",
        ],
        ids=["missing-hostname", "invalid-port", "unknown-scheme"],
    )
    def test_password_never_in_warning(self, uri, caplog):
        """The raw config value (which may contain a password) must not appear in logs."""
        args = [f"--remoteAutoTuningConfig={uri}"]
        with (
            caplog.at_level(logging.WARNING, logger="modelopt.onnx"),
            patch.object(bm, "_try_connect", return_value=None),
        ):
            _check_remote_connectivity(args)

        full_log = caplog.text
        assert self._SECRET not in full_log, f"Password leaked into log output: {full_log!r}"

    def test_successful_uri_with_creds_never_logs_password(self, caplog):
        """Even on a successful parse+connect, the password must not appear in logs."""
        uri = f"ssh://admin:{self._SECRET}@board:22"
        args = [f"--remoteAutoTuningConfig={uri}"]
        with (
            caplog.at_level(logging.DEBUG, logger="modelopt.onnx"),
            patch.object(bm, "_try_connect", return_value=None),
        ):
            _check_remote_connectivity(args)

        full_log = caplog.text
        assert self._SECRET not in full_log, f"Password leaked into log output: {full_log!r}"


# ============================================================================
# _try_connect — success, retry exhaustion, backoff
# ============================================================================


class TestTryConnect:
    """TCP connection logic with retries and back-off."""

    def test_success_on_first_attempt(self):
        """If create_connection succeeds immediately, return None."""
        mock_conn = MagicMock()
        with patch("socket.create_connection", return_value=mock_conn):
            result = _try_connect("board", 22, retries=3)
        assert result is None
        mock_conn.close.assert_called_once()

    def test_success_after_transient_failure(self):
        """Succeed on 2nd attempt after 1st times out -> return None."""
        mock_conn = MagicMock()
        with (
            patch(
                "socket.create_connection",
                side_effect=[TimeoutError("timeout"), mock_conn],
            ),
            patch("time.sleep"),
        ):
            result = _try_connect("board", 22, retries=3)
        assert result is None

    def test_all_retries_exhausted_returns_last_error(self):
        """All attempts fail -> return the last exception."""
        errors = [OSError(f"fail {i}") for i in range(3)]
        with patch("socket.create_connection", side_effect=errors), patch("time.sleep"):
            result = _try_connect("board", 22, retries=3)
        assert result is errors[-1]

    def test_backoff_sleep_between_retries(self):
        """Each failed attempt (except the last) sleeps 2 s before retrying."""
        with (
            patch("socket.create_connection", side_effect=OSError("fail")),
            patch("time.sleep") as mock_sleep,
        ):
            _try_connect("board", 22, retries=3)
        # Sleeps after attempt 1 and 2, not after final attempt 3.
        assert mock_sleep.call_args_list == [call(2), call(2)]

    def test_raises_remote_connection_error_on_exhaustion(self):
        """Full _check_remote_connectivity raises after retry exhaustion."""
        args = ["--remoteAutoTuningConfig=ssh://board:22"]
        with (
            patch("socket.create_connection", side_effect=OSError("unreachable")),
            patch("time.sleep"),
            pytest.raises(RemoteConnectionError, match="board:22"),
        ):
            _check_remote_connectivity(args, retries=2)


# ============================================================================
# _benchmark_failed — post-failure re-check
# ============================================================================


class TestBenchmarkFailed:
    """TrtExecBenchmark._benchmark_failed() re-checks connectivity after trtexec error."""

    @staticmethod
    def _make_bench(trtexec_args=None, retries=1):
        """Build a TrtExecBenchmark without hitting real trtexec init checks."""
        bench = bm.TrtExecBenchmark.__new__(bm.TrtExecBenchmark)
        bench._base_cmd = trtexec_args or ["--remoteAutoTuningConfig=ssh://board:22"]
        bench._remote_connection_retries = retries
        bench.logger = MagicMock()
        return bench

    def test_board_reachable_returns_inf(self):
        """Board still up after trtexec error -> genuine failure -> float('inf')."""
        bench = self._make_bench()
        with patch.object(bm, "_try_connect", return_value=None):
            assert bench._benchmark_failed() == float("inf")

    def test_board_unreachable_raises(self):
        """Board dropped mid-trtexec -> RemoteConnectionError propagates."""
        bench = self._make_bench()
        with (
            patch.object(bm, "_try_connect", return_value=OSError("gone")),
            pytest.raises(RemoteConnectionError),
        ):
            bench._benchmark_failed()

    def test_run_nonzero_exit_triggers_recheck(self):
        """trtexec exit code != 0 -> two connectivity calls: pre-check + post-failure."""
        bench = self._make_bench()
        bench.timing_cache_file = "/tmp/cache.bin"
        bench.temp_model_path = "/tmp/model.onnx"
        bench.latency_pattern = r"median\s*=\s*([\d.]+)\s*ms"
        bench.warmup_runs = 1
        bench.timing_runs = 1

        mock_result = MagicMock(returncode=1, stderr="error", stdout="")
        with (
            patch("subprocess.run", return_value=mock_result),
            patch("os.path.exists", return_value=True),
            patch.object(bm, "_try_connect", return_value=None) as mock_conn,
        ):
            result = bench.run("/tmp/model.onnx")
        assert result == float("inf")
        # Pre-check before trtexec + post-failure re-check = 2 calls.
        assert mock_conn.call_count == 2

    def test_run_unparseable_output_triggers_recheck(self):
        """trtexec succeeds but output has no latency -> two connectivity calls."""
        bench = self._make_bench()
        bench.timing_cache_file = "/tmp/cache.bin"
        bench.temp_model_path = "/tmp/model.onnx"
        bench.latency_pattern = r"\[I\]\s+Latency:.*?median\s*=\s*([\d.]+)\s*ms"
        bench.warmup_runs = 1
        bench.timing_runs = 1

        mock_result = MagicMock(returncode=0, stderr="", stdout="no latency here")
        with (
            patch("subprocess.run", return_value=mock_result),
            patch("os.path.exists", return_value=True),
            patch.object(bm, "_try_connect", return_value=None) as mock_conn,
        ):
            result = bench.run("/tmp/model.onnx")
        assert result == float("inf")
        # Pre-check + post-failure re-check = 2 calls.
        assert mock_conn.call_count == 2

    def test_run_board_lost_mid_trtexec_raises(self):
        """
        Given a remote board that is reachable before trtexec but drops during execution,
        When run() is called and trtexec fails (returncode != 0),
        Then RemoteConnectionError is raised, connectivity is checked exactly twice
        (pre-check + post-failure), and trtexec is invoked exactly once.
        """
        bench = self._make_bench()
        bench.timing_cache_file = "/tmp/cache.bin"
        bench.temp_model_path = "/tmp/model.onnx"
        bench.latency_pattern = r"median\s*=\s*([\d.]+)\s*ms"
        bench.warmup_runs = 1
        bench.timing_runs = 1

        mock_result = MagicMock(returncode=1, stderr="error", stdout="")
        # _try_connect returns None on success or an exception on failure (it
        # does not raise).  side_effect=[None, OSError(...)] would *raise* the
        # OSError instead of returning it, so use a callable that returns it.
        connect_returns = iter([None, OSError("gone")])
        with (
            patch.object(bm, "_run_trtexec", return_value=mock_result) as mock_trtexec,
            patch("os.path.exists", return_value=True),
            patch.object(
                bm, "_try_connect", side_effect=lambda *a, **kw: next(connect_returns)
            ) as mock_conn,
            pytest.raises(RemoteConnectionError),
        ):
            bench.run("/tmp/model.onnx")
        assert mock_conn.call_count == 2
        assert mock_trtexec.call_count == 1

    def test_no_remote_config_skips_recheck(self):
        """Without --remoteAutoTuningConfig, _benchmark_failed returns inf without TCP check."""
        bench = self._make_bench(trtexec_args=["--fp16", "--safe"])
        with patch.object(bm, "_try_connect") as mock_conn:
            result = bench._benchmark_failed()
        assert result == float("inf")
        mock_conn.assert_not_called()


# ============================================================================
# _benchmark_or_save — workflow-level state saving
# ============================================================================


class TestBenchmarkOrSave:
    """_benchmark_or_save saves autotuner state on RemoteConnectionError."""

    @staticmethod
    def _make_autotuner_and_path(tmp_path):
        """Create a real autotuner + state path for checkpoint round-trip tests."""
        model = _create_simple_conv_onnx_model()
        autotuner = QDQAutotuner(model)
        config = Config(verbose=False, maximum_sequence_region_size=50)
        autotuner.initialize(config)
        state_path = tmp_path / "state.yaml"
        return autotuner, state_path

    def test_success_returns_latency(self, tmp_path):
        """Normal benchmark returns latency without touching state."""
        autotuner, state_path = self._make_autotuner_and_path(tmp_path)

        with patch(
            "modelopt.onnx.quantization.autotune.workflows.benchmark_onnx_model",
            return_value=5.0,
        ):
            result = _benchmark_or_save(autotuner, state_path, "/fake.onnx")
        assert result == 5.0
        assert not state_path.exists()

    def test_genuine_failure_returns_inf(self, tmp_path):
        """Non-connection failure -> float('inf') returned, no state save."""
        autotuner, state_path = self._make_autotuner_and_path(tmp_path)

        with patch(
            "modelopt.onnx.quantization.autotune.workflows.benchmark_onnx_model",
            return_value=float("inf"),
        ):
            result = _benchmark_or_save(autotuner, state_path, "/fake.onnx")
        assert result == float("inf")
        assert not state_path.exists()

    def test_connection_error_saves_state_and_raises(self, tmp_path):
        """RemoteConnectionError -> state saved to disk -> re-raised."""
        autotuner, state_path = self._make_autotuner_and_path(tmp_path)

        with (
            patch(
                "modelopt.onnx.quantization.autotune.workflows.benchmark_onnx_model",
                side_effect=RemoteConnectionError("board lost"),
            ),
            pytest.raises(RemoteConnectionError, match="board lost"),
        ):
            _benchmark_or_save(autotuner, state_path, "/fake.onnx")
        assert state_path.exists()


# ============================================================================
# Workflow integration — recovery boundary and restart behaviour
# ============================================================================


class TestBenchmarkOnnxModelPropagation:
    """benchmark_onnx_model must propagate RemoteConnectionError from run()."""

    def test_remote_connection_error_propagates(self):
        """RemoteConnectionError from TrtExecBenchmark.run() passes through."""
        mock_instance = MagicMock()
        mock_instance.run.side_effect = RemoteConnectionError("board lost")
        old = wf._benchmark_instance
        try:
            wf._benchmark_instance = mock_instance
            with pytest.raises(RemoteConnectionError, match="board lost"):
                wf.benchmark_onnx_model("/fake.onnx")
        finally:
            wf._benchmark_instance = old

    def test_generic_exception_returns_inf(self):
        """Non-connection exceptions from run() are caught and return inf."""
        mock_instance = MagicMock()
        mock_instance.run.side_effect = RuntimeError("engine build failed")
        old = wf._benchmark_instance
        try:
            wf._benchmark_instance = mock_instance
            result = wf.benchmark_onnx_model("/fake.onnx")
        finally:
            wf._benchmark_instance = old
        assert result == float("inf")


class TestWorkflowInterruption:
    """Verify that RemoteConnectionError during scheme profiling does NOT submit
    an errored result.  Recovery boundary: committed regions survive restart;
    the active region is re-profiled from scratch (its in-progress candidates
    and measurements are not serialized)."""

    @staticmethod
    def _make_autotuner(config=None):
        model = _create_simple_conv_onnx_model()
        autotuner = QDQAutotuner(model)
        if config is None:
            config = Config(verbose=False, maximum_sequence_region_size=50)
        autotuner.initialize(config)
        return autotuner, model, config

    def test_connection_loss_skips_submit_and_generates_candidate_after_reload(self, tmp_path):
        """Baseline succeeds, first scheme benchmark raises -> submit never called,
        state saved; after reload the active region is re-profiled from scratch
        (no committed patterns for it) with all measurements reset."""
        autotuner, model, config = self._make_autotuner()
        state_path = tmp_path / "state.yaml"

        # Submit baseline.
        autotuner.submit(10.0)

        regions = autotuner.regions
        assert len(regions) > 0
        region = regions[0]
        autotuner.set_profile_region(region)

        # Generate a scheme.
        scheme_idx = autotuner.generate()
        assert scheme_idx >= 0
        ps = autotuner.current_profile_pattern_schemes
        interrupted_scheme = ps.schemes[scheme_idx]

        # Spy on submit to prove it is never called during the error path.
        with patch.object(autotuner, "submit", wraps=autotuner.submit) as spy_submit:
            with (
                patch(
                    "modelopt.onnx.quantization.autotune.workflows.benchmark_onnx_model",
                    side_effect=RemoteConnectionError("board lost"),
                ),
                pytest.raises(RemoteConnectionError),
            ):
                _benchmark_or_save(autotuner, state_path, b"fake-model-bytes")

            # submit() must not have been called — error interrupted before measurement.
            spy_submit.assert_not_called()

        # The scheme was never profiled.
        assert interrupted_scheme.latency_ms == float("inf")
        assert interrupted_scheme.error is False
        assert state_path.exists()

        # --- Reload into a fresh autotuner ---
        autotuner2 = QDQAutotuner(model)
        autotuner2.initialize(config)
        autotuner2.load_state(str(state_path))

        assert autotuner2.baseline_latency_ms == 10.0

        # The region must still be eligible (no committed profiled_patterns for it).
        assert len(autotuner2.profiled_patterns) == 0

        region2 = autotuner2.regions[0]
        autotuner2.set_profile_region(region2)
        ps2 = autotuner2.current_profile_pattern_schemes
        assert ps2 is not None

        # All seeded measurements must be reset (latency_ms=inf, error=False).
        for s in ps2.schemes:
            assert s.latency_ms == float("inf")
            assert s.error is False

        # Must be able to generate a new candidate for the region (the
        # interrupted candidate is NOT guaranteed to be regenerated — cache
        # seeding may or may not reproduce it).
        new_idx = autotuner2.generate()
        assert new_idx >= 0

    def test_baseline_failure_through_workflow(self, tmp_path):
        """region_pattern_autotuning_workflow: connection loss during baseline
        benchmark saves state; reloaded autotuner has no baseline recorded."""
        model = _create_simple_conv_onnx_model()

        with (
            patch(
                "modelopt.onnx.quantization.autotune.workflows.benchmark_onnx_model",
                side_effect=RemoteConnectionError("board lost"),
            ),
            pytest.raises(RemoteConnectionError, match="board lost"),
        ):
            region_pattern_autotuning_workflow(
                model,
                output_dir=tmp_path / "output",
                num_schemes_per_region=5,
                state_file=str(tmp_path / "state.yaml"),
            )

        assert (tmp_path / "state.yaml").exists()

        # Reload and verify no baseline was committed.
        autotuner2 = QDQAutotuner(model)
        config = Config(verbose=False, maximum_sequence_region_size=50)
        autotuner2.initialize(config)
        autotuner2.load_state(str(tmp_path / "state.yaml"))
        assert autotuner2.baseline_latency_ms is None
        assert len(autotuner2.profiled_patterns) == 0

    def test_workflow_mid_scheme_interruption_no_submit(self, tmp_path):
        """region_pattern_autotuning_workflow: connection loss mid-scheme does not
        submit a failed result — spy on submit() proves no errored measurement
        was recorded."""
        model = _create_simple_conv_onnx_model()
        call_count = 0

        def _mock_benchmark(model_path, log_file=None, flush_timing_cache=False):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                # Baseline succeeds.
                return 10.0
            # First scheme benchmark -> board lost.
            raise RemoteConnectionError("board lost")

        submit_calls = []
        _original_submit = QDQAutotuner.submit

        def _tracking_submit(self, latency_ms, success=True):
            submit_calls.append((latency_ms, success))
            return _original_submit(self, latency_ms, success=success)

        with (
            patch(
                "modelopt.onnx.quantization.autotune.workflows.benchmark_onnx_model",
                side_effect=_mock_benchmark,
            ),
            patch.object(QDQAutotuner, "submit", _tracking_submit),
            pytest.raises(RemoteConnectionError, match="board lost"),
        ):
            region_pattern_autotuning_workflow(
                model,
                output_dir=tmp_path / "output",
                num_schemes_per_region=5,
                state_file=str(tmp_path / "state.yaml"),
            )

        # Only the baseline submit (10.0) should have been recorded — never a
        # scheme measurement, especially not an errored one.
        assert len(submit_calls) == 1
        assert submit_calls[0] == (10.0, True)

        assert (tmp_path / "state.yaml").exists()

    def test_workflow_final_measurement_interruption(self, tmp_path):
        """Connection loss during the final optimized-model benchmark saves state;
        reloaded autotuner retains all committed patterns and their measurements."""
        model = _create_simple_conv_onnx_model()
        call_count = 0

        def _mock_benchmark(model_path, log_file=None, flush_timing_cache=False):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                return 10.0  # baseline
            # All scheme benchmarks succeed, then final fails.
            if isinstance(model_path, bytes):
                return 9.0  # scheme measurement (model_bytes)
            # Final model path is a string ending in optimized_final.onnx.
            if isinstance(model_path, str) and "optimized_final" in model_path:
                raise RemoteConnectionError("board lost during final")
            return 9.0

        with (
            patch(
                "modelopt.onnx.quantization.autotune.workflows.benchmark_onnx_model",
                side_effect=_mock_benchmark,
            ),
            pytest.raises(RemoteConnectionError, match="board lost during final"),
        ):
            region_pattern_autotuning_workflow(
                model,
                output_dir=tmp_path / "output",
                num_schemes_per_region=2,
                state_file=str(tmp_path / "state.yaml"),
            )

        state_path = tmp_path / "state.yaml"
        assert state_path.exists()

        # Reload and verify committed patterns and measurements survived.
        autotuner2 = QDQAutotuner(model)
        config = Config(verbose=False, maximum_sequence_region_size=50)
        autotuner2.initialize(config)
        autotuner2.load_state(str(state_path))

        assert autotuner2.baseline_latency_ms == 10.0
        # At least one pattern must have been committed with measured schemes.
        assert len(autotuner2.profiled_patterns) > 0
        for ps in autotuner2.profiled_patterns:
            measured = [s for s in ps.schemes if s.latency_ms != float("inf")]
            assert len(measured) > 0, "Committed pattern has no measured schemes"

    def test_restart_skips_committed_regions_resumes_active(self, tmp_path):
        """Committed region + mid-scheme interruption in the active region:
        on resume the committed pattern and its measurements are retained
        without re-benchmarking, while the active region is re-profiled."""
        model = _create_branch_merge_onnx_model()

        # Verify the model produces at least 2 distinct region patterns so
        # the first can be committed while the second is still active.
        probe = QDQAutotuner(model)
        probe.initialize(Config(verbose=False, maximum_sequence_region_size=50))
        assert len(probe.regions) >= 2, (
            f"Model must produce >=2 regions for this test, got {len(probe.regions)}"
        )
        del probe

        # --- First run: commit region 0, measure one scheme in region 1,
        #     then raise RemoteConnectionError. ---
        first_run_scheme_calls = 0  # byte-valued benchmark calls in first run

        def _mock_benchmark_initial(model_path, log_file=None, flush_timing_cache=False):
            nonlocal first_run_scheme_calls
            if isinstance(model_path, bytes):
                first_run_scheme_calls += 1
                # Let region 0's schemes (calls 1-2) and region 1's first
                # scheme (call 3) succeed.  Fail on region 1's second scheme.
                if first_run_scheme_calls <= 3:
                    return 9.0
                raise RemoteConnectionError("board lost mid-region")
            # Baseline (str path).
            return 10.0

        state_path = tmp_path / "state.yaml"
        with (
            patch(
                "modelopt.onnx.quantization.autotune.workflows.benchmark_onnx_model",
                side_effect=_mock_benchmark_initial,
            ),
            pytest.raises(RemoteConnectionError, match="board lost mid-region"),
        ):
            region_pattern_autotuning_workflow(
                model,
                output_dir=tmp_path / "output1",
                num_schemes_per_region=2,
                state_file=str(state_path),
            )

        assert state_path.exists()
        # Region 0 was committed (set_profile_region(region1, commit=True)
        # triggers commit of region 0).  Region 1 was active at crash time.
        checkpoint_autotuner = QDQAutotuner(model)
        checkpoint_autotuner.initialize(Config(verbose=False, maximum_sequence_region_size=50))
        checkpoint_autotuner.load_state(str(state_path))
        assert checkpoint_autotuner.baseline_latency_ms == 10.0
        num_committed = len(checkpoint_autotuner.profiled_patterns)
        assert num_committed >= 1, "Region 0 must have been committed before the crash"
        # Verify committed patterns have actual measurements (not all inf).
        for ps in checkpoint_autotuner.profiled_patterns:
            measured = [s for s in ps.schemes if s.latency_ms != float("inf")]
            assert len(measured) > 0, "Committed pattern has no measured schemes"

        # Snapshot committed pattern signatures, scheme hashes, and latencies
        # so we can verify they survive the resume unchanged.
        committed_snapshot = {}
        for ps in checkpoint_autotuner.profiled_patterns:
            sig = ps.pattern_signature
            committed_snapshot[sig] = {
                s.hash: s.latency_ms for s in ps.schemes if s.latency_ms != float("inf")
            }

        # Identify the interrupted pattern: load the state file directly to
        # read the active pattern signature at crash time.
        with open(str(state_path)) as f:
            raw_state = yaml.safe_load(f)
        interrupted_sig = raw_state.get("current_profile_pattern_schemes_signature")
        # The state file must record the active pattern at crash time.
        assert interrupted_sig is not None, (
            "State file must record current_profile_pattern_schemes_signature at crash time"
        )
        # The interrupted pattern must NOT be one of the committed patterns.
        assert interrupted_sig not in committed_snapshot, (
            "Interrupted pattern should not already be in committed set"
        )
        del checkpoint_autotuner

        # --- Second run: resume from checkpoint. ---
        # Spy on which pattern signature each benchmark call belongs to by
        # wrapping set_profile_region on the class to track the active pattern.
        active_pattern_sig = [None]  # mutable container for closure
        original_set_profile_region = QDQAutotuner.set_profile_region.__wrapped__

        def _spy_set_profile_region(self_inner, region, commit=True):
            result = original_set_profile_region(self_inner, region, commit=commit)
            if self_inner.current_profile_pattern_schemes is not None:
                active_pattern_sig[0] = self_inner.current_profile_pattern_schemes.pattern_signature
            else:
                active_pattern_sig[0] = None
            return result

        resume_benchmarked_patterns = []  # pattern sig for each scheme benchmark

        def _mock_benchmark_resume(model_path, log_file=None, flush_timing_cache=False):
            if isinstance(model_path, bytes):
                resume_benchmarked_patterns.append(active_pattern_sig[0])
                return 8.0
            # Baseline (str path) — should not be reached because checkpoint
            # already has baseline_latency_ms, but return a value if it is.
            return 10.0

        with (
            patch(
                "modelopt.onnx.quantization.autotune.workflows.benchmark_onnx_model",
                side_effect=_mock_benchmark_resume,
            ),
            patch.object(
                QDQAutotuner,
                "set_profile_region",
                _spy_set_profile_region,
            ),
        ):
            autotuner2 = region_pattern_autotuning_workflow(
                model,
                output_dir=tmp_path / "output2",
                num_schemes_per_region=2,
                state_file=str(state_path),
            )

        # --- Verify the recovery contract. ---

        # Build a signature → PatternSchemes mapping from the resumed result.
        profiled = autotuner2.profiled_patterns
        profiled_by_sig = {}
        for ps in profiled:
            profiled_by_sig[ps.pattern_signature] = ps

        assert len(profiled) >= num_committed + 1, (
            f"Expected at least {num_committed + 1} profiled patterns after resume, "
            f"got {len(profiled)}"
        )

        # 1. Every committed pattern survives the resume with identical
        #    scheme hashes and latencies.
        for sig, original_schemes in committed_snapshot.items():
            assert sig in profiled_by_sig, f"Committed pattern {sig[:16]}… disappeared after resume"
            resumed_ps = profiled_by_sig[sig]
            for scheme_hash, expected_latency in original_schemes.items():
                match = next((s for s in resumed_ps.schemes if s.hash == scheme_hash), None)
                assert match is not None, (
                    f"Committed scheme {scheme_hash[:8]}… missing after resume"
                )
                assert match.latency_ms == expected_latency, (
                    f"Committed scheme {scheme_hash[:8]}… latency changed: "
                    f"{expected_latency} -> {match.latency_ms}"
                )

        # 2. No resumed benchmark call targeted a committed pattern.
        assert len(resume_benchmarked_patterns) > 0, "Active region must be re-profiled on resume"
        for pat_sig in resume_benchmarked_patterns:
            assert pat_sig not in committed_snapshot, (
                f"Resume re-benchmarked committed pattern {pat_sig[:16]}…"
            )

        # 3. The interrupted pattern was specifically re-measured.
        assert interrupted_sig in resume_benchmarked_patterns, (
            f"Interrupted pattern {interrupted_sig[:16]}… was not re-measured on resume"
        )
