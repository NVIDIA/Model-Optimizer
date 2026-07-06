# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

import sys
import warnings

import pytest
import tqdm

from modelopt.torch.utils.logging import (
    capture_io,
    no_stdout,
    silence_matched_warnings,
    warn_rank_0,
)


def test_no_stdout_disables_tqdm_and_restores_it():
    with no_stdout():
        bar = tqdm.tqdm(range(2))
        assert bar.disable
        bar.close()
    bar2 = tqdm.tqdm(range(2))
    assert not bar2.disable  # original tqdm.__init__ restored on exit
    bar2.close()


def test_warn_rank_0_wraps_message_in_yellow_on_tty(monkeypatch):
    monkeypatch.setattr(sys.stderr, "isatty", lambda: True)
    with pytest.warns(UserWarning) as record:
        warn_rank_0("colored message")
    assert str(record[0].message) == "\033[33mcolored message\033[0m"


def test_capture_io_captures_stdout_and_stderr():
    with capture_io() as buf:
        print("to stdout")
        print("to stderr", file=sys.stderr)
    assert "to stdout" in buf.getvalue()
    assert "to stderr" in buf.getvalue()


def test_capture_io_can_leave_stderr_alone(capsys):
    with capture_io(capture_stderr=False) as buf:
        print("out")
        print("err", file=sys.stderr)
    assert buf.getvalue() == "out\n"
    assert "err" in capsys.readouterr().err


def test_silence_matched_warnings_filters_by_pattern():
    seen = []

    def recorder(message, category, filename, lineno, file=None, line=None):
        seen.append(str(message))

    with warnings.catch_warnings():
        warnings.simplefilter("always")
        warnings.showwarning = recorder  # restored by catch_warnings on exit
        with silence_matched_warnings("skip"):
            warnings.warn("please skip me")
            warnings.warn("keep me")
    assert seen == ["keep me"]


def test_silence_matched_warnings_restores_showwarning():
    original = warnings.showwarning
    with silence_matched_warnings("pattern"):
        assert warnings.showwarning is not original
    assert warnings.showwarning is original


@pytest.mark.parametrize("pattern", [None, 123])
def test_silence_matched_warnings_invalid_pattern_is_noop(pattern):
    # None or an un-compilable pattern leaves warnings.showwarning untouched
    original = warnings.showwarning
    with silence_matched_warnings(pattern):
        assert warnings.showwarning is original
    assert warnings.showwarning is original
