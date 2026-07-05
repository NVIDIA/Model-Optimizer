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

import argparse
import sys
import warnings

import pytest
import tqdm

from modelopt.torch.utils.logging import (
    DeprecatedError,
    atomic_print,
    capture_io,
    no_stdout,
    num2hrb,
    print_args,
    print_rank_0,
    silence_matched_warnings,
    warn_rank_0,
)


@pytest.mark.parametrize(
    ("num", "expected"),
    [
        (0, "0.00"),
        (999, "999.00"),
        (1000, "1.00K"),
        (1500, "1.50K"),
        (-1500, "-1.50K"),
        (1e6, "1.00M"),
        (2.5e9, "2.50B"),
        (1e12, "1.00T"),
        (1e15, "1.00P"),
        (1e18, "1.00E"),
        (1e21, "1000.00E"),  # units exhausted, number keeps growing
    ],
)
def test_num2hrb_exact(num, expected):
    assert num2hrb(num) == expected


def test_num2hrb_suffix():
    assert num2hrb(1000, suffix="B") == "1.00KB"
    assert num2hrb(1, suffix="s") == "1.00s"


def test_print_rank_0_prints_in_single_process(capsys):
    print_rank_0("hello", "world")
    assert capsys.readouterr().out == "hello world\n"


def test_print_rank_0_forwards_print_kwargs(capsys):
    print_rank_0("a", "b", sep="-", end="!")
    assert capsys.readouterr().out == "a-b!"


def test_print_args_sorted_with_header_and_footer(capsys):
    print_args(argparse.Namespace(beta=2, alpha="x"), title="Test")
    lines = capsys.readouterr().out.strip().splitlines()
    header = "=" * 20 + " Test " + "=" * 20
    assert lines[0] == header
    assert lines[1] == f"{'alpha':<35} x"  # keys sorted alphabetically
    assert lines[2] == f"{'beta':<35} 2"
    assert lines[3] == "=" * len(header)


def test_warn_rank_0_emits_user_warning():
    with pytest.warns(UserWarning, match="something happened"):
        warn_rank_0("something happened")


def test_warn_rank_0_no_ansi_codes_when_stderr_not_a_tty(monkeypatch):
    # Pin the non-tty branch explicitly so the test also holds under pytest -s
    # in a real terminal.
    monkeypatch.setattr(sys.stderr, "isatty", lambda: False)
    with pytest.warns(UserWarning) as record:
        warn_rank_0("plain message")
    assert str(record[0].message) == "plain message"


def test_warn_rank_0_stacklevel_points_at_caller():
    with pytest.warns(UserWarning) as record:
        warn_rank_0("stack check")
    assert record[0].filename == __file__


def test_no_stdout_silences_prints(capsys):
    print("before")
    with no_stdout():
        print("hidden")
    print("after")
    assert capsys.readouterr().out == "before\nafter\n"


def test_no_stdout_disables_tqdm_and_restores_it():
    with no_stdout():
        bar = tqdm.tqdm(range(2))
        assert bar.disable
        bar.close()
    bar2 = tqdm.tqdm(range(2))
    assert not bar2.disable  # original tqdm.__init__ restored on exit
    bar2.close()


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


def test_atomic_print_returns_result_and_prints_once(capsys):
    @atomic_print
    def announce():
        print("part1")
        print("part2")
        return 42

    assert announce() == 42
    assert capsys.readouterr().out == "part1\npart2\n"


def test_atomic_print_nested_does_not_deadlock(capsys):
    @atomic_print
    def inner():
        print("inner")

    @atomic_print
    def outer():
        print("outer")
        inner()

    outer()  # reentrant lock allows nesting
    out = capsys.readouterr().out
    assert "inner" in out
    assert "outer" in out


def test_atomic_print_restores_stdout_on_exception(capsys):
    @atomic_print
    def broken():
        print("lost output")
        raise ValueError("boom")

    original_stdout = sys.stdout
    with pytest.raises(ValueError, match="boom"):
        broken()
    assert sys.stdout is original_stdout


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


def test_deprecated_error_is_not_implemented_error():
    assert issubclass(DeprecatedError, NotImplementedError)
    with pytest.raises(NotImplementedError, match="gone"):
        raise DeprecatedError("gone")
