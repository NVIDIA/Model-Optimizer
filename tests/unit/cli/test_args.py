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

"""Tests for modelopt/cli/args.py."""

import argparse

import pytest

from modelopt.cli import add_model_args


def _parser(**kwargs) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    add_model_args(parser, **kwargs)
    return parser


@pytest.mark.parametrize(
    "flag", ["--model_name_or_path", "--model-name-or-path", "--model", "--pyt_ckpt_path"]
)
def test_every_spelling_sets_model_name_or_path(flag):
    args = _parser(aliases=("--pyt_ckpt_path",)).parse_args([flag, "org/model"])
    assert args.model_name_or_path == "org/model"
    assert not args.trust_remote_code


@pytest.mark.parametrize("flag", ["--trust_remote_code", "--trust-remote-code"])
def test_trust_remote_code_is_opt_in(flag):
    args = _parser().parse_args(["--model", "org/model", flag])
    assert args.trust_remote_code


@pytest.mark.parametrize("alias", ["--model_name_or_path", "--model"])
def test_an_alias_repeating_a_standard_flag_is_rejected(alias):
    with pytest.raises(ValueError, match=alias):
        _parser(aliases=(alias,))


def test_model_is_required_unless_said_otherwise():
    with pytest.raises(SystemExit):
        _parser().parse_args([])
    assert _parser(required=False).parse_args([]).model_name_or_path is None


def test_roles_compose_into_one_parser():
    parser = argparse.ArgumentParser()
    add_model_args(parser, role="student")
    add_model_args(parser, role="teacher", trust_remote_code=False)

    args = parser.parse_args(
        ["--student_model_name_or_path", "org/small", "--teacher-model-name-or-path", "org/big"]
    )

    assert (args.student_model_name_or_path, args.teacher_model_name_or_path) == (
        "org/small",
        "org/big",
    )
    # A role takes no --model alias: it would be ambiguous between roles.
    with pytest.raises(SystemExit):
        parser.parse_args(["--model", "org/small", "--teacher_model_name_or_path", "org/big"])
