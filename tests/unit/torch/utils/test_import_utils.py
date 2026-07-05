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

import warnings

import pytest

from modelopt.torch.utils.import_utils import import_plugin


def _import_missing_module(tracker=None, **kwargs):
    """Run import_plugin around an import that always fails with ModuleNotFoundError."""
    with import_plugin("foo", **kwargs):
        __import__("definitely_not_a_real_module_xyz")
        if tracker is not None:
            tracker.append("ran past import")


def _raise_runtime_error(**kwargs):
    with import_plugin("myplugin", **kwargs):
        raise RuntimeError("boom")


def test_no_error_and_no_success_msg_is_silent():
    with warnings.catch_warnings():
        warnings.simplefilter("error")  # any warning would fail the test
        with import_plugin("foo"):
            pass


def test_success_msg_emitted_on_clean_import():
    with (
        pytest.warns(UserWarning, match="foo plugin loaded"),
        import_plugin("foo", success_msg="foo plugin loaded"),
    ):
        pass


def test_module_not_found_suppressed_with_message():
    tracker = []
    with pytest.warns(UserWarning, match="foo is not installed"):
        _import_missing_module(tracker=tracker, msg_if_missing="foo is not installed")
    assert not tracker  # code after the failing import must not run


def test_module_not_found_without_message_is_silent():
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        _import_missing_module()


def test_module_not_found_verbose_false_is_silent():
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        _import_missing_module(msg_if_missing="should not appear", verbose=False)


def test_generic_exception_suppressed_with_warning():
    with pytest.warns(UserWarning, match="Failed to import modelopt myplugin plugin") as record:
        _raise_runtime_error()
    assert "RuntimeError('boom')" in str(record[0].message)


def test_generic_exception_verbose_false_is_silent():
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        _raise_runtime_error(verbose=False)


def test_success_msg_not_emitted_when_import_fails():
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        _import_missing_module(success_msg="should not appear", verbose=True)
