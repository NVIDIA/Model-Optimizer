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

"""Tests for the configuration-redaction helpers in specdec_bench/utils.py.

The redaction surface is the only thing standing between user-supplied secrets
(HF tokens, AWS keys pasted into runtime_params or serving_config) and an S3
bucket of configuration.json files. Worth exercising it explicitly.
"""

import pytest

pytest.importorskip("transformers")  # utils.py imports AutoTokenizer at module load
from specdec_bench.utils import (
    _SENSITIVE_KEY_ALLOWLIST,
    _is_sensitive_key,
    _redact_argv,
    _redact_value,
)

REDACTED = "***REDACTED***"


class TestIsSensitiveKey:
    @pytest.mark.parametrize(
        "key",
        ["hf_token", "HF_TOKEN", "api_key", "aws_secret_access_key", "password", "secret"],
    )
    def test_sensitive_names_match(self, key):
        assert _is_sensitive_key(key) is True

    @pytest.mark.parametrize("key", list(_SENSITIVE_KEY_ALLOWLIST))
    def test_allowlist_overrides(self, key):
        """tokenizer / tokenizer_path / tokenizer_mode / tokenizer_revision
        all contain the substring 'token' but are not secrets."""
        assert _is_sensitive_key(key) is False

    @pytest.mark.parametrize("key", ["model_dir", "concurrency", "engine", "speculative_algorithm"])
    def test_non_sensitive_names_pass(self, key):
        assert _is_sensitive_key(key) is False


class TestRedactValue:
    def test_top_level_secret(self):
        out = _redact_value({"hf_token": "abc", "model_dir": "/m"})
        assert out == {"hf_token": REDACTED, "model_dir": "/m"}

    def test_nested_secret_in_dict(self):
        """Nested hf_token inside serving_config must be redacted."""
        cfg = {
            "serving_config": {
                "model_loader_extra_config": {"hf_token": "supersecret"},
                "tokenizer": "/path/to/tok",
            }
        }
        out = _redact_value(cfg)
        assert out["serving_config"]["model_loader_extra_config"]["hf_token"] == REDACTED
        # tokenizer is in the allowlist — must survive
        assert out["serving_config"]["tokenizer"] == "/path/to/tok"

    def test_nested_secret_in_list(self):
        cfg = {"runtime_params": [{"aws_secret_access_key": "s"}, {"endpoint": "e"}]}
        out = _redact_value(cfg)
        assert out["runtime_params"][0]["aws_secret_access_key"] == REDACTED
        assert out["runtime_params"][1] == {"endpoint": "e"}

    def test_tuple_preserved(self):
        out = _redact_value(("a", {"hf_token": "x"}))
        assert out == ("a", {"hf_token": REDACTED})

    def test_primitive_passthrough(self):
        for v in [None, 0, 1.5, "plain-string", True]:
            assert _redact_value(v) == v


class TestRedactArgv:
    def test_long_flag_separate_value(self):
        out = _redact_argv(["run.py", "--hf_token", "abcdef", "--model", "/m"])
        assert out == ["run.py", "--hf_token", REDACTED, "--model", "/m"]

    def test_long_flag_equals_form(self):
        out = _redact_argv(["run.py", "--api_key=topsecret", "--concurrency", "1"])
        assert out == ["run.py", f"--api_key={REDACTED}", "--concurrency", "1"]

    def test_non_sensitive_long_flag_unchanged(self):
        argv = ["run.py", "--model", "/m", "--concurrency", "4"]
        assert _redact_argv(argv) == argv

    def test_allowlist_keys_not_redacted_in_argv(self):
        out = _redact_argv(["run.py", "--tokenizer", "/path/tok"])
        assert out == ["run.py", "--tokenizer", "/path/tok"]

    def test_prev_is_sensitive_carry_resets(self):
        """After a sensitive flag's value is consumed, the next token must NOT
        be redacted just because it follows a now-consumed sensitive flag."""
        out = _redact_argv(["run.py", "--hf_token", "abc", "--model", "/m"])
        assert out[-1] == "/m"  # /m must survive

    def test_empty_argv(self):
        assert _redact_argv([]) == []

    def test_non_string_tokens_handled(self):
        """argv usually arrives as strings but defend against int/Path/etc."""
        out = _redact_argv(["run.py", "--concurrency", 4])
        assert out == ["run.py", "--concurrency", "4"]
