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
"""Failures that a rerun clears. ``tests/conftest.py`` reruns a test once when it fails with one."""

# Substrings that identify a *transient* failure: a HuggingFace Hub / dataset access error, or a
# rendezvous port that another process bound between its selection and its use
TRANSIENT_MARKERS = (
    "HfHubHTTPError",
    "Too Many Requests",
    "500 Server Error",
    "502 Bad Gateway",
    "503 Server Error",
    "504 Server Error",
    "Bad Gateway",
    "Service Unavailable",
    "Gateway Time-out",
    "ConnectionError",
    "ReadTimeout",
    "ConnectTimeout",
    "Max retries exceeded",
    "NewConnectionError",
    "Connection reset by peer",
    "Connection aborted",
    "Consistency check failed",  # partial / interrupted HF download
    "couldn't connect to 'https://huggingface.co'",
    "Couldn't reach",  # datasets: "Couldn't reach <repo> on the Hub"
    "Temporary failure in name resolution",  # transient DNS
    "Name or service not known",  # transient DNS
    "EADDRINUSE",  # torch.distributed's TCPStore: "name: EADDRINUSE, message: address already in use"
    "Address already in use",  # the same from Python's OSError, gloo and NCCL
)


class TransientError(RuntimeError):
    """A command failed and its output shows one of ``TRANSIENT_MARKERS``."""


def find_transient_marker(text: str) -> str | None:
    """Return the first of ``TRANSIENT_MARKERS`` found in ``text``, if any."""
    return next((marker for marker in TRANSIENT_MARKERS if marker in text), None)
