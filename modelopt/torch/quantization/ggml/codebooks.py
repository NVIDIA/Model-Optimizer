# This file includes the IQ1_S and IQ2_XS codebooks adapted from:
# https://github.com/ggml-org/llama.cpp/blob/9b05354ec6fb58b4e665e9a39ebc40285c015638/ggml/src/ggml-common.h
#
# MIT License
#
# Copyright (c) 2023-2026 The ggml authors
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
#
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0 AND MIT
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

"""Canonical IQ codebooks, carried verbatim from the GGML tables.

This module holds only data and its decoding. The tables are the one piece of GGML material
reproduced in this package, so keeping them here bounds the MIT-licensed surface to a single
file and leaves the codecs beside it as reviewable logic.
"""

import base64
import zlib
from functools import cache

# zlib-compressed little-endian bytes of the canonical uint64_t table. The
# decoded int8 values are -1, 0, and 1.
_IQ1_S_GRID_ZLIB_B64 = (
    "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"
    "jDMYrMeWQNZ5e68uzuCwPq7TO3/ss/TvHzM5DA8="
)

# Compact byte representation of the canonical [512, 8] grid. Values are only
# 8, 25, and 43. Keeping this as checkpoint-independent package data avoids
# adding a pickle-backed torch.save artifact to the wheel.
_IQ2_XS_GRID_B64 = (
    "CAgICAgICAgrCAgICAgICBkZCAgICAgICCsICAgICAgrKwgICAgICBkIGQgICAgICBkZCAgICAgrGRkICAgICBkrGQgICAgICAgr"
    "CAgICAgrCCsICAgICBkZKwgICAgICCsrCAgICAgZCAgZCAgICAgZCBkICAgIKxkIGQgICAgZKwgZCAgICAgIGRkICAgIKwgZGQgI"
    "CAgZGRkZCAgICAgrGRkICAgIGQgrGQgICAgIGSsZCAgICAgICCsICAgIKwgIKwgICAgZGQgrCAgICAgrCCsICAgIGQgZKwgICAgI"
    "GRkrCAgICBkrGSsICAgICAgrKwgICAgZCAgIGQgICAgZCAgZCAgIKxkICBkICAgZKwgIGQgICAgIGQgZCAgIKwgZCBkICAgZGRkI"
    "GQgICAgrGQgZCAgIKysZCBkICAgZCCsIGQgICAgZKwgZCAgICAgIGRkICAgrCAgZGQgICBkZCBkZCAgICCsIGRkICAgZCBkZGQgI"
    "CAgZGRkZCAgICAgrGRkICAgIKysZGQgICBkICCsZCAgICBkIKxkICAgICBkrGQgICAgICAgrCAgIKwgICCsICAgZGQgIKwgICAgr"
    "CAgrCAgIGQgZCCsICAgIGRkIKwgICAgIKwgrCAgIGQgIGSsICAgIGQgZKwgICAgIGRkrCAgIGRkZGSsICAgICAgrKwgICCsrCCsr"
    "CAgIGQgICAgZCAgIGQgICBkICCsZCAgIGQgIGSsICAgZCAgICBkICBkICCsIGQgIGQgIGRkZCAgZCAgIKxkICBkICBkIKwgIGQgI"
    "CBkrCAgZCAgICAgZCBkICCsICBkIGQgIGRkIGQgZCAgIKwgZCBkICBkIGRkIGQgICBkZGQgZCAgrGRkZCBkICAgIKxkIGQgIGQgI"
    "KwgZCAgIGQgrCBkICAgIGSsIGQgICAgICBkZCAgrCAgIGRkICBkZCAgZGQgICCsICBkZCAgZCBkIGRkICAgZGQgZGQgICAgrCBkZ"
    "CAgZCAgZGRkICAgZCBkZGQgICAgZGRkZCAgZCCsZGRkICAgICCsZGQgIGQgICCsZCAgIGQgIKxkICAgIGQgrGQgIKxkrCCsZCAgI"
    "CAgZKxkICCsICBkrGQgICBkIKysZCAgICAgICCsICCsICAgIKwgIGRkICAgrCAgIKwgICCsICCsrCAgIKwgIGQgZCAgrCAgIGRkI"
    "CCsICAgIKwgIKwgIGRkrCAgrCAgZCAgZCCsICAgZCBkIKwgICAgZGQgrCAgIKxkZCCsICAgICCsIKwgICAgrKwgrCAgrKysrCCsI"
    "CBkICAgZKwgICBkICBkrCAgICBkIGSsICAgICBkZKwgIGQgIKxkrCAgZKwgrGSsICAgICAgrKwgICAgrCCsrCAgIKysIKysICCsZ"
    "GSsrKwgICAgrKysrCAgZCAgICAgZCAgZCAgICBkIKxkICAgIGQgZKwgICAgZCAgIGQgICBkIKwgZCAgIGQgZGRkICAgZCAgrGQgI"
    "CBkIGQgrCAgIGQgIGSsICAgZCAgICBkICBkIKwgIGQgIGQgZGQgZCAgZCAgrCBkICBkIGQgZGQgIGQgIGRkZCAgZCAgIKxkICBkI"
    "KysrGQgIGQgZCAgrCAgZCAgZCCsICBkICAgZKwgIGQgICAgIGQgZCCsICAgZCBkIGRkICBkIGQgIKwgIGQgZCBkIGQgZCBkICBkZ"
    "CBkIGQgICCsIGQgZCBkICBkZCBkICBkIGRkIGQgICBkZGQgZCAgICCsZCBkICBkZKxkIGQgrGRkrGQgZCBkICAgrCBkICBkICCsI"
    "GQgrGQgIKwgZCAgIGQgrCBkICAgIGSsIGQgICCsZKwgZCAgICAgIGRkIKwgICAgZGQgZGQgICBkZCAgrCAgIGRkIGQgZCAgZGQgI"
    "GRkICBkZCAgIKwgIGRkIGQgIGQgZGQgIGQgZCBkZCBkrCBkIGRkICAgZGQgZGQgIGSsZCBkZCAgICCsIGRkIGQgICBkZGQgIGQgI"
    "GRkZCAgIGQgZGRkICAgIGRkZGQgICAgIKxkZCAgZGQgrGRkIGSsIGSsZGQgZCAgICCsZCAgZCAgIKxkICAgZCAgrGQgrCBkICCsZ"
    "CAgICBkIKxkICBkZGQgrGQgrGQgrCCsZCAgICAgZKxkIGRkICBkrGQgrGSsZGSsZCBkIGRkrKxkIGSsrKysrGQgICAgICAgrCCsI"
    "CAgICCsIGRkICAgIKwgIKwgICAgrCCsrCAgICCsIGQgZCAgIKwgIGRkICAgrCAgIKwgICCsIGQgIGQgIKwgIGQgZCAgrCAgIGRkI"
    "CCsICAgIKwgIKwgICCsrCAgrCBkICAgZCCsICBkICBkIKwgICBkIGQgrCAgICBkZCCsICCsIGRkIKwgZGSsZGQgrCAgICAgrCCsI"
    "KwgrCCsIKwgICAgrKwgrCAgrKysrCCsIGQgICAgZKwgIGQgICBkrCAgIGQgIGSsIGSsrCAgZKwgICAgZCBkrCAgICAgZGSsIGQgI"
    "GRkZKwgrCBkZGRkrCBkrGSsZGSsIGQgICCsZKwgrKxkIKxkrCCsZKysrGSsICAgICAgrKwgIKwgICCsrCCsrCAgIKysICAgrCAgr"
    "KwgZGRkZCCsrCAgrCCsIKysIKwgrKwgrKwgIKysZGSsrCAgIGSsZKysICCsICCsrKwgICCsIKysrCCsICCsrKysICCsIKysrKwgr"
    "KwgrKysrCBkICAgICAgZCBkICAgICBkrGQgICAgIGRkrCAgICAgZCAgZCAgICBkrCBkICAgIGRkZGQgICAgZCCsZCAgICBkZCCsI"
    "CAgIGQgZKwgICAgZCAgIGQgICBkrCAgZCAgIGRkZCBkICAgZCCsIGQgICBkrKwgZCAgIGRkIGRkICAgZCBkZGQgICBkICCsZCAgI"
    "GRkZKxkICAgZGQgIKwgICBkIGQgrCAgIGQgIGSsICAgZCAgICBkICBkrCAgIGQgIGRkZCAgZCAgZCCsICBkICBkZCBkIGQgIGQgZ"
    "GQgZCAgZCAgrCBkICBkZCAgZGQgIGQgZCBkZCAgZCAgZGRkICBkICAgrGQgIGRkZCCsZCAgZKwgrKxkICBkZCAgIKwgIGQgZCAgr"
    "CAgZCAgZCCsICBkrCBkIKwgIGRkrKwgrCAgZCAgIGSsICBkICAgICBkIGSsICAgIGQgZGRkICAgZCBkIKwgICBkIGRkIGQgIGQgZ"
    "CBkZCAgZCBkZKxkICBkIGQgIKwgIGQgZGQgIGQgZCBkIGQgZCBkIGQgIGRkIGQgZCAgIKwgZCBkIGRkrCBkIGRkICAgZGQgZCBkI"
    "CBkZCBkICBkIGRkIGQgZKwgZGQgZCAgIGRkZCBkrKxkrGRkIGQgICAgrGQgZKysICCsZCBkIGQgZKxkIGQgIGRkrGQgZGQgICAgr"
    "CBkIGQgICCsIGQgIGQgIKwgZCAgIGQgrCBkZGQgZCCsIGQgZGRkIKwgZKwgrGQgrCBkICAgIGSsIGRkIGQgZKwgZCBkIGRkrCBkI"
    "CBkZGSsIGRkrKxkZKwgZCBkICCsrCBkICAgICAgZGSsICAgICBkZGRkICAgIGRkIKwgICAgZGRkIGQgICBkZCBkZCAgIGRkICCsI"
    "CAgZGQgrKwgICBkZGQgIGQgIGRkIGQgZCAgZGQgIGRkICBkZCAgIKwgIGRkZCAgIGQgZGQgZCAgZCBkZCAgZCBkIGRkZGRkIGQgZ"
    "GQgICBkZCBkZKwgIGRkIGRkICAgIKwgZGQgZCBkrCBkZKysrKysIGRkZCAgICBkZGQgZCAgIGRkZCAgZCAgZGRkZCCsICBkZGQgI"
    "CBkIGRkZCAgrGQgZGRkZCAgrCBkZGRkIKysIGRkZCAgICBkZGRkIKwgIGRkZGQgICCsZGRkZCCsIKxkZGRkZCCsIKxkZGQgrKxkr"
    "GRkZGQgrKysZGRkICAgICCsZGQgZGQgIKxkZGQgIGQgrGRkICBkZCCsZGRkrGSsIKxkZKysZCBkrGRkICAgZGSsZGSsICBkZKxkZ"
    "GRkIKysrGRkZCAgICAgrGQgZCAgICCsZCAgZCAgIKxkICAgZCAgrGQgZGRkICCsZKwgrGQgIKxkrGQgrCAgrGRkrKysICCsZCAgI"
    "CBkIKxkIGSsIKwgrGSsrCBkrCCsZKwgZKysIKxkICAgICBkrGSsZGQgIGSsZCAgZCBkZKxkICAgZGRkrGRkZCBkZGSsZCBkrKxkZ"
    "KxkZCAgICCsrGSsrKxkIKysZGRkrCBkrKxkrGQgIKysrGQgZGRkrKysZKwgrGSsrKxkICAgICAgIKysICAgICAgrGRkICAgICCsI"
    "KwgICAgIKxkIGQgICAgrCBkZCAgICCsICCsICAgIKysrKwgICAgrGQgIGQgICCsIGQgZCAgIKwgIGRkICAgrCAgIKwgICCsrCAgr"
    "CAgIKwgrKysICAgrKysrKwgICCsZCAgIGQgIKwgZCAgZCAgrKxkICBkICCsICBkIGQgIKwgICBkZCAgrGQgZGRkICCsZKxkZGQgI"
    "KwgICAgrCAgrCAgrCCsICCsICAgrKwgIKysICCsrCAgrCAgrKysICCsIKysrKwgIKxkICAgIGQgrCBkICAgZCCsICBkICBkIKysI"
    "GQgIGQgrGRkZCAgZCCsICAgZCBkIKwgIKxkIGQgrGSsIKwgZCCsICAgIGRkIKwgZCBkZGQgrGRkrKxkZCCsIKxkIKxkIKysrKxkr"
    "GQgrCAgICAgrCCsIKwgICCsIKxkZKwgIKwgrKysZGQgrCCsICAgrCCsIKysICCsIKwgrCCsrKwgrCCsrGQgIGSsIKysIKwgrKwgr"
    "CAgIKysrCCsIKwgrKysIKysZGSsrKwgrCCsrKysrCCsZCAgICAgZKwgZCAgICBkrCAgZCAgIGSsICAgZCAgZKysZGRkICBkrCBkI"
    "KwgIGSsICAgIGQgZKysIKwgZCBkrCBkrGRkIGSsrGRkZKwgZKxkrCCsrCBkrCAgICAgZGSsZGQgICBkZKwgZCBkIGRkrCAgZGQgZ"
    "GSsIKxkZCBkZKxkrKwgZGRkrCAgZKxkZGSsrCBkrGRkZKxkICBkrGRkrGQgZGQgrGSsrGSsrCCsZKxkrCBkZKxkrGRkZCCsrGSsI"
    "CCsZKysZKwgICAgICCsrKwgICAgIKysIKwgICAgrKysrCAgICCsrCAgrCAgIKysrKysICAgrKwgIKysICCsrGQgZGRkIKysZKxkZ"
    "GQgrKysZKysZCCsrCAgICCsIKysrCAgIKwgrKwgrCAgrCCsrKysrCCsIKysICAgrKwgrKwgIKysrCCsrCAgIGQgZKysZGRkrCBkr"
    "KxkZKxkrGSsrCCsZKysZKysrKwgICCsrKwgIKwgIKysrKwgrCAgrKysIKysICCsrKwgIKysIKysrCCsrKwgrKysIGQgIGSsrKwgZ"
    "CCsZKysrKxkIKxkrKysIKysIKysrKysrKwgrKysrGQgZKysrKysrKysrKysrKw=="
)


@cache
def iq1_s_grid_bytes() -> bytes:
    """Decoded little-endian int8 bytes of the [2048, 8] IQ1_S ternary table."""
    return zlib.decompress(base64.b64decode(_IQ1_S_GRID_ZLIB_B64))


@cache
def iq2_xs_grid_bytes() -> bytes:
    """Decoded bytes of the [512, 8] IQ2_XS magnitude table."""
    return base64.b64decode(_IQ2_XS_GRID_B64)
