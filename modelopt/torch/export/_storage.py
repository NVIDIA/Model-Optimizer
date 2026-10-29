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

"""Conservative byte reservations before writing an isolated export."""

import json
import os
import shutil
from pathlib import Path

from safetensors.torch import save_file

__all__ = []


class ExportStorageBudget:
    def __init__(self, limit):
        if type(limit) is not int or limit < 1:
            raise ValueError("Export storage limit must be a positive byte count")
        self.remaining = limit

    def reserve(self, size):
        if size > self.remaining:
            raise ValueError("Export exceeds its storage byte limit")
        self.remaining -= size

    def write_text(self, path, content):
        data = content.encode()
        self.reserve(len(data))
        Path(path).write_bytes(data)

    def copy(self, source, destination):
        with Path(source).open("rb") as reader:
            size = os.fstat(reader.fileno()).st_size
            self.reserve(size)
            with Path(destination).open("xb") as writer:
                remaining = size
                while remaining:
                    chunk = reader.read(min(remaining, 1024**2))
                    if not chunk:
                        raise ValueError("Source file shrank during export")
                    writer.write(chunk)
                    remaining -= len(chunk)
                if reader.read(1):
                    raise ValueError("Source file grew during export")
        shutil.copystat(source, destination)

    def save_safetensors(self, tensors, path):
        offset, header = 0, {}
        for name, tensor in tensors.items():
            header[name] = {
                # All supported safetensors dtype tags are shorter than this reservation.
                "dtype": "x" * 32,
                "shape": list(tensor.shape),
                "data_offsets": [offset, offset + tensor.nbytes],
            }
            offset += tensor.nbytes
        # ASCII escaping and default JSON whitespace conservatively bound the compact
        # safetensors header; account for its length prefix and alignment without tensor copies.
        self.reserve(offset + 8 + ((len(json.dumps(header).encode()) + 7) // 8) * 8)
        save_file(tensors, str(path))
