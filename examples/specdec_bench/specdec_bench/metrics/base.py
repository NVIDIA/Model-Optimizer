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

import json
import os


class Metric:
    directory = "./"

    def __init__(self):
        self.out = {}
        self.name = "metric"

    def process_final(self, text_outputs, request_records):
        raise NotImplementedError

    def clear(self):
        self.out = {}

    @staticmethod
    def chunk_lengths(record):
        chunk_lengths = record.get("chunk_lengths")
        if chunk_lengths is not None:
            return chunk_lengths

        output_ids = record.get("output_ids", [])
        if output_ids and output_ids[0] and isinstance(output_ids[0][0], list):
            return [[len(chunk) for chunk in beam_output] for beam_output in output_ids]
        return [[] for _ in output_ids]

    @classmethod
    def output_chunks(cls, record):
        output_ids = record.get("output_ids", [])
        if output_ids and output_ids[0] and isinstance(output_ids[0][0], list):
            return output_ids

        chunks = []
        chunk_lengths = cls.chunk_lengths(record)
        for beam_idx, beam_lengths in enumerate(chunk_lengths):
            beam_tokens = output_ids[beam_idx] if beam_idx < len(output_ids) else []
            beam_chunks = []
            cursor = 0
            for chunk_length in beam_lengths:
                beam_chunks.append(beam_tokens[cursor : cursor + chunk_length])
                cursor += chunk_length
            chunks.append(beam_chunks)
        return chunks

    @staticmethod
    def trim_beam_lengths(beam_lengths):
        if not beam_lengths:
            return [1]
        start_idx = 1 if beam_lengths[0] == 1 else 0
        end_idx = len(beam_lengths) - 1
        if end_idx <= start_idx:
            return [1]
        return beam_lengths[start_idx:end_idx]

    @staticmethod
    def trim_beam_chunks(beam_chunks):
        if not beam_chunks:
            return []
        start_idx = 1 if len(beam_chunks[0]) == 1 else 0
        end_idx = len(beam_chunks) - 1
        if end_idx <= start_idx:
            return []
        return beam_chunks[start_idx:end_idx]

    @classmethod
    def trimmed_chunk_lengths(cls, record):
        return [cls.trim_beam_lengths(beam_lengths) for beam_lengths in cls.chunk_lengths(record)]

    @classmethod
    def trimmed_output_chunks(cls, record):
        return [cls.trim_beam_chunks(beam_chunks) for beam_chunks in cls.output_chunks(record)]

    @staticmethod
    def flat_output_tokens(record):
        output_ids = record.get("output_ids", [])
        if output_ids and output_ids[0] and isinstance(output_ids[0][0], list):
            return [token for beam_output in output_ids for chunk in beam_output for token in chunk]
        return [token for beam_output in output_ids for token in beam_output]

    def write(self):
        os.makedirs(self.directory, exist_ok=True)
        if self.out:
            filename = os.path.join(self.directory, f"{self.name}.json")
            if os.path.exists(filename):
                with open(filename) as json_file:
                    existing_data = json.load(json_file)
                existing_data.append(self.out)
            else:
                existing_data = [self.out]

            with open(filename, "w") as json_file:
                json.dump(existing_data, json_file, indent=4)

    @classmethod
    def update_directory(cls, new_dir):
        cls.directory = new_dir
