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

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class Request:
    question_id: int | None = None
    category: str | None = None
    system_prompt: str | None = None
    turns: list[str] = field(default_factory=list)
    mm_content: Any | None = None  # TODO

    # not to be set by user
    output_turn_ids = None
    output_turn_text: list[str] = field(default_factory=list)


class Dataset:
    def __init__(self, path, **kwargs):
        self.data: list[Request] = []
        raise NotImplementedError

    def _preprocess(self):
        raise NotImplementedError

    @classmethod
    def prepare_data(cls, output_dir: str | Path, **kwargs) -> Path:
        """Prepare and save the dataset to the specified output directory.

        Downloads any external data, resolves all references, and persists
        the fully-resolved dataset so that subsequent loads are self-contained.

        Args:
            output_dir: Directory where the prepared data file will be saved.
            **kwargs: Dataset-specific parameters (e.g. config_name, category).

        Returns:
            Path to the saved dataset file.

        Raises:
            NotImplementedError: Subclasses must override this method.
        """
        raise NotImplementedError
