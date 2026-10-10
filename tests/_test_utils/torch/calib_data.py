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

"""Local calibration text for tests that only need some text, not a particular Hub dataset."""

import json
import random
from pathlib import Path

__all__ = ["write_calib_jsonl"]

_TEXT = (
    "the of and to in is that for it as was with be by on not he this are or his from at which "
    "but have an had they you were their one all we can her has there been if more when will "
    "would who so no model layer weight scale range value sample batch token cache quantize"
)
_WORDS = _TEXT.split()


def write_calib_jsonl(path: Path | str, num_samples: int = 64, words_per_sample: int = 150) -> str:
    """Write ``num_samples`` random-word texts to ``path`` as ``{"text": ...}`` JSON lines.

    Every text is long enough to fill the default 512-token calibration window with a character
    level tokenizer. The texts differ from each other and are the same on every call.
    """
    rng = random.Random(0)
    texts = [" ".join(rng.choices(_WORDS, k=words_per_sample)) for _ in range(num_samples)]
    Path(path).write_text("".join(json.dumps({"text": t}) + "\n" for t in texts), encoding="utf-8")
    return str(path)
