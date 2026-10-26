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

import asyncio
import importlib.util
import json
import sys
from pathlib import Path

_EXAMPLE = Path(__file__).parents[3] / "examples" / "dataset"
sys.path.insert(0, str(_EXAMPLE))
_SPEC = importlib.util.spec_from_file_location("sonar_make_dataset", _EXAMPLE / "make_dataset.py")
assert _SPEC is not None and _SPEC.loader is not None
make_dataset = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = make_dataset
_SPEC.loader.exec_module(make_dataset)


def test_dataset_threaded_io_preserves_message_format_and_ids(tmp_path):
    conversation = [{"role": "user", "content": "héllo"}]
    entries = [
        {"conversations": conversation},
        {"conversation_id": "existing", "conversations": conversation},
    ]
    output = tmp_path / "data.jsonl"
    asyncio.run(asyncio.to_thread(make_dataset._write_conversations, output, entries))
    restored = asyncio.run(asyncio.to_thread(make_dataset._read_jsonl, output))
    assert [entry["messages"] for entry in restored] == [conversation, conversation]
    assert restored[0]["conversation_id"] == make_dataset.id_for_conversation(conversation)
    assert restored[1]["conversation_id"] == "existing"
    assert all("conversations" not in entry for entry in restored)
    source = tmp_path / "data.json"
    source.write_text(json.dumps(entries), encoding="utf-8")
    assert asyncio.run(asyncio.to_thread(make_dataset._read_json, source)) == entries
