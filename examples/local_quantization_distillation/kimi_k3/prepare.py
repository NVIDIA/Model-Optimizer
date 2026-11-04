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


"""Prepare only the source shards and disjoint tokens needed for Kimi's first MoE."""

import argparse
import json
from pathlib import Path
from types import SimpleNamespace

from huggingface_hub import hf_hub_download, snapshot_download
from transformers import AutoTokenizer

from examples.local_quantization_distillation.prepare_data import (
    _conversations,
    prepare_packed_corpus,
)

MODEL = "moonshotai/Kimi-K3"
REVISION = "f831ab66814297da540d832a5235f8e904f29d06"
DATASET = "nvidia/Nemotron-Post-Training-Dataset-v2"
DATA_REVISION = "5c89e01dd720ae0f4058445ed49c5fb68a03c76e"


def selected_source_key(key):
    """Select embeddings and the two native blocks required to reach MoE layer1."""
    return key == "language_model.model.embed_tokens.weight" or key.startswith(
        ("language_model.model.layers.0.", "language_model.model.layers.1.")
    )


def main():
    """Download three source shards and prepare the study's immutable packed splits."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    root = Path(args.output).resolve()
    root.mkdir(parents=True, exist_ok=True)
    index = hf_hub_download(MODEL, "model.safetensors.index.json", revision=REVISION)
    weight_map = json.loads(Path(index).read_text())["weight_map"]
    shards = sorted({file for key, file in weight_map.items() if selected_source_key(key)})
    source = snapshot_download(
        MODEL,
        revision=REVISION,
        allow_patterns=[
            *shards,
            "config.json",
            "tokenizer*",
            "special_tokens_map.json",
            "*.jinja",
            "chat_templates/*",
            "model.safetensors.index.json",
            "tokenization_kimi.py",
            "encoding_k3.py",
            "tiktoken.model",
        ],
        max_workers=3,
    )
    metadata = {
        "model": MODEL,
        "revision": REVISION,
        "source": source,
        "shards": shards,
        "layer": 1,
        "teacher": "released MXFP4 routed experts decoded to BF16",
        "capture_sequence_length": 8192,
        "repair_sequence_length": 2048,
        "updates": 512,
        "global_batch_size": 8,
        "seed": 42,
    }
    (root / "source.json").write_text(json.dumps(metadata, indent=2) + "\n")
    data = root / "data"
    if not (data / "manifest.json").is_file():
        tokenizer = AutoTokenizer.from_pretrained(source, trust_remote_code=True)
        options = SimpleNamespace(
            dataset=DATASET,
            splits=["stem", "chat", "math", "code"],
            seed=42,
            max_conversations=1_000_000,
        )
        prepare_packed_corpus(
            _conversations(options, tokenizer, DATA_REVISION, native_chat_tokenization=True),
            data,
            sequence_length=8192,
            pack_counts={"train": 1024, "calibration": 64, "validation": 32, "diagnostic": 64},
            seed=42,
            padding_idx=tokenizer.pad_token_id or 0,
            source_metadata={
                "dataset": DATASET,
                "revision": DATA_REVISION,
                "splits": options.splits,
                "tokenizer": MODEL,
                "tokenizer_revision": REVISION,
            },
        )
    (root / "PREPARED.json").write_text(json.dumps(metadata, indent=2) + "\n")
    print(json.dumps({"prepared": str(root), "downloaded_shards": len(shards)}), flush=True)


if __name__ == "__main__":
    main()
