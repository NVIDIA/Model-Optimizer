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


"""Cache only first-MoE inputs, with exact BF16 bits and disjoint token provenance."""

import argparse
import hashlib
import json
import os
from pathlib import Path

import numpy as np
import torch

from examples.local_quantization_distillation.prepare_data import PackedBatchSequence

from .source import capture_inputs, prefix_model


def main():
    """Capture unique packed rows independently across GPUs, then unload the prefix."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", required=True)
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    root = Path(args.root)
    rank, size = int(os.environ["LOCAL_RANK"]), int(os.environ["WORLD_SIZE"])
    torch.cuda.set_device(rank)
    source = json.loads((root / "source.json").read_text())
    output = root / ("smoke-activations" if args.smoke else "activations")
    output.mkdir(exist_ok=True)
    fingerprint = hashlib.sha256((root / "data/manifest.json").read_bytes()).hexdigest()
    marker = output / f"rank{rank}-COMMITTED.json"
    if marker.is_file():
        previous = json.loads(marker.read_text())
        if (
            previous["data_fingerprint"] != fingerprint
            or previous["model_revision"] != source["revision"]
            or previous["world_size"] != size
        ):
            raise ValueError("Existing activation capture has different data/source/topology.")
        for partition, info in previous["partitions"].items():
            array = np.load(output / f"{partition}-rank{rank}.npy", mmap_mode="r")
            if array.shape != (info["rows"], 8192, previous["hidden_size"]):
                raise ValueError("Committed activation shape differs from its receipt.")
        print(json.dumps({"capture_reused": rank}), flush=True)
        return
    model = prefix_model(source["source"])
    hidden = model.config.hidden_size
    receipt = {
        "data_fingerprint": fingerprint,
        "model_revision": source["revision"],
        "world_size": size,
        "rank": rank,
        "hidden_size": hidden,
        "partitions": {},
    }
    for partition in ("calibration", "train", "validation", "diagnostic"):
        data = PackedBatchSequence(root / "data", partition, global_batch_size=1, device="cuda")
        rows = list(range(rank, min(len(data), size * 2) if args.smoke else len(data), size))
        array = np.lib.format.open_memmap(
            output / f"{partition}-rank{rank}.npy",
            mode="w+",
            dtype=np.uint16,
            shape=(len(rows), 8192, hidden),
        )
        for local, index in enumerate(rows):
            batch = data[index]
            if not batch["valid_mask"].all():
                raise ValueError("The source corpus must contain fully packed valid rows.")
            inputs = capture_inputs(model, batch)
            array[local] = inputs.squeeze(0).cpu().contiguous().view(torch.uint16).numpy()
            if local == 0 or (local + 1) % 32 == 0:
                print(
                    json.dumps(
                        {
                            "capture": partition,
                            "rank": rank,
                            "rows": local + 1,
                            "total_rows": len(rows),
                        }
                    ),
                    flush=True,
                )
        array.flush()
        array._mmap.close()
        receipt["partitions"][partition] = {"rows": len(rows), "tokens": len(rows) * 8192}
    (output / f"rank{rank}-COMMITTED.json").write_text(json.dumps(receipt, indent=2) + "\n")


if __name__ == "__main__":
    main()
