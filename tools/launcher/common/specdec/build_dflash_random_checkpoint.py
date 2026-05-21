#!/usr/bin/env python3
"""Generate a random-weight DFlash checkpoint matching Qwen3-8B-DFlash-b16.

Writes to /scratchspace/dflash_random_b16/exported-checkpoint-0 by default.
"""

import argparse
import json
import os
from pathlib import Path
import struct
import urllib.request

import torch
from safetensors.torch import save_file


REPO_ID = "z-lab/Qwen3-8B-DFlash-b16"
CONFIG_URL = f"https://huggingface.co/{REPO_ID}/resolve/main/config.json"
MODEL_URL = f"https://huggingface.co/{REPO_ID}/resolve/main/model.safetensors"


def fetch_config() -> dict:
    with urllib.request.urlopen(CONFIG_URL) as resp:
        return json.load(resp)


def fetch_safetensors_index() -> dict:
    req = urllib.request.Request(MODEL_URL)
    req.add_header("Range", "bytes=0-1048575")
    with urllib.request.urlopen(req) as resp:
        data = resp.read()

    if len(data) < 8:
        raise RuntimeError("Failed to read safetensors header")
    header_len = struct.unpack("<Q", data[:8])[0]
    need = header_len + 8
    if len(data) < need:
        req = urllib.request.Request(MODEL_URL)
        req.add_header("Range", f"bytes=0-{need-1}")
        with urllib.request.urlopen(req) as resp:
            data = resp.read()

    header = data[8 : 8 + header_len]
    meta = json.loads(header.decode("utf-8"))
    meta.pop("__metadata__", None)
    return meta


def build_random_checkpoint(output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    config = fetch_config()
    with open(output_dir / "config.json", "w") as f:
        json.dump(config, f, indent=2)

    meta = fetch_safetensors_index()
    tensors = {}
    for name, info in meta.items():
        shape = info["shape"]
        tensors[name] = torch.randn(*shape, dtype=torch.bfloat16)

    safetensors_path = output_dir / "model.safetensors"
    save_file(tensors, safetensors_path)

    # Write HF-style index with weight_map
    weight_map = {name: "model.safetensors" for name in meta.keys()}
    index = {
        "metadata": {"total_size": os.path.getsize(safetensors_path)},
        "weight_map": weight_map,
    }
    with open(output_dir / "model.safetensors.index.json", "w") as f:
        json.dump(index, f, indent=2)

    # Save a raw tensor list for debugging/repro
    with open(output_dir / "model.safetensors.tensor_shapes.json", "w") as f:
        json.dump(meta, f, indent=2)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output",
        default="/scratchspace/dflash_random_b16/exported-checkpoint-0",
        help="Output directory for the random DFlash checkpoint",
    )
    args = parser.parse_args()

    output_dir = Path(args.output)
    build_random_checkpoint(output_dir)
    print(f"Wrote random DFlash checkpoint to {output_dir}")


if __name__ == "__main__":
    main()
