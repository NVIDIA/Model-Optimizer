#!/usr/bin/env python3
"""Create a deterministic prefix slice of a Puzzletron packed-token cache."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--num-samples", required=True, type=int)
    parser.add_argument("--seq-length", required=True, type=int)
    args = parser.parse_args()

    source_metadata_path = args.input.with_suffix(args.input.suffix + ".json")
    source_metadata = json.loads(source_metadata_path.read_text())
    source_samples = int(source_metadata["num_samples"])
    source_seq_length = int(source_metadata["seq_length"])
    if args.num_samples <= 0 or args.num_samples > source_samples:
        raise ValueError(
            f"num_samples must be in [1, {source_samples}], got {args.num_samples}"
        )
    if args.seq_length <= 0 or args.seq_length > source_seq_length:
        raise ValueError(
            f"seq_length must be in [1, {source_seq_length}], got {args.seq_length}"
        )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    output_metadata_path = args.output.with_suffix(args.output.suffix + ".json")
    temporary_data = args.output.with_name(f".{args.output.name}.{os.getpid()}.tmp")
    temporary_metadata = output_metadata_path.with_name(
        f".{output_metadata_path.name}.{os.getpid()}.tmp"
    )
    source_row_bytes = (source_seq_length + 1) * 4
    output_row_bytes = (args.seq_length + 1) * 4
    with args.input.open("rb") as source, temporary_data.open("wb") as target:
        for sample_index in range(args.num_samples):
            source.seek(sample_index * source_row_bytes)
            row = source.read(output_row_bytes)
            if len(row) != output_row_bytes:
                raise RuntimeError(
                    f"short read for sample {sample_index}: {len(row)} != {output_row_bytes}"
                )
            target.write(row)

    output_metadata = dict(source_metadata)
    output_metadata.update(
        {
            "status": "complete",
            "num_samples": args.num_samples,
            "seq_length": args.seq_length,
            "source_cache": str(args.input.resolve()),
            "source_sample_indices": list(range(args.num_samples)),
        }
    )
    temporary_metadata.write_text(
        json.dumps(output_metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    os.replace(temporary_data, args.output)
    os.replace(temporary_metadata, output_metadata_path)
    print(
        json.dumps(
            {
                "output": str(args.output),
                "num_samples": args.num_samples,
                "seq_length": args.seq_length,
                "bytes": args.output.stat().st_size,
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
