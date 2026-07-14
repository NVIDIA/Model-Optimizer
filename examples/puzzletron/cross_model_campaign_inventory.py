"""Stdlib-only launcher inventory read from the verified preflight artifact."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def inventory_rows(path: str | Path) -> tuple[tuple[str, int, int, int, bool], ...]:
    raw = json.loads(Path(path).read_text())
    rows = []
    for model in raw.get("models", []):
        topology = model["topology"]
        world_size = 1
        for key in ("tp", "cp", "pp", "fsdp", "ep"):
            world_size *= int(topology[key])
        if world_size > 16:
            raise ValueError(
                f"{model['model_id']}: topology needs {world_size} ranks but only 16 GPUs are allowed"
            )
        nodes = 1 if world_size <= 8 else 2
        if world_size % nodes:
            raise ValueError(
                f"{model['model_id']}: world size {world_size} cannot be spread over {nodes} nodes"
            )
        gpus_per_node = world_size // nodes
        rows.append(
            (
                str(model["model_id"]),
                nodes,
                gpus_per_node,
                gpus_per_node,
                gpus_per_node == 8,
            )
        )
    if not rows:
        raise ValueError("preflight contains no campaign models")
    return tuple(rows)


def filter_rows(rows, *, start_model: str | None = None):
    rows = tuple(rows)
    if not start_model:
        return rows
    for index, row in enumerate(rows):
        if row[0] == start_model:
            return rows[index:]
    raise ValueError(f"unknown campaign start model: {start_model}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("preflight", type=Path)
    parser.add_argument("--start-model")
    args = parser.parse_args()
    for row in filter_rows(
        inventory_rows(args.preflight), start_model=args.start_model
    ):
        print(*row[:-1], int(row[-1]), sep="\t")


if __name__ == "__main__":
    main()
