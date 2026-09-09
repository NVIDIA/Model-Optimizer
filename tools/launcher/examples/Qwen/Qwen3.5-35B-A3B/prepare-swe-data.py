#!/usr/bin/env python3
"""Convert Prime-RL train traces into exact pretokenized DSpark conversations.

Usage:
    python prepare-swe-data.py <rollout-root> <output-dir>
"""

from __future__ import annotations

import argparse
import json
import os
from collections import Counter
from pathlib import Path


def branch_nodes(nodes: list[dict]) -> list[dict]:
    """Return the single root-to-leaf branch stored by these SWE rollouts."""
    parents = {node["parent"] for node in nodes if node["parent"] is not None}
    leaves = [index for index in range(len(nodes)) if index not in parents]
    if len(leaves) != 1:
        raise ValueError(f"expected one branch, found {len(leaves)}")
    path = []
    node_id = leaves[0]
    while node_id is not None:
        path.append(nodes[node_id])
        node_id = nodes[node_id]["parent"]
    return path[::-1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("rollout_root", type=Path)
    parser.add_argument("output_dir", type=Path)
    parser.add_argument("--first-step", type=int, default=1)
    parser.add_argument("--last-step", type=int, default=300)
    parser.add_argument("--records-per-shard", type=int, default=1024)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.output_dir.exists():
        raise FileExistsError(args.output_dir)
    staging = args.output_dir.with_name(f".{args.output_dir.name}.tmp-{os.getpid()}")
    staging.mkdir(parents=True)

    counts = Counter()
    policy_versions = Counter()
    tasks = Counter()
    trace_ids: set[str] = set()
    shard = None
    shard_index = -1

    try:
        for step in range(args.first_step, args.last_step + 1):
            path = args.rollout_root / f"step_{step}" / "train" / "all" / "traces.jsonl"
            if not path.is_file():
                raise FileNotFoundError(path)
            counts["input_bytes"] += path.stat().st_size
            with path.open() as source:
                for line in source:
                    counts["train_traces"] += 1
                    trace = json.loads(line)
                    if not trace.get("ok", False):
                        counts["excluded"] += 1
                        continue
                    trace_id = trace["id"]
                    if trace_id in trace_ids:
                        raise ValueError(f"duplicate trace id: {trace_id}")
                    trace_ids.add(trace_id)

                    nodes = branch_nodes(trace["nodes"])
                    token_ids = [token for node in nodes for token in node["token_ids"]]
                    loss_mask = [int(mask) for node in nodes for mask in node["mask"]]
                    if len(token_ids) != len(loss_mask) or not any(loss_mask):
                        raise ValueError(f"invalid token/mask data: {trace_id}")

                    if counts["converted"] % args.records_per_shard == 0:
                        if shard is not None:
                            shard.close()
                        shard_index += 1
                        shard = (staging / f"shard_{shard_index:05d}.jsonl").open("w")
                    record = {
                        "conversation_id": trace_id,
                        "token_ids": token_ids,
                        "loss_mask": loss_mask,
                        "step": step,
                        "task_id": trace["task"]["data"]["name"],
                        "policy_version": trace["info"]["policy_version"],
                    }
                    shard.write(json.dumps(record, separators=(",", ":")) + "\n")
                    counts["converted"] += 1
                    counts["tokens"] += len(token_ids)
                    counts["supervised_tokens"] += sum(loss_mask)
                    policy_versions[str(record["policy_version"])] += 1
                    tasks[record["task_id"]] += 1
            if step % 25 == 0 or step == args.last_step:
                print(f"step {step}: {counts['converted']} converted", flush=True)
        if shard is not None:
            shard.close()
            shard = None

        manifest = {
            "steps": [args.first_step, args.last_step],
            "counts": dict(counts),
            "num_shards": shard_index + 1,
            "num_tasks": len(tasks),
            "policy_versions": dict(sorted(policy_versions.items(), key=lambda item: int(item[0]))),
            "task_trace_counts": dict(sorted(tasks.items())),
        }
        (staging / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
        staging.rename(args.output_dir)
        print(json.dumps(manifest["counts"], indent=2))
        print(f"output: {args.output_dir}")
    except BaseException:
        if shard is not None:
            shard.close()
        raise


if __name__ == "__main__":
    main()
