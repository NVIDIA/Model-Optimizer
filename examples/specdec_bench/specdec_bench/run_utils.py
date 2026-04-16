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

import asyncio
import os
import re
from datetime import datetime
from typing import Any

import yaml
from tqdm.auto import tqdm

SWEEP_ALLOWED_ENTRY_KEYS = {
    "dataset",
    "dataset_path",
    "concurrency",
    "concurrencies",
    "random_isl",
    "num_requests",
    "output_length",
    "temperature",
    "category",
}
SWEEP_ALLOWED_TOP_LEVEL_KEYS = {"runs", "datasets"}


def _normalize_concurrency_values(values, context):
    if isinstance(values, int):
        values = [values]
    if not isinstance(values, list) or not values:
        raise ValueError(f"{context}: concurrency must be an int or non-empty list of ints")
    normalized = []
    for value in values:
        if not isinstance(value, int):
            raise ValueError(f"{context}: concurrency values must be integers")
        if value <= 0:
            raise ValueError(f"{context}: concurrency must be > 0, got {value}")
        normalized.append(value)
    return sorted(set(normalized))


def _normalize_positive_int_or_list(values, n_concurrencies, context, field):
    if isinstance(values, int):
        if values <= 0:
            raise ValueError(f"{context}: {field} must be > 0, got {values}")
        return [values] * n_concurrencies
    if isinstance(values, list):
        if len(values) != n_concurrencies:
            raise ValueError(
                f"{context}: {field} list length ({len(values)}) must match "
                f"number of concurrency values ({n_concurrencies})"
            )
        for v in values:
            if not isinstance(v, int) or v <= 0:
                raise ValueError(f"{context}: {field} values must be positive integers, got {v}")
        return list(values)
    raise ValueError(f"{context}: {field} must be an int or list of ints")


def _normalize_non_negative_float_or_list(values, n_concurrencies, context, field):
    if isinstance(values, (int, float)):
        if values < 0:
            raise ValueError(f"{context}: {field} must be >= 0, got {values}")
        return [float(values)] * n_concurrencies
    if isinstance(values, list):
        if len(values) != n_concurrencies:
            raise ValueError(
                f"{context}: {field} list length ({len(values)}) must match "
                f"number of concurrency values ({n_concurrencies})"
            )
        normalized = []
        for v in values:
            if not isinstance(v, (int, float)) or v < 0:
                raise ValueError(f"{context}: {field} values must be non-negative numbers, got {v}")
            normalized.append(float(v))
        return normalized
    raise ValueError(f"{context}: {field} must be a number or list of numbers")


def _parse_sweep_entry(entry, index, datasets_available):
    if not isinstance(entry, dict):
        raise ValueError(f"sweep entry {index} must be a mapping")
    unknown_keys = set(entry.keys()) - SWEEP_ALLOWED_ENTRY_KEYS
    if unknown_keys:
        raise ValueError(
            f"sweep entry {index} has unsupported keys: {sorted(unknown_keys)}. "
            "Only dataset/dataset_path/random_isl, concurrency, and per-run "
            "num_requests/output_length/temperature/category overrides are supported."
        )
    if "dataset" not in entry:
        raise ValueError(f"sweep entry {index} must include 'dataset'")
    dataset_name = entry["dataset"]
    if dataset_name not in datasets_available:
        raise ValueError(f"sweep entry {index} has unsupported dataset '{dataset_name}'")
    if "concurrency" in entry and "concurrencies" in entry:
        raise ValueError(f"sweep entry {index} cannot set both 'concurrency' and 'concurrencies'")
    concurrency_values = entry.get("concurrency", entry.get("concurrencies"))
    if concurrency_values is None:
        raise ValueError(f"sweep entry {index} must include 'concurrency' or 'concurrencies'")
    normalized_concurrencies = _normalize_concurrency_values(
        concurrency_values, f"sweep entry {index}"
    )
    n = len(normalized_concurrencies)
    ctx = f"sweep entry {index}"
    raw_num_requests = entry.get("num_requests")
    num_requests_per_concurrency = (
        _normalize_positive_int_or_list(raw_num_requests, n, ctx, "num_requests")
        if raw_num_requests is not None
        else None
    )
    raw_output_length = entry.get("output_length")
    output_length_per_concurrency = (
        _normalize_positive_int_or_list(raw_output_length, n, ctx, "output_length")
        if raw_output_length is not None
        else None
    )
    raw_temperature = entry.get("temperature")
    temperature_per_concurrency = (
        _normalize_non_negative_float_or_list(raw_temperature, n, ctx, "temperature")
        if raw_temperature is not None
        else None
    )
    category = entry.get("category")
    if category is not None and not isinstance(category, str):
        raise ValueError(f"{ctx}: category must be a string")
    return {
        "dataset": dataset_name,
        "dataset_path": entry.get("dataset_path"),
        "random_isl": entry.get("random_isl"),
        "category": category,
        "concurrency_values": normalized_concurrencies,
        "num_requests_per_concurrency": num_requests_per_concurrency,
        "output_length_per_concurrency": output_length_per_concurrency,
        "temperature_per_concurrency": temperature_per_concurrency,
    }


def load_sweep_entries(path, datasets_available):
    with open(path) as f:
        raw = yaml.safe_load(f)
    if raw is None:
        raise ValueError("Sweep config is empty")

    if isinstance(raw, list):
        entries = raw
    elif isinstance(raw, dict):
        unknown_top_level_keys = set(raw.keys()) - SWEEP_ALLOWED_TOP_LEVEL_KEYS
        if unknown_top_level_keys:
            raise ValueError(
                f"Sweep config has unsupported top-level keys: {sorted(unknown_top_level_keys)}. "
                "Use either 'runs' or 'datasets'."
            )
        if "runs" in raw and "datasets" in raw:
            raise ValueError("Sweep config can define only one of 'runs' or 'datasets'")
        if "runs" in raw:
            entries = raw["runs"]
            if not isinstance(entries, list):
                raise ValueError("'runs' must be a list")
        elif "datasets" in raw:
            datasets_block = raw["datasets"]
            if isinstance(datasets_block, list):
                entries = datasets_block
            elif isinstance(datasets_block, dict):
                entries = []
                for dataset_name, entry in datasets_block.items():
                    if entry is None:
                        entry = {}
                    if not isinstance(entry, dict):
                        raise ValueError(
                            f"Dataset '{dataset_name}' in sweep config must map to a dict"
                        )
                    entry = dict(entry)
                    entry["dataset"] = dataset_name
                    entries.append(entry)
            else:
                raise ValueError("'datasets' must be either a list or a mapping")
        else:
            raise ValueError("Sweep config must contain either 'runs' or 'datasets'")
    else:
        raise ValueError("Sweep config must be a list or mapping")

    return [_parse_sweep_entry(entry, i, datasets_available) for i, entry in enumerate(entries)]


def build_cli_dataset_defaults(args):
    defaults = {}
    if args.mtbench is not None:
        defaults["mtbench"] = {"dataset_path": args.mtbench}
    if args.specbench is not None:
        defaults["specbench"] = {"dataset_path": args.specbench}
    if args.dataset is not None and args.dataset_path is not None:
        defaults[args.dataset] = {"dataset_path": args.dataset_path}
    if args.random_isl is not None:
        defaults["random"] = {"random_isl": args.random_isl}
    return defaults


def resolve_single_run_spec(args):
    if args.dataset is not None:
        dataset_name = args.dataset
        dataset_path = args.dataset_path
        random_isl = args.random_isl
    elif args.mtbench is not None:
        dataset_name = "mtbench"
        dataset_path = args.mtbench
        random_isl = None
    elif args.random_isl is not None:
        dataset_name = "random"
        dataset_path = None
        random_isl = args.random_isl
    elif args.specbench is not None:
        dataset_name = "specbench"
        dataset_path = args.specbench
        random_isl = None
    else:
        raise ValueError("A dataset must be specified")

    if dataset_name == "random":
        if random_isl is None and dataset_path is not None:
            random_isl = int(dataset_path)
        if random_isl is None:
            raise ValueError("Random dataset requires --random_isl")
        if int(random_isl) <= 0:
            raise ValueError("--random_isl must be > 0")
        dataset_path = None
    elif dataset_path is None:
        raise ValueError(f"Dataset '{dataset_name}' requires a dataset path")

    if args.concurrency <= 0:
        raise ValueError("--concurrency must be > 0")

    return [
        {
            "dataset": dataset_name,
            "dataset_path": dataset_path,
            "random_isl": random_isl,
            "concurrency": args.concurrency,
        }
    ]


def expand_sweep_run_specs(args, datasets_available):
    entries = load_sweep_entries(args.sweep_config, datasets_available)
    defaults = build_cli_dataset_defaults(args)
    run_specs = []
    for entry in entries:
        dataset_name = entry["dataset"]
        dataset_path = entry["dataset_path"]
        random_isl = entry["random_isl"]
        default_entry = defaults.get(dataset_name, {})
        if dataset_path is None:
            dataset_path = default_entry.get("dataset_path")
        if random_isl is None:
            random_isl = default_entry.get("random_isl")

        if dataset_name == "random":
            if random_isl is None and dataset_path is not None:
                random_isl = int(dataset_path)
            if random_isl is None:
                raise ValueError(
                    "Random dataset in sweep config requires 'random_isl' or a fallback --random_isl"
                )
            if int(random_isl) <= 0:
                raise ValueError("random_isl must be > 0")
            dataset_path = None
        elif dataset_path is None:
            raise ValueError(
                f"Sweep dataset '{dataset_name}' is missing dataset_path "
                "(set it in sweep config or pass the corresponding CLI dataset path)"
            )

        num_requests_list = entry["num_requests_per_concurrency"]
        output_length_list = entry["output_length_per_concurrency"]
        temperature_list = entry["temperature_per_concurrency"]
        for i, concurrency in enumerate(entry["concurrency_values"]):
            run_specs.append(
                {
                    "dataset": dataset_name,
                    "dataset_path": dataset_path,
                    "random_isl": random_isl,
                    "category": entry["category"],
                    "concurrency": concurrency,
                    "num_requests": num_requests_list[i] if num_requests_list is not None else None,
                    "output_length": output_length_list[i]
                    if output_length_list is not None
                    else None,
                    "temperature": temperature_list[i] if temperature_list is not None else None,
                }
            )
    if not run_specs:
        raise ValueError("Sweep config produced no runs")
    return run_specs


def _sanitize_for_path(value):
    return re.sub(r"[^A-Za-z0-9._-]+", "_", str(value)).strip("_") or "run"


def resolve_sweep_output_root(args):
    if args.sweep_output_root is not None:
        return args.sweep_output_root
    if args.save_dir is not None:
        return args.save_dir
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return os.path.join("sweep_outputs", timestamp)


def resolve_run_save_dir(args, run_spec, run_index, is_sweep, sweep_output_root):
    if not is_sweep:
        return args.save_dir
    dataset_part = _sanitize_for_path(run_spec["dataset"])
    return os.path.join(
        sweep_output_root, f"{run_index:03d}_{dataset_part}_c{run_spec['concurrency']}"
    )


def build_dataset(run_spec, tokenizer, dataset_kwargs, datasets_available, datasets_module):
    dataset_name = run_spec["dataset"]
    if dataset_name == "random":
        return datasets_module.RandomToken(tokenizer, int(run_spec["random_isl"]), **dataset_kwargs)
    return datasets_available[dataset_name](run_spec["dataset_path"], **dataset_kwargs)


def build_metrics(args, tokenizer, dataset_name, dataset, metrics_module):
    metrics_list = [metrics_module.Timing(args.tp_size)]
    if args.aa_timing:
        metrics_list.append(metrics_module.AATiming(tokenizer))
    if args.stop_think_id is not None:
        metrics_list.append(
            metrics_module.ThinkingAcceptance(thinking_end_token=args.stop_think_id)
        )
    if dataset_name == "mtbench":
        metrics_list.insert(0, metrics_module.MTBench(requests=dataset.data))
    elif dataset_name in {"specbench", "speed"}:
        metrics_list.insert(0, metrics_module.SpecBench(requests=dataset.data))
    else:
        metrics_list.insert(0, metrics_module.AcceptanceRate())
    return metrics_list


async def gather_limited(items, worker, concurrency, show_progress=False, progress_desc=None):
    if concurrency <= 0:
        raise ValueError("concurrency must be > 0")

    results: list[Any] = [None] * len(items)
    queue = asyncio.Queue()
    for index, item in enumerate(items):
        queue.put_nowait((index, item))

    progress = None
    if show_progress:
        progress = tqdm(total=len(items), desc=progress_desc)

    async def worker_loop():
        while True:
            try:
                index, item = queue.get_nowait()
            except asyncio.QueueEmpty:
                return
            try:
                results[index] = await worker(item, index)
            except Exception as exc:
                results[index] = exc
            finally:
                queue.task_done()
                if progress is not None:
                    progress.update(1)

    tasks = [
        asyncio.create_task(worker_loop()) for _ in range(min(concurrency, max(1, len(items))))
    ]
    try:
        await queue.join()
        await asyncio.gather(*tasks)
    finally:
        for task in tasks:
            if not task.done():
                task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)
        if progress is not None:
            progress.close()

    return results
