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

"""Evaluate MMLU-Pro subjects concurrently across local GPUs."""

import argparse
import json
import multiprocessing
import os
import subprocess  # nosec B404 - commands use fixed argument lists without a shell.
import sys
import tempfile
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
TASKS = [
    "mmlu_pro_math",
    "mmlu_pro_physics",
    "mmlu_pro_chemistry",
    "mmlu_pro_law",
    "mmlu_pro_engineering",
    "mmlu_pro_other",
    "mmlu_pro_economics",
    "mmlu_pro_health",
    "mmlu_pro_psychology",
    "mmlu_pro_business",
    "mmlu_pro_biology",
    "mmlu_pro_philosophy",
    "mmlu_pro_computer_science",
    "mmlu_pro_history",
]


def _result_exists(output_dir: Path, task: str) -> bool:
    return any((output_dir / "subjects" / task).glob("**/results_*.json"))


def _worker(
    gpu: int,
    queue: multiprocessing.Queue,
    model: Path,
    output_dir: Path,
    limit: int,
    batch_size: int,
) -> None:
    scheduler_log = output_dir / f"scheduler_gpu{gpu}.log"
    scheduler_log.write_text("")
    triton_cache = tempfile.TemporaryDirectory(prefix=f"triton_mmlu_pro_gpu{gpu}_")
    env = os.environ.copy()
    for key in ("RANK", "LOCAL_RANK", "WORLD_SIZE", "MASTER_ADDR", "MASTER_PORT"):
        env.pop(key, None)
    env.update(
        {
            "CUDA_VISIBLE_DEVICES": str(gpu),
            "HF_HOME": "/workspace/hf_cache",
            "PYTHONPATH": f"{REPO}:{env.get('PYTHONPATH', '')}",
            "TRITON_CACHE_DIR": triton_cache.name,
        }
    )
    while (task := queue.get()) is not None:
        if _result_exists(output_dir, task):
            with scheduler_log.open("a") as stream:
                stream.write(f"skipping completed {task}\n")
            continue
        task_output = output_dir / "subjects" / task
        task_output.mkdir(parents=True, exist_ok=True)
        command = [
            sys.executable,
            str(REPO / "examples/llm_eval/lm_eval_hf.py"),
            "run",
            "--model",
            "hf",
            "--model_args",
            f"pretrained={model},dtype=bfloat16",
            "--device",
            "cuda:0",
            "--tasks",
            task,
            "--num_fewshot",
            "5",
            "--batch_size",
            str(batch_size),
            "--apply_chat_template",
            "--fewshot_as_multiturn",
            "true",
            "--confirm_run_unsafe_code",
            "--output_path",
            str(task_output),
            "--limit",
            str(limit),
        ]
        with scheduler_log.open("a") as stream:
            stream.write(f"starting {task} on GPU {gpu}\n")
        with (output_dir / "logs" / f"{task}.log").open("w") as stream:
            return_code = subprocess.run(  # nosec B603 - executable and flags are controlled here.
                command, cwd=REPO, env=env, stdout=stream, stderr=subprocess.STDOUT, check=False
            ).returncode
        with scheduler_log.open("a") as stream:
            stream.write(f"finished {task} rc={return_code} on GPU {gpu}\n")


def main() -> None:
    """Run MMLU-Pro subjects on the requested local GPUs."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model", type=Path)
    parser.add_argument("--limit", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--gpus", type=int, nargs="+", default=[0])
    args = parser.parse_args()
    model = args.model.resolve()
    output_dir = model / "eval_results" / f"mmlu_pro_limit_{args.limit}"
    (output_dir / "logs").mkdir(parents=True, exist_ok=True)
    (output_dir / "subjects").mkdir(parents=True, exist_ok=True)
    manifest = {
        "model": str(model),
        "task_group": "mmlu_pro",
        "num_fewshot": 5,
        "batch_size_per_gpu": args.batch_size,
        "apply_chat_template": True,
        "fewshot_as_multiturn": True,
        "limit_per_subject": args.limit,
        "gpus": args.gpus,
        "subjects": TASKS,
    }
    (output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")

    queue = multiprocessing.Queue()
    for task in TASKS:
        queue.put(task)
    for _ in args.gpus:
        queue.put(None)
    workers = [
        multiprocessing.Process(
            target=_worker,
            args=(gpu, queue, model, output_dir, args.limit, args.batch_size),
        )
        for gpu in args.gpus
    ]
    for worker in workers:
        worker.start()
    for worker in workers:
        worker.join()

    failed = [task for task in TASKS if not _result_exists(output_dir, task)]
    if failed:
        raise SystemExit(f"Failed subjects: {', '.join(failed)}")
    print(f"Completed all subjects: {output_dir}")


if __name__ == "__main__":
    main()
