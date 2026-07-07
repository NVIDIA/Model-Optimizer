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

import importlib.util
import json
from pathlib import Path

import pytest
from _test_utils.examples.run_command import run_example_command
from datasets import Dataset

ENGINE_IMPORTS = {
    "SGLANG": "sglang",
    "TRTLLM": "tensorrt_llm.bindings.executor",
    "VLLM": "vllm",
}


def _create_dummy_speed_bench_qualitative_split(output_dir: Path) -> Path:
    prompts = [
        ("coding", ["Python code"]),
        ("math", ["17"]),
        ("reasoning", ["strawberries"]),
        ("writing", ["movie"]),
    ]
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "test.parquet"
    Dataset.from_dict(
        {
            "question_id": list(range(len(prompts))),
            "category": [category for category, _ in prompts],
            "turns": [turns for _, turns in prompts],
            "source": ["mock_qualitative"] * len(prompts),
        }
    ).to_parquet(output_path)
    return output_path


def _find_module(name: str):
    try:
        return importlib.util.find_spec(name)
    except ModuleNotFoundError:
        return None


def _require_engine(engine: str) -> None:
    if _find_module(ENGINE_IMPORTS[engine]) is None:
        raise RuntimeError(f"{engine} runtime is not installed")


@pytest.fixture(scope="module")
def dummy_speed_bench_qualitative_split(tmp_path_factory) -> Path:
    split_dir = tmp_path_factory.mktemp("specdec_bench_data") / "speed" / "qualitative"
    parquet_path = _create_dummy_speed_bench_qualitative_split(split_dir)
    assert parquet_path.is_file()
    return split_dir


def _run_speed_bench(
    *,
    tiny_qwen3_path: str,
    tmp_path: Path,
    engine: str,
    dataset_path: Path,
    num_gpus: int,
) -> Path:
    save_dir = tmp_path / f"{engine.lower()}_qualitative"

    cmd = [
        "python3",
        "run.py",
        "--model_dir",
        tiny_qwen3_path,
        "--tokenizer",
        tiny_qwen3_path,
        "--dataset",
        "speed",
        "--dataset_path",
        str(dataset_path),
        "--engine",
        engine,
        "--speculative_algorithm",
        "NONE",
        "--output_length",
        "2",
        "--tp_size",
        str(num_gpus),
        "--ep_size",
        "1",
        "--concurrency",
        "10",
        "--max_seq_len",
        "20",
        "--save_dir",
        str(save_dir),
    ]
    run_example_command(cmd, "specdec_bench")
    return save_dir


def _read_json(path: Path):
    with open(path) as f:
        return json.load(f)


def _read_jsonl(path: Path) -> list[dict]:
    with open(path) as f:
        return [json.loads(line) for line in f]


@pytest.mark.parametrize("engine", ["VLLM"])
def test_speed_bench_qualitative_runs_example_script(
    tiny_qwen3_path, dummy_speed_bench_qualitative_split, tmp_path, engine, num_gpus
):
    _require_engine(engine)
    save_dir = _run_speed_bench(
        tiny_qwen3_path=tiny_qwen3_path,
        tmp_path=tmp_path,
        engine=engine,
        dataset_path=dummy_speed_bench_qualitative_split,
        num_gpus=num_gpus,
    )

    assert (save_dir / "configuration.json").is_file()
    assert (save_dir / "specbench_results.json").is_file()
    assert (save_dir / "timing.json").is_file()
    assert (save_dir / "specbench_responses.jsonl").is_file()

    config = _read_json(save_dir / "configuration.json")
    specbench = _read_json(save_dir / "specbench_results.json")
    timing = _read_json(save_dir / "timing.json")
    responses = _read_jsonl(save_dir / "specbench_responses.jsonl")

    assert config["dataset"] == "speed"
    assert config["dataset_path"] == str(dummy_speed_bench_qualitative_split)
    assert config["engine"] == engine
    assert specbench["Average_AL"] > 0
    assert len(specbench["Request_AL"]) == 4
    assert len(responses) == 4
    assert timing[0]["Number of Output Tokens"]["mean"]
