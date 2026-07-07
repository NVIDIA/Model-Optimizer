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

ENGINE_IMPORTS = {
    "SGLANG": "sglang",
    "TRTLLM": "tensorrt_llm.bindings.executor",
    "VLLM": "vllm",
}


def _create_dummy_speed_bench_qualitative_split(output_dir: Path) -> Path:
    pd = pytest.importorskip("pandas")
    pytest.importorskip("pyarrow")

    prompts = [
        ("coding", ["Write a Python function that returns the square of an integer."]),
        ("math", ["What is 17 plus 25? Explain in one short sentence."]),
        ("reasoning", ["List two reasons caching can improve application latency."]),
        ("writing", ["Rewrite this sentence to be more concise: The system is not unavailable."]),
        ("summarization", ["Summarize in five words: GPUs accelerate matrix multiplication."]),
        ("qa", ["Name one benefit of batching inference requests."]),
        ("coding", ["Write a Python expression that checks whether x is even."]),
        ("math", ["What is 9 multiplied by 8?"]),
        ("reasoning", ["Which is usually faster for lookup, a list scan or a hash map?"]),
        ("writing", ["Convert this to active voice: The benchmark was run by the engineer."]),
    ]
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "test.parquet"
    pd.DataFrame(
        [
            {
                "question_id": question_id,
                "category": category,
                "turns": turns,
                "source": "mock_qualitative",
            }
            for question_id, (category, turns) in enumerate(prompts)
        ]
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
    tiny_llama_path: str,
    tmp_path: Path,
    engine: str,
    dataset_path: Path,
) -> Path:
    save_dir = tmp_path / f"{engine.lower()}_qualitative"

    cmd = [
        "python",
        "run.py",
        "--model_dir",
        tiny_llama_path,
        "--tokenizer",
        tiny_llama_path,
        "--dataset",
        "speed",
        "--dataset_path",
        str(dataset_path),
        "--engine",
        engine,
        "--speculative_algorithm",
        "NONE",
        "--output_length",
        "100",
        "--tp_size",
        "1",
        "--ep_size",
        "1",
        "--concurrency",
        "10",
        "--max_seq_len",
        "8192",
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


@pytest.mark.parametrize("engine", ["TRTLLM"])
def test_speed_bench_qualitative_runs_example_script(
    tiny_llama_path, dummy_speed_bench_qualitative_split, tmp_path, engine
):
    _require_engine(engine)
    save_dir = _run_speed_bench(
        tiny_llama_path=tiny_llama_path,
        tmp_path=tmp_path,
        engine=engine,
        dataset_path=dummy_speed_bench_qualitative_split,
    )

    config = _read_json(save_dir / "configuration.json")
    specbench = _read_json(save_dir / "specbench_results.json")
    timing = _read_json(save_dir / "timing.json")
    responses = _read_jsonl(save_dir / "specbench_responses.jsonl")

    assert config["dataset"] == "speed"
    assert config["dataset_path"] == str(dummy_speed_bench_qualitative_split)
    assert config["engine"] == engine
    assert specbench["Average_AL"] > 0
    assert len(specbench["Request_AL"]) == 10
    assert len(responses) == 10
    assert timing[0]["Number of Output Tokens"]["mean"]
