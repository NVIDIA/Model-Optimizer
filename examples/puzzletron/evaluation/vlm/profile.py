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

"""Pinned dataset contracts for the Qwen 3.5 VLM benchmark profile."""

from dataclasses import dataclass

__all__ = [
    "VLM_BENCHMARK_DATASETS",
    "VLM_BENCHMARK_TASKS",
    "VLM_BENCHMARK_VIDEO_DATASETS",
    "BenchmarkDataset",
]


@dataclass(frozen=True)
class BenchmarkDataset:
    """One pinned VLM benchmark task and its local media contract."""

    repository: str
    revision: str
    task_config: str
    max_new_tokens: int
    media_dir: str | None = None
    preparation_dir: str | None = None


VLM_BENCHMARK_DATASETS = {
    "realworldqa": BenchmarkDataset(
        repository="lmms-lab/RealWorldQA",
        revision="907c4e5228fd1703c710ed937601cb5f89ab8d5c",
        task_config="tasks/realworldqa/realworldqa.yaml",
        max_new_tokens=16,
    ),
    "mmmu_val": BenchmarkDataset(
        repository="lmms-lab/MMMU",
        revision="364f2e2eb107b36e07ff4c5a15f5947a759cef47",
        task_config="tasks/mmmu/mmmu_val.yaml",
        max_new_tokens=128,
    ),
    "video_mmmu": BenchmarkDataset(
        repository="lmms-lab/VideoMMMU",
        revision="d1c35ac933123d79e877b7f1b9506afb0309cf1b",
        task_config="tasks/videommmu/video_mmmu.yaml",
        max_new_tokens=1024,
        media_dir="video_mmmu",
        preparation_dir="video_mmmu",
    ),
    "mvbench": BenchmarkDataset(
        repository="OpenGVLab/MVBench",
        revision="a776e554280b99b70f00cc3eacd69a65e0727efc",
        task_config="tasks/mvbench/mvbench.yaml",
        max_new_tokens=16,
        media_dir="mvbench_video",
        preparation_dir="mvbench_video",
    ),
    "mmvu_val": BenchmarkDataset(
        repository="lmms-lab/MMVU",
        revision="7537bc8a4b6716be5a9995e022c295679f4af616",
        task_config="tasks/mmvu/mmvu_val.yaml",
        max_new_tokens=1024,
        media_dir="mmvu",
        preparation_dir="mmvu",
    ),
    "videomme": BenchmarkDataset(
        repository="lmms-lab/Video-MME",
        revision="ead1408f75b618502df9a1d8e0950166bf0a2a0b",
        task_config="tasks/videomme/videomme.yaml",
        max_new_tokens=16,
        media_dir="videomme/data",
        preparation_dir="videomme",
    ),
    "longvideobench_val_v": BenchmarkDataset(
        repository="longvideobench/LongVideoBench",
        revision="60d1c89c1919a198b73be39c2babb213b29d6a5c",
        task_config="tasks/longvideobench/longvideobench_val_v.yaml",
        max_new_tokens=32,
        media_dir="datasets/longvideobench/videos",
        preparation_dir="datasets/longvideobench",
    ),
    "mlvu_dev": BenchmarkDataset(
        repository="sy1998/MLVU_dev",
        revision="96207eb9aa7101e2a495dd147684a7e618c79e12",
        task_config="tasks/mlvu/mlvu_dev.yaml",
        max_new_tokens=16,
        media_dir="mlvu",
        preparation_dir="mlvu",
    ),
    "perceptiontest_val_mc": BenchmarkDataset(
        repository="lmms-lab/PerceptionTest_Val",
        revision="c5e520d8c4167fb1f135c36e9d6e67312b4f8e6b",
        task_config="tasks/perceptiontest/val/perceptiontest_mc.yaml",
        max_new_tokens=16,
        media_dir="perceptiontest_val/videos",
        preparation_dir="perceptiontest_val",
    ),
}
VLM_BENCHMARK_TASKS = tuple(VLM_BENCHMARK_DATASETS)
VLM_BENCHMARK_VIDEO_DATASETS = {
    task: dataset
    for task, dataset in VLM_BENCHMARK_DATASETS.items()
    if dataset.preparation_dir is not None
}
