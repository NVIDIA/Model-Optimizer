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

"""``training.dataloader_in_order`` reaches the torch loader the speculative-decoding trainer iterates.

It starts DataLoader worker processes, so it lives here rather than in ``tests/unit``.
"""

import importlib.util
import multiprocessing
import sys
from pathlib import Path

import pytest
import torch

_EXAMPLE_DIR = Path(__file__).resolve().parents[3] / "examples" / "speculative_decoding"


@pytest.fixture(scope="module")
def main_mod():
    """Import the entrypoint by path; its sibling modules need the example dir on sys.path."""
    sys.path.insert(0, str(_EXAMPLE_DIR))
    try:
        spec = importlib.util.spec_from_file_location("sd_main", _EXAMPLE_DIR / "main.py")
        assert spec is not None and spec.loader is not None
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
    finally:
        sys.path.pop(0)
    return module


class _FirstIndexWaits(torch.utils.data.Dataset):
    """Index 0 waits on ``release``, so only a loader free to reorder yields the others first."""

    def __init__(self, release, timeout):
        self.release, self.timeout = release, timeout

    def __len__(self):
        return 8

    def __getitem__(self, idx):
        if idx == 0:
            self.release.wait(self.timeout)
        return idx


def _collate_ids(samples):
    return {"idx": torch.tensor(samples)}


@pytest.mark.parametrize("in_order", [False, True])
def test_dataloader_in_order_reaches_the_torch_loader(main_mod, tmp_path, in_order):
    """``Trainer`` forwards the flag only from transformers 5.15; accelerate's wrapper drops writes."""
    release = multiprocessing.Event()
    # In order, nothing can overtake index 0 to release it, so it waits out a short timeout.
    dataset = _FirstIndexWaits(release, timeout=1 if in_order else 10)
    trainer = main_mod.EagleTrainerWithAccLog(
        model=torch.nn.Linear(1, 1),
        args=main_mod.HfTrainingArguments(
            output_dir=str(tmp_path),
            per_device_train_batch_size=1,
            dataloader_num_workers=2,
            dataloader_in_order=in_order,
            train_sampling_strategy="sequential",
            remove_unused_columns=False,
            report_to="none",
            use_cpu=True,
        ),
        train_dataset=dataset,
        data_collator=_collate_ids,
    )
    order = []
    for batch in trainer.get_train_dataloader():
        order += batch["idx"].tolist()
        release.set()
    assert sorted(order) == list(range(8))
    assert (order == sorted(order)) == in_order, order
