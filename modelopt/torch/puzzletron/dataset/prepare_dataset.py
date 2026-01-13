# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

import os

import datasets
import fire
import numpy as np

from modelopt.torch.puzzletron.tools.logger import mprint


def process_and_save_dataset(
    dataset_name: str,
    output_dir: str,
    split: tuple = ("code", "math", "stem", "chat"),
    overwrite: bool = False,
):
    # Check if output_dir contains an existing dataset
    dataset_dict_path = os.path.join(output_dir, "dataset_dict.json")
    if os.path.exists(output_dir) and os.path.exists(dataset_dict_path):
        if not overwrite:
            mprint(
                f"Output directory '{output_dir}' already contains a dataset. "
                "Use '--overwrite True' to overwrite existing data."
            )
            return

    ds = datasets.load_dataset(dataset_name, split=split)
    ds = datasets.concatenate_datasets(ds)
    # Filter out samples with reasoning = on
    ds = ds.filter(lambda x: x["reasoning"] == "off")
    # Hardcoded for dynamically create a deterministic train-val split
    seed = 408
    generator = np.random.RandomState(seed=seed)
    ds_split = ds.train_test_split(test_size=0.05, shuffle=True, generator=generator)
    # Rename dataset names to follow previous conventions
    ds_dict = datasets.DatasetDict(
        {
            "train": ds_split["train"],
            "valid": ds_split["test"],
        }
    )
    # Save locally
    os.makedirs(output_dir, exist_ok=True)
    ds_dict.save_to_disk(output_dir)

    mprint(f"Dataset splits:\n{ds_dict}")
    mprint(f"Saved processed datasets to {output_dir}")


if __name__ == "__main__":
    fire.Fire(process_and_save_dataset)
