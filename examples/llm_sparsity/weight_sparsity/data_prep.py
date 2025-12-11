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

import argparse
import json
import os

from datasets import load_dataset

dataset_id = "cnn_dailymail"
dataset_config = "3.0.0"
text_column = "article"
summary_column = "highlights"
instruction_template = "Summarize the following news article:"


def preprocess_function(sample):
    # create list of samples
    inputs = [
        {"instruction": instruction_template, "input": text, "output": summary}
        for text, summary in zip(sample[text_column], sample[summary_column])
    ]
    model_inputs = {"text": inputs}
    return model_inputs


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--save_path", type=str, default="data")
    return parser.parse_args()


def main():
    args = parse_args()

    # Load dataset from the hub
    dataset = load_dataset(dataset_id, name=dataset_config)

    # process dataset
    tokenized_dataset = dataset.map(
        preprocess_function, batched=True, remove_columns=list(dataset["train"].features)
    )

    # save dataset to disk
    os.makedirs(args.save_path, exist_ok=True)

    with open(os.path.join(args.save_path, "cnn_train.json"), "w") as write_f:
        json.dump(list(tokenized_dataset["train"]["text"]), write_f, indent=4, ensure_ascii=False)
    with open(os.path.join(args.save_path, "cnn_eval.json"), "w") as write_f:
        json.dump(list(tokenized_dataset["test"]["text"]), write_f, indent=4, ensure_ascii=False)


if __name__ == "__main__":
    main()
