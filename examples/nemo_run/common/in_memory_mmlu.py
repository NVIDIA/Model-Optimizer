# SPDX-FileCopyrightText: Copyright (c) 2023-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

from nemo.collections.llm.modelopt import setup_trainer_and_restore_model_with_modelopt_spec

from modelopt.torch.export.plugins.nemo_run import _get_most_recent_ckpt
from modelopt.torch.utils.plugins.megatron_mmlu import megatron_mmlu


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Run MMLU evaluation with ModelOpt Megatron model. Provide either --nemo_ckpt"
            "or --finetuned_ckpt_dir"
        )
    )
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--nemo_ckpt", type=str, required=False, help="Path to NeMo checkpoint.")
    group.add_argument(
        "--finetuned_ckpt_dir",
        required=False,
        type=str,
        help="Checkpoint directory of 1 or more finetuned models",
    )
    parser.add_argument(
        "--tensor_parallelism", type=int, default=1, help="Tensor parallelism size."
    )
    parser.add_argument(
        "--pipeline_parallelism", type=int, default=1, help="Pipeline parallelism size."
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    ckpt_path = args.nemo_ckpt
    if args.finetuned_ckpt_dir:
        ckpt_path = _get_most_recent_ckpt(args.finetuned_ckpt_dir)
    model, trainer = setup_trainer_and_restore_model_with_modelopt_spec(
        ckpt_path,
        tensor_model_parallel_size=args.tensor_parallelism,
        pipeline_model_parallel_size=args.pipeline_parallelism,
        devices=args.tensor_parallelism * args.pipeline_parallelism,
    )
    tokenizer = model.tokenizer.tokenizer
    megatron_mmlu(model.module, tokenizer)
