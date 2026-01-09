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

"""Export a HF checkpoint (with ModelOpt state) for deployment."""

import argparse

import torch
from transformers import AutoModelForCausalLM

import modelopt.torch.opt as mto
from modelopt.torch.export import export_hf_checkpoint


def parse_args():
    parser = argparse.ArgumentParser(
        description="Export a HF checkpoint (with ModelOpt state) for deployment."
    )
    parser.add_argument("--model_path", type=str, default="Path of the trained checkpoint.")
    parser.add_argument(
        "--export_path", type=str, default="Destination directory for exported files."
    )
    return parser.parse_args()


mto.enable_huggingface_checkpointing()

args = parse_args()
model = AutoModelForCausalLM.from_pretrained(args.model_path, torch_dtype="auto")
model.eval()
with torch.inference_mode():
    export_hf_checkpoint(
        model,  # The quantized model.
        export_dir=args.export_path,  # The directory where the exported files will be stored.
    )
print(f"Exported checkpoint to {args.export_path}")
