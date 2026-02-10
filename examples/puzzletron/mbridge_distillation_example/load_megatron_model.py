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

"""
Load a Megatron-Bridge checkpoint using load_megatron_model.

IMPORTANT: load_megatron_model() expects the path to point directly to the
iter_XXXXXXX/ directory (the distributed checkpoint directory), not the parent directory.

To create a Megatron-Bridge checkpoint from a HuggingFace model:
  AutoBridge.import_ckpt(
      hf_model_id="meta-llama/Llama-3.2-1B",  # or local HF path
      megatron_path="./megatron_checkpoint",
      dtype="bfloat16"
  )

Then load it by pointing to the iter_XXXXXXX/ directory:
  from megatron.bridge.training.model_load_save import load_megatron_model
  model = load_megatron_model("./megatron_checkpoint/iter_0000000")
"""

from megatron.bridge.training.model_load_save import load_megatron_model

# Load model from Megatron-Bridge checkpoint
# NOTE: Path must point to the iter_XXXXXXX/ directory, not the parent
checkpoint_path = ".../mbridge_models/meta-llama/Llama-3.2-3B-Instruct/iter_0000000"

print(f"Loading Megatron-Bridge checkpoint from: {checkpoint_path}")
model = load_megatron_model(checkpoint_path)
print(f"✓ Successfully loaded model: {type(model)}")
