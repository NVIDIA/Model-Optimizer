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
This script was used to truncate the tokenizer.json file from Llama 3.1 8B model
to keep only the top 100 most common tokens.
"""

import json

# Path to your original and new tokenizer.json
in_path = "./tokenizer.json"
out_path = "./tokenizer_truncated.json"

# How many top tokens to keep
NUM_TO_KEEP = 100

with open(in_path, encoding="utf-8") as f:
    tokenizer_data = json.load(f)

# Get and sort the original vocab by index (frequency proxy)
orig_vocab = tokenizer_data["model"]["vocab"]

# Sort tokens by their original index (lowest index = assumed most common/important)
sorted_tokens = sorted(orig_vocab.items(), key=lambda item: item[1])

# Keep the top N tokens
tokens_to_keep = [tok for tok, idx in sorted_tokens[:NUM_TO_KEEP]]

# Re-index the selected tokens: 0..N-1
small_vocab = {tok: i for i, tok in enumerate(tokens_to_keep)}
tokenizer_data["model"]["vocab"] = small_vocab

# Update vocab size
if "vocab_size" in tokenizer_data["model"]:
    tokenizer_data["model"]["vocab_size"] = len(small_vocab)

# Optionally remove merges if present and unneeded (mostly for BPE/WordPiece)
if "merges" in tokenizer_data["model"]:
    tokenizer_data["model"]["merges"] = []

# Remove added_tokens if not needed
if "added_tokens" in tokenizer_data:
    tokenizer_data["added_tokens"] = []

# Write out the truncated tokenizer.json
with open(out_path, "w", encoding="utf-8") as f:
    json.dump(tokenizer_data, f, indent=2, ensure_ascii=False)

print(f"Truncated tokenizer saved to: {out_path}")
