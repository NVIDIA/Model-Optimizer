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

"""Add the NVFP4 KV-cache scales the recipe implies to an exported HF checkpoint.

The recipe's K/V quantizers use a constant amax (448) and keep no calibrated amax, so the export
writes NVFP4 KV metadata but no ``k_scale`` / ``v_scale`` tensors. The scale that amax implies is
448 / (6 * 448) = 1/6. This writes it for every attention layer into one extra shard and adds the
shard to the index. Re-running is a no-op.

Usage: python add_nvfp4_kv_scales.py <exported_hf_dir>
"""

import json
import os
import re
import sys

import torch
from safetensors.torch import save_file

SHARD = "model-kv-scales.safetensors"
# Backbone attention only: the MTP head stays BF16 and has no quantized KV cache.
K_PROJ = re.compile(r"(.*\.backbone\.layers\.\d+\.mixer)\.k_proj\.weight$")


def main(hf_dir: str):
    with open(os.path.join(hf_dir, "hf_quant_config.json")) as f:
        kv_algo = json.load(f)["quantization"].get("kv_cache_quant_algo")
    if kv_algo != "NVFP4":
        sys.exit(f"expected kv_cache_quant_algo NVFP4, found {kv_algo}")

    index_path = os.path.join(hf_dir, "model.safetensors.index.json")
    with open(index_path) as f:
        index = json.load(f)
    weight_map = index["weight_map"]
    attention = sorted({m.group(1) for k in weight_map if (m := K_PROJ.match(k))})
    scales = {
        f"{prefix}.{proj}.{name}": torch.tensor(1.0 / 6.0, dtype=torch.float32)
        for prefix in attention
        for proj, name in (("k_proj", "k_scale"), ("v_proj", "v_scale"))
    }
    scales = {k: v for k, v in scales.items() if k not in weight_map}
    if scales:
        save_file(scales, os.path.join(hf_dir, SHARD), metadata={"format": "pt"})
        weight_map.update(dict.fromkeys(scales, SHARD))
        index.setdefault("metadata", {})["total_size"] = index.get("metadata", {}).get(
            "total_size", 0
        ) + 4 * len(scales)
        with open(index_path, "w") as f:
            json.dump(index, f, indent=2)
    print(f"{len(attention)} attention layers; added {len(scales)} k/v scales")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        sys.exit(__doc__)
    main(sys.argv[1])
