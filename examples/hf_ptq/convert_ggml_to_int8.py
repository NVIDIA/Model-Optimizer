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

"""Expand a GGML checkpoint exported by hf_ptq.py into an INT8 checkpoint that vLLM serves.

python hf_ptq.py --pyt_ckpt_path <model> --recipe general/ptq/iq1_s --export_path <ggml_dir>
python convert_ggml_to_int8.py <ggml_dir> <int8_dir>
vllm serve <int8_dir>
"""

import argparse

from modelopt.torch.export import convert_ggml_export_to_int8


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("ggml_dir", help="Checkpoint exported with a GGML recipe (IQ, q8_0).")
    parser.add_argument("output_dir", help="Where to write the INT8 checkpoint; must be empty.")
    parser.add_argument(
        "--max_shard_size_gb", type=float, default=10, help="Size of each output shard in GiB."
    )
    args = parser.parse_args()
    convert_ggml_export_to_int8(
        args.ggml_dir, args.output_dir, max_shard_size=int(args.max_shard_size_gb * 2**30)
    )
    print(f"INT8 checkpoint written to {args.output_dir}")


if __name__ == "__main__":
    main()
