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
"""Helper (run as a subprocess) that loads an exported HuggingFace checkpoint with vLLM and runs a
short generation. A non-zero exit fails the calling test.

vLLM is intentionally run in its own process: its (large) footprint would otherwise stack on top
of the test process's torch / Megatron / transformers imports and OOM-kill memory-constrained CI
runners. As a subprocess, its memory is also fully reclaimed on exit.

Usage: python _vllm_generate.py <hf_checkpoint_dir>
"""

import sys

from vllm import LLM, SamplingParams


def main(model_dir: str) -> None:
    # Keep the footprint minimal: no CPU swap space, a single sequence slot, eager execution
    # (no CUDA-graph capture), and a small KV-cache reservation.
    llm = LLM(
        model=model_dir,
        tensor_parallel_size=1,
        enforce_eager=True,
        gpu_memory_utilization=0.3,
        max_model_len=128,
        max_num_seqs=1,
        swap_space=0,
        dtype="bfloat16",
    )
    outputs = llm.generate(["Hello!"], SamplingParams(max_tokens=4))
    assert outputs and outputs[0].outputs[0].text is not None, "vLLM produced no output"
    print("VLLM_GENERATE_OK")


if __name__ == "__main__":
    main(sys.argv[1])
