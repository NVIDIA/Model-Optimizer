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

"""Parity test for compute_hidden_states_vllm.py vs compute_hidden_states_hf.py.

Runs both example scripts on the shared tiny dataset and asserts the per-conversation
.pt outputs agree. input_ids / loss_mask must match exactly; hidden_states and
aux_hidden_states are compared with cosine similarity because vLLM and HF use
different attention kernels and bf16 rounding will differ.
"""

import pytest
import torch
from _test_utils.examples.run_command import run_example_command

pytest.importorskip("vllm")

COS_THRESHOLD = 0.95


def _cos_sim(a: torch.Tensor, b: torch.Tensor) -> float:
    a = a.flatten().float()
    b = b.flatten().float()
    return ((a @ b) / (a.norm() * b.norm() + 1e-12)).item()


@pytest.fixture(scope="module")
def hidden_states_dirs(tmp_path_factory):
    return {
        "hf": tmp_path_factory.mktemp("hf_hidden_states"),
        "vllm": tmp_path_factory.mktemp("vllm_hidden_states"),
    }


def test_vllm_hf_parity(tiny_llama_path, tiny_conversations_path, hidden_states_dirs):
    common_args = [
        "--model",
        tiny_llama_path,
        "--input-data",
        str(tiny_conversations_path),
        "--max-seq-len",
        "32",
        "--debug-max-num-conversations",
        "2",
    ]

    run_example_command(
        [
            "python",
            "collect_hidden_states/compute_hidden_states_hf.py",
            *common_args,
            "--output-dir",
            str(hidden_states_dirs["hf"]),
        ],
        "speculative_decoding",
    )

    run_example_command(
        [
            "python",
            "collect_hidden_states/compute_hidden_states_vllm.py",
            *common_args,
            "--output-dir",
            str(hidden_states_dirs["vllm"]),
            "--min-seq-len",
            "1",
            "--gpu-memory-util",
            "0.3",
            "--enforce-eager",
        ],
        "speculative_decoding",
    )

    hf_files = sorted(hidden_states_dirs["hf"].glob("*.pt"))
    vllm_files = sorted(hidden_states_dirs["vllm"].glob("*.pt"))
    assert hf_files, "HF stage produced no .pt files"
    assert vllm_files, "vLLM stage produced no .pt files"
    assert {f.name for f in hf_files} == {f.name for f in vllm_files}, (
        "HF and vLLM produced different conversation IDs"
    )

    for f_hf in hf_files:
        f_vl = hidden_states_dirs["vllm"] / f_hf.name
        pt_hf = torch.load(f_hf, map_location="cpu", weights_only=False)
        pt_vl = torch.load(f_vl, map_location="cpu", weights_only=False)

        assert torch.equal(pt_hf["input_ids"], pt_vl["input_ids"]), (
            f"input_ids mismatch in {f_hf.name}"
        )
        assert torch.equal(pt_hf["loss_mask"], pt_vl["loss_mask"]), (
            f"loss_mask mismatch in {f_hf.name}"
        )

        for key in ("hidden_states", "aux_hidden_states"):
            h_hf, h_vl = pt_hf[key], pt_vl[key]
            assert h_hf.shape == h_vl.shape, (
                f"{key} shape mismatch in {f_hf.name}: {tuple(h_hf.shape)} vs {tuple(h_vl.shape)}"
            )
            cs = _cos_sim(h_hf, h_vl)
            assert cs >= COS_THRESHOLD, (
                f"{key} cosine similarity {cs:.4f} below {COS_THRESHOLD} in {f_hf.name}"
            )
