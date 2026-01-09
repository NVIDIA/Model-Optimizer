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


from _test_utils.examples.run_command import run_example_command


# fmt: off
def _run_hf_ptq(model_path, output_dir, qformat):
    run_example_command(
        [
            "python", "hf_ptq.py",
            "--pyt_ckpt_path", model_path,
            "--batch_size", "1",
            "--calib_size", "64",
            "--export_path", output_dir,
            "--qformat", qformat,
        ],
        "llm_ptq",
    )


def test_llama_medusa_fp8_qat(tiny_llama_path, num_gpus, tiny_daring_anteater_path, tmp_path):
    medusa_path = tmp_path / "medusa-tinyllama"

    # Test Medusa
    run_example_command(
        [
            "./launch_train.sh",
            "--model", tiny_llama_path,
            "--data", tiny_daring_anteater_path,
            "--num_epochs", "1",
            "--lr", "1e-5",
            "--num_gpu", str(num_gpus),
            "--mode", "medusa",
            "--output_dir", medusa_path,
            "--medusa_num_heads", "2",
            "--medusa_num_layers", "1",
        ],
        "speculative_decoding",
    )

    # Test PTQ on Medusa
    _run_hf_ptq(medusa_path, tmp_path / "medusa-tinyllama-hf", "fp8")

    # Test QAT on Medusa
    run_example_command(
        [
            "./launch.sh",
            "--model", medusa_path,
            "--num_epochs", "1",
            "--train_size", "128",
            "--eval_size", "64",
            "--lr", "1e-5",
            "--output_dir", tmp_path / "medusa-tinyllama-qat-finetune",
            "--quant_cfg", "FP8_DEFAULT_CFG",
            "--calib_size", "64",
            "--backend", "fsdp2",
        ],
        "llm_qat",
    )
