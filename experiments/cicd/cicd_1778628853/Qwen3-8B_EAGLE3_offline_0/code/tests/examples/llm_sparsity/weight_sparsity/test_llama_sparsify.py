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


import pytest
from _test_utils.examples.run_command import extend_cmd_parts, run_example_command
from _test_utils.torch.misc import minimum_gpu


def run_llm_sparsity_command(
    *, model: str, output_dir: str, sparsity_fmt: str = "sparsegpt", **kwargs
):
    kwargs.update(
        {"model_name_or_path": model, "sparsity_fmt": sparsity_fmt, "output_dir": output_dir}
    )
    kwargs.setdefault("calib_size", 16)
    kwargs.setdefault("device", "cuda")
    kwargs.setdefault("dtype", "fp16")
    kwargs.setdefault("model_max_length", 1024)

    cmd_parts = extend_cmd_parts(["python", "hf_pts.py"], **kwargs)
    run_example_command(cmd_parts, "llm_sparsity/weight_sparsity")


def run_llm_sparsity_ft_command(
    *, model: str, restore_path: str, output_dir: str, data_path: str, **kwargs
):
    kwargs.update(
        {
            "model": model,
            "restore_path": restore_path,
            "output_dir": output_dir,
            "data_path": data_path,
        }
    )
    kwargs.setdefault("num_epochs", 0.01)
    kwargs.setdefault("max_length", 128)
    kwargs.setdefault("train_bs", 1)
    kwargs.setdefault("eval_bs", 1)

    cmd_parts = extend_cmd_parts(["bash", "launch_finetune.sh"], **kwargs)
    run_example_command(cmd_parts, "llm_sparsity/weight_sparsity")


@pytest.fixture(scope="session")
def data_path(tmp_path_factory):
    data_path = tmp_path_factory.mktemp("data")
    run_example_command(
        ["python", "data_prep.py", "--save_path", data_path], "llm_sparsity/weight_sparsity"
    )

    # Copy eval data to train path for faster test
    run_example_command(
        ["cp", data_path / "cnn_eval.json", data_path / "cnn_train.json"], "llm_sparsity"
    )

    return data_path


@pytest.mark.parametrize(
    ("sparsity_fmt", "dtype"),
    [
        ("sparse_magnitude", "bf16"),
    ],
)
def test_llama_sparsity(tiny_llama_path, tmp_path, sparsity_fmt, dtype):
    run_llm_sparsity_command(
        model=tiny_llama_path,
        output_dir=tmp_path,
        sparsity_fmt=sparsity_fmt,
        dtype=dtype,
        calib_size=8,
        model_max_length=128,
    )


def _test_llama_sparsity_finetune(tiny_llama_path, tmp_path, data_path, sparsity_fmt, dtype):
    pts_output = tmp_path / "pts_output"
    finetune_output = tmp_path / "finetune_output"

    # First do sparsification
    run_llm_sparsity_command(
        model=tiny_llama_path,
        output_dir=pts_output,
        sparsity_fmt=sparsity_fmt,
        dtype=dtype,
        calib_size=8,
        model_max_length=128,
    )

    # Then do finetuning using the sparsified model
    restore_path = pts_output / "pts_modelopt_state.pth"
    run_llm_sparsity_ft_command(
        model=tiny_llama_path,
        restore_path=restore_path,
        output_dir=finetune_output,
        data_path=data_path,
        num_epochs=0.001,
        max_length=128,
    )


@pytest.mark.parametrize(
    ("sparsity_fmt", "dtype"),
    [
        ("sparsegpt", "bf16"),
    ],
)
def test_llama_sparsity_finetune(tiny_llama_path, tmp_path, data_path, sparsity_fmt, dtype):
    _test_llama_sparsity_finetune(tiny_llama_path, tmp_path, data_path, sparsity_fmt, dtype)


@pytest.mark.parametrize(
    ("sparsity_fmt", "dtype"),
    [
        ("sparsegpt", "fp16"),
    ],
)
@minimum_gpu(2)
def test_llama_sparsity_finetune_multi_gpu(
    tiny_llama_path, tmp_path, data_path, sparsity_fmt, dtype
):
    _test_llama_sparsity_finetune(tiny_llama_path, tmp_path, data_path, sparsity_fmt, dtype)
