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

import os
from functools import partial

import pytest
import torch
from _test_utils.torch.distributed.utils import spawn_multiprocess_job

import modelopt.torch.quantization as mtq

pytest.importorskip("transformers")
from _test_utils.torch.transformers_models import create_tiny_llama_dir
from transformers import AutoModelForCausalLM


def _test_transformers_tp(model_path, rank, size):
    torch.manual_seed(0)
    os.environ["LOCAL_RANK"] = str(rank)
    torch.set_default_device(f"cuda:{rank}")
    model_tp = AutoModelForCausalLM.from_pretrained(model_path, tp_plan="auto")
    input_ids = torch.randint(0, model_tp.config.vocab_size, (10, 512))
    mtq.quantize(model_tp, mtq.NVFP4_AWQ_LITE_CFG, lambda model: model(input_ids))
    outputs_ref = model_tp(input_ids)  # Test that the model forward pass works

    mtq.fold_weight(model_tp)
    outputs_test = model_tp(input_ids)  # Test that the model forward pass works
    assert torch.allclose(outputs_ref.logits, outputs_test.logits, atol=1e-4)


def test_transformers_tp(need_2_gpus, tmp_path):
    model_path = create_tiny_llama_dir(tmp_path)
    spawn_multiprocess_job(size=2, job=partial(_test_transformers_tp, model_path), backend="nccl")
