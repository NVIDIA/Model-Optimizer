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

import traceback
from concurrent.futures import ThreadPoolExecutor, as_completed

import pytest
from _test_utils.torch.opt.utils import apply_mode_with_sampling
from _test_utils.torch.transformers_models import (
    create_tiny_llama_dir,
    tf_modelopt_state_and_output_tester,
)
from transformers import AutoConfig, AutoModelForCausalLM, LlamaForCausalLM


@pytest.mark.parametrize("model_cls", [LlamaForCausalLM, AutoModelForCausalLM])
def test_causal_lm_save_restore(tmp_path, model_cls):
    tiny_llama_dir = create_tiny_llama_dir(tmp_path, hidden_size=128)
    model_ref = model_cls.from_pretrained(tiny_llama_dir)
    # TODO: Add calibrate, compress mode to the test
    model_ref = apply_mode_with_sampling(
        model_ref, ["sparse_magnitude", "export_sparse", "quantize"]
    )
    model_ref.save_pretrained(tiny_llama_dir / "modelopt_model")

    model_test = model_cls.from_pretrained(tiny_llama_dir / "modelopt_model")
    tf_modelopt_state_and_output_tester(model_ref, model_test)


def test_causal_lm_from_config(tmp_path):
    """Test loading a model using from_config after applying optimizations"""
    tiny_llama_dir = create_tiny_llama_dir(tmp_path, hidden_size=128)

    model_ref = AutoModelForCausalLM.from_pretrained(tiny_llama_dir)
    model_ref = apply_mode_with_sampling(
        model_ref, ["sparse_magnitude", "export_sparse", "quantize"]
    )
    model_ref.save_pretrained(tiny_llama_dir / "modelopt_model")

    config = AutoConfig.from_pretrained(tiny_llama_dir / "modelopt_model")

    model_test = AutoModelForCausalLM.from_config(config)

    # from_config doesn't load weights, need to load state_dict separately
    state_dict = model_ref.state_dict()
    model_test.load_state_dict(state_dict)

    tf_modelopt_state_and_output_tester(model_ref, model_test)


# This test is flaky and causes other tests to fail; This seems to run fine in isolation
@pytest.mark.manual(
    reason="Flaky test causing other tests to fail, run this test manually with --run-manual"
)
def test_transformers_load_with_multi_thread(tmp_path):
    """Multi-threaded test for save/restore functionality"""
    tiny_llama_dir = create_tiny_llama_dir(tmp_path)
    workers = 2
    exceptions = []

    def worker_func(worker_id):
        try:
            _ = AutoModelForCausalLM.from_pretrained(tiny_llama_dir)
        except Exception as e:
            traceback.print_exc()
            return e

    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = [executor.submit(worker_func, i) for i in range(workers)]

        for future in as_completed(futures):
            result = future.result()
            if isinstance(result, Exception):
                exceptions.append(result)

    assert len(exceptions) == 0, "Parallel model loading tests failed, check error log"
