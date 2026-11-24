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

"""``load_model`` in ``examples/llm_qat/utils.py`` on an MXFP4 checkpoint.

Lives under ``tests/unit`` rather than ``tests/examples/`` so CI runs it on every change and every
transformers it tests: the save it guards depends on transformers' internals.
"""

import importlib.util
from pathlib import Path

import pytest
import torch

transformers = pytest.importorskip("transformers")
pytest.importorskip("peft")

from _test_utils.torch.transformers_models import get_tiny_gpt_oss

import modelopt.torch.quantization as mtq
from modelopt.recipe import load_recipe
from modelopt.torch.export import export_hf_checkpoint

_UTILS = Path(__file__).resolve().parents[3] / "examples" / "llm_qat" / "utils.py"


def test_mxfp4_checkpoint_loads_dequantized_and_saves_its_experts(tmp_path):
    """Saving a model transformers dequantized must keep every layer's experts under their names."""
    spec = importlib.util.spec_from_file_location("llm_qat_utils", _UTILS)
    assert spec is not None and spec.loader is not None
    utils = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(utils)
    # An MXFP4 checkpoint as openai/gpt-oss-* ships, written by the export.
    model = get_tiny_gpt_oss(num_hidden_layers=2, dtype=torch.float32)
    mtq.quantize(model, load_recipe("general/ptq/mxfp4_mlp_weight_only").quantize)
    export_hf_checkpoint(model, export_dir=tmp_path / "mxfp4")

    loaded = utils.load_model(str(tmp_path / "mxfp4"), dtype=torch.float32)
    loaded.save_pretrained(tmp_path / "saved")
    reloaded = transformers.AutoModelForCausalLM.from_pretrained(
        tmp_path / "saved", dtype=torch.float32
    )

    assert type(loaded.model.layers[0].mlp.experts).__name__ == "GptOssExperts"
    for name, tensor in loaded.state_dict().items():
        assert torch.equal(reloaded.state_dict()[name], tensor), name
