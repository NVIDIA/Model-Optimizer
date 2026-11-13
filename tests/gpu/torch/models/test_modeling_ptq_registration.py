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

"""A model's ``<model_type>/modeling_ptq.py`` is imported just before it is converted, and only then.

The check runs in a fresh interpreter: a module registers once per process, so an in-process
test would see whatever earlier tests imported.
"""

import os
import subprocess
import sys

import pytest

pytest.importorskip("transformers")

CHECK = """
import sys
from types import SimpleNamespace

from transformers import FalconConfig, FalconForCausalLM

import modelopt.torch.export
from modelopt.torch.models import import_model_modules
from modelopt.torch.quantization.conversion import replace_quant_module
from modelopt.torch.quantization.nn import QuantModule
from modelopt.torch.quantization.plugins.huggingface import is_homogeneous_hf_model
from modelopt.torch.quantization.utils.layerwise_calib import LayerActivationCollector


def loaded():
    return {
        name.split(".")[-2]
        for name in sys.modules
        if name.startswith("modelopt.torch.models.") and name.endswith(".modeling_ptq")
    }


assert not loaded(), f"imported before any model was converted: {sorted(loaded())}"

# Converting a model imports its own first, so the conversion uses its wrappers: only Falcon's
# registers FalconLinear.
config = FalconConfig(vocab_size=32, hidden_size=16, num_hidden_layers=1, num_attention_heads=2)
model = FalconForCausalLM(config)
replace_quant_module(model)
dense = model.transformer.h[0].self_attention.dense
assert isinstance(dense, QuantModule), type(dense).__mro__
assert loaded() == {"falcon"}, sorted(loaded())

# Types whose PTQ code is another's, Falcon's original remote-code types (whose packages import
# Falcon's, loaded above), and a VLM's language model, found through its nested config.
for config, expected in [
    (SimpleNamespace(model_type="llama4_text"), {"llama4_text", "llama4"}),
    (SimpleNamespace(model_type="step3p7"), {"step3p7", "step3p5"}),
    (SimpleNamespace(model_type="RefinedWeb"), {"refined_web"}),
    (SimpleNamespace(model_type="RefinedWebModel"), {"refined_web_model"}),
    (
        SimpleNamespace(
            model_type="NemotronH_Nano_VL_V2", llm_config=SimpleNamespace(model_type="nemotron_h")
        ),
        {"nemotron_h"},
    ),
    (SimpleNamespace(model_type="llama"), set()),
]:
    before = loaded()
    import_model_modules(SimpleNamespace(config=config), "modeling_ptq")
    assert loaded() - before == expected, (config, sorted(loaded() - before))

# The first matching decoder discoverer wins, so Nemotron-H's, registered just now, must still
# precede the generic one.
from modelopt.torch.models.nemotron_h.modeling_ptq import is_nemotron_h_model

predicates = [
    is_supported
    for is_supported, _ in (
        *LayerActivationCollector._decoder_layer_support,
        *LayerActivationCollector._decoder_layer_fallbacks,
    )
]
assert predicates.index(is_nemotron_h_model) < predicates.index(is_homogeneous_hf_model)
"""


# A fresh interpreter importing torch and transformers can outlast the default 60 s per-test cap
# on a small CI runner.
@pytest.mark.timeout(300)
def test_modeling_ptq_is_imported_on_conversion():
    # The child only asserts; tracing its heavy imports for coverage (inherited through
    # COVERAGE_PROCESS_START) would just slow it down.
    env = {k: v for k, v in os.environ.items() if k != "COVERAGE_PROCESS_START"}
    result = subprocess.run(
        [sys.executable, "-c", CHECK], capture_output=True, text=True, env=env, check=False
    )
    assert result.returncode == 0, result.stderr
