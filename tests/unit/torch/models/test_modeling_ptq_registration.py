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

"""Model-specific PTQ modeling registers on import, whichever module is imported first.

The HF quantization plugin imports every ``<model_type>/modeling_ptq.py`` from an explicit
list, and those modules import back from the plugin, so registration depends on import
order. Each first import runs in a fresh interpreter: registration happens once per process,
so an in-process test would only ever see the import order of whichever test ran first.
"""

import os
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor

import pytest

pytest.importorskip("transformers")

FIRST_IMPORTS = [
    "modelopt.torch.models.falcon.modeling_ptq",
    "modelopt.torch.models.gpt_oss.modeling_ptq",
    "modelopt.torch.models.llama4.modeling_ptq",
    "modelopt.torch.models.nemotron_h.modeling_ptq",
    "modelopt.torch.models.step3p5.modeling_ptq",
    "modelopt.torch.models.step3p7.modeling_ptq",
    "modelopt.torch.quantization",
    "modelopt.torch.export",
]

# Runs in the subprocess after importing ``sys.argv[1]`` first.
CHECK = """
import importlib
import sys

importlib.import_module(sys.argv[1])

from modelopt.torch.models.falcon.modeling_ptq import register_falcon_linears_on_the_fly
from modelopt.torch.models.nemotron_h.modeling_ptq import is_nemotron_h_model
from modelopt.torch.models.step3p5.modeling_ptq import register_moe_linear_on_the_fly
from modelopt.torch.quantization.nn import QuantModuleRegistry
from modelopt.torch.quantization.plugins.custom import CUSTOM_MODEL_PLUGINS
from modelopt.torch.quantization.plugins.huggingface import is_homogeneous_hf_model
from modelopt.torch.quantization.utils.layerwise_calib import LayerActivationCollector

try:
    from transformers.models.falcon.modeling_falcon import FalconLinear
except ImportError:
    pass
else:
    assert FalconLinear in QuantModuleRegistry, "FalconLinear not registered"
try:
    from transformers.models.llama4.modeling_llama4 import Llama4TextExperts
except ImportError:
    pass
else:
    assert Llama4TextExperts in QuantModuleRegistry, "Llama4TextExperts not registered"

try:
    from transformers.models.gpt_oss.modeling_gpt_oss import GptOssExperts
except ImportError:
    pass
else:
    assert GptOssExperts in QuantModuleRegistry, "GptOssExperts not registered"

assert register_falcon_linears_on_the_fly in CUSTOM_MODEL_PLUGINS, "Falcon callback missing"
assert register_moe_linear_on_the_fly in CUSTOM_MODEL_PLUGINS, "Step callback missing"

# The first matching decoder discoverer wins, so Nemotron-H's must precede the generic one.
predicates = [is_supported for is_supported, _ in LayerActivationCollector._decoder_layer_support]
assert predicates.count(is_nemotron_h_model) == 1, predicates
assert predicates.count(is_homogeneous_hf_model) == 1, predicates
assert predicates.index(is_nemotron_h_model) < predicates.index(is_homogeneous_hf_model)
"""


def _run_check(first_import):
    # The child only asserts; tracing its heavy imports for coverage (inherited through
    # COVERAGE_PROCESS_START) would just slow it down.
    env = {k: v for k, v in os.environ.items() if k != "COVERAGE_PROCESS_START"}
    result = subprocess.run(
        [sys.executable, "-c", CHECK, first_import],
        capture_output=True,
        text=True,
        env=env,
        check=False,
    )
    return first_import, result


# Several fresh interpreters each import torch and transformers, which outlasts the default
# 60 s per-test cap on a small CI runner.
@pytest.mark.timeout(300)
def test_registration_is_independent_of_import_order():
    # Each first import needs its own interpreter; run them in parallel, at most one per CPU,
    # rather than one after another.
    with ThreadPoolExecutor(max_workers=os.cpu_count() or 1) as pool:
        results = list(pool.map(_run_check, FIRST_IMPORTS))
    failures = {name: result.stderr for name, result in results if result.returncode != 0}
    assert not failures, "\n\n".join(
        f"importing {name} first:\n{err}" for name, err in failures.items()
    )
