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

import importlib.util
import sys
from pathlib import Path
from types import SimpleNamespace

import torch
from transformers import TrainerControl

_EXAMPLE = Path(__file__).parents[3] / "examples" / "speculative_decoding"
sys.path.insert(0, str(_EXAMPLE))
_SPEC = importlib.util.spec_from_file_location("sonar_eagle_utils", _EXAMPLE / "eagle_utils.py")
assert _SPEC is not None and _SPEC.loader is not None
eagle_utils = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(eagle_utils)


def test_callbacks_preserve_control_and_conditional_side_effects(tmp_path):
    control = TrainerControl()
    args = SimpleNamespace(output_dir=str(tmp_path))
    state = SimpleNamespace(global_step=1)
    model = torch.nn.Linear(2, 2)
    model._lora_cotraining_active = False
    warmup = eagle_utils.LoRAWarmupCallback(warmup_steps=2)
    assert warmup.on_step_begin(args, state, control, model=model) is control
    assert not model._lora_cotraining_active
    state.global_step = 2
    assert warmup.on_step_begin(args, state, control, model=model) is control
    assert model._lora_cotraining_active
    assert warmup.on_step_begin(args, state, control, model=model) is control
    export = eagle_utils.DFlashFSDP2ShardedSDExportCallback()
    assert export.on_save(args, state, control, model=model) is control
    plot = eagle_utils.EagleTrainingPlot(estimate_ar=False, ar_validate_steps=0)
    assert plot.on_log(args, state, control) is control
    state.training_accs = [[[0.5, 0.25]]]
    logs = {"loss": 1.0}
    assert plot.on_log(args, state, control, logs=logs) is control
    assert logs["train_acc/parallel_0_step_1"] == 0.25
    assert state.training_accs == []
    assert plot.on_step_end(args, state, control) is control
