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

# Adapted from https://github.com/EleutherAI/lm-evaluation-harness/tree/aa457edc3d64d81530159cd3a182932320c78f8c

# MIT License
#
# Copyright (c) 2020 EleutherAI
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.


"""Run lm-eval directly on AnyModel (Puzzletron) checkpoints without a deployment server.

Patches lm-eval's HFLM to wrap model loading with deci_x_patcher so AnyModel
Puzzletron checkpoints load correctly. Model descriptor is auto-detected from the
checkpoint's config.json model_type.
"""

from lm_eval.__main__ import cli_evaluate
from lm_eval.api.model import T
from lm_eval.models.huggingface import HFLM
from transformers import AutoConfig

# Trigger factory registration for all model descriptors
import modelopt.torch.puzzletron.anymodel.models  # noqa: F401
from modelopt.torch.puzzletron.anymodel.model_descriptor import ModelDescriptorFactory
from modelopt.torch.puzzletron.anymodel.puzzformer import deci_x_patcher

# Map from HuggingFace config.model_type (in checkpoint config.json) to ModelDescriptorFactory name.
# Local to this script; add entries when supporting new model types for auto-detection.
_MODEL_TYPE_TO_DESCRIPTOR = {
    "llama": "llama",
    "mistral": "mistral_small",
    "qwen2": "qwen2",
    "qwen3": "qwen3",
    "nemotron_h": "nemotron_h",
    "nemotron_h_v2": "nemotron_h_v2",
    "gpt_oss_20b": "gpt_oss_20b",
}


def _resolve_descriptor_from_pretrained(pretrained: str | None):
    """Resolve the model descriptor by loading the checkpoint config and mapping model_type."""
    if not pretrained:
        raise ValueError(
            "pretrained must be set in --model_args "
            "(e.g. --model_args pretrained=/path/to/checkpoint,dtype=bfloat16)."
        )

    config = AutoConfig.from_pretrained(pretrained, trust_remote_code=True)
    model_type = getattr(config, "model_type", None)

    if model_type and model_type in _MODEL_TYPE_TO_DESCRIPTOR:
        detected = _MODEL_TYPE_TO_DESCRIPTOR[model_type]
        print(
            f"[lm_eval_anymodel] Auto-detected model_type='{model_type}' → descriptor='{detected}'"
        )
        return ModelDescriptorFactory.get(detected)

    known = sorted(_MODEL_TYPE_TO_DESCRIPTOR.keys())
    raise ValueError(
        f"Cannot auto-detect descriptor for model_type='{model_type}'. "
        f"Known model types: {known}. Add this model_type to _MODEL_TYPE_TO_DESCRIPTOR if supported."
    )


def create_from_arg_obj(cls: type[T], arg_dict: dict, additional_config: dict | None = None) -> T:
    """Override HFLM.create_from_arg_obj to wrap model loading with deci_x_patcher."""
    pretrained = arg_dict.get("pretrained")
    descriptor = _resolve_descriptor_from_pretrained(pretrained)

    additional_config = {} if additional_config is None else additional_config
    additional_config = {k: v for k, v in additional_config.items() if v is not None}

    # The patcher must be active during HFLM.__init__ because that's where
    # AutoModelForCausalLM.from_pretrained() is called internally.
    with deci_x_patcher(model_descriptor=descriptor):
        model_obj = cls(**arg_dict, **additional_config)

    return model_obj


# Monkey-patch HFLM so lm-eval uses our patched model loading
HFLM.create_from_arg_obj = classmethod(create_from_arg_obj)


if __name__ == "__main__":
    cli_evaluate()
