# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

import json
import os
import sys

from transformers import AutoTokenizer

_SENSITIVE_SUBSTRINGS = ("token", "key", "secret", "password")


def get_tokenizer(path, trust_remote_code=False):
    return AutoTokenizer.from_pretrained(path, trust_remote_code=trust_remote_code)


def encode_chat(tokenizer, messages, chat_template_args=None, completions=False):
    if chat_template_args is None:
        chat_template_args = {}
    if completions:
        return tokenizer.encode(messages[-1]["content"], add_special_tokens=False)
    return tokenizer.encode(
        tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True, **chat_template_args
        ),
        add_special_tokens=False,
    )


def decode_chat(tokenizer, out_tokens):
    return tokenizer.decode(out_tokens)


def read_json(path):
    if path is not None:
        with open(path) as f:
            data = json.load(f)
        return data
    return {}


def postprocess_base(text):
    return text


def _get_engine_version(engine):
    """Try to import the engine package and return its __version__, or None on failure."""
    try:
        if engine in ("TRTLLM", "AUTO_DEPLOY"):
            import tensorrt_llm

            return tensorrt_llm.__version__
        elif engine == "VLLM":
            import vllm

            return vllm.__version__
        elif engine == "SGLANG":
            import sglang

            return sglang.__version__
        elif engine == "ATOM":
            import atom

            return atom.__version__
    except Exception:
        pass
    return None


def _get_gpu_name():
    """Return the name of GPU 0 if CUDA is available, else None."""
    try:
        import torch

        if torch.cuda.is_available():
            return torch.cuda.get_device_name(0)
    except Exception:
        pass
    return None


def _redact_config(config):
    return {
        key: (
            "***REDACTED***"
            if any(part in key.lower() for part in _SENSITIVE_SUBSTRINGS)
            else value
        )
        for key, value in config.items()
    }


def dump_env(args, save_dir, overrides=None):
    """Write a configuration.json to save_dir capturing the run args and engine version."""
    config = _redact_config(vars(args).copy())
    if overrides:
        config.update(_redact_config(overrides))
    config["engine_version"] = _get_engine_version(config.get("engine"))
    config["gpu"] = _get_gpu_name()
    config["python_version"] = sys.version
    config["argv"] = sys.argv[:]
    os.makedirs(save_dir, exist_ok=True)
    with open(os.path.join(save_dir, "configuration.json"), "w") as f:
        json.dump(config, f, indent=4)


def postprocess_gptoss(text):
    final_message = text.split("<|channel|>final<|message|>")[-1]
    if "<|end|>" in final_message:
        final_message = final_message.split("<|end|>")[0]
    if "<|return|>" in final_message:
        final_message = final_message.split("<|return|>")[0]
    if "<|channel|>" in final_message:
        final_message = final_message.split("<|channel|>")[0]
    return final_message
