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

"""Self-implemented final (pre-lm_head) norms for the offline/streaming fake base model.

FakeBaseModel reconstructs base logits from vLLM-captured final hidden states, which are
*un-normed* — so it must re-apply the base model's final norm before lm_head. We reimplement a
small, explicit set of norm variants here (rather than importing each base model's real module)
to keep the fake base lightweight, and map base ``model_type`` → norm type so a norm is applied
only when we know which one the model uses.
"""

import torch
from transformers.models.llama.modeling_llama import LlamaRMSNorm


class _FinalRMSNorm(LlamaRMSNorm):
    """Canonical transformers RMSNorm with an added ``dtype`` to build the weight in that dtype.

    Llama/Qwen/Mistral/Kimi all share this exact module. The forward is inherited unchanged —
    float32 reduction, ``x * rsqrt(mean(x^2) + eps) * weight``. We only add the dtype convenience
    because FakeBaseModel constructs its submodules without a model-wide ``.to(dtype)``; a float32
    weight here would promote the output to float32 and mismatch the bf16 lm_head. ``weight`` is
    loaded from the base checkpoint.
    """

    def __init__(self, hidden_size, eps=1e-6, dtype=torch.bfloat16):
        super().__init__(hidden_size, eps)
        self.to(dtype)


# Registry of self-implemented final-norm variants. We deliberately reimplement these
# (rather than importing the base model's actual module) to keep FakeBaseModel lightweight.
# Only a small, explicit set is supported; add a class here when a new type is needed.
_FINAL_NORM_CLASSES = {
    "rmsnorm": _FinalRMSNorm,
}

# Base ``model_type`` → final-norm type. ONLY listed models get a final norm: we apply it
# solely when we know which norm a model uses, because the vLLM-captured final hidden is
# un-normed and applying the wrong (or an un-loaded) norm would silently corrupt the
# distillation target. Unlisted models get NO ``norm`` module (logits reconstructed without
# it — the pre-final-norm-feature behavior). This mapping is hardcoded, NOT auto-detected:
# add an entry to enable a model, plus a class in ``_FINAL_NORM_CLASSES`` if it needs a new
# norm flavor (e.g. Gemma's ``(1 + weight)`` RMSNorm, GPT-2/OPT-style LayerNorm).
_FINAL_NORM_TYPE_BY_MODEL_TYPE: dict[str, str] = {
    "llama": "rmsnorm",
    "mistral": "rmsnorm",
    "mixtral": "rmsnorm",
    "qwen2": "rmsnorm",
    "qwen3": "rmsnorm",
    "qwen3_moe": "rmsnorm",
    "deepseek_v3": "rmsnorm",
    "kimi_k25": "rmsnorm",  # Kimi-K2.5 / K2.6 / K2.7 all report model_type "kimi_k25"
    "gpt_oss": "rmsnorm",
}


def _select_final_norm_type(model_type: str | None) -> str | None:
    """Return the final-norm type for a base ``model_type``, or ``None`` if unknown.

    ``None`` means we don't know the model's final norm, so FakeBaseModel builds no norm.
    """
    return _FINAL_NORM_TYPE_BY_MODEL_TYPE.get(model_type or "")
