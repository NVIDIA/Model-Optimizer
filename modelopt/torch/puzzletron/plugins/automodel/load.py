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

"""Load a Puzzletron AnyModel teacher checkpoint into a NeMo AutoModel.

Thin convenience over :func:`apply_patch` +
``NeMoAutoModelForCausalLM.from_pretrained`` for activation scoring. Parallelism
(``device_mesh``, ``distributed_setup``) is forwarded via ``**from_pretrained_kwargs``
so the scoring recipe controls FSDP/TP/CP/EP/PP; for a standalone forward / parity
check, call it with no mesh to get a single-device model.
"""

import logging

from .patch import apply_patch

logger = logging.getLogger(__name__)

__all__ = ["load_anymodel_for_scoring", "validate_force_hf_ep"]


def validate_force_hf_ep(force_hf: bool, ep_size: int | None) -> None:
    """Reject ``force_hf=True`` with expert parallel.

    NeMo's MoE expert parallelism requires NeMo-native MoE layers, which are only
    built on the ``force_hf=False`` (custom-model) path. Mirrors the guard in the
    reference ``run.py``.
    """
    if force_hf and (ep_size or 1) > 1:
        raise ValueError(
            f"ep_size={ep_size} is not supported with force_hf=True (AnyModel via HF layers). "
            "NeMo's MoE expert parallelism requires NeMo-native MoE layers, which are not used "
            "when force_hf=True. Set ep_size=1, or use force_hf=False with a NeMo custom model."
        )


def load_anymodel_for_scoring(
    checkpoint_path: str,
    *,
    anymodel_descriptor: str,
    force_hf: bool = True,
    torch_dtype="auto",
    block_configs_path: str | None = None,
    ep_size: int | None = 1,
    **from_pretrained_kwargs,
):
    """Load an AnyModel teacher checkpoint as a NeMo AutoModel for scoring.

    Args:
        checkpoint_path: Path to the converted AnyModel teacher checkpoint
            (``config.json`` plus standard HuggingFace safetensors artifacts).
        anymodel_descriptor: Registered descriptor name (e.g. ``"llama"``, ``"qwen3_5"``).
        force_hf: ``True`` loads the HF model + ``deci_x_patcher`` (all descriptors, no EP);
            ``False`` uses the NeMo custom model (added in a later milestone).
        torch_dtype: Passed to ``from_pretrained`` (``"auto"`` honors the checkpoint).
        block_configs_path: Optional explicit path; otherwise auto-detected.
        ep_size: Expert-parallel degree, used only for the ``force_hf`` guard.
        **from_pretrained_kwargs: Forwarded to ``from_pretrained`` (e.g. ``device_mesh``,
            ``distributed_setup``, ``attn_implementation``).

    Returns:
        The constructed NeMo AutoModel (``PreTrainedModel``).
    """
    validate_force_hf_ep(force_hf, ep_size)
    apply_patch()

    from nemo_automodel import NeMoAutoModelForCausalLM

    logger.info(
        "Loading AnyModel checkpoint %s (descriptor=%s, force_hf=%s)",
        checkpoint_path,
        anymodel_descriptor,
        force_hf,
    )
    return NeMoAutoModelForCausalLM.from_pretrained(
        checkpoint_path,
        anymodel_descriptor=anymodel_descriptor,
        force_hf=force_hf,
        torch_dtype=torch_dtype,
        block_configs_path=block_configs_path,
        **from_pretrained_kwargs,
    )
