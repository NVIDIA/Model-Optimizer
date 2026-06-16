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

"""Factories for tied-weight test scenarios.

These build small synthetic modules whose ``.weight`` :class:`nn.Parameter` is
shared between two sibling modules — mimicking HuggingFace's
``_tied_weights_keys`` machinery — for unit-testing the export-time dedup,
canonical-side naming, and per-side ``input_quantizer.amax`` merge logic in
the HF export path.

Every factory returns CPU-resident, float32-default modules; no GPU required.
Each factory asserts its own post-conditions before returning, so a broken
tie surfaces as a clear factory-side error rather than as a downstream test
failure with an ambiguous cause.
"""

import re

import torch.nn as nn


def make_tied_linear_pair(
    in_features: int = 16,
    out_features: int = 32,
    bias: bool = False,
) -> tuple[nn.Linear, nn.Linear]:
    """Two :class:`nn.Linear` modules whose ``.weight`` Parameter is shared.

    Mimics what HuggingFace's :meth:`PreTrainedModel.tie_weights` does after
    ``__init__``: one extra ``setattr`` so that both modules' ``.weight``
    attributes resolve to the same :class:`nn.Parameter` and therefore the
    same underlying storage. The modules are otherwise independent — separate
    biases (if requested), separate forward/training state, separate
    quantizer slots when ``mtq.quantize`` inserts them later.
    """
    enc = nn.Linear(in_features, out_features, bias=bias)
    dec = nn.Linear(in_features, out_features, bias=bias)
    dec.weight = enc.weight  # mimics HF tie_weights()

    # Post-conditions — fail loudly if the tie was somehow lost.
    assert enc.weight is dec.weight, "Linear weights not tied (object identity)"
    assert enc.weight.data_ptr() == dec.weight.data_ptr(), (
        "Linear weights tied at object level but storage diverged"
    )
    return enc, dec


def tie_fused_experts_3d_params(enc: nn.Module, dec: nn.Module) -> None:
    """Tie ``gate_up_proj`` and ``down_proj`` between two fused-experts modules.

    Mutates ``dec`` in place. After calling, ``dec.gate_up_proj`` IS
    ``enc.gate_up_proj`` (same :class:`nn.Parameter`) and likewise for
    ``down_proj``. Used by MoE-dedup tests together with the
    ``_SyntheticFusedExperts`` fixture defined in
    ``tests/unit/torch/quantization/plugins/test_fused_experts.py``.
    """
    dec.gate_up_proj = enc.gate_up_proj
    dec.down_proj = enc.down_proj

    assert enc.gate_up_proj is dec.gate_up_proj, "gate_up_proj not tied"
    assert enc.down_proj is dec.down_proj, "down_proj not tied"
    assert enc.gate_up_proj.data_ptr() == dec.gate_up_proj.data_ptr()
    assert enc.down_proj.data_ptr() == dec.down_proj.data_ptr()


def wrap_in_parent_with_tied_keys(
    enc: nn.Module,
    dec: nn.Module,
    *,
    decoder_canonical: bool = True,
    weight_attr: str = "weight",
) -> nn.Module:
    """Wrap two tied modules in a parent that declares HF ``_tied_weights_keys``.

    Returns a parent :class:`nn.Module` with:

    - ``parent.encoder = enc`` — registered as a submodule (alias side).
    - ``parent.decoder = dec`` — registered as a submodule (canonical side
      when ``decoder_canonical=True``, the default and DiffusionGemma-like case).
    - ``parent._tied_weights_keys``: dict-style ``{alias_regex: canonical}``
      when ``decoder_canonical=True``, list-style (legacy, no canonical/alias
      distinction) when ``decoder_canonical=False``.

    Used by tests for :func:`_collect_canonical_tied_patterns` and
    :func:`_reorder_canonical_first`. The legacy list-style branch exercises
    the "no patterns extracted" negative case.

    The parent's class name contains ``DiffusionGemma`` so the model_type
    gate inside :func:`_reorder_canonical_first` (mirrors the existing
    whisper / nemotron-vl dispatch in ``unified_export_hf.py``) passes for
    test parents — without this, the function early-returns before
    reaching the patterns step.
    """
    parent_cls = type("DiffusionGemmaTestParent", (nn.Module,), {})
    parent = parent_cls()
    parent.encoder = enc
    parent.decoder = dec

    if decoder_canonical:
        # Dict-style: regex pattern → canonical path. Mimics HF's per-class
        # ``_tied_weights_keys`` declaration for an encoder/decoder model.
        parent._tied_weights_keys = {
            rf"^encoder\.{re.escape(weight_attr)}$": f"decoder.{weight_attr}",
        }
    else:
        # Legacy list-style: just a list of tied paths, no canonical info.
        parent._tied_weights_keys = [f"encoder.{weight_attr}"]

    return parent
