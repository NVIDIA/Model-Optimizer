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

"""Every ``MODELOPT_*`` environment variable ModelOpt reads, each through one accessor.

An accessor reads ``os.environ`` when it is called, not at import: a launcher sets these before it starts
the workers, and a test may set one after importing ModelOpt, so the live value is what counts. Call one
as ``from modelopt import env`` then ``env.ep_degree()``.

An accessor is named after its variable, lowercased and without ``MODELOPT_``. A blank value reads as
unset, and a value the accessor cannot parse raises a ``ValueError`` naming the variable. A rule relating
a variable to another one or to the run is checked where the value is used.

The first read of a variable also raises if a ``MODELOPT_*`` variable ModelOpt does not read is set under
a name closest to it, such as ``MODELOPT_EP_DEGRE``: most likely it is that variable misspelled, which
would otherwise leave it at its default. Other names ModelOpt does not read are left alone, since
launchers and tools set ``MODELOPT_*`` variables of their own.

This module needs only the standard library, so code that must not import torch reads its variables here
too. A new variable gets its accessor here: the ``check-env-knobs`` pre-commit hook refuses a
``MODELOPT_*`` read anywhere else under ``modelopt/``.
"""

import difflib
import os

__all__ = [
    "ep_degree",
    "fsdp2_export",
    "fsdp_size",
    "nvfp4_triton_sweep",
    "skip_weight_quant_check",
    "tegrouped_compile_weight_loop",
]

_PREFIX = "MODELOPT_"
# Each accessor is named after its variable, so the accessors name every variable read here.
_KNOBS = frozenset(_PREFIX + name.upper() for name in __all__)
_YES = ("1", "true", "yes", "on")
_NO = ("0", "false", "no", "off")
# Variables whose first read found no misspelled twin set; the check scans the whole environment.
_CHECKED: set[str] = set()


def _text(name: str) -> str:
    """``name``'s value without surrounding whitespace; ``""`` when it is unset or blank."""
    if name not in _CHECKED:
        _refuse_misspelling_of(name)
        _CHECKED.add(name)
    return (os.environ.get(name) or "").strip()


def _refuse_misspelling_of(name: str) -> None:
    """Raise if a variable ModelOpt does not read is set under a name closest to ``name``."""
    suffixes = [knob.removeprefix(_PREFIX) for knob in _KNOBS]
    for key in os.environ:
        if key.startswith(_PREFIX) and key not in _KNOBS:
            # Compared without the shared prefix, at a cutoff no variable set by a launcher or a tool reaches.
            closest = difflib.get_close_matches(
                key.removeprefix(_PREFIX), suffixes, n=1, cutoff=0.85
            )
            if closest == [name.removeprefix(_PREFIX)]:
                raise ValueError(
                    f"{key} is set, but ModelOpt reads no such variable: did you mean {name}?"
                )


def _flag(name: str, default: bool) -> bool:
    """``name`` as a yes/no switch in any case; blank or unset is ``default``."""
    value = _text(name).lower()
    if not value:
        return default
    if value not in _YES + _NO:
        raise ValueError(
            f"{name}={os.environ[name]!r}: expected 1 or 0 (or true/false, yes/no, on/off)."
        )
    return value in _YES


def _choice(name: str, *choices: str) -> str:
    """``name`` as one of ``choices`` in any case; blank or unset is the first."""
    value = _text(name).lower() or choices[0]
    if value not in choices:
        raise ValueError(
            f"{name}={os.environ[name]!r}: expected {' or '.join(choices)}, {choices[0]} being the "
            "default."
        )
    return value


# Distributed loading (see ``resolve_distribution_plan``)


def ep_degree() -> str:
    """``MODELOPT_EP_DEGREE``: the expert-parallel degree, a number or ``auto``; unset is ``auto``."""
    return _text("MODELOPT_EP_DEGREE") or "auto"


def fsdp_size() -> str:
    """``MODELOPT_FSDP_SIZE``: the layout of the dense part; unset is ``auto``.

    ``1`` replicates it; ``world``, or the world size as a number, FSDP2-shards it over every rank.
    """
    return _text("MODELOPT_FSDP_SIZE") or "auto"


# Export


def fsdp2_export() -> str:
    """``MODELOPT_FSDP2_EXPORT``: the writer of an FSDP2-sharded export; ``streaming`` by default.

    ``streaming`` all-gathers each FSDP2 unit on every rank; ``nogather`` moves each tensor once, to the
    one rank that writes it.
    """
    return _choice("MODELOPT_FSDP2_EXPORT", "streaming", "nogather")


# Quantization


def nvfp4_triton_sweep() -> bool:
    """``MODELOPT_NVFP4_TRITON_SWEEP``: use the fused Triton NVFP4 scale sweep; on by default.

    ``0`` makes NVFP4 MSE calibration sweep the scales in PyTorch instead.
    """
    return _flag("MODELOPT_NVFP4_TRITON_SWEEP", True)


def skip_weight_quant_check() -> bool:
    """``MODELOPT_SKIP_WEIGHT_QUANT_CHECK``: skip a check of ``mtq.quantize``; off by default.

    The check raises when a config asks for weight quantization but enables no weight quantizer.
    """
    return _flag("MODELOPT_SKIP_WEIGHT_QUANT_CHECK", False)


def tegrouped_compile_weight_loop() -> bool:
    """``MODELOPT_TEGROUPED_COMPILE_WEIGHT_LOOP``: compile a weight-quantizer loop; off by default.

    ``1`` runs the per-expert weight quantizers of a Transformer Engine grouped linear under
    ``torch.compile``.
    """
    return _flag("MODELOPT_TEGROUPED_COMPILE_WEIGHT_LOOP", False)
