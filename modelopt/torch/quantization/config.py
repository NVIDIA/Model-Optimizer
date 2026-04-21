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

"""This document lists the quantization formats supported by Model Optimizer and example quantization configs.

.. _quantization-formats:

Quantization Formats
==========================================

The following table lists the quantization formats supported by Model Optimizer and the corresponding quantization
config. See :ref:`Quantization Configs <example-quantization-configs>` for the
specific quantization config definitions.

Please see :doc:`choosing the right quantization formats <../../guides/_choosing_quant_methods>` to
learn more about the formats and their use-cases.

.. note::

    The recommended configs given below are for LLM models. For CNN models, only INT8 quantization
    is supported. Please use quantization config ``INT8_DEFAULT_CFG`` for CNN models.

=================================   =======================================================
Quantization  Format                Model Optimizer config
=================================   =======================================================
INT8                                ``INT8_SMOOTHQUANT_CFG``

FP8                                 ``FP8_DEFAULT_CFG``

INT4 Weights only AWQ (W4A16)       ``INT4_AWQ_CFG``

INT4-FP8 AWQ (W4A8)                 ``W4A8_AWQ_BETA_CFG``

=================================   =======================================================

.. _quantization-configs:

Quantization Configs
================================

Quantization config is a dictionary with two top-level keys:

- ``"quant_cfg"``: an ordered list of :class:`QuantizerCfgEntry` dicts that specify which
  quantizers to configure and how.
- ``"algorithm"``: the calibration algorithm passed to
  :meth:`calibrate <modelopt.torch.quantization.model_calib.calibrate>`.

Please see :class:`QuantizeConfig` for the full config schema.

``quant_cfg`` — Entry Format
-----------------------------

Each entry in the ``quant_cfg`` list is a :class:`QuantizerCfgEntry` with the following fields:

- ``quantizer_name`` *(required)*: a wildcard string matched against quantizer module names.
  Quantizer modules are instances of
  :class:`TensorQuantizer <modelopt.torch.quantization.nn.modules.tensor_quantizer.TensorQuantizer>`
  and have names ending with ``weight_quantizer``, ``input_quantizer``, etc.
- ``parent_class`` *(optional)*: restricts matching to quantizers whose immediate parent module is
  of this PyTorch class (e.g. ``"nn.Linear"``). If omitted, all matching quantizers are targeted
  regardless of their parent class.
- ``cfg`` *(optional)*: a dict of quantizer attributes as defined by
  :class:`QuantizerAttributeConfig`, or a list of such dicts. When a list is given, the matched
  :class:`TensorQuantizer <modelopt.torch.quantization.nn.modules.tensor_quantizer.TensorQuantizer>`
  is replaced with a
  :class:`SequentialQuantizer <modelopt.torch.quantization.nn.modules.tensor_quantizer.SequentialQuantizer>`
  that applies each format in sequence. This is used for example in W4A8 quantization where weights
  are quantized first in INT4 and then in FP8.
- ``enable`` *(optional)*: toggles matched quantizers on (``True``) or off (``False``),
  independently of ``cfg``. When ``cfg`` is present and ``enable`` is absent, the quantizer is
  implicitly enabled. When ``enable`` is the only field (no ``cfg``), it only flips the on/off
  state — all other attributes remain unchanged.

``quant_cfg`` — Ordering and Precedence
-----------------------------------------

Entries are applied **in list order**; later entries override earlier ones for any quantizer they
match. The recommended pattern is:

1. Start with a deny-all entry ``{"quantizer_name": "*", "enable": False}`` (provided as
   :data:`_base_disable_all`) to disable every quantizer by default.
2. Follow with format-specific entries that selectively enable and configure the desired quantizers.
3. Append :data:`_default_disabled_quantizer_cfg` to enforce standard exclusions (e.g. BatchNorm
   layers, LM head, MoE routers).

To get the string representation of a module class for use in ``parent_class``, do:

.. code-block::

    from modelopt.torch.quantization import QuantModuleRegistry

    # Get the class name for nn.Conv2d
    class_name = QuantModuleRegistry.get_key(nn.Conv2d)

Here is an example of a quantization config:

.. code-block::

    MY_QUANT_CFG = {
        "quant_cfg": [
            # Deny all quantizers by default
            {"quantizer_name": "*", "enable": False},

            # Enable and configure weight and input quantizers
            {"quantizer_name": "*weight_quantizer", "cfg": {"num_bits": 8, "axis": 0}},
            {"quantizer_name": "*input_quantizer", "cfg": {"num_bits": 8, "axis": None}},

            # Disable input quantizers specifically for LeakyReLU layers
            {"quantizer_name": "*input_quantizer", "parent_class": "nn.LeakyReLU", "enable": False},
        ]
    }

.. _example-quantization-configs:

Example Quantization Configurations
==========================================

These example configs can be accessed as attributes of ``modelopt.torch.quantization`` and can be given as
input to :meth:`mtq.quantize() <modelopt.torch.quantization.model_quant.quantize>`. For example:

.. code-block::

    import modelopt.torch.quantization as mtq
    model = mtq.quantize(model, mtq.INT8_DEFAULT_CFG, forward_loop)

You can also create your own config by following these examples.
For instance, if you want to quantize a model with int4 AWQ algorithm, but need to skip quantizing
the layer named ``lm_head``,  you can create a custom config and quantize your model as following:

.. code-block::

    # Create custom config
    CUSTOM_INT4_AWQ_CFG = copy.deepcopy(mtq.INT4_AWQ_CFG)
    CUSTOM_INT4_AWQ_CFG["quant_cfg"].append({"quantizer_name": "*lm_head*", "enable": False})

    # quantize model
    model = mtq.quantize(model, CUSTOM_INT4_AWQ_CFG, forward_loop)

"""

import copy
import warnings
from typing import Any, Literal, cast

from pydantic import ValidationInfo, field_validator, model_validator
from typing_extensions import Required, TypedDict

from modelopt.torch.opt.config import ModeloptBaseConfig, ModeloptField
from modelopt.torch.utils.network import ConstructorLike


class QuantizerCfgEntry(TypedDict, total=False):
    """A single entry in a ``quant_cfg`` list."""

    quantizer_name: Required[str]  # matched against quantizer module names
    parent_class: str | None  # optional; filters by pytorch module class name (e.g. "nn.Linear")
    cfg: dict[str, Any] | list[dict[str, Any]] | None  # quantizer attribute config(s)
    enable: bool | None  # toggles matched quantizers on/off; independent of cfg


def find_quant_cfg_entry_by_path(
    quant_cfg_list: list[QuantizerCfgEntry], quantizer_name: str
) -> QuantizerCfgEntry:
    """Find the last entry in a ``quant_cfg`` list whose ``quantizer_name`` key equals the query.

    This performs an **exact string comparison** against the ``quantizer_name`` field of each
    entry — it does *not* apply ``fnmatch`` pattern matching.  For example, passing
    ``"*input_quantizer"`` will only match entries whose ``quantizer_name`` is literally
    ``"*input_quantizer"``, not entries with a different wildcard that would match the same
    module names at apply time.

    Returns the *last* match because entries are applied in list order and later entries
    override earlier ones, so the last match represents the effective configuration.

    Args:
        quant_cfg_list: A list of :class:`QuantizerCfgEntry` dicts.
        quantizer_name: The exact ``quantizer_name`` string to search for.

    Returns:
        The last entry whose ``quantizer_name`` equals *quantizer_name*.

    Raises:
        KeyError: If no entry with the given ``quantizer_name`` is found.
    """
    result = None
    for entry in quant_cfg_list:
        if isinstance(entry, dict) and entry.get("quantizer_name") == quantizer_name:
            result = entry
    if result is None:
        raise KeyError(f"No quant_cfg entry with quantizer_name={quantizer_name!r}")
    return result


_base_disable_all: list[QuantizerCfgEntry] = [
    {"quantizer_name": "*", "enable": False},
]

_default_disabled_quantizer_cfg: list[QuantizerCfgEntry] = [
    {"parent_class": "nn.BatchNorm1d", "quantizer_name": "*", "enable": False},
    {"parent_class": "nn.BatchNorm2d", "quantizer_name": "*", "enable": False},
    {"parent_class": "nn.BatchNorm3d", "quantizer_name": "*", "enable": False},
    {"parent_class": "nn.LeakyReLU", "quantizer_name": "*", "enable": False},
    {"quantizer_name": "*lm_head*", "enable": False},
    {
        "quantizer_name": "*proj_out.*",
        "enable": False,
    },  # In Whisper model, lm_head has key name proj_out
    {
        "quantizer_name": "*block_sparse_moe.gate*",
        "enable": False,
    },  # Skip the MOE router
    {"quantizer_name": "*router*", "enable": False},  # Skip the MOE router
    {"quantizer_name": "*mlp.gate.*", "enable": False},  # Skip the MOE router
    {
        "quantizer_name": "*mlp.shared_expert_gate.*",
        "enable": False,
    },  # Skip the MOE router
    {"quantizer_name": "*linear_attn.conv1d*", "enable": False},
    {"quantizer_name": "*mixer.conv1d*", "enable": False},  # Skip mamba conv1d
    {"quantizer_name": "*output_layer*", "enable": False},
    {"quantizer_name": "output.*", "enable": False},
]

_mamba_moe_disabled_quantizer_cfg: list[QuantizerCfgEntry] = [
    {"quantizer_name": "*fc1_latent_proj*", "enable": False},  # Skip Latent MOE
    {"quantizer_name": "*fc2_latent_proj*", "enable": False},  # Skip Latent MOE
    {"quantizer_name": "*q_proj*", "enable": False},  # Skip QKV Linear
    {"quantizer_name": "*k_proj*", "enable": False},  # Skip QKV Linear
    {"quantizer_name": "*v_proj*", "enable": False},  # Skip QKV Linear
    {"quantizer_name": "*o_proj*", "enable": False},  # Skip QKV Output Projection
]

INT8_DEFAULT_CFG = {
    "quant_cfg": [
        *_base_disable_all,
        {"quantizer_name": "*weight_quantizer", "cfg": {"num_bits": 8, "axis": 0}},
        {"quantizer_name": "*input_quantizer", "cfg": {"num_bits": 8, "axis": None}},
        *_default_disabled_quantizer_cfg,
    ],
    "algorithm": "max",
}

INT8_SMOOTHQUANT_CFG = {
    "quant_cfg": [
        *_base_disable_all,
        {"quantizer_name": "*weight_quantizer", "cfg": {"num_bits": 8, "axis": 0}},
        {"quantizer_name": "*input_quantizer", "cfg": {"num_bits": 8, "axis": None}},
        *_default_disabled_quantizer_cfg,
    ],
    "algorithm": "smoothquant",
}

INT8_WEIGHT_ONLY_CFG = {
    "quant_cfg": [
        *_base_disable_all,
        {"quantizer_name": "*weight_quantizer", "cfg": {"num_bits": 8, "axis": 0}},
        {"quantizer_name": "*input_quantizer", "enable": False},
        *_default_disabled_quantizer_cfg,
    ],
    "algorithm": "max",
}

FP8_DEFAULT_CFG = {
    "quant_cfg": [
        *_base_disable_all,
        {
            "quantizer_name": "*weight_quantizer",
            "cfg": {"num_bits": (4, 3), "axis": None},
        },
        {
            "quantizer_name": "*input_quantizer",
            "cfg": {"num_bits": (4, 3), "axis": None},
        },
        *_default_disabled_quantizer_cfg,
    ],
    "algorithm": "max",
}

MAMBA_MOE_FP8_AGGRESSIVE_CFG = {
    "quant_cfg": [
        *_base_disable_all,
        {
            "quantizer_name": "*weight_quantizer",
            "cfg": {"num_bits": (4, 3), "axis": None},
        },
        {
            "quantizer_name": "*input_quantizer",
            "cfg": {"num_bits": (4, 3), "axis": None},
        },
        *_default_disabled_quantizer_cfg,
        *_mamba_moe_disabled_quantizer_cfg,
    ],
    "algorithm": "max",
}

MAMBA_MOE_FP8_CONSERVATIVE_CFG = {
    "quant_cfg": [
        *_base_disable_all,
        {
            "quantizer_name": "*weight_quantizer",
            "cfg": {"num_bits": (4, 3), "axis": None},
        },
        {
            "quantizer_name": "*input_quantizer",
            "cfg": {"num_bits": (4, 3), "axis": None},
        },
        *_default_disabled_quantizer_cfg,
        *_mamba_moe_disabled_quantizer_cfg,
        {"quantizer_name": "*mixer.in_proj*", "enable": False},  # Skip mamba linear
        {"quantizer_name": "*mixer.out_proj*", "enable": False},  # Skip mamba linear
    ],
    "algorithm": "max",
}

FP8_PER_CHANNEL_PER_TOKEN_CFG = {
    "quant_cfg": [
        *_base_disable_all,
        {"quantizer_name": "*weight_quantizer", "cfg": {"num_bits": (4, 3), "axis": 0}},
        {
            "quantizer_name": "*input_quantizer",
            "cfg": {
                "num_bits": (4, 3),
                "type": "dynamic",
                "block_sizes": {-1: None},
            },
        },
        *_default_disabled_quantizer_cfg,
    ],
    "algorithm": "max",
}

# FP8 2D blockwise fake quantization config for deepseek models
FP8_2D_BLOCKWISE_WEIGHT_ONLY_CFG = {
    "quant_cfg": [
        *_base_disable_all,
        {
            "quantizer_name": "*weight_quantizer",
            "cfg": {
                "num_bits": (4, 3),
                "block_sizes": {-1: 128, -2: 128},
            },
        },
        {"quantizer_name": "*input_quantizer", "enable": False},
        *_default_disabled_quantizer_cfg,
    ],
    "algorithm": "max",
}

INT4_BLOCKWISE_WEIGHT_ONLY_CFG = {
    "quant_cfg": [
        *_base_disable_all,
        {
            "quantizer_name": "*weight_quantizer",
            "cfg": {
                "num_bits": 4,
                "block_sizes": {-1: 128},
            },
        },
        {"quantizer_name": "*input_quantizer", "enable": False},
        *_default_disabled_quantizer_cfg,
    ],
    "algorithm": "max",
}


INT4_AWQ_CFG = {
    "quant_cfg": [
        *_base_disable_all,
        {
            "quantizer_name": "*weight_quantizer",
            "cfg": {
                "num_bits": 4,
                "block_sizes": {-1: 128, "type": "static"},
            },
        },
        {"quantizer_name": "*input_quantizer", "enable": False},
        *_default_disabled_quantizer_cfg,
    ],
    "algorithm": {"method": "awq_lite", "alpha_step": 0.1},
    # "algorithm": {"method": "awq_full", "alpha_step": 0.1, "max_co_batch_size": 1024},
    # "algorithm": {"method": "awq_clip", "max_co_batch_size": 2048},
}

# W4A8 currently uses INT4 blockwise quantization (block size = 128) followed by FP8 quantization
# for weights. This could change in the future
W4A8_AWQ_BETA_CFG = {
    "quant_cfg": [
        *_base_disable_all,
        {
            "quantizer_name": "*weight_quantizer",
            "cfg": [
                {
                    "num_bits": 4,
                    "block_sizes": {-1: 128, "type": "static"},
                },
                {
                    "num_bits": (4, 3),
                },
            ],
        },
        {
            "quantizer_name": "*input_quantizer",
            "cfg": {
                "num_bits": (4, 3),
            },
        },
        *_default_disabled_quantizer_cfg,
    ],
    "algorithm": "awq_lite",
}

MXFP8_DEFAULT_CFG = {
    "quant_cfg": [
        *_base_disable_all,
        {
            "quantizer_name": "*weight_quantizer",
            "cfg": {
                "num_bits": (4, 3),
                "block_sizes": {-1: 32, "type": "dynamic", "scale_bits": (8, 0)},
            },
        },
        {
            "quantizer_name": "*input_quantizer",
            "cfg": {
                "num_bits": (4, 3),
                "block_sizes": {-1: 32, "type": "dynamic", "scale_bits": (8, 0)},
            },
        },
        *_default_disabled_quantizer_cfg,
    ],
    "algorithm": None,
}

MXFP6_DEFAULT_CFG = {
    "quant_cfg": [
        *_base_disable_all,
        {
            "quantizer_name": "*weight_quantizer",
            "cfg": {
                "num_bits": (3, 2),
                "block_sizes": {-1: 32, "type": "dynamic", "scale_bits": (8, 0)},
            },
        },
        {
            "quantizer_name": "*input_quantizer",
            "cfg": {
                "num_bits": (3, 2),
                "block_sizes": {-1: 32, "type": "dynamic", "scale_bits": (8, 0)},
            },
        },
        *_default_disabled_quantizer_cfg,
    ],
    "algorithm": None,
}

MXFP4_DEFAULT_CFG = {
    "quant_cfg": [
        *_base_disable_all,
        {
            "quantizer_name": "*weight_quantizer",
            "cfg": {
                "num_bits": (2, 1),
                "block_sizes": {-1: 32, "type": "dynamic", "scale_bits": (8, 0)},
            },
        },
        {
            "quantizer_name": "*input_quantizer",
            "cfg": {
                "num_bits": (2, 1),
                "block_sizes": {-1: 32, "type": "dynamic", "scale_bits": (8, 0)},
            },
        },
        *_default_disabled_quantizer_cfg,
    ],
    "algorithm": None,
}

W4A8_MXFP4_FP8_CFG = {
    "quant_cfg": [
        *_base_disable_all,
        {
            "quantizer_name": "*weight_quantizer",
            "cfg": {
                "num_bits": (2, 1),
                "block_sizes": {-1: 32, "type": "dynamic", "scale_bits": (8, 0)},
            },
        },
        {
            "quantizer_name": "*input_quantizer",
            "cfg": {"num_bits": (4, 3), "axis": None},
        },
        *_default_disabled_quantizer_cfg,
    ],
    "algorithm": None,
}

MXINT8_DEFAULT_CFG = {
    "quant_cfg": [
        *_base_disable_all,
        {
            "quantizer_name": "*weight_quantizer",
            "cfg": {
                "num_bits": 8,
                "block_sizes": {-1: 32, "type": "dynamic", "scale_bits": (8, 0)},
            },
        },
        {
            "quantizer_name": "*input_quantizer",
            "cfg": {
                "num_bits": 8,
                "block_sizes": {-1: 32, "type": "dynamic", "scale_bits": (8, 0)},
            },
        },
        *_default_disabled_quantizer_cfg,
    ],
    "algorithm": None,
}

# KV-cache configs are designed to be merged with a primary quantization config (e.g.
# FP8_DEFAULT_CFG) that already contains _base_disable_all.  They intentionally omit both
# _base_disable_all and "algorithm" because these are provided by the primary config.
FP8_KV_CFG = {
    "quant_cfg": [
        {
            "quantizer_name": "*[kv]_bmm_quantizer",
            "cfg": {"num_bits": (4, 3)},
        },
    ]
}

FP8_AFFINE_KV_CFG = {
    "quant_cfg": [
        {
            "quantizer_name": "*[kv]_bmm_quantizer",
            "cfg": {
                "num_bits": (4, 3),
                "bias": {-2: None, -4: None, "type": "static"},
            },
        },
    ]
}

_nvfp4_cfg = {
    "num_bits": (2, 1),
    "block_sizes": {-1: 16, "type": "dynamic", "scale_bits": (4, 3)},
}

_nvfp4_cfg_bs32 = {
    "num_bits": (2, 1),
    "block_sizes": {-1: 32, "type": "dynamic", "scale_bits": (4, 3)},
}


def _nvfp4_selective_quant_cfg(
    layer_patterns: list[str],
    *,
    quantizer: dict = _nvfp4_cfg,
    weight_only: bool = False,
    algorithm: str | dict = "max",
) -> dict:
    """Build an NVFP4 config that quantizes only the specified layer patterns."""
    quant_cfg: list[QuantizerCfgEntry] = []
    quant_cfg.extend(_base_disable_all)
    for pattern in layer_patterns:
        # Deep-copy the quantizer dict so each config constant gets its own instance.
        quant_cfg.append(
            {"quantizer_name": f"{pattern}weight_quantizer", "cfg": copy.deepcopy(quantizer)}
        )
        if not weight_only:
            quant_cfg.append(
                {"quantizer_name": f"{pattern}input_quantizer", "cfg": copy.deepcopy(quantizer)}
            )
    quant_cfg.extend(_default_disabled_quantizer_cfg)
    return {"quant_cfg": quant_cfg, "algorithm": algorithm}


NVFP4_DEFAULT_CFG = _nvfp4_selective_quant_cfg(["*"])

NVFP4_W4A4_WEIGHT_MSE_FP8_SWEEP_CFG = {
    "quant_cfg": [
        *_base_disable_all,
        {
            "quantizer_name": "*weight_quantizer",
            "cfg": {
                "num_bits": (2, 1),
                "block_sizes": {-1: 16, "type": "static", "scale_bits": (4, 3)},
            },
        },
        {"quantizer_name": "*input_quantizer", "cfg": _nvfp4_cfg},
        *_default_disabled_quantizer_cfg,
    ],
    "algorithm": {
        "method": "mse",
        "fp8_scale_sweep": True,
    },
}

NVFP4_W4A4_WEIGHT_LOCAL_HESSIAN_CFG = {
    "quant_cfg": [
        *_base_disable_all,
        {
            "quantizer_name": "*weight_quantizer",
            "cfg": {
                "num_bits": (2, 1),
                "block_sizes": {-1: 16, "type": "static", "scale_bits": (4, 3)},
            },
        },
        {"quantizer_name": "*input_quantizer", "cfg": _nvfp4_cfg},
        *_default_disabled_quantizer_cfg,
    ],
    "algorithm": {
        "method": "local_hessian",
        "fp8_scale_sweep": True,
    },
}

MAMBA_MOE_NVFP4_AGGRESSIVE_CFG = {
    "quant_cfg": [
        *_base_disable_all,
        {"quantizer_name": "*weight_quantizer", "cfg": _nvfp4_cfg},
        {"quantizer_name": "*input_quantizer", "cfg": _nvfp4_cfg},
        *_default_disabled_quantizer_cfg,
        *_mamba_moe_disabled_quantizer_cfg,
    ],
    "algorithm": "max",
}
MAMBA_MOE_NVFP4_CONSERVATIVE_CFG = {
    "quant_cfg": [
        *_base_disable_all,
        {"quantizer_name": "*weight_quantizer", "cfg": _nvfp4_cfg},
        {"quantizer_name": "*input_quantizer", "cfg": _nvfp4_cfg},
        *_default_disabled_quantizer_cfg,
        *_mamba_moe_disabled_quantizer_cfg,
        {"quantizer_name": "*mixer.in_proj*", "enable": False},  # Skip mamba linear
        {"quantizer_name": "*mixer.out_proj*", "enable": False},  # Skip mamba linear
    ],
    "algorithm": "max",
}

NVFP4_AWQ_LITE_CFG = _nvfp4_selective_quant_cfg(["*"], algorithm="awq_lite")

NVFP4_AWQ_CLIP_CFG = _nvfp4_selective_quant_cfg(["*"], algorithm={"method": "awq_clip"})

NVFP4_AWQ_FULL_CFG = _nvfp4_selective_quant_cfg(
    ["*"], algorithm={"method": "awq_full", "alpha_step": 0.1}
)

# See comment above FP8_KV_CFG — KV-cache configs omit _base_disable_all and "algorithm".
NVFP4_AFFINE_KV_CFG = {
    "quant_cfg": [
        {
            "quantizer_name": "*[kv]_bmm_quantizer",
            "cfg": {
                **_nvfp4_cfg,
                "bias": {-2: None, -4: None, "type": "static"},
            },
        },
    ]
}

NVFP4_KV_CFG = {
    "quant_cfg": [
        {"quantizer_name": "*[kv]_bmm_quantizer", "cfg": _nvfp4_cfg},
    ]
}

# Moved from examples/diffusers/quantization/config.py to here
NVFP4_FP8_MHA_CONFIG = {
    "quant_cfg": [
        *_base_disable_all,
        {"quantizer_name": "*weight_quantizer", "cfg": _nvfp4_cfg},
        {"quantizer_name": "*input_quantizer", "cfg": _nvfp4_cfg},
        {"quantizer_name": "*output_quantizer", "enable": False},
        {
            "quantizer_name": "*q_bmm_quantizer",
            "cfg": {
                "num_bits": (4, 3),
            },
        },
        {
            "quantizer_name": "*k_bmm_quantizer",
            "cfg": {
                "num_bits": (4, 3),
            },
        },
        {
            "quantizer_name": "*v_bmm_quantizer",
            "cfg": {
                "num_bits": (4, 3),
            },
        },
        {
            "quantizer_name": "*softmax_quantizer",
            "cfg": {
                "num_bits": (4, 3),
            },
        },
        {
            "quantizer_name": "transformer_blocks*bmm2_output_quantizer",
            "cfg": {
                "num_bits": (4, 3),
            },
        },
    ],
    "algorithm": "max",
}

# See comment above FP8_KV_CFG — KV-cache configs omit _base_disable_all and "algorithm".
NVFP4_KV_ROTATE_CFG = {
    "quant_cfg": [
        {
            # q_bmm is disabled but pre-configured with rotate=True so that downstream
            # code can inspect the rotate flag even while the quantizer is off.
            "quantizer_name": "*q_bmm_quantizer",
            "cfg": {
                "rotate": True,
            },
            "enable": False,
        },
        {
            "quantizer_name": "*k_bmm_quantizer",
            "cfg": {
                **_nvfp4_cfg,
                "rotate": True,
            },
        },
        {"quantizer_name": "*v_bmm_quantizer", "cfg": _nvfp4_cfg},
    ],
    "algorithm": "max",
}

NVFP4_SVDQUANT_DEFAULT_CFG = _nvfp4_selective_quant_cfg(
    ["*"], algorithm={"method": "svdquant", "lowrank": 32}
)

W4A8_NVFP4_FP8_CFG = {
    "quant_cfg": [
        *_base_disable_all,
        {
            "quantizer_name": "*weight_quantizer",
            "cfg": {
                "num_bits": (2, 1),
                "block_sizes": {-1: 32, "type": "dynamic", "scale_bits": (4, 3)},
            },
        },
        {
            "quantizer_name": "*input_quantizer",
            "cfg": {
                "num_bits": (4, 3),
            },
        },
        *_default_disabled_quantizer_cfg,
    ],
    "algorithm": "max",
}

MXFP4_MLP_WEIGHT_ONLY_CFG = {
    "quant_cfg": [
        *_base_disable_all,
        {
            "quantizer_name": "*mlp*weight_quantizer",
            "cfg": {
                "num_bits": (2, 1),
                "block_sizes": {-1: 32, "type": "dynamic", "scale_bits": (8, 0)},
            },
        },
        {
            "quantizer_name": "*block_sparse_moe*weight_quantizer",
            "cfg": {
                "num_bits": (2, 1),
                "block_sizes": {-1: 32, "type": "dynamic", "scale_bits": (8, 0)},
            },
        },
        *_default_disabled_quantizer_cfg,
    ],
    "algorithm": None,
}

NVFP4_MLP_WEIGHT_ONLY_CFG = _nvfp4_selective_quant_cfg(
    ["*mlp*", "*block_sparse_moe*"], quantizer=_nvfp4_cfg_bs32, weight_only=True
)
NVFP4_EXPERTS_ONLY_CFG = _nvfp4_selective_quant_cfg(["*mlp.experts*", "*block_sparse_moe*"])
NVFP4_MLP_ONLY_CFG = _nvfp4_selective_quant_cfg(["*mlp*", "*block_sparse_moe*"])
NVFP4_OMLP_ONLY_CFG = _nvfp4_selective_quant_cfg(["*o_proj*", "*mlp*", "*block_sparse_moe*"])

# DO NOT ADD NEW CONFIGS HERE. If you want to add a new general recipe, add it to
# modelopt_recipes/general/ptq/ as a yaml file
choices: set[str] = {
    "FP8_2D_BLOCKWISE_WEIGHT_ONLY_CFG",
    "FP8_AFFINE_KV_CFG",
    "FP8_DEFAULT_CFG",
    "FP8_KV_CFG",
    "FP8_PER_CHANNEL_PER_TOKEN_CFG",
    "INT4_AWQ_CFG",
    "INT4_BLOCKWISE_WEIGHT_ONLY_CFG",
    "INT8_DEFAULT_CFG",
    "INT8_SMOOTHQUANT_CFG",
    "INT8_WEIGHT_ONLY_CFG",
    "MXFP4_DEFAULT_CFG",
    "MXFP8_DEFAULT_CFG",
    "MXINT8_DEFAULT_CFG",
    "NVFP4_AFFINE_KV_CFG",
    "NVFP4_AWQ_CLIP_CFG",
    "NVFP4_AWQ_FULL_CFG",
    "NVFP4_AWQ_LITE_CFG",
    "NVFP4_DEFAULT_CFG",
    "NVFP4_FP8_MHA_CONFIG",
    "NVFP4_KV_CFG",
    "NVFP4_KV_ROTATE_CFG",
    "W4A8_NVFP4_FP8_CFG",
    "NVFP4_SVDQUANT_DEFAULT_CFG",
    "W4A8_AWQ_BETA_CFG",
    "W4A8_MXFP4_FP8_CFG",
    "NVFP4_MLP_WEIGHT_ONLY_CFG",
    "MXFP4_MLP_WEIGHT_ONLY_CFG",
    "NVFP4_MLP_ONLY_CFG",
    "NVFP4_EXPERTS_ONLY_CFG",
    "NVFP4_OMLP_ONLY_CFG",
    "MAMBA_MOE_NVFP4_CONSERVATIVE_CFG",
    "MAMBA_MOE_NVFP4_AGGRESSIVE_CFG",
    "MAMBA_MOE_FP8_CONSERVATIVE_CFG",
    "MAMBA_MOE_FP8_AGGRESSIVE_CFG",
    "NVFP4_W4A4_WEIGHT_MSE_FP8_SWEEP_CFG",
}

BiasType = Literal["static", "dynamic"]
BiasMethod = Literal["mean", "max_min"]


class RotateConfig(ModeloptBaseConfig):
    """Configuration for rotating quantizer input via Hadamard transform (RHT/QuaRot/SpinQuant).

    See :func:`normalized_hadamard_transform <modelopt.torch.quantization.nn.functional.normalized_hadamard_transform>`
    for transform details.
    """

    enable: bool = False
    rotate_fp32: bool = False
    block_size: int | None = None

    @field_validator("block_size", mode="before")
    @classmethod
    def validate_block_size(cls, v):
        """Validate block_size is a positive int (mode=before to catch bool before int coercion)."""
        if v is not None and (isinstance(v, bool) or not isinstance(v, int) or v <= 0):
            raise ValueError(f"block_size must be a positive int, got {v!r}")
        return v


class QuantizerAttributeConfig(ModeloptBaseConfig):
    """Quantizer attribute type."""

    enable: bool = ModeloptField(
        default=True,
        title="Enable quantizer.",
        description="""If True, enables the quantizer. If False, by-pass the quantizer and returns the input tensor.""",
    )

    num_bits: int | tuple[int, int] | str = ModeloptField(
        default=8,
        title="An integer or a tuple of two integers specifying the number of quantization bits.",
        description="""`num_bits` can be:

        #. A positive integer argument for integer quantization. `num_bits` specify
            the number of bits used for integer quantization.

        #. Constant integer tuple (E,M) for floating point quantization emulating
            Nvidia's FPx quantization. E is the number of exponent bits and M is the number
            of mantissa bits. Supported FPx quantization formats: FP8 (E4M3, E5M2), FP6(E3M2, E2M3), FP4(E2M1).

        #. String specifying the quantization format. This is current used only for custom backends.""",
    )

    @model_validator(mode="before")
    @classmethod
    def validate_config(cls, values):
        """Validate quantizer config."""

        def _validate_recursive(value):
            """Recursively validate config structure."""
            if value is None:
                return

            if isinstance(value, list):
                for item in value:
                    _validate_recursive(item)
            elif isinstance(value, dict):
                if len(value) == 1 and "enable" in value and value["enable"] is True:
                    raise ValueError(
                        "Invalid quantizer config: Cannot specify only {'enable': True}. "
                        "Additional parameters are required when enabling quantization."
                    )
                # Recurse into nested dicts
                for v in value.values():
                    _validate_recursive(v)

        _validate_recursive(values)
        return values

    @model_validator(mode="after")
    def validate_num_bits(self):
        """Validate `num_bits`."""
        if self.backend is not None:
            # For custom backends, we don't need to validate num_bits
            return self

        num_bits = self.num_bits

        if isinstance(num_bits, int) and num_bits < 1:
            raise ValueError(
                f"num_bits must be a positive integer or a tuple of positive integers. {num_bits}"
            )

        if not isinstance(num_bits, tuple):
            return self

        if not all(x > 0 for x in num_bits):
            raise ValueError("num_bits must be a positive integer or a tuple of positive integers.")

        block_sizes = self.block_sizes
        if num_bits not in [
            (4, 3),
            (5, 2),
            (2, 1),
            (1, 2),
            (0, 3),
            (3, 0),
            (3, 2),
            (2, 3),
        ]:
            raise ValueError(
                "Supported FPx quantization formats: FP8 (E4M3, E5M2), FP6(E3M2, E2M3), FP4(E2M1)."
            )
        elif num_bits not in [(4, 3), (2, 1)] and (
            block_sizes is None or block_sizes.get("type", None) != "dynamic"
        ):
            raise ValueError(
                "Only blockwise dynamic quantization is supported with quantization "
                "formats E{num_bis[0]}M{num_bits[1]}."
            )
        return self

    axis: int | tuple[int, ...] | None = ModeloptField(
        default=None,
        title="None, integer or a tuple of integers specifying the axis to quantize.",
        description="""This field is for static per-channel quantization. *It cannot coexist with `block_sizes`*.
            You should set axis if you want a fixed shape of scale factor.

            For example, if axis is set to None, the scale factor will be a scalar (per-tensor quantization)
            if the axis is set to 0, the scale factor will be a vector of shape (dim0, ) (per-channel quantization).
            if the axis is set to (-2, -1), the scale factor will be a vector of shape (dim-2, dim-1)

            axis value must be in the range [-rank(input_tensor), rank(input_tensor))
        """,
    )

    fake_quant: bool = ModeloptField(
        default=True,
        title="Enable fake quantization.",
        description="""If True, enable fake quantization.""",
    )

    unsigned: bool = ModeloptField(
        default=False,
        title="Enable unsigned quantization.",
        description="""If True, enable unsigned quantization. Used only for integer quantization.""",
    )

    narrow_range: bool = ModeloptField(
        default=False,
        title="Enable narrow range quantization.",
        description="""If True, enable narrow range quantization. Used only for integer quantization.""",
    )

    learn_amax: bool = ModeloptField(
        default=False,
        title="Enable learning amax.",
        description="""``learn_amax`` is deprecated and reserved for backward compatibility.""",
    )

    @field_validator("learn_amax")
    @classmethod
    def validate_learn_amax(cls, v):
        """Validate learn_amax."""
        assert v is not True, "learn_amax is deprecated and reserved for backward compatibility."
        return v

    type: str = ModeloptField(
        default="static",
        title="""Specify whether the quantization is static or dynamic.""",
        description="""The value is a string from ``["static", "dynamic"]``.
            If ``"dynamic"``, dynamic quantization will be enabled which does not collect any statistics during
            calibration.""",
        pattern=r"^static$|^dynamic$",
    )

    block_sizes: dict[int | str, int | tuple[int, int] | str | dict[int, int] | None] | None = (
        ModeloptField(
            default=None,
            title="Optional dictionary specifying block quantization parameters.",
            description="""This field is for static or dynamic block quantization. *It cannot coexist with ``axis``*.
            You should set block_sizes if you want fixed number of elements to share every scale factor.

            The keys are the axes for block quantization and the
            values are block sizes for quantization along the respective axes. Keys must be in the
            range ``[-tensor.dim(), tensor.dim())``. Values, which are the block sizes for quantization must be
            positive integers or ``None``. A positive block size specifies the block size for quantization along that
            axis. ``None`` means that the block size will be the maximum possible size in that dimension - this is
            useful for specifying certain quantization formats such per-token dynamic quantization which has the `amax`
            shared along the last dimension.

            In addition, there can be special string keys ``"type"``, ``"scale_bits"`` and ``"scale_block_sizes"``.

            Key ``"type"`` should map to ``"dynamic"`` or ``"static"`` where ``"dynamic"``
            indicates dynamic block quantization and "static"
            indicates static calibrated block quantization. By default, the type is ``"static"``.

            Key ``"scale_bits"`` specify the quantization bits for the per-block quantization scale factor
            (i.e a double quantization scheme).

            Key ``"scale_block_sizes"`` specify the block size for double quantization.
            By default per-block quantization scale is not quantized.

            For example, ``block_sizes = {-1: 32}`` will quantize the last axis of the input tensor in
            blocks of size 32 with static calibration, with a total of ``numel(tensor) / 32`` scale factors.
            ``block_sizes = {-1: 32, "type": "dynamic"}`` will perform dynamic block quantization.
            ``block_sizes = {-1: None, "type": "dynamic"}`` can be used to
            specify per-token dynamic quantization.
        """,
        )
    )

    bias: dict[int | str, BiasType | BiasMethod | tuple[int, ...] | bool | int | None] | None = (
        ModeloptField(
            default=None,
            title="Bias configuration.",
            description="""Configuration for bias handling in affine quantization. The keys are:
            - "enable": Boolean to enable/disable bias handling, default is False
            - "type": Specify the type of bias ["static", "dynamic"], default is "static"
            - "method": Specify the method of bias calibration ["mean", "max_min"], default is "mean"
            - "axis": Tuple of integers specifying axes for bias computation, default is None

            Examples:
            bias = {"enable": True}
            bias = {"enable": True, "type": "static", "axis": -1}
            bias = {"enable": True, "type": "dynamic", "axis": (-1, -3)}
        """,
        )
    )

    @staticmethod
    def _get_block_quant_axes_and_sizes(block_sizes):
        if block_sizes is None:
            return None
        return {
            k: v
            for k, v in block_sizes.items()
            if k not in ["type", "scale_bits", "scale_block_sizes"]
        }

    @field_validator("block_sizes")
    @classmethod
    def validate_block_sizes(cls, v, info: ValidationInfo):
        """Validate block sizes."""
        if v is None:
            return v
        assert info.data["axis"] is None, "axis must be None when block_sizes is not None."
        if v.get("type", None) == "dynamic":
            assert len(cls._get_block_quant_axes_and_sizes(v)) == 1, (
                "Dynamic block quantization only supports quantization last axis."
            )
        for _k, _v in v.items():
            if isinstance(_k, str):
                assert _k in ["type", "scale_bits", "scale_block_sizes"]
            else:
                assert isinstance(_k, int) and (_v is None or isinstance(_v, int))
        return v

    @field_validator("bias")
    @classmethod
    def validate_bias(cls, v):
        """Validate bias."""
        if v is None:
            return v

        if "type" in v and v["type"] not in ["static", "dynamic"]:
            raise ValueError(f"Invalid bias type: {v['type']}, expected 'static' or 'dynamic'")

        if "method" in v and v["method"] not in ["mean", "max_min"]:
            raise ValueError(f"Invalid bias method: {v['method']}, expected 'mean' or 'max_min'")

        axis = [k for k in v.keys() if k not in ["type", "method"]]  # noqa: SIM118
        assert len(axis) > 0, "The axis for bias computation is not specified."
        for x in axis:
            if not isinstance(x, int):
                raise ValueError(f"Invalid axis type {type(axis)}, expected int")

        return v

    trt_high_precision_dtype: str = ModeloptField(
        default="Float",
        title="TRT StronglyType requires all weights and amax to be in the same dtype.",
        description="""The value is a string from ``["Float", "Half", "BFloat16"]``.
            The QDQs will be assigned the appropriate data type, and this variable will only be
            used when the user is exporting the quantized ONNX model.""",
        pattern=r"^Float$|^Half$|^BFloat16$",
    )

    calibrator: str | ConstructorLike = ModeloptField(
        default="max",
        title="""Specify the calibrator to use.""",
        description="""The calibrator can be a string from ``["max", "histogram"]`` or a constructor
        to create a calibrator which subclasses :class:`_Calibrator <modelopt.torch.quantization.calib._Calibrator>`.
        See :meth:`standardize_constructor_args <modelopt.torch.utils.network.standardize_constructor_args>`
        for more information on how to specify the constructor.""",
    )

    @field_validator("calibrator")
    @classmethod
    def validate_calibrator(cls, v, info: ValidationInfo):
        """Validate calibrator."""
        if isinstance(v, str):
            assert v in ["max", "histogram"]
        return v

    rotate: bool | RotateConfig = ModeloptField(
        default=False,
        title="""Configuration for rotating the input before quantization.""",
        description="""Can be a boolean or a :class:`RotateConfig` instance (or equivalent dict).

        If a boolean, it is treated as :attr:`RotateConfig.enable` with all other fields defaulting.

        This can be used for rotation based PTQ methods, e.g. QuaRot or SpinQuant.
        See https://arxiv.org/abs/2404.00456 for example.""",
    )

    pass_through_bwd: bool = ModeloptField(
        default=True,
        title="If set to true, fake quantization will be a pass through for gradient computation.",
        description="""
        Gradient computation where fake quantization is pass through is called
        'Straight-Through Estimator (STE)'. STE does not require saving of the input tensor for
        performing backward pass and hence consumes less memory.

        If set to False, we will use STE with zeroed outlier gradients. This setting may
        yield better QAT accuracy depending on the quantization format. However, this setting
        requires saving of the input tensor for computing gradients which uses more memory.

        For dynamic quantization formats like MXFP4, STE with zeroed outlier gradients
        is not needed since fake quantization with dynamic amax results in minimal/no clipping.
        """,
    )

    backend: str | None = ModeloptField(
        default=None,
        title="Name of custom quantization functional backend.",
        description="""
            Selects a non-default quantization functional backend by name. See
            :meth:`register_quant_backend <modelopt.torch.nn.modules.tensor_quantizer.register_quant_backend>`
            for more details on how to register a custom quantization backend.
        """,
    )
    backend_extra_args: dict | None = ModeloptField(
        default=None,
        title="Extra arguments for the selected backend.",
        description="""The extra arguments will saved on to the quantizer instance - this wont be
        passed directly to the backend entrypoint. Can be any serializable dictionary.

        Please use `backend_extra_args` to pass arguments that are not already supported by
        `QuantizerAttributeConfig`. This will ensure maximum compatibility with the other modelopt
        features such as modelopt's calibration algorithms.
        """,
    )

    use_constant_amax: bool = ModeloptField(
        default=False,
        title="Use constant amax for the quantizer.",
        description="""If True, set the amax to FP8 E4M3 max (448.0) and skip calibration.
        This is used for KV cache quantization where the downstream engine uses FP8 attention
        math for both FP8 and NVFP4 quantization, so the amax is hardcoded to the FP8 range.
        """,
    )


class QuantizeAlgorithmConfig(ModeloptBaseConfig):
    """Calibration algorithm config base."""

    method: Literal[None] = ModeloptField(
        None,
        title="This field specifies the name of the calibration algorithm. If None, no calibration is performed.",
    )

    moe_calib_experts_ratio: float | None = ModeloptField(
        default=None,
        gt=0.0,
        le=1.0,
        title="% of experts to calibrate during forward pass.",
        description=(
            "If specified, we force forward tokens to % of experts during the calibration"
            " pass. This forward is for calibration purpose only and will not affect the"
            " actual inference. NOTE: when set, ``layer_sync_moe_local_experts_amax`` is"
            " disabled so each expert maintains its own calibration statistics. Not"
            " supported for all MoE architectures; currently works with a few HuggingFace"
            " models such as Mixtral, Qwen3Moe, MiniMax."
        ),
    )

    layerwise: bool = ModeloptField(
        default=False,
        title="Enable layerwise (layer-by-layer) calibration.",
        description=(
            "If True, the calibration algorithm is applied layer by layer. "
            "Each layer's inputs are captured via a forward pass that reflects the "
            "quantization of all preceding layers, incurring O(N) forward passes for N layers."
        ),
    )

    layerwise_checkpoint_dir: str | None = ModeloptField(
        default=None,
        title="Checkpoint directory for layerwise calibration.",
        description=(
            "If set together with layerwise=True, per-layer checkpoints are saved to this "
            "directory during calibration. On restart, calibration resumes from the last "
            "completed layer."
        ),
    )

    @model_validator(mode="after")
    def validate_layerwise_checkpoint_dir(self):
        """Raise if layerwise_checkpoint_dir is set but layerwise is False."""
        if self.layerwise_checkpoint_dir is not None and not self.layerwise:
            raise ValueError(
                "layerwise_checkpoint_dir requires layerwise=True. "
                "Set layerwise=True or remove layerwise_checkpoint_dir."
            )
        return self


class MaxCalibConfig(QuantizeAlgorithmConfig):
    """The config for max calibration algorithm.

    Max calibration estimates max values of activations or weights and use this max values
    to set the quantization scaling factor.
    See `Integer Quantization <https://arxiv.org/pdf/2004.09602>`_ for the concepts.
    """

    method: Literal["max"] = ModeloptField("max")

    distributed_sync: bool | None = ModeloptField(
        default=True,
        title="Whether to sync the amax across the distributed processes.",
        description="If True, the amax will be synced across the distributed processes.",
    )

    sync_expert_weight_amax: bool = ModeloptField(
        default=False,
        title="Sync weight quantizer amax across MoE experts",
        description=(
            "If True, the weight quantizer amax values are synchronized (max) across "
            "local experts in SequentialMLP layers during calibration. This matches "
            "TEGroupedMLP behavior where all experts share a single weight quantizer. "
            "Only affects MoE models with SequentialMLP experts."
        ),
    )


class MseCalibConfig(QuantizeAlgorithmConfig):
    """Configuration for per-tensor MSE calibration.

    Finds a scale s (via amax a, with s = a / q_max) that minimizes the
    reconstruction error of a tensor after uniform Q→DQ:

        s* = argmin_s  E[(X - DQ(Q(X; s)))^2],   X ∈ {weights | activations}

    When fp8_scale_sweep is enabled, step_size is ignored.
    """

    method: Literal["mse"] = ModeloptField("mse")

    step_size: float | None = ModeloptField(
        default=0.1,
        gt=0.0,
        title="Step size for amax search.",
        description="Step size between amax candidates. The number of candidates is computed as "
        "ceil((stop_multiplier - start_multiplier) / step_size) + 1.",
    )

    start_multiplier: float | None = ModeloptField(
        default=0.25,
        gt=0.0,
        title="Starting multiplier for amax search.",
        description="Starting multiplier for amax search range (multiplies initial amax).",
    )

    stop_multiplier: float | None = ModeloptField(
        default=4.0,
        gt=0.0,
        title="Ending multiplier for amax search.",
        description="Ending multiplier for amax search range (multiplies initial amax).",
    )

    fp8_scale_sweep: bool | None = ModeloptField(
        default=False,
        title="Enable FP8 scale sweep for NVFP4 per-block quantization.",
        description="If True, sweep all 128 FP8 E4M3 scale values instead of using multipliers. "
        "Only applies to NVFP4 weight quantization. When enabled, num_steps, step_size, "
        "start_multiplier, and stop_multiplier are ignored.",
    )

    distributed_sync: bool | None = ModeloptField(
        default=True,
        title="Whether to sync the amax across the distributed processes.",
        description="If True, the amax will be synced across the distributed processes.",
    )


class LocalHessianCalibConfig(QuantizeAlgorithmConfig):
    """Configuration for local Hessian-weighted MSE calibration.

    This algorithm uses activation information to optimize per-block scales for weight
    quantization. It minimizes the output reconstruction error by weighting the loss
    with the local Hessian matrix computed from input activations.

    The local Hessian loss for each block is: ``(dw @ H @ dw.T)`` where:
    - ``dw = weight - quantized_weight`` (weight reconstruction error per block)
    - ``H = X @ X.T`` is the local Hessian computed from input activations X

    """

    method: Literal["local_hessian"] = ModeloptField("local_hessian")

    step_size: float | None = ModeloptField(
        default=0.1,
        gt=0.0,
        title="Step size for amax search.",
        description="Step size between amax candidates. The number of candidates is computed as "
        "ceil((stop_multiplier - start_multiplier) / step_size) + 1.",
    )

    start_multiplier: float | None = ModeloptField(
        default=0.25,
        gt=0.0,
        title="Starting multiplier for amax search.",
        description="Starting multiplier for amax search range (multiplies initial amax).",
    )

    stop_multiplier: float | None = ModeloptField(
        default=4.0,
        gt=0.0,
        title="Ending multiplier for amax search.",
        description="Ending multiplier for amax search range (multiplies initial amax).",
    )

    fp8_scale_sweep: bool | None = ModeloptField(
        default=True,
        title="Enable FP8 scale sweep for NVFP4 per-block quantization.",
        description="If True, sweep over all 128 possible FP8 E4M3 scale values "
        "for NVFP4 per-block quantization instead of using multipliers. "
        "This is the recommended setting for NVFP4 quantization.",
    )

    block_size: int | None = ModeloptField(
        default=16,
        gt=0,
        title="Block size for local Hessian computation.",
        description="The block size used for computing the local Hessian matrix. "
        "This should match the block size used in the quantization config. "
        "Default is 16 for NVFP4.",
    )

    distributed_sync: bool | None = ModeloptField(
        default=True,
        title="Whether to sync the amax across the distributed processes.",
        description="If True, the amax will be synced across the distributed processes.",
    )

    debug: bool | None = ModeloptField(
        default=False,
        title="Debug mode.",
        description="If True, module's local Hessian metadata will be kept as a module attribute.",
    )


class SmoothQuantCalibConfig(QuantizeAlgorithmConfig):
    """The config for ``smoothquant`` algorithm (SmoothQuant).

    SmoothQuant applies a smoothing factor which balances the scale of outliers in weights and activations.
    See `SmoothQuant paper <https://arxiv.org/pdf/2211.10438>`_ for more details.
    """

    method: Literal["smoothquant"] = ModeloptField("smoothquant")

    alpha: float | None = ModeloptField(
        default=1.0,
        ge=0.0,
        le=1.0,
        title="SmoothQuant hyper-parameter alpha.",
        description=(
            "This hyper-parameter controls the migration strength."
            "The migration strength is within [0, 1], "
            "a larger value migrates more quantization difficulty to weights."
        ),
    )


class AWQLiteCalibConfig(QuantizeAlgorithmConfig):
    """The config for ``awq_lite`` (AWQ lite) algorithm.

    AWQ lite applies a channel-wise scaling factor which minimizes the output difference after quantization.
    See `AWQ paper <https://arxiv.org/pdf/2306.00978>`_ for more details.
    """

    method: Literal["awq_lite"] = ModeloptField("awq_lite")

    alpha_step: float | None = ModeloptField(
        default=0.1,
        gt=0.0,
        le=1.0,
        title="Step size for the searching alpha.",
        description="The alpha will be searched from 0 to 1 with the step size specified.",
    )

    debug: bool | None = ModeloptField(
        default=False,
        title="Debug mode.",
        description="If True, module's search metadata will be kept as a module attribute named `awq_lite`.",
    )


class AWQClipCalibConfig(QuantizeAlgorithmConfig):
    """The config for ``awq_clip`` (AWQ clip) algorithm.

    AWQ clip searches clipped amax for per-group quantization, This search requires much more compute
    compared to AWQ lite. To avoid any OOM, the linear layer weights are batched along the ``out_features``
    dimension of batch size ``max_co_batch_size``. AWQ clip calibration also takes longer than AWQ lite.
    """

    method: Literal["awq_clip"] = ModeloptField("awq_clip")

    max_co_batch_size: int | None = ModeloptField(
        default=1024,
        title="Maximum output channel batch size while searching clip values.",
        description="Reduce this number if CUDA Out of Memory error occurs.",
    )

    max_tokens_per_batch: int | None = ModeloptField(
        default=64,
        title="Maximum tokens per batch while searching clip values.",
        description="""The total tokens used for clip search would be ``max_tokens_per_batch * number of batches``.
        Original AWQ uses a total of 512 tokens to search for clip values.""",
    )

    min_clip_ratio: float | None = ModeloptField(
        default=0.5,
        gt=0.0,
        lt=1.0,
        title="Minimum clip ratio to search for.",
        description="""It should be in (0, 1.0). Clip will search for the optimal clipping value in the range
        ``[original block amax * min_clip_ratio, original block amax]``.""",
    )

    shrink_step: float | None = ModeloptField(
        default=0.05,
        gt=0.0,
        le=1.0,
        title="Step size to search for clip values.",
        description="""It should be in range (0, 1.0]. The clip ratio will be searched from ``min_clip_ratio`` to 1
        with the step size specified.""",
    )

    debug: bool | None = ModeloptField(
        default=False,
        title="Debug mode.",
        description="If True, module's search metadata will be kept as a module attribute named ``awq_clip``.",
    )


class AWQFullCalibConfig(AWQLiteCalibConfig, AWQClipCalibConfig):
    """The config for ``awq`` or ``awq_full`` algorithm (AWQ full).

    AWQ full performs ``awq_lite`` followed by ``awq_clip``.
    """

    method: Literal["awq_full"] = ModeloptField("awq_full")

    debug: bool | None = ModeloptField(
        default=False,
        title="Debug mode.",
        description=(
            "If True, module's search metadata will be kept as "
            "module attributes named ``awq_lite`` and ``awq_clip``."
        ),
    )


class SVDQuantConfig(QuantizeAlgorithmConfig):
    """The config for SVDQuant.

    Refer to the `SVDQuant paper <https://arxiv.org/pdf/2411.05007>`_ for more details.
    """

    method: Literal["svdquant"] = ModeloptField("svdquant")

    lowrank: int | None = ModeloptField(
        default=32,
        title="Low-rank dimension for the SVD LoRA",
        description=(
            "Specifies the rank of the LoRA used in the SVDQuant method, "
            "which captures outliers from the original weights."
        ),
    )


class GPTQCalibConfig(QuantizeAlgorithmConfig):
    """The config for GPTQ quantization.

    GPTQ minimizes the layer-wise quantization error by using second-order (Hessian) information
    to perform blockwise weight updates that compensate for rounding loss. Layers are quantized
    sequentially so that each layer's Hessian is computed from activations that already reflect
    the quantization of preceding layers.

    The default values are taken from the official GPTQ implementation:
    https://github.com/IST-DASLab/FP-Quant/blob/d2e3092f968262c4de5fb050e1aef568a280dadd/src/quantization/gptq.py#L35
    """

    method: Literal["gptq"] = ModeloptField("gptq")
    perc_damp: float | None = ModeloptField(
        default=0.01,
        gt=0.0,
        le=1.0,
        title="Percentage damping factor.",
        description="The percentage of average Hessian diagonal used for damping.",
    )
    block_size: int | None = ModeloptField(
        default=128,
        title="Block size for GPTQ weight update.",
        description="""The block size for GPTQ weight update, which must be a multiple of the
        group_size used in the quantization.""",
    )
    fused: bool = ModeloptField(
        default=False,
        title="Use fused Triton kernel for GPTQ.",
        description="""When True, use a fused Triton kernel that combines quantization and
        per-column error propagation into one launch per GPTQ block.""",
    )


QuantizeQuantCfgType = list[QuantizerCfgEntry]

_QuantizeAlgoCfgType = str | dict | QuantizeAlgorithmConfig | None

QuantizeAlgoCfgType = _QuantizeAlgoCfgType | list[_QuantizeAlgoCfgType] | None


def normalize_quant_cfg_list(v: dict | list) -> list[QuantizerCfgEntry]:
    """Normalize a raw quant_cfg into a list of :class:`QuantizerCfgEntry` dicts.

    Supports the following input forms:

    - A ``list`` of entries in any of the per-entry forms below.
    - A legacy flat ``dict`` (``{"*": ..., "*weight_quantizer": ...}``) — each key/value pair is
      converted to a single-key dict entry and then normalized.

    Per-entry forms (when input is a list):

    - New format: ``{"quantizer_name": ..., "enable": ..., "cfg": ...}`` — passed through.
    - Legacy single-key format: ``{"<quantizer_name>": <cfg_or_dict>}`` — converted to new format.
    - Legacy ``nn.*``-scoped format: ``{"nn.<Class>": {"<quantizer_name>": <cfg>}}`` — converted
      to a new-format entry with ``parent_class`` set.

    **Validation** — an entry is rejected if it carries no instruction, i.e. it specifies neither
    ``cfg`` nor ``enable``.  Concretely, the following are invalid:

    - An empty entry ``{}``.
    - An entry with only ``quantizer_name`` and no other keys — the only effect would be an
      implicit ``enable=True``, which must be stated explicitly.
    - An entry with ``enable=True`` (explicit or implicit) whose ``cfg`` is not a non-empty
      ``dict`` or ``list`` — e.g. ``{"quantizer_name": "*", "cfg": {}}`` or
      ``{"quantizer_name": "*", "cfg": 42}``.  An enabled quantizer must have a valid
      configuration.

    **Normalization** — after conversion and validation every entry is put into canonical form:

    - ``enable`` is set to ``True`` if not explicitly specified.
    - ``cfg`` is set to ``None`` if not present in the entry.

    Every returned entry is therefore guaranteed to have the keys ``quantizer_name``, ``enable``,
    and ``cfg`` (plus optionally ``parent_class``).

    Args:
        v: A list of raw quant_cfg entries in any supported format, or a legacy flat dict.

    Returns:
        A list of :class:`QuantizerCfgEntry` dicts in canonical normalized form.

    Raises:
        ValueError: If any entry has only ``quantizer_name`` with neither ``cfg`` nor ``enable``,
            if ``enable=True`` with an empty or non-dict/list ``cfg``, or if the entry format
            is not recognized.
    """

    def _warn_legacy():
        warnings.warn(
            "Passing quant_cfg in the legacy dict format is deprecated and will be removed in "
            "a future release. Use the list-of-dicts format with explicit 'quantizer_name' "
            "keys instead. See the quant_cfg documentation for the new format and migration "
            "guide.",
            DeprecationWarning,
            stacklevel=4,
        )

    # Legacy flat-dict format: {"*": {...}, "*weight_quantizer": {...}} → list of single-key dicts.
    if isinstance(v, dict):
        _warn_legacy()
        v = [{k: val} for k, val in v.items()]

    def _dict_to_entry(key: str, value) -> list[QuantizerCfgEntry]:
        """Convert a single legacy key-value pair to one or more QuantizerCfgEntry dicts."""
        # Legacy "default" key was a catch-all applied as "*" in the old conversion code.
        if key == "default":
            key = "*"

        if isinstance(key, str) and key.startswith("nn."):
            if not isinstance(value, dict):
                raise ValueError(f"For 'nn.*' scoped format, value must be a dict, got {value!r}")
            # Support multi-key nn.*-scoped dicts by emitting one entry per sub-key.
            entries: list[QuantizerCfgEntry] = []
            for q_path, sub_cfg in value.items():
                sub_cfg = dict(sub_cfg)
                enable = sub_cfg.pop("enable", None)
                cfg = sub_cfg or None
                entry: QuantizerCfgEntry = {
                    "parent_class": key,
                    "quantizer_name": q_path,
                    "cfg": cfg,
                }
                if enable is not None:
                    entry["enable"] = enable
                entries.append(entry)
            return entries
        else:
            if isinstance(value, dict):
                cfg = {k: val for k, val in value.items() if k != "enable"} or None
                enable = value.get("enable")
            else:
                cfg = value
                enable = None
            entry = {"quantizer_name": key, "cfg": cfg}
            if enable is not None:
                entry["enable"] = enable
            return [entry]

    result: list[QuantizerCfgEntry] = []
    _warned_legacy = False
    for raw in v:
        if isinstance(raw, dict) and "quantizer_name" in raw:
            entries = [dict(raw)]  # copy to avoid mutating caller's data
        elif isinstance(raw, dict) and len(raw) == 1:
            key, val = next(iter(raw.items()))
            entries = [dict(e) for e in _dict_to_entry(key, val)]
            if not _warned_legacy:
                _warn_legacy()
                _warned_legacy = True
        elif isinstance(raw, dict) and len(raw) > 1 and any(k.startswith("nn.") for k in raw):
            # Legacy flat dict with nn.*-scoped keys mixed with other keys — expand all pairs.
            entries = []
            for k, val in raw.items():
                entries.extend(dict(e) for e in _dict_to_entry(k, val))
            if not _warned_legacy:
                _warn_legacy()
                _warned_legacy = True
        else:
            raise ValueError(f"Invalid quant_cfg entry: {raw!r}.")

        for entry in entries:
            # Validate: must carry at least one instruction beyond the path selector.
            if "cfg" not in entry and "enable" not in entry:
                raise ValueError(
                    f"Invalid quant_cfg entry: {raw!r} — each entry must specify 'cfg', 'enable', "
                    "or both. An entry with only 'quantizer_name' has no effect (implicit "
                    "enable=True is not allowed; set it explicitly)."
                )

            # Validate: when cfg is present and enable=True, cfg must be a non-empty
            # dict or list.  An empty cfg would attempt to create a
            # QuantizerAttributeConfig with no actual configuration.
            cfg = entry.get("cfg")
            enable = entry.get("enable", True)
            if enable and cfg is not None:
                if isinstance(cfg, dict):
                    is_invalid = len(cfg) == 0
                elif isinstance(cfg, list):
                    is_invalid = len(cfg) == 0 or any(
                        not isinstance(item, dict) or len(item) == 0 for item in cfg
                    )
                else:
                    is_invalid = True
                if is_invalid:
                    raise ValueError(
                        f"Invalid quant_cfg entry: {raw!r} — 'cfg' must be a non-empty dict "
                        f"or a non-empty list of non-empty dicts when enabling a quantizer "
                        f"(got {type(cfg).__name__}: {cfg!r}). Either provide quantizer "
                        "attributes in 'cfg' or remove 'cfg' and set 'enable' explicitly."
                    )

            # Normalize: make enable and cfg always explicit.
            entry.setdefault("enable", True)
            entry.setdefault("cfg", None)

            result.append(cast("QuantizerCfgEntry", entry))
    return result


class QuantizeConfig(ModeloptBaseConfig):
    """Default configuration for ``quantize`` mode."""

    quant_cfg: QuantizeQuantCfgType = ModeloptField(
        default=[{"quantizer_name": "*", "cfg": {"num_bits": 8, "axis": None}}],
        title="Quantization configuration",
        validate_default=True,
    )

    algorithm: QuantizeAlgoCfgType = ModeloptField(
        default="max",
        title="Calibration algorithm, see :meth:`calibrate <modelopt.torch.quantization.model_quant.calibrate>` "
        "for more details.",
        validate_default=True,
    )

    @field_validator("quant_cfg", mode="before")
    @classmethod
    def normalize_quant_cfg(cls, v):
        """Normalize quant_cfg entries: convert dict and tuple forms to QuantizerCfgEntry dicts."""
        if not isinstance(v, (list, dict)):
            return v
        return normalize_quant_cfg_list(v)

    @field_validator("quant_cfg", mode="after")
    @classmethod
    def validate_quant_cfg_entries(cls, v):
        """Validate quantizer attribute configs to surface errors (e.g. invalid axis/block_sizes)."""
        qac_fields = set(QuantizerAttributeConfig.model_fields.keys())
        for entry in v:
            cfg = entry.get("cfg")
            if cfg is None:
                continue
            cfgs = cfg if isinstance(cfg, list) else [cfg]
            for c in cfgs:
                if isinstance(c, dict) and qac_fields & set(c.keys()):
                    QuantizerAttributeConfig.model_validate(c)
        return v


class CompressConfig(ModeloptBaseConfig):
    """Default configuration for ``compress`` mode."""

    compress: dict[str, bool] = ModeloptField(
        default={"*": True},
        title="""Enable weight compression for the given pattern. Default is False for all weights.
        Call `compress` function to compress the model weights.""",
    )

    quant_gemm: bool = ModeloptField(
        default=True,
        title="Enable quantized GEMM.",
        description="If True, quantized GEMM compute will be enabled. Otherwise, we only do weight-only quantization.",
    )


CompressCfgType = dict[str, bool] | None | CompressConfig


class _QuantizeExportConfig(ModeloptBaseConfig):
    """An empty config."""


def need_calibration(config):
    """Check if calibration is needed for the given config."""
    if config["algorithm"] is not None and config["algorithm"] != "max":
        return True

    def _not_dynamic(cfg):
        return cfg.get("enable", True) and cfg.get("type", "") != "dynamic"

    quant_cfg: list = config.get("quant_cfg") or []
    quant_cfg = normalize_quant_cfg_list(quant_cfg)
    for entry in quant_cfg:
        name = entry["quantizer_name"]
        raw_cfg = entry.get("cfg")
        if "weight_quantizer" in name:
            # We don't calibrate weight quantizer
            continue
        # Sequential quantizers (e.g. W4A8) have a list of cfg dicts
        if isinstance(raw_cfg, list):
            for _config in raw_cfg:
                if _not_dynamic(_config):
                    return True
            continue
        cfg = dict(raw_cfg or {})
        if "enable" in entry:
            cfg["enable"] = entry["enable"]
        if _not_dynamic(cfg):
            return True

    return False
