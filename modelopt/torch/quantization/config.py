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

import re
import warnings
from collections.abc import Mapping, Sequence
from typing import Any, Literal

from pydantic import AliasChoices, Field, ValidationInfo, field_validator, model_validator

from modelopt.torch.opt.config import ModeloptBaseConfig, ModeloptField
from modelopt.torch.opt.config_loader import load_config
from modelopt.torch.utils.network import ConstructorLike


class QuantizerCfgEntry(ModeloptBaseConfig):
    """A single entry in a ``quant_cfg`` list."""

    quantizer_name: str = ModeloptField(
        default=...,
        title="Quantizer name pattern.",
        description="Glob pattern matched against quantizer module names.",
    )
    parent_class: str | None = ModeloptField(
        default=None,
        title="Optional parent-class filter.",
        description="If provided, only quantizers whose parent module matches this PyTorch class "
        "name (e.g. ``'nn.Linear'``) are affected.",
    )
    cfg: "QuantizerAttributeConfig | list[QuantizerAttributeConfig] | None" = ModeloptField(
        default=None,
        title="Quantizer attribute config.",
        description="A :class:`QuantizerAttributeConfig` (or a mapping that validates as one), "
        "or a list of such for sequential quantizers.  ``None`` leaves the existing attribute "
        "config untouched.",
    )
    enable: bool = ModeloptField(
        default=True,
        title="Enable the quantizer.",
        description="Toggle matched quantizers on/off; independent of ``cfg``.",
    )

    @model_validator(mode="before")
    @classmethod
    def _normalize_cfg_shape(cls, values):
        """Pre-validation shape rules for ``cfg``.

        Runs against the raw input mapping, before pydantic coerces ``cfg`` into a
        :class:`QuantizerAttributeConfig` (which would fill in schema defaults and erase the
        distinction between "user typed nothing" and "user typed `{}`").  Two rules:

        1. ``enable=False`` with an empty ``cfg`` — empty dict, empty list, or list of empty
           dicts — is normalized to ``cfg=None``.  Downstream applies any non-``None`` ``cfg``
           as a full quantizer-attribute replacement, so without this an entry like
           ``{cfg: {}, enable: False}`` would reset attributes to schema defaults and a later
           re-enable would bring the quantizer back with defaults instead of its original config.

        2. ``enable=True`` (explicit or implicit) with an empty ``cfg`` — same shapes — is
           rejected.  Pydantic would otherwise coerce ``{}`` into ``QuantizerAttributeConfig()``
           with all defaults, silently turning a likely typo (``cfg: {}``) into "quantize with
           schema defaults."  Callers who really want defaults should drop ``cfg`` entirely and
           rely on ``enable=True``; an empty ``cfg`` always indicates missing input.
        """
        if not isinstance(values, dict):
            return values
        cfg = values.get("cfg")
        cfg_is_empty = (isinstance(cfg, dict) and len(cfg) == 0) or (
            isinstance(cfg, list)
            and (len(cfg) == 0 or all(isinstance(item, dict) and len(item) == 0 for item in cfg))
        )
        if cfg_is_empty:
            if values.get("enable") is False:
                values = {**values, "cfg": None}
            else:
                raise ValueError(
                    f"QuantizerCfgEntry 'cfg' must specify at least one quantizer attribute; "
                    f"got an empty mapping/list for quantizer "
                    f"{values.get('quantizer_name')!r}.  To keep existing attributes, drop "
                    f"'cfg' and rely on 'enable=True'; to disable, set 'enable=False'."
                )
        return values

    @model_validator(mode="after")
    def _validate_instruction(self):
        """Reject entries that carry no instruction beyond the path selector."""
        fields_set = self.model_fields_set
        if "cfg" not in fields_set and "enable" not in fields_set:
            raise ValueError(
                f"QuantizerCfgEntry must specify 'cfg', 'enable', or both. An entry with only "
                f"'quantizer_name'={self.quantizer_name!r} has no effect (implicit enable=True "
                "is not allowed; set it explicitly)."
            )
        return self


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
        if entry.get("quantizer_name") == quantizer_name:
            result = entry
    if result is None:
        raise KeyError(f"No quant_cfg entry with quantizer_name={quantizer_name!r}")
    return result


BiasType = Literal["static", "dynamic"]
BiasMethod = Literal["mean", "max_min"]


class RotateConfig(ModeloptBaseConfig):
    """Configuration for rotating quantizer input via Hadamard transform (RHT/QuaRot/SpinQuant).

    See :func:`normalized_hadamard_transform <modelopt.torch.quantization.nn.functional.normalized_hadamard_transform>`
    for transform details.
    """

    enable: bool = ModeloptField(
        default=False,
        title="Enable input rotation.",
        description="If True, applies a normalized Hadamard transform before quantization.",
    )
    mode: Literal["rotate", "rotate_back"] = ModeloptField(
        default="rotate",
        title="Rotation mode.",
        description=(
            "Use 'rotate' for input rotation only, or 'rotate_back' to apply the transform "
            "again after fake quantization."
        ),
    )
    rotate_fp32: bool = ModeloptField(
        default=False,
        title="Run rotation in float32.",
        description="If True, computes the rotation in float32 before casting back to the input dtype.",
    )
    block_size: int | None = ModeloptField(
        default=None,
        title="Rotation block size.",
        description="Positive block size for block-wise rotation, or None to rotate the full input.",
    )

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

    effective_bits: float | None = ModeloptField(
        default=None,
        title="Effective bits per element (autoquant cost).",
        description=(
            "Per-format effective bits for the autoquant cost model; overrides the "
            "``num_bits`` heuristic for this entry (e.g. NVFP4 = 4.5). Must be in (0, 16]."
        ),
    )

    @field_validator("effective_bits")
    @classmethod
    def _validate_effective_bits(cls, v: float | None) -> float | None:
        if v is not None and not (0 < v <= 16):
            raise ValueError(f"effective_bits must be in (0, 16], got {v}")
        return v

    @model_validator(mode="before")
    @classmethod
    def validate_config(cls, values):
        """Validate quantizer config."""

        def _validate_recursive(value, field_name=None):
            """Recursively validate config structure."""
            if value is None:
                return

            if isinstance(value, list):
                for item in value:
                    _validate_recursive(item)
            elif isinstance(value, dict):
                if field_name == "rotate":
                    return
                if len(value) == 1 and "enable" in value and value["enable"] is True:
                    raise ValueError(
                        "Invalid quantizer config: Cannot specify only {'enable': True}. "
                        "Additional parameters are required when enabling quantization."
                    )
                # Recurse into nested dicts
                for k, v in value.items():
                    _validate_recursive(v, k)

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
            if k not in ["type", "scale_bits", "scale_block_sizes", "four_over_six"]
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
                assert _k in ["type", "scale_bits", "scale_block_sizes", "four_over_six"]
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


class LayerwiseConfig(ModeloptBaseConfig):
    """Nested config for layer-by-layer calibration behavior."""

    enable: bool = ModeloptField(
        default=False,
        title="Enable layerwise (layer-by-layer) calibration.",
        description=(
            "If True, the calibration algorithm is applied layer by layer. "
            "Each layer's inputs are captured via a forward pass that reflects the "
            "quantization of all preceding layers, incurring O(N) forward passes for N layers."
        ),
    )

    get_qdq_activations_from_prev_layer: bool = ModeloptField(
        default=False,
        title="Cache next-layer inputs from QDQ outputs of prior layers.",
        description=(
            "If True (GPTQ default), capture each layer's next-layer inputs "
            "after it is calibrated, so QDQ error and in-place weight updates "
            "propagate forward. If False (max/mse default), capture before, so "
            "the next layer sees the same FP activations as a non-layerwise pass."
        ),
    )

    checkpoint_dir: str | None = ModeloptField(
        default=None,
        title="Per-layer checkpoint directory (resume on restart).",
        description=(
            "If set, per-layer checkpoints are saved here during calibration. "
            "On restart, calibration resumes from the last completed layer."
        ),
    )

    save_every: int = ModeloptField(
        default=1,
        ge=1,
        title="Flush resume metadata every N layers (final layer always flushes).",
        description=(
            "Only the boundary layer of each window writes the large "
            "``next_inputs.pt`` activation cache; other per-layer files are "
            "still written for every layer (resume needs them to replay skips). "
            "Mid-window interrupts re-calibrate the unfinished window on resume."
        ),
    )


def _coerce_layerwise_input(value):
    """Normalize a raw ``layerwise`` value to a dict; warn on deprecated bool."""
    if isinstance(value, bool):
        warnings.warn(
            "Passing the layerwise field as a bool is deprecated; use a dict, "
            "e.g. `{'enable': True}`.",
            DeprecationWarning,
            stacklevel=2,
        )
        return {"enable": value}
    if value is None:
        return {}
    if isinstance(value, LayerwiseConfig):
        # ``exclude_unset=True`` so downstream ``model_fields_set`` reflects the
        # user's actual input
        return value.model_dump(exclude_unset=True)
    return value


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

    layerwise: LayerwiseConfig = Field(
        default_factory=LayerwiseConfig,
        validation_alias=AliasChoices("layerwise", "use_sequential"),
        title="Layerwise calibration configuration.",
        description=(
            "Nested config controlling layer-by-layer calibration. Pass a dict, "
            "e.g. ``{'enable': True, 'checkpoint_dir': '/path'}``. Bool input is "
            "accepted for backward compatibility but deprecated."
        ),
    )

    @model_validator(mode="before")
    @classmethod
    def _migrate_layerwise_checkpoint_dir(cls, data):
        """Merge the legacy flat ``layerwise_checkpoint_dir`` key into ``layerwise``.

        Raises if both the flat key and a nested ``checkpoint_dir`` are set with conflicting values.
        """
        if not isinstance(data, dict) or "layerwise_checkpoint_dir" not in data:
            return data
        warnings.warn(
            "Passing `layerwise_checkpoint_dir` at the top level is deprecated; "
            "nest it under `layerwise.checkpoint_dir` instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        data = dict(data)
        flat_dir = data.pop("layerwise_checkpoint_dir")
        # Resolve the legacy ``use_sequential`` alias before writing ``layerwise``,
        # otherwise the alias value is silently dropped when AliasChoices picks the
        # newly-written ``layerwise`` key over ``use_sequential``.
        raw_layerwise = data.pop("layerwise", data.pop("use_sequential", None))
        layerwise = _coerce_layerwise_input(raw_layerwise)
        existing = layerwise.get("checkpoint_dir")
        if existing is not None and existing != flat_dir:
            raise ValueError(
                f"Conflicting checkpoint_dir: layerwise_checkpoint_dir={flat_dir!r} "
                f"differs from layerwise.checkpoint_dir={existing!r}. Set only one."
            )
        data["layerwise"] = {**layerwise, "checkpoint_dir": flat_dir}
        return data

    @field_validator("layerwise", mode="before")
    @classmethod
    def _coerce_layerwise(cls, value):
        """Coerce ``layerwise=bool/None`` to dict form; also handles the alias path."""
        return _coerce_layerwise_input(value)

    @model_validator(mode="after")
    def validate_layerwise_checkpoint_dir(self):
        """Raise if layerwise.checkpoint_dir is set but layerwise.enable is False."""
        if self.layerwise.checkpoint_dir is not None and not self.layerwise.enable:
            raise ValueError(
                "layerwise.checkpoint_dir requires layerwise.enable=True. "
                "Set layerwise.enable=True or remove layerwise.checkpoint_dir."
            )
        return self


class _SharedStatesConfig(ModeloptBaseConfig):
    """The ``shared_states`` grouping knob, shared by max / mse / local_hessian calibration."""

    shared_states: dict[str, dict[str, list[str]]] | None = ModeloptField(
        default=None,
        title="Concrete shared quantization states and their grouping patterns",
        description=(
            "Optional dict keyed by shared-state name. ``'weight_global_amax'`` is implemented "
            "today and accepts ``{'patterns': [...]}``, where patterns are full-match regexes "
            "against module fully-qualified names. Omitted patterns use the state's defaults; "
            "an empty pattern list disables that state."
        ),
    )

    @field_validator("shared_states")
    @classmethod
    def validate_shared_states(cls, v):
        """Reject unknown shared-state names, fields, and invalid regexes."""
        if v is None:
            return v
        supported = {"weight_global_amax"}
        unknown = set(v) - supported
        if unknown:
            raise ValueError(
                f"shared_states has unsupported state(s) {sorted(unknown)}; "
                f"expected keys from {sorted(supported)}."
            )

        offending = ("", "")
        try:
            for name, state_cfg in v.items():
                unknown_fields = set(state_cfg) - {"patterns"}
                if unknown_fields:
                    raise ValueError(
                        f"shared_states[{name!r}] has unsupported field(s) "
                        f"{sorted(unknown_fields)}; expected ['patterns']."
                    )
                for pattern in state_cfg.get("patterns", []):
                    offending = (name, pattern)
                    re.compile(pattern)
        except re.error as e:
            bad_state, bad_pattern = offending
            raise ValueError(
                f"shared_states[{bad_state!r}]['patterns'] has an invalid regex "
                f"{bad_pattern!r}: {e}"
            ) from e
        return v


class MaxCalibConfig(_SharedStatesConfig, QuantizeAlgorithmConfig):
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
        title="Share one weight amax across local experts in a SequentialMLP MoE layer.",
        description=(
            "If True, max-calibration synchronizes the weight quantizer amax across local "
            "experts within each SequentialMLP layer, so all experts in that layer share "
            "one effective weight amax. TEGroupedMLP already fuses experts into a single "
            "GEMM with one weight quantizer, so this flag is irrelevant there."
        ),
    )


class MseCalibConfig(_SharedStatesConfig, QuantizeAlgorithmConfig):
    """Configuration for per-tensor MSE calibration.

    Finds a scale s (via amax a, with s = a / q_max) that minimizes the
    reconstruction error of a tensor after uniform Q→DQ:

        s* = argmin_s  E[(W - DQ(Q(W; s)))^2],   W ∈ weights

    When fp8_scale_sweep is enabled for a supported FP8-scale format, step_size is ignored.
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
        "Applies to ModelOpt static NVFP4 weight quantizers and registered custom backends with "
        "FP8 sweep support. Other weight quantizers use the multiplier search.",
    )

    distributed_sync: bool | None = ModeloptField(
        default=True,
        title="Whether to sync the amax across the distributed processes.",
        description="If True, the amax will be synced across the distributed processes.",
    )


class LocalHessianCalibConfig(_SharedStatesConfig, QuantizeAlgorithmConfig):
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

    skip_layers: list[str] | None = ModeloptField(
        default=None,
        title="Module-name wildcard patterns excluded from the SVDQuant algorithm",
        description=(
            "Quantized linears whose module name matches any of these fnmatch-style wildcard "
            "patterns (e.g. ``'*.attn.add_q_proj'``) keep their quantizer config but skip the "
            "SVDQuant algorithm entirely: no AWQ smoothing (``pre_quant_scale``) and no "
            "low-rank branch, leaving their weights unchanged. They are max-calibrated "
            "instead, i.e. quantized like a plain max recipe."
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

    @model_validator(mode="after")
    def _gptq_qdq_default(self):
        """Inject ``get_qdq_activations_from_prev_layer=True`` unless the user set it.

        GPTQ's Hessian correctness depends on prior-layer QDQ activations, so the
        default differs from the base class. Uses ``model_fields_set`` to detect
        whether the user explicitly set the field — covers every input shape
        (empty constructor, bool, dict) without a per-shape special case.
        """
        if "get_qdq_activations_from_prev_layer" not in self.layerwise.model_fields_set:
            self.layerwise = self.layerwise.model_copy(
                update={"get_qdq_activations_from_prev_layer": True}
            )
        return self


QuantizeQuantCfgType = list[QuantizerCfgEntry]
QuantizerCfgListConfig = QuantizeQuantCfgType

# Pre-normalization input shape: either a sequence of already-validated
# :class:`QuantizerCfgEntry` instances, or a sequence of raw mappings (any of the legacy /
# new dict forms).  Splitting the union into two ``Sequence[...]`` arms — rather than
# ``Sequence[QuantizerCfgEntry | Mapping[str, Any]]`` — keeps each arm covariant in its
# element type, so callers can pass ``list[QuantizerCfgEntry]`` or ``list[dict]`` without
# tripping invariance.
RawQuantizeQuantCfgType = Sequence[QuantizerCfgEntry] | Sequence[Mapping[str, Any]]

# Legacy flat-dict input shape (``{"*": ..., "*weight_quantizer": ...}``).  Accepted by
# ``normalize_quant_cfg_list`` for backward compatibility but emits a DeprecationWarning;
# new code should use a list of :class:`QuantizerCfgEntry`-shaped entries instead.
DeprecatedQuantCfgType = Mapping[str, Any]

_QuantizeAlgoCfgType = str | dict | QuantizeAlgorithmConfig | None

QuantizeAlgoCfgType = _QuantizeAlgoCfgType | list[_QuantizeAlgoCfgType] | None


def normalize_quant_cfg_list(
    v: RawQuantizeQuantCfgType | DeprecatedQuantCfgType,
) -> list[QuantizerCfgEntry]:
    """Normalize a raw quant_cfg into a list of :class:`QuantizerCfgEntry` instances.

    Supports the following input forms:

    - A ``list`` of entries in any of the per-entry forms below.
    - A legacy flat ``dict`` (``{"*": ..., "*weight_quantizer": ...}``) — each key/value pair is
      converted to a single-key dict entry and then normalized.

    Per-entry forms (when input is a list):

    - New format: ``{"quantizer_name": ..., "enable": ..., "cfg": ...}`` — passed through.
    - Legacy single-key format: ``{"<quantizer_name>": <cfg_or_dict>}`` — converted to new format.
    - Legacy ``nn.*``-scoped format: ``{"nn.<Class>": {"<quantizer_name>": <cfg>}}`` — converted
      to a new-format entry with ``parent_class`` set.

    Each normalized dict is then constructed into a :class:`QuantizerCfgEntry`, whose own
    validator enforces that every entry specifies ``cfg``, ``enable``, or both, and that any
    ``cfg`` for an enabled quantizer is a non-empty dict or non-empty list of non-empty dicts.

    Args:
        v: A list of raw quant_cfg entries in any supported format, or a legacy flat dict.

    Returns:
        A list of validated :class:`QuantizerCfgEntry` instances.

    Raises:
        ValueError: If any entry's shape is not recognized, or if it fails
            :class:`QuantizerCfgEntry` validation (missing instruction or invalid ``cfg``).
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
    if isinstance(v, Mapping):
        _warn_legacy()
        v = [{k: val} for k, val in v.items()]
    elif not isinstance(v, Sequence) or isinstance(v, (str, bytes)):
        raise ValueError(
            f"quant_cfg must be a sequence of entries (or a legacy flat mapping), got "
            f"{type(v).__name__}: {v!r}."
        )

    def _dict_to_entry(key: str, value) -> list[dict[str, Any]]:
        """Convert a single legacy key-value pair to one or more entry dicts."""
        # Legacy "default" key was a catch-all applied as "*" in the old conversion code.
        if key == "default":
            key = "*"

        if isinstance(key, str) and key.startswith("nn."):
            if not isinstance(value, Mapping):
                raise ValueError(
                    f"For 'nn.*' scoped format, value must be a mapping, got {value!r}"
                )
            # Support multi-key nn.*-scoped dicts by emitting one entry per sub-key.
            entries: list[dict[str, Any]] = []
            for q_path, sub_cfg in value.items():
                sub_cfg = dict(sub_cfg)
                enable = sub_cfg.pop("enable", None)
                cfg = sub_cfg or None
                entry: dict[str, Any] = {
                    "parent_class": key,
                    "quantizer_name": q_path,
                    "cfg": cfg,
                }
                if enable is not None:
                    entry["enable"] = enable
                entries.append(entry)
            return entries
        else:
            if isinstance(value, Mapping):
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
        # Already-validated QuantizerCfgEntry instances (e.g. produced by load_config on a
        # snippet schematized with `# modelopt-schema: QuantizerCfgEntry`, then spread into
        # a quant_cfg list) are passed through unchanged.
        if isinstance(raw, QuantizerCfgEntry):
            result.append(raw)
            continue
        if isinstance(raw, Mapping) and "quantizer_name" in raw:
            entries: list[dict[str, Any]] = [dict(raw)]  # copy to avoid mutating caller's data
        elif isinstance(raw, Mapping) and len(raw) == 1:
            key, val = next(iter(raw.items()))
            entries = [dict(e) for e in _dict_to_entry(key, val)]
            if not _warned_legacy:
                _warn_legacy()
                _warned_legacy = True
        elif isinstance(raw, Mapping) and len(raw) > 1 and any(k.startswith("nn.") for k in raw):
            # Legacy flat dict with nn.*-scoped keys mixed with other keys — expand all pairs.
            entries = []
            for k, val in raw.items():
                entries.extend(dict(e) for e in _dict_to_entry(k, val))
            if not _warned_legacy:
                _warn_legacy()
                _warned_legacy = True
        else:
            raise ValueError(f"Invalid quant_cfg entry: {raw!r}.")

        # Constructing each QuantizerCfgEntry runs its model_validator, which enforces the
        # at-least-one-of('cfg', 'enable') and cfg-shape constraints. Defaults for absent
        # 'cfg' / 'enable' are filled by the pydantic field defaults.
        result.extend(QuantizerCfgEntry(**entry) for entry in entries)
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

    effective_bits: float | None = ModeloptField(
        default=None,
        title="Effective bits per element (autoquant cost override)",
        description=(
            "Recipe-level override for the autoquant cost model; replaces the per-entry "
            "``num_bits`` heuristic for the whole config. Must be in (0, 16]."
        ),
    )

    @field_validator("effective_bits")
    @classmethod
    def _validate_effective_bits(cls, v: float | None) -> float | None:
        if v is not None and not (0 < v <= 16):
            raise ValueError(f"effective_bits must be in (0, 16], got {v}")
        return v

    @field_validator("quant_cfg", mode="before")
    @classmethod
    def normalize_quant_cfg(
        cls, v: RawQuantizeQuantCfgType | DeprecatedQuantCfgType
    ) -> QuantizeQuantCfgType:
        """Normalize raw quant_cfg input into a ``list[QuantizerCfgEntry]``.

        Delegates to :func:`normalize_quant_cfg_list`, which accepts every supported input
        shape (new-format list, legacy single-key-dict list, legacy flat dict, and lists
        containing already-validated ``QuantizerCfgEntry`` instances) and rejects anything
        else with a clear ``ValueError`` before pydantic's field-type check would see it.
        """
        return normalize_quant_cfg_list(v)


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


def _load_quantizer_attribute_dict(config_path: str) -> dict[str, Any]:
    """Load a schema-backed QuantizerAttributeConfig YAML as a public dict."""
    config = load_config(config_path, schema_type=QuantizerAttributeConfig)
    if isinstance(config, QuantizerAttributeConfig):
        return config.model_dump(exclude_unset=True)
    if isinstance(config, Mapping):
        return dict(config)
    raise TypeError(f"{config_path} must declare QuantizerAttributeConfig.")


def _load_quantize_config_dict(config_path: str) -> dict[str, Any]:
    """Load a schema-backed QuantizeConfig YAML as a public legacy-shape dict."""
    return load_config(config_path, schema_type=QuantizeConfig).model_dump(exclude_unset=True)


def _load_quantizer_cfg_dict_list(config_path: str) -> list[dict[str, Any]]:
    """Load a QuantizerCfgEntry or QuantizerCfgListConfig snippet as public dict entries."""
    config = load_config(config_path)
    entries = config if isinstance(config, list) else [config]
    return [e.model_dump(exclude_unset=True) for e in entries]


_base_disable_all: list[dict[str, Any]] = _load_quantizer_cfg_dict_list(
    "configs/ptq/units/base_disable_all"
)

_default_disabled_quantizer_cfg: list[dict[str, Any]] = _load_quantizer_cfg_dict_list(
    "configs/ptq/units/default_disabled_quantizers"
)

_mamba_moe_disabled_quantizer_cfg: list[dict[str, Any]] = _load_quantizer_cfg_dict_list(
    "configs/ptq/units/mamba_moe_disabled_quantizers"
)

_nvfp4_cfg: dict[str, Any] = _load_quantizer_attribute_dict("configs/numerics/nvfp4")

_nvfp4_cfg_bs32: dict[str, Any] = _load_quantizer_attribute_dict("configs/numerics/nvfp4_bs32")

INT8_DEFAULT_CFG: dict[str, Any] = _load_quantize_config_dict("configs/ptq/presets/model/int8")
INT8_SMOOTHQUANT_CFG: dict[str, Any] = _load_quantize_config_dict(
    "configs/ptq/presets/model/int8_smoothquant"
)
INT8_WEIGHT_ONLY_CFG: dict[str, Any] = _load_quantize_config_dict(
    "configs/ptq/presets/model/int8_weight_only"
)
FP8_DEFAULT_CFG: dict[str, Any] = _load_quantize_config_dict("configs/ptq/presets/model/fp8")
MAMBA_MOE_FP8_AGGRESSIVE_CFG: dict[str, Any] = _load_quantize_config_dict(
    "configs/ptq/presets/model/mamba_moe_fp8_aggressive"
)
MAMBA_MOE_FP8_CONSERVATIVE_CFG: dict[str, Any] = _load_quantize_config_dict(
    "configs/ptq/presets/model/mamba_moe_fp8_conservative"
)
FP8_PER_CHANNEL_PER_TOKEN_CFG: dict[str, Any] = _load_quantize_config_dict(
    "configs/ptq/presets/model/fp8_per_channel_per_token"
)
FP8_2D_BLOCKWISE_WEIGHT_ONLY_CFG: dict[str, Any] = _load_quantize_config_dict(
    "configs/ptq/presets/model/fp8_2d_blockwise_weight_only"
)
INT4_BLOCKWISE_WEIGHT_ONLY_CFG: dict[str, Any] = _load_quantize_config_dict(
    "configs/ptq/presets/model/int4_blockwise_weight_only"
)
INT4_AWQ_CFG: dict[str, Any] = _load_quantize_config_dict("configs/ptq/presets/model/int4_awq")
W4A8_AWQ_BETA_CFG: dict[str, Any] = _load_quantize_config_dict(
    "configs/ptq/presets/model/w4a8_awq_beta"
)
MXFP8_DEFAULT_CFG: dict[str, Any] = _load_quantize_config_dict("configs/ptq/presets/model/mxfp8")
MXFP6_DEFAULT_CFG: dict[str, Any] = _load_quantize_config_dict("configs/ptq/presets/model/mxfp6")
MXFP4_DEFAULT_CFG: dict[str, Any] = _load_quantize_config_dict("configs/ptq/presets/model/mxfp4")
W4A8_MXFP4_FP8_CFG: dict[str, Any] = _load_quantize_config_dict(
    "configs/ptq/presets/model/w4a8_mxfp4_fp8"
)
MXINT8_DEFAULT_CFG: dict[str, Any] = _load_quantize_config_dict("configs/ptq/presets/model/mxint8")

# KV-cache configs are designed to be merged with a primary quantization config (e.g.
# FP8_DEFAULT_CFG) that already contains _base_disable_all.  They intentionally omit both
# _base_disable_all and "algorithm" because these are provided by the primary config.
FP8_KV_CFG: dict[str, Any] = _load_quantize_config_dict("configs/ptq/presets/kv/fp8")
FP8_AFFINE_KV_CFG: dict[str, Any] = _load_quantize_config_dict("configs/ptq/presets/kv/fp8_affine")

NVFP4_DEFAULT_CFG: dict[str, Any] = _load_quantize_config_dict("configs/ptq/presets/model/nvfp4")
NVFP4_FOUR_OVER_SIX_CFG: dict[str, Any] = _load_quantize_config_dict(
    "configs/ptq/presets/model/nvfp4_four_over_six"
)
NVFP4_W4A4_WEIGHT_MSE_FP8_SWEEP_CFG: dict[str, Any] = _load_quantize_config_dict(
    "configs/ptq/presets/model/nvfp4_w4a4_weight_mse_fp8_sweep"
)
NVFP4_W4A4_WEIGHT_LOCAL_HESSIAN_CFG: dict[str, Any] = _load_quantize_config_dict(
    "configs/ptq/presets/model/nvfp4_w4a4_weight_local_hessian"
)
MAMBA_MOE_NVFP4_AGGRESSIVE_CFG: dict[str, Any] = _load_quantize_config_dict(
    "configs/ptq/presets/model/mamba_moe_nvfp4_aggressive"
)
MAMBA_MOE_NVFP4_CONSERVATIVE_CFG: dict[str, Any] = _load_quantize_config_dict(
    "configs/ptq/presets/model/mamba_moe_nvfp4_conservative"
)
NVFP4_AWQ_LITE_CFG: dict[str, Any] = _load_quantize_config_dict(
    "configs/ptq/presets/model/nvfp4_awq_lite"
)
NVFP4_AWQ_CLIP_CFG: dict[str, Any] = _load_quantize_config_dict(
    "configs/ptq/presets/model/nvfp4_awq_clip"
)
NVFP4_AWQ_FULL_CFG: dict[str, Any] = _load_quantize_config_dict(
    "configs/ptq/presets/model/nvfp4_awq_full"
)
NVFP4_AFFINE_KV_CFG: dict[str, Any] = _load_quantize_config_dict(
    "configs/ptq/presets/kv/nvfp4_affine"
)
NVFP4_KV_CFG: dict[str, Any] = _load_quantize_config_dict("configs/ptq/presets/kv/nvfp4")
NVFP4_FP8_MHA_CONFIG: dict[str, Any] = _load_quantize_config_dict(
    "configs/ptq/presets/model/nvfp4_fp8_mha"
)
NVFP4_KV_ROTATE_CFG: dict[str, Any] = _load_quantize_config_dict(
    "configs/ptq/presets/kv/nvfp4_rotate"
)
NVFP4_SVDQUANT_DEFAULT_CFG: dict[str, Any] = _load_quantize_config_dict(
    "configs/ptq/presets/model/nvfp4_svdquant"
)
W4A8_NVFP4_FP8_CFG: dict[str, Any] = _load_quantize_config_dict(
    "configs/ptq/presets/model/w4a8_nvfp4_fp8"
)
W4A16_NVFP4_CFG: dict[str, Any] = _load_quantize_config_dict(
    "configs/ptq/presets/model/w4a16_nvfp4"
)
MXFP4_MLP_WEIGHT_ONLY_CFG: dict[str, Any] = _load_quantize_config_dict(
    "configs/ptq/presets/model/mxfp4_mlp_weight_only"
)
NVFP4_MLP_WEIGHT_ONLY_CFG: dict[str, Any] = _load_quantize_config_dict(
    "configs/ptq/presets/model/nvfp4_mlp_weight_only"
)
NVFP4_EXPERTS_ONLY_CFG: dict[str, Any] = _load_quantize_config_dict(
    "configs/ptq/presets/model/nvfp4_experts_only"
)
NVFP4_MLP_ONLY_CFG: dict[str, Any] = _load_quantize_config_dict(
    "configs/ptq/presets/model/nvfp4_mlp_only"
)
NVFP4_OMLP_ONLY_CFG: dict[str, Any] = _load_quantize_config_dict(
    "configs/ptq/presets/model/nvfp4_omlp_only"
)

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
    "MXFP6_DEFAULT_CFG",
    "MXFP8_DEFAULT_CFG",
    "MXINT8_DEFAULT_CFG",
    "NVFP4_AFFINE_KV_CFG",
    "NVFP4_AWQ_CLIP_CFG",
    "NVFP4_AWQ_FULL_CFG",
    "NVFP4_AWQ_LITE_CFG",
    "NVFP4_DEFAULT_CFG",
    "NVFP4_FOUR_OVER_SIX_CFG",
    "NVFP4_FP8_MHA_CONFIG",
    "NVFP4_KV_CFG",
    "NVFP4_KV_ROTATE_CFG",
    "W4A16_NVFP4_CFG",
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
    "NVFP4_W4A4_WEIGHT_LOCAL_HESSIAN_CFG",
    "NVFP4_W4A4_WEIGHT_MSE_FP8_SWEEP_CFG",
}


def need_calibration(config: QuantizeConfig | Mapping[str, Any]) -> bool:
    """Check if calibration is needed for the given config."""
    if config["algorithm"] is not None and config["algorithm"] != "max":
        return True

    def _not_dynamic(cfg):
        return cfg.get("enable", True) and cfg.get("type", "") != "dynamic"

    raw_quant_cfg: RawQuantizeQuantCfgType | DeprecatedQuantCfgType = config.get("quant_cfg") or []
    quant_cfg: list[QuantizerCfgEntry] = normalize_quant_cfg_list(raw_quant_cfg)
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
