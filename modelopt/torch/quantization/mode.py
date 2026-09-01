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

"""This module contains the mode descriptor for the quantization mode."""

from abc import abstractmethod
from collections.abc import Callable

from modelopt.torch.opt.config import ModeloptBaseConfig
from modelopt.torch.opt.conversion import ModelLikeModule
from modelopt.torch.opt.mode import (
    ConvertEntrypoint,
    ConvertReturnType,
    ModeConfigList,
    ModeDescriptor,
    RestoreEntrypoint,
    UpdateEntrypoint,
    _ModeRegistryCls,
)
from modelopt.torch.opt.searcher import ForwardLoop

from .algo_cfg import compile_algo_cfg, derive_handoff, describe_plan, plan_hash, stage_predicate
from .compress import compress_convert, compress_restore, update_compress_metadata
from .config import (
    AWQClipCalibConfig,
    AWQFullCalibConfig,
    AWQLiteCalibConfig,
    CalibrationPlanConfig,
    CompressConfig,
    GPTQCalibConfig,
    LocalHessianCalibConfig,
    LSQConfig,
    MaxCalibConfig,
    MseCalibConfig,
    NVFP4ActHeadroomCalibConfig,
    QuantizeAlgoCfgType,
    QuantizeAlgorithmConfig,
    QuantizeConfig,
    SmoothQuantCalibConfig,
    SVDQuantConfig,
    _QuantizeExportConfig,
)
from .conversion import (
    convert_to_quantized_model,
    export_quantized_model,
    restore_export_quantized_model,
    restore_quantized_model,
    restore_quantizer_state,
    restore_svdquant_model,
    update_quantize_metadata,
)
from .model_calib import (
    awq,
    gptq,
    layerwise_calibrate,
    local_hessian_calibrate,
    lsq,
    max_calibrate,
    mse_calibrate,
    nvfp4_act_headroom_calibrate,
    smoothquant,
    svdquant,
)
from .utils import print_rank_0

__all__ = ["BaseCalibrateModeDescriptor"]

QuantizeModeRegistry = _ModeRegistryCls("quantization")


# TODO: OMNIML-717 Reuse search infra for quantization calibration algorithms
@QuantizeModeRegistry.register_mode
class QuantizeModeDescriptor(ModeDescriptor):
    """Class to describe the ``"quant"`` mode.

    The properties of this mode can be inspected via the source code.
    """

    @property
    def name(self) -> str:
        """Returns the value (str representation) of the mode."""
        return "quantize"

    @property
    def config_class(self) -> type[ModeloptBaseConfig]:
        """Specifies the config class for the mode."""
        return QuantizeConfig

    @property
    def next_prohibited_modes(self) -> set[str] | None:
        """Modes that should not be applied after this mode."""
        return {"sparsity", "autonas", "fastnas"}

    @property
    def export_mode(self) -> str | None:
        """The mode that corresponds to the export mode of this mode."""
        return "export_quantize"

    @property
    def convert(self) -> ConvertEntrypoint:
        """The mode's entrypoint for converting a model."""
        return convert_to_quantized_model

    @property
    def restore(self) -> RestoreEntrypoint:
        """The mode's entrypoint for restoring a model."""
        return restore_quantized_model

    @property
    def update_for_save(self) -> UpdateEntrypoint:
        """The mode's entrypoint for updating the models state before saving."""
        return update_quantize_metadata

    @property
    def update_for_new_mode(self) -> UpdateEntrypoint:
        """The mode's entrypoint for updating the models state before new mode."""
        return update_quantize_metadata


@QuantizeModeRegistry.register_mode
class QuantizeExportModeDescriptor(ModeDescriptor):
    """Class to describe the export of quantization mode.

    Note that this mode is just a placeholder to throw an error since we don't support exporting
    quantized models right now. It is used to properly indicate that the ``quantize`` mode does
    require an export mode if we ever wanted to do chaining/stacking of modes with it.
    """

    @property
    def name(self) -> str:
        """Returns the value (str representation) of the mode."""
        return "quantize_export"

    @property
    def config_class(self) -> type[ModeloptBaseConfig]:
        """Specifies the config class for the mode."""
        return _QuantizeExportConfig

    @property
    def is_export_mode(self) -> bool:
        """Specifies whether the mode is an export mode."""
        return True

    @property
    def convert(self) -> ConvertEntrypoint:
        """The mode's entrypoint for converting a model."""
        return export_quantized_model

    @property
    def restore(self) -> RestoreEntrypoint:
        """The mode's entrypoint for restoring a model."""
        return restore_export_quantized_model


@QuantizeModeRegistry.register_mode
class RealQuantizeModeDescriptor(ModeDescriptor):
    """Mode for real quantization."""

    @property
    def name(self) -> str:
        """Returns the value (str representation) of the mode."""
        return "real_quantize"

    @property
    def next_modes(self) -> set[str] | None:
        """Real quantization should be the last mode in the chain."""
        # TODO: update this to support QLoRA
        return {"max_calibrate", "calibration_plan", "eagle"}

    @property
    def config_class(self) -> type[ModeloptBaseConfig]:
        """Specifies the config class for the mode."""
        return CompressConfig

    @property
    def convert(self) -> ConvertEntrypoint:
        """The mode's entrypoint for converting a model."""
        return compress_convert

    @property
    def restore(self) -> RestoreEntrypoint:
        """The mode's entrypoint for restoring a model."""
        return compress_restore

    @property
    def update_for_save(self) -> UpdateEntrypoint:
        """The mode's entrypoint for updating the models state before saving."""
        return update_compress_metadata

    @property
    def update_for_new_mode(self) -> UpdateEntrypoint:
        """The mode's entrypoint for updating the models state before new mode."""
        return update_compress_metadata


@QuantizeModeRegistry.register_mode
class AutoQuantizeModeDescriptor(QuantizeModeDescriptor):
    """Mode for autoquantize."""

    @property
    def name(self) -> str:
        """Returns the value (str representation) of the mode."""
        return "auto_quantize"


def wrapped_calib_func(
    model: ModelLikeModule,
    config: QuantizeAlgorithmConfig,
    forward_loop: ForwardLoop | None = None,
    func: Callable | None = None,
    supports_layerwise: bool = True,
    should_process: Callable[[str], bool] | None = None,
) -> ConvertReturnType:
    """Wrap the calibration function to be compatible with the ModelOpt convert entrypoint.

    The calibration algorithms in ..model_calib.py are designed to be called directly with the model,
    forward_loop and the relevant kwargs and are independent of the ModelOpt framework.
    So lets wrap them to be compatible with the ModelOpt convert entrypoint.
    """
    kwargs = config.model_dump()
    method = kwargs.pop("method")
    layerwise_cfg = kwargs.pop("layerwise", None) or {}
    layerwise = layerwise_cfg.get("enable", False)
    checkpoint_dir = layerwise_cfg.get("checkpoint_dir")
    export_dir = layerwise_cfg.get("export_dir")
    qdq_from_prev = layerwise_cfg.get("get_qdq_activations_from_prev_layer", False)
    save_every = layerwise_cfg.get("save_every", 1)
    calib_mutates_weights = layerwise_cfg.get("calib_mutates_weights", True)
    if method is not None and "awq" in method:
        # For backward compatibility
        kwargs["algorithm"] = method

    # The scoping write-mask (see `algo_cfg.stage_predicate`). `None` means "whole model",
    # which is exactly today's behaviour, so it is not forwarded in that case.
    if should_process is not None:
        kwargs["should_process"] = should_process

    moe_calib_experts_ratio = kwargs.pop("moe_calib_experts_ratio", None)
    if moe_calib_experts_ratio is not None:
        assert (
            isinstance(moe_calib_experts_ratio, (int, float)) and 0 < moe_calib_experts_ratio <= 1
        ), f"Invalid moe_calib_experts_ratio {moe_calib_experts_ratio!r}"
        for module in model.modules():
            if hasattr(module, "_moe_calib_experts_ratio"):
                module._moe_calib_experts_ratio = moe_calib_experts_ratio

    if func is not None:
        if layerwise:
            # All currently implemented PTQ algorithms support layerwise calibration;
            # future algorithms that need full-model context must add a guard here.
            if not supports_layerwise:
                raise ValueError(
                    f"Calibration algorithm '{method}' does not support layerwise.enable=True. "
                    "Set layerwise.enable=False, or override `_supports_layerwise = True` on the "
                    "corresponding CalibrateModeDescriptor once the algorithm is made "
                    "compatible with per-layer calibration."
                )
            if forward_loop is None:
                raise ValueError("forward_loop is required for calibration but got None.")
            # Wrap with layerwise processing
            layerwise_calibrate(
                model,
                forward_loop=forward_loop,
                calib_func=func,
                checkpoint_dir=checkpoint_dir,
                export_dir=export_dir,
                get_qdq_activations_from_prev_layer=qdq_from_prev,
                save_every=save_every,
                calib_mutates_weights=calib_mutates_weights,
                **kwargs,
            )
        else:
            # Direct calibration (existing behavior)
            func(model, forward_loop=forward_loop, **kwargs)

    # Lets get the latest metadata for the quantizer states
    metadata = {}
    update_quantize_metadata(model, config, metadata)
    return model, metadata


class BaseCalibrateModeDescriptor(ModeDescriptor):
    """Base class for quantization calibration algorithm modes.

    All calibration algorithm modes must be derived from this base class.
    In addition, the `config_class` for the mode must return a subclass of :class:`QuantizeAlgorithmConfig`.

    This base class also provides some convenient wrappers/utilities for calibration algorithms to be
    translated into ModelOpt mode.

    It includes:
        1. A utility to convert the algorithm name to a mode name. This is useful since many algorithm names
            are trivial and not a good fit as a mode name. For example, ``"max"`` or ``None``.
        2. Conversion of the ``algorithm`` and ``kwargs`` arguments of
            :meth:`calibrate <modelopt.torch.quantization.model_quant.calibrate>` API to a mode config
            list compatible with :meth:`apply_mode <modelopt.torch.opt.conversion.apply_mode>`.
        3. Wrapper for the calibration functions in :mod:`modelopt.torch.quantization.model_calib` to be
            compatible with the ModelOpt convert entrypoint.
    """

    _calib_func: Callable | None

    # Override to False when the algorithm requires full-model context and
    # cannot run per decoder layer (e.g. needs ModeloptStateManager on the root).
    _supports_layerwise: bool = True

    def __init__(self, *args, **kwargs):
        """Initialize Base calibrate mode descriptor."""
        assert issubclass(self.config_class, QuantizeAlgorithmConfig), (
            f"`config_class` of {self.__class__} must be a subclass of `QuantizeAlgorithmConfig`!, "
            f"got {self.config_class}!"
        )
        super().__init__(*args, **kwargs)

    @classmethod
    def _get_mode_name(cls, algo_name: str | None = None, check: bool = False) -> str:
        mode_name = algo_name + "_calibrate" if algo_name else "_no_calibrate"
        if check:
            assert mode_name in CalibrateModeRegistry, (
                f"Algorithm {algo_name} not found in CalibrateModeRegistry!"
            )
        return mode_name

    @property
    def name(self) -> str:
        """Returns the value (str representation) of the mode."""
        return self._get_mode_name(self.config_class().method)

    @property
    @abstractmethod
    def config_class(self) -> type[QuantizeAlgorithmConfig]:
        """Specifies the config class for the mode."""

    @property
    def convert(self) -> ConvertEntrypoint:
        """The calibrate algorithm mode's entrypoint for converting a model.

        This method is called by the ModelOpt framework when applying this calibration mode to a model.
        See :meth:`wrapped_calib_func` for more details on the logic.

        Note: Subclasses must specify the `_calib_func` class attribute with the appropriate
        calibration function to be used or override this method.
        """
        assert hasattr(self.__class__, "_calib_func"), (
            f"Calibration function '_calib_func' not defined for {self.__class__}, "
            "either define it or override the `convert` method!"
        )

        def wrapped_func(model, config, forward_loop=None):
            # Access _calib_func as a class attribute to avoid binding
            # Check if _calib_func is defined as a class attribute
            return wrapped_calib_func(
                model,
                config,
                forward_loop,
                func=self.__class__._calib_func,
                supports_layerwise=self.__class__._supports_layerwise,
            )

        return wrapped_func

    @property
    def restore(self) -> RestoreEntrypoint:
        """The mode's entrypoint for restoring a model."""
        return restore_quantizer_state

    @property
    def update_for_save(self) -> UpdateEntrypoint:
        """The mode's entrypoint for updating the models state before saving."""
        return update_quantize_metadata

    @property
    def update_for_new_mode(self) -> UpdateEntrypoint:
        """The mode's entrypoint for updating the models state before new mode."""
        return update_quantize_metadata


def get_modelike_from_algo_cfg(algo_cfg: QuantizeAlgoCfgType) -> ModeConfigList:
    """Get the mode like from the algorithm config."""
    if isinstance(algo_cfg, list):
        assert not any(isinstance(c, list) for c in algo_cfg), (
            f"Nested lists received as config! config: {algo_cfg}"
        )
        return [get_modelike_from_algo_cfg(c)[0] for c in algo_cfg]
    if isinstance(algo_cfg, QuantizeAlgorithmConfig):
        algo_cfg = algo_cfg.model_dump()
    if algo_cfg is None or isinstance(algo_cfg, str):
        algo_name, algo_cfg = algo_cfg, {}
    elif isinstance(algo_cfg, dict):
        algo_name = algo_cfg["method"]
    else:
        raise ValueError(f"Invalid config type: {type(algo_cfg)}")
    return [(BaseCalibrateModeDescriptor._get_mode_name(algo_name, check=True), algo_cfg)]


class _CalibrateModeRegistryCls(_ModeRegistryCls):
    def register_mode(self, cls_descriptor: type[_ModeRegistryCls.T]) -> type[_ModeRegistryCls.T]:
        """Register a new mode with the given descriptor."""
        assert issubclass(cls_descriptor, BaseCalibrateModeDescriptor), (
            f"Mode descriptor for `_CalibrateModeRegistryCls` must be a subclass of `BaseCalibrateModeDescriptor`! "
            f"Got: {cls_descriptor}"
        )
        return super().register_mode(cls_descriptor)


CalibrateModeRegistry = _CalibrateModeRegistryCls("calibrate_algos")


@CalibrateModeRegistry.register_mode
class NoneCalibrateModeDescriptor(BaseCalibrateModeDescriptor):
    """Mode for no calibration algorithm."""

    @property
    def config_class(self) -> type[QuantizeAlgorithmConfig]:
        """Specifies the config class for the mode."""
        return QuantizeAlgorithmConfig

    _calib_func = None


@CalibrateModeRegistry.register_mode
class MaxCalibrateModeDescriptor(BaseCalibrateModeDescriptor):
    """Mode for max calibration algorithm."""

    @property
    def config_class(self) -> type[QuantizeAlgorithmConfig]:
        """Specifies the config class for the mode."""
        return MaxCalibConfig

    _calib_func = max_calibrate


@CalibrateModeRegistry.register_mode
class NVFP4ActHeadroomCalibrateModeDescriptor(BaseCalibrateModeDescriptor):
    """Mode for the ``nvfp4_act_headroom`` calibration algorithm.

    Headroom-aware global scales for NVFP4 activation quantizers; plain max for everything
    else (see :class:`NVFP4ActHeadroomCalibConfig
    <modelopt.torch.quantization.config.NVFP4ActHeadroomCalibConfig>`).
    """

    @property
    def config_class(self) -> type[QuantizeAlgorithmConfig]:
        """Specifies the config class for the mode."""
        return NVFP4ActHeadroomCalibConfig

    _calib_func = nvfp4_act_headroom_calibrate


@CalibrateModeRegistry.register_mode
class MseCalibrateModeDescriptor(BaseCalibrateModeDescriptor):
    """Mode for mse calibration algorithm."""

    @property
    def config_class(self) -> type[QuantizeAlgorithmConfig]:
        """Specifies the config class for the mode."""
        return MseCalibConfig

    _calib_func = mse_calibrate


@CalibrateModeRegistry.register_mode
class LocalHessianModeDescriptor(BaseCalibrateModeDescriptor):
    """Mode for local Hessian-weighted MSE calibration algorithm.

    This algorithm uses activation information to optimize per-block scales for weight
    quantization by minimizing output reconstruction error instead of weight reconstruction error.
    """

    @property
    def config_class(self) -> type[QuantizeAlgorithmConfig]:
        """Specifies the config class for the mode."""
        return LocalHessianCalibConfig

    _calib_func = local_hessian_calibrate


@CalibrateModeRegistry.register_mode
class SmoothQuantModeDescriptor(BaseCalibrateModeDescriptor):
    """Mode for smoothquant calibration algorithm."""

    @property
    def config_class(self) -> type[QuantizeAlgorithmConfig]:
        """Specifies the config class for the mode."""
        return SmoothQuantCalibConfig

    _calib_func = smoothquant


@CalibrateModeRegistry.register_mode
class AWQLiteModeDescriptor(BaseCalibrateModeDescriptor):
    """Mode for AWQ lite calibration algorithm."""

    @property
    def config_class(self) -> type[QuantizeAlgorithmConfig]:
        """Specifies the config class for the mode."""
        return AWQLiteCalibConfig

    _calib_func = awq


@CalibrateModeRegistry.register_mode
class AWQClipModeDescriptor(BaseCalibrateModeDescriptor):
    """Mode for AWQ clip calibration algorithm."""

    @property
    def config_class(self) -> type[QuantizeAlgorithmConfig]:
        """Specifies the config class for the mode."""
        return AWQClipCalibConfig

    _calib_func = awq


@CalibrateModeRegistry.register_mode
class AWQFullModeDescriptor(BaseCalibrateModeDescriptor):
    """Mode for AWQ full calibration algorithm."""

    @property
    def config_class(self) -> type[QuantizeAlgorithmConfig]:
        """Specifies the config class for the mode."""
        return AWQFullCalibConfig

    _calib_func = awq


@CalibrateModeRegistry.register_mode
class SVDQuantModeDescriptor(BaseCalibrateModeDescriptor):
    """Mode for SVDQuant calibration algorithm."""

    @property
    def config_class(self) -> type[QuantizeAlgorithmConfig]:
        """Specifies the config class for the mode."""
        return SVDQuantConfig

    _calib_func = svdquant
    # create_and_replace_svdquant_linear_on_the_fly reads ModeloptStateManager from the
    # root model, which is not present when layerwise_calibrate dispatches per decoder layer.
    _supports_layerwise = False

    @property
    def restore(self) -> RestoreEntrypoint:
        """The mode's entrypoint for restoring a model."""
        return restore_svdquant_model


@CalibrateModeRegistry.register_mode
class GPTQModeDescriptor(BaseCalibrateModeDescriptor):
    """Mode for GPTQ calibration algorithm."""

    @property
    def config_class(self) -> type[QuantizeAlgorithmConfig]:
        """Specifies the config class for the mode."""
        return GPTQCalibConfig

    _calib_func = gptq


def calibration_plan_convert(
    model: ModelLikeModule,
    config: CalibrationPlanConfig,
    forward_loop: ForwardLoop | None = None,
) -> ConvertReturnType:
    """Run a compiled calibration plan and record it as a single mode.

    Compile (pure) then execute (effectful):

    1. :func:`compile_algo_cfg <modelopt.torch.quantization.algo_cfg.compile_algo_cfg>` lowers
       ``algo_cfg`` + ``algorithm`` into one ordered stage list and validates it against the
       model structure.
    2. Each stage runs in order through the same :func:`wrapped_calib_func` the whole-model
       algorithms use, with a ``should_process`` write-mask built from the stage's scope and
       any handoff kwargs implied by what earlier stages produced.

    Stages are serial in this phase — each runs its own forward.  Batching independent
    stages onto a shared forward is a later optimization the declared capabilities already
    carry enough information for (``shareable_forward``).
    """
    plan = compile_algo_cfg(
        {"algo_cfg": config.algo_cfg, "algorithm": config.algorithm},
        model,
        strict=config.strict,
    )
    print_rank_0(
        f"calibration_plan: {len(plan)} stage(s), hash {plan_hash(plan)}\n"
        + describe_plan(plan, model)
    )

    for i, stage in enumerate(plan):
        if stage.algo is None:
            continue
        descriptor = CalibrateModeRegistry[
            BaseCalibrateModeDescriptor._get_mode_name(stage.algo, check=True)
        ]
        stage_kwargs = {**stage.cfg, **derive_handoff(model, plan, i)}
        stage_config = descriptor.config_class(**stage_kwargs)
        wrapped_calib_func(
            model,
            stage_config,
            forward_loop,
            func=type(descriptor)._calib_func,
            supports_layerwise=type(descriptor)._supports_layerwise,
            should_process=stage_predicate(model, stage),
        )

    metadata = {}
    update_quantize_metadata(model, config, metadata)
    return model, metadata


@CalibrateModeRegistry.register_mode
class CalibrationPlanModeDescriptor(BaseCalibrateModeDescriptor):
    """Mode for a compiled, scoped calibration plan.

    One mode covers an arbitrary number of stages: recording one mode per stage would
    bloat the saved state (``auto_quantize`` would emit hundreds).  Restore needs nothing
    algorithm-specific — the generic quantizer-state snapshot already captures amax,
    pre_quant_scale, num_bits and friends.
    """

    _calib_func = None

    @property
    def name(self) -> str:
        """Returns the value (str representation) of the mode."""
        return "calibration_plan"

    @property
    def config_class(self) -> type[QuantizeAlgorithmConfig]:
        """Specifies the config class for the mode."""
        return CalibrationPlanConfig

    @property
    def convert(self) -> ConvertEntrypoint:
        """The mode's entrypoint for converting a model."""
        return calibration_plan_convert


@CalibrateModeRegistry.register_mode
class LSQModeDescriptor(BaseCalibrateModeDescriptor):
    """Mode for LSQ (Learned Scale Quantization) algorithm."""

    @property
    def config_class(self) -> type[QuantizeAlgorithmConfig]:
        """Specifies the config class for the mode."""
        return LSQConfig

    _calib_func = lsq
