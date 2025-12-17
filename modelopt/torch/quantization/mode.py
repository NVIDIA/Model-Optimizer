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

from .compress import compress_convert, compress_restore, update_compress_metadata
from .config import (
    AWQClipCalibConfig,
    AWQFullCalibConfig,
    AWQLiteCalibConfig,
    CompressConfig,
    GPTQLiteConfig,
    MaxCalibConfig,
    MseCalibConfig,
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
from .model_calib import awq, gptq_lite, max_calibrate, mse_calibrate, smoothquant, svdquant

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
        return {"sparsity", "autonas", "fastnas", "gradnas"}

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
        return {"max_calibrate", "eagle"}

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
) -> ConvertReturnType:
    """Wrap the calibration function to be compatible with the ModelOpt convert entrypoint.

    The calibration algorithms in ..model_calib.py are designed to be called directly with the model,
    forward_loop and the relevant kwargs and are independent of the ModelOpt framework.
    So lets wrap them to be compatible with the ModelOpt convert entrypoint.
    """
    kwargs = config.model_dump()
    method = kwargs.pop("method")
    if method is not None and "awq" in method:
        # For backward compatibility
        kwargs["algorithm"] = method

    if func is not None:
        # Call the function with forward_loop as a separate argument
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
            return wrapped_calib_func(model, config, forward_loop, func=self.__class__._calib_func)

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
class MseCalibrateModeDescriptor(BaseCalibrateModeDescriptor):
    """Mode for mse calibration algorithm."""

    @property
    def config_class(self) -> type[QuantizeAlgorithmConfig]:
        """Specifies the config class for the mode."""
        return MseCalibConfig

    _calib_func = mse_calibrate


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

    @property
    def restore(self) -> RestoreEntrypoint:
        """The mode's entrypoint for restoring a model."""
        return restore_svdquant_model


@CalibrateModeRegistry.register_mode
class GPTQLiteModeDescriptor(BaseCalibrateModeDescriptor):
    """Mode for GPTQ calibration algorithm."""

    @property
    def config_class(self) -> type[QuantizeAlgorithmConfig]:
        """Specifies the config class for the mode."""
        return GPTQLiteConfig

    _calib_func = gptq_lite
