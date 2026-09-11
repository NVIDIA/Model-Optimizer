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

"""Decay analysis and policy selection for GDN state sparsity."""

import hashlib
import importlib
import importlib.util
import json
import math
import warnings
from collections.abc import Iterable
from functools import lru_cache
from typing import Literal

import torch
import torch.nn.functional as F
from torch import nn

from modelopt.torch.opt.conversion import ApplyModeError
from modelopt.torch.opt.dynamic import DynamicModule
from modelopt.torch.utils import unwrap_model

from .config import DASCCalibrationMeasurement, DASCConfig, DASCLayerPolicy, DASCPolicy

__all__ = ["analyze_gdn_decay", "compute_gdn_decay_horizons"]

_SUPPORTED_GDN_CLASS_PATHS = (
    ("megatron.core.ssm.gated_delta_net", "GatedDeltaNet"),
    ("transformers.models.qwen3_next.modeling_qwen3_next", "Qwen3NextGatedDeltaNet"),
)
_STORAGE_DTYPES = {
    "float16": torch.float16,
    "bfloat16": torch.bfloat16,
    "float32": torch.float32,
}


class _DASCModelStructureMismatchError(ApplyModeError):
    """Identify recoverable policy-versus-GDN-geometry drift during restore."""


def _validated_gdn_decay_tensors(
    a_log: torch.Tensor, dt_bias: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """Validate decay tensors and return deterministic CPU float64 values."""
    if a_log.ndim != 1 or dt_bias.ndim != 1 or a_log.shape != dt_bias.shape or not a_log.numel():
        raise ValueError(
            "GDN A_log and dt_bias must be non-empty one-dimensional tensors of equal shape"
        )
    if not a_log.dtype.is_floating_point or not dt_bias.dtype.is_floating_point:
        raise ValueError("GDN A_log and dt_bias must use floating-point dtypes")

    a_log_cpu = a_log.detach().to(device="cpu", dtype=torch.float64)
    dt_bias_cpu = dt_bias.detach().to(device="cpu", dtype=torch.float64)
    if not torch.isfinite(a_log_cpu).all() or not torch.isfinite(dt_bias_cpu).all():
        raise ValueError("GDN decay parameters must be finite")
    return a_log_cpu, dt_bias_cpu


@lru_cache(maxsize=1)
def _supported_gdn_classes() -> tuple[type[nn.Module], ...]:
    """Resolve installed GDN implementations without making either framework mandatory."""
    classes = []
    for module_name, class_name in _SUPPORTED_GDN_CLASS_PATHS:
        try:
            candidate = getattr(importlib.import_module(module_name), class_name)
        except ModuleNotFoundError as error:
            root_module = module_name.partition(".")[0]
            if importlib.util.find_spec(root_module) is None:
                continue
            warnings.warn(
                f"DASC could not resolve {module_name}.{class_name}: {error!r}", stacklevel=2
            )
            continue
        except Exception as error:
            warnings.warn(
                f"DASC could not resolve {module_name}.{class_name}: {error!r}", stacklevel=2
            )
            continue
        if isinstance(candidate, type) and issubclass(candidate, nn.Module):
            classes.append(candidate)
        else:
            warnings.warn(
                f"DASC resolved {module_name}.{class_name}, but it is not an nn.Module class",
                stacklevel=2,
            )
    return tuple(classes)


def compute_gdn_decay_horizons(
    a_log: torch.Tensor,
    dt_bias: torch.Tensor,
    *,
    epsilon: float = 1e-3,
    static_gate_input: float = -0.3,
) -> torch.Tensor:
    """Compute one static retention horizon per GDN head in CPU float64."""
    if not 0.0 < epsilon < 1.0:
        raise ValueError("epsilon must be in (0, 1)")

    a_log_cpu, dt_bias_cpu = _validated_gdn_decay_tensors(a_log, dt_bias)

    decay = -torch.exp(a_log_cpu) * F.softplus(dt_bias_cpu + static_gate_input)
    horizons = torch.log(torch.tensor(epsilon, dtype=torch.float64)) / decay
    if not torch.isfinite(horizons).all() or not torch.all(horizons > 0):
        raise ValueError("GDN decay parameters produced non-finite or non-positive horizons")
    return horizons


def _has_supported_gdn_identity(
    module: nn.Module, supported_classes: tuple[type[nn.Module], ...]
) -> bool:
    """Return whether a module has an exact or ModelOpt-generated supported GDN identity."""
    module_class = type(module)
    return module_class in supported_classes or (
        isinstance(module, DynamicModule)
        and any(base in supported_classes for base in module_class.__mro__)
    )


def _reject_incomplete_gdn_modules(identity_modules: list[tuple[str, nn.Module]]) -> None:
    """Reject supported identities that do not expose both required decay tensors."""
    missing_decay_parameters = [
        name or "<root>"
        for name, module in identity_modules
        if not all(
            isinstance(getattr(module, parameter, None), torch.Tensor)
            for parameter in ("A_log", "dt_bias")
        )
    ]
    if missing_decay_parameters:
        raise ApplyModeError(
            "DASC found supported GDN modules without A_log and dt_bias tensors at: "
            f"{', '.join(missing_decay_parameters)}"
        )


def _reject_unconverted_gdn_subclasses(
    named_modules: list[tuple[str, nn.Module]],
    supported_classes: tuple[type[nn.Module], ...],
) -> None:
    """Reject ordinary subclasses that would otherwise be silently omitted from the policy."""
    unsupported_subclasses = [
        name or "<root>"
        for name, module in named_modules
        if not isinstance(module, DynamicModule)
        and type(module) not in supported_classes
        and any(base in supported_classes for base in type(module).__mro__[1:])
    ]
    if unsupported_subclasses:
        raise ApplyModeError(
            "DASC found GDN subclasses that are not ModelOpt dynamic modules at: "
            f"{', '.join(unsupported_subclasses)}; convert the module with ModelOpt or use a "
            "supported class directly"
        )


def _get_gdn_modules(model: nn.Module) -> dict[str, nn.Module]:
    """Find supported GDN layers after removing a recognized model wrapper."""
    model = unwrap_model(model, force_unwrap=True)
    supported_classes = _supported_gdn_classes()
    named_modules = list(model.named_modules())
    identity_modules = [
        (name, module)
        for name, module in named_modules
        if _has_supported_gdn_identity(module, supported_classes)
    ]
    _reject_incomplete_gdn_modules(identity_modules)
    _reject_unconverted_gdn_subclasses(named_modules, supported_classes)
    if not identity_modules:
        supported = ", ".join(
            f"{module_name}.{class_name}" for module_name, class_name in _SUPPORTED_GDN_CLASS_PATHS
        )
        raise ApplyModeError(f"DASC found no supported GDN modules; expected one of: {supported}")
    return dict(sorted(identity_modules, key=lambda item: item[0]))


def _analyze_gdn_modules(
    modules: dict[str, nn.Module],
    *,
    epsilon: float,
    static_gate_input: float,
    storage_dtype: torch.dtype | None = None,
) -> dict[str, list[float]]:
    """Compute horizons, optionally from checkpoint-storage-canonical parameters."""
    horizons = {}
    for name, module in modules.items():
        a_log = module.A_log
        dt_bias = module.dt_bias
        try:
            if storage_dtype is not None:
                _validated_gdn_decay_tensors(a_log, dt_bias)
                a_log = a_log.detach().to(device="cpu", dtype=storage_dtype)
                dt_bias = dt_bias.detach().to(device="cpu", dtype=storage_dtype)
            layer_horizons = compute_gdn_decay_horizons(
                a_log,
                dt_bias,
                epsilon=epsilon,
                static_gate_input=static_gate_input,
            )
        except ValueError as error:
            raise ApplyModeError(
                f"Invalid GDN decay parameters in module {name!r}: {error}"
            ) from error
        horizons[name] = layer_horizons.tolist()
    return horizons


def analyze_gdn_decay(
    model: nn.Module,
    *,
    epsilon: float = 1e-3,
    static_gate_input: float = -0.3,
    decay_parameter_storage_dtype: Literal["float16", "bfloat16", "float32"] | None = None,
) -> dict[str, list[float]]:
    """Return per-head horizons, optionally canonicalized to a checkpoint storage dtype."""
    storage_dtype = None
    if decay_parameter_storage_dtype is not None:
        try:
            storage_dtype = _STORAGE_DTYPES[decay_parameter_storage_dtype]
        except KeyError as error:
            supported = ", ".join(_STORAGE_DTYPES)
            raise ValueError(
                f"decay_parameter_storage_dtype must be one of: {supported}"
            ) from error
    return _analyze_gdn_modules(
        _get_gdn_modules(model),
        epsilon=epsilon,
        static_gate_input=static_gate_input,
        storage_dtype=storage_dtype,
    )


def _canonical_sha256(value: object) -> str:
    """Hash a JSON value with deterministic ordering and no non-finite numbers."""
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(payload.encode()).hexdigest()


def _model_structure(modules: dict[str, nn.Module]) -> list[dict[str, object]]:
    """Describe the layer names and head counts that define policy geometry."""
    return [
        {
            "name": name,
            "num_heads": int(module.A_log.numel()),
        }
        for name, module in modules.items()
    ]


def _decay_parameters(
    modules: dict[str, nn.Module], storage_dtype: torch.dtype
) -> list[dict[str, object]]:
    """Serialize a storage-dtype-canonicalized calibration snapshot for provenance."""
    return [
        {
            "name": name,
            "A_log": module.A_log.detach()
            .to(device="cpu", dtype=storage_dtype)
            .to(dtype=torch.float32)
            .tolist(),
            "dt_bias": module.dt_bias.detach()
            .to(device="cpu", dtype=storage_dtype)
            .to(dtype=torch.float32)
            .tolist(),
        }
        for name, module in modules.items()
    ]


def _storage_rounding_radius(tensor: torch.Tensor, storage_dtype: torch.dtype) -> torch.Tensor:
    """Compose inverse error bounds for storage and live-dtype materialization casts."""
    values = tensor.detach().to(device="cpu", dtype=torch.float64).abs()
    upper = values
    cast_dtypes = tuple(dict.fromkeys((storage_dtype, tensor.dtype)))
    for dtype in reversed(cast_dtypes):
        dtype_info = torch.finfo(dtype)
        unit_roundoff = dtype_info.eps / 2.0
        smallest_subnormal = dtype_info.tiny * dtype_info.eps
        upper = (upper + smallest_subnormal) / (1.0 - unit_roundoff)
    return upper - values


def _storage_cast_horizon_bounds(
    module: nn.Module,
    *,
    epsilon: float,
    static_gate_input: float,
    storage_dtype: torch.dtype,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Bound horizons compatible with the current parameters before one storage cast."""
    a_log = module.A_log.detach().to(device="cpu", dtype=torch.float64)
    dt_bias = module.dt_bias.detach().to(device="cpu", dtype=torch.float64)
    a_radius = _storage_rounding_radius(module.A_log, storage_dtype)
    dt_radius = _storage_rounding_radius(module.dt_bias, storage_dtype)
    scale = -math.log(epsilon)
    lower = scale / (
        torch.exp(a_log + a_radius) * F.softplus(dt_bias + dt_radius + static_gate_input)
    )
    upper = scale / (
        torch.exp(a_log - a_radius) * F.softplus(dt_bias - dt_radius + static_gate_input)
    )
    return lower, upper


def _validate_measurement_coverage(
    config: DASCConfig, measurements: list[DASCCalibrationMeasurement]
) -> None:
    """Require exactly one matching measurement for every configured window."""
    measured = [measurement.wmax for measurement in measurements]
    unexpected = sorted(set(measured) - set(config.wmax_candidates))
    missing = sorted(set(config.wmax_candidates) - set(measured))
    if unexpected or missing or len(measured) != len(set(measured)):
        raise ApplyModeError(
            "DASC measurements must contain exactly one result per configured candidate; "
            f"missing={missing}, unexpected={unexpected}, duplicates={len(measured) != len(set(measured))}"
        )
    if any(measurement.variant != config.variant for measurement in measurements):
        raise ApplyModeError(
            f"All DASC measurements must use configured variant {config.variant!r}"
        )


def _validate_measurement_geometry(
    horizons: dict[str, list[float]], measurements: list[DASCCalibrationMeasurement]
) -> None:
    """Bind caller-reported retained and total head counts to the analyzed model."""
    total_heads = sum(len(layer_horizons) for layer_horizons in horizons.values())
    for measurement in measurements:
        retained_heads = sum(
            horizon > measurement.wmax
            for layer_horizons in horizons.values()
            for horizon in layer_horizons
        )
        if measurement.total_heads != total_heads or measurement.retained_heads != retained_heads:
            raise ApplyModeError(
                f"DASC measurement geometry for Wmax={measurement.wmax} does not match the model; "
                f"expected retained/total={retained_heads}/{total_heads}, got "
                f"{measurement.retained_heads}/{measurement.total_heads}"
            )


def _candidate_passes(config: DASCConfig, measurement: DASCCalibrationMeasurement) -> bool:
    """Return whether one candidate passes every configured evidence gate."""
    return measurement.checkpoint_savings >= config.min_checkpoint_savings and all(
        result.perplexity_retention >= config.min_perplexity_retention
        and result.top1_agreement >= config.min_top1_agreement
        and result.finite_continuation_logits
        and result.retained_state_exact
        and result.omitted_state_matches_recovery
        and result.convolution_state_exact
        for result in measurement.quality
    )


def build_dasc_policy(
    model: nn.Module,
    config: DASCConfig,
    measurements: Iterable[DASCCalibrationMeasurement | dict],
) -> DASCPolicy:
    """Build a checkpoint-specific policy from decay parameters and measured quality."""
    try:
        validated_measurements = [
            measurement
            if isinstance(measurement, DASCCalibrationMeasurement)
            else DASCCalibrationMeasurement(**measurement)
            for measurement in measurements
        ]
    except (TypeError, ValueError) as error:
        raise ApplyModeError(f"Invalid DASC calibration measurements: {error}") from error
    _validate_measurement_coverage(config, validated_measurements)
    validated_measurements.sort(key=lambda measurement: measurement.wmax)

    modules = _get_gdn_modules(model)
    horizons = _analyze_gdn_modules(
        modules,
        epsilon=config.epsilon,
        static_gate_input=config.static_gate_input,
        storage_dtype=_STORAGE_DTYPES[config.decay_parameter_storage_dtype],
    )
    _validate_measurement_geometry(horizons, validated_measurements)

    passing = [
        measurement.wmax
        for measurement in validated_measurements
        if _candidate_passes(config, measurement)
    ]
    if not passing:
        raise ApplyModeError(
            "No DASC Wmax candidate passed every configured quality and storage gate"
        )
    selected_wmax = max(passing)

    layers = {}
    for name, values in horizons.items():
        retained = [head for head, horizon in enumerate(values) if horizon > selected_wmax]
        layers[name] = DASCLayerPolicy(
            num_heads=len(values),
            static_horizons=values,
            retained_heads=retained,
            omitted_heads=[head for head in range(len(values)) if head not in retained],
        )

    return DASCPolicy(
        variant=config.variant,
        recovery="zero" if config.variant == "dasc_nr" else "suffix_replay",
        epsilon=config.epsilon,
        static_gate_input=config.static_gate_input,
        decay_parameter_storage_dtype=config.decay_parameter_storage_dtype,
        selected_wmax=selected_wmax,
        wmax_candidates=config.wmax_candidates,
        quality_gates={
            "min_perplexity_retention": config.min_perplexity_retention,
            "min_top1_agreement": config.min_top1_agreement,
            "min_checkpoint_savings": config.min_checkpoint_savings,
        },
        model_id=config.model_id,
        model_revision=config.model_revision,
        model_config_id=config.model_config_id,
        calibration_data_id=config.calibration_data_id,
        granularity=config.granularity,
        preserve_convolution_state=config.preserve_convolution_state,
        model_structure_sha256=_canonical_sha256(_model_structure(modules)),
        decay_parameters_sha256=_canonical_sha256(
            _decay_parameters(modules, _STORAGE_DTYPES[config.decay_parameter_storage_dtype])
        ),
        layers=layers,
        measurements=validated_measurements,
    )


def validate_dasc_model_structure(model: nn.Module, policy: DASCPolicy) -> None:
    """Reject restoring a policy onto a different GDN module structure."""
    modules = _get_gdn_modules(model)
    actual_structure = _model_structure(modules)
    policy_structure = [
        {"name": name, "num_heads": layer.num_heads}
        for name, layer in sorted(policy.layers.items())
    ]
    if (
        actual_structure != policy_structure
        or _canonical_sha256(actual_structure) != policy.model_structure_sha256
    ):
        raise _DASCModelStructureMismatchError(
            "DASC policy does not match the model's GDN module structure"
        )


def validate_dasc_decay_parameters(model: nn.Module, policy: DASCPolicy) -> None:
    """Reject deployment when current decay parameters no longer derive the stored policy.

    Numerical validation uses inverse cast bounds rather than the provenance digest because an
    FP16 or BF16 storage cast is lossy. A stored mask is rejected only when its head's complete
    admissible horizon interval lies on the opposite side of the strict ``horizon > Wmax`` rule.
    """
    modules = _get_gdn_modules(model)
    for name, module in modules.items():
        layer = policy.layers[name]
        try:
            _validated_gdn_decay_tensors(module.A_log, module.dt_bias)
        except ValueError as error:
            raise ApplyModeError(
                f"Invalid GDN decay parameters in module {name!r}: {error}"
            ) from error
        lower, upper = _storage_cast_horizon_bounds(
            module,
            epsilon=policy.epsilon,
            static_gate_input=policy.static_gate_input,
            storage_dtype=_STORAGE_DTYPES[policy.decay_parameter_storage_dtype],
        )
        declared_retained = set(layer.retained_heads)
        for head, (head_lower, head_upper) in enumerate(zip(lower, upper)):
            retained_is_impossible = (
                head in declared_retained and head_upper <= policy.selected_wmax
            )
            omitted_is_impossible = (
                head not in declared_retained and head_lower > policy.selected_wmax
            )
            if retained_is_impossible or omitted_is_impossible:
                raise ApplyModeError(
                    "DASC policy head mask does not match current decay parameters in layer "
                    f"{name!r}"
                )
        stored = torch.tensor(layer.static_horizons, dtype=torch.float64)
        numerical_slack = 32.0 * torch.finfo(torch.float64).eps
        if torch.any(stored < lower * (1.0 - numerical_slack)) or torch.any(
            stored > upper * (1.0 + numerical_slack)
        ):
            raise ApplyModeError(
                f"DASC policy horizons do not match current decay parameters in layer {name!r}"
            )
