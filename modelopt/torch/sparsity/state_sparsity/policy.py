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
import json
from collections.abc import Iterable

import torch
import torch.nn.functional as F
from torch import nn

from modelopt.torch.opt.conversion import ApplyModeError

from .config import DASCCalibrationMeasurement, DASCConfig, DASCLayerPolicy, DASCPolicy

__all__ = ["analyze_gdn_decay", "compute_gdn_decay_horizons"]


def compute_gdn_decay_horizons(
    a_log: torch.Tensor,
    dt_bias: torch.Tensor,
    *,
    epsilon: float = 1e-3,
    static_gate_input: float = -0.3,
) -> torch.Tensor:
    """Compute one static retention horizon per GDN head in CPU float64."""
    if a_log.ndim != 1 or dt_bias.ndim != 1 or a_log.shape != dt_bias.shape or not a_log.numel():
        raise ValueError(
            "GDN A_log and dt_bias must be non-empty one-dimensional tensors of equal shape"
        )
    if not 0.0 < epsilon < 1.0:
        raise ValueError("epsilon must be in (0, 1)")

    a_log_cpu = a_log.detach().to(device="cpu", dtype=torch.float64)
    dt_bias_cpu = dt_bias.detach().to(device="cpu", dtype=torch.float64)
    if not torch.isfinite(a_log_cpu).all() or not torch.isfinite(dt_bias_cpu).all():
        raise ValueError("GDN decay parameters must be finite")

    decay = -torch.exp(a_log_cpu) * F.softplus(dt_bias_cpu + static_gate_input)
    horizons = torch.log(torch.tensor(epsilon, dtype=torch.float64)) / decay
    if not torch.isfinite(horizons).all() or not torch.all(horizons > 0):
        raise ValueError("GDN decay parameters produced non-finite or non-positive horizons")
    return horizons


def _is_gdn_module(module: nn.Module) -> bool:
    class_name = "".join(
        character for character in type(module).__name__.lower() if character.isalnum()
    )
    return (
        "gateddeltanet" in class_name
        and isinstance(getattr(module, "A_log", None), torch.Tensor)
        and isinstance(getattr(module, "dt_bias", None), torch.Tensor)
    )


def _get_gdn_modules(model: nn.Module) -> dict[str, nn.Module]:
    modules = {name: module for name, module in model.named_modules() if _is_gdn_module(module)}
    if not modules:
        raise ApplyModeError("DASC found no GatedDeltaNet modules; only GDN is supported")
    return dict(sorted(modules.items()))


def analyze_gdn_decay(
    model: nn.Module,
    *,
    epsilon: float = 1e-3,
    static_gate_input: float = -0.3,
) -> dict[str, list[float]]:
    """Return deterministic per-head horizons for every GDN module in a model."""
    horizons = {}
    for name, module in _get_gdn_modules(model).items():
        try:
            layer_horizons = compute_gdn_decay_horizons(
                module.A_log,
                module.dt_bias,
                epsilon=epsilon,
                static_gate_input=static_gate_input,
            )
        except ValueError as error:
            raise ApplyModeError(
                f"Invalid GDN decay parameters in module {name!r}: {error}"
            ) from error
        horizons[name] = layer_horizons.tolist()
    return horizons


def _canonical_sha256(value: object) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(payload.encode()).hexdigest()


def _model_structure(modules: dict[str, nn.Module]) -> list[dict[str, object]]:
    return [
        {
            "name": name,
            "num_heads": int(module.A_log.numel()),
        }
        for name, module in modules.items()
    ]


def _decay_parameters(modules: dict[str, nn.Module]) -> list[dict[str, object]]:
    return [
        {
            "name": name,
            "A_log": module.A_log.detach().to(device="cpu", dtype=torch.float64).tolist(),
            "dt_bias": module.dt_bias.detach().to(device="cpu", dtype=torch.float64).tolist(),
        }
        for name, module in modules.items()
    ]


def _validate_measurement_coverage(
    config: DASCConfig, measurements: list[DASCCalibrationMeasurement]
) -> None:
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
    horizons = analyze_gdn_decay(
        model, epsilon=config.epsilon, static_gate_input=config.static_gate_input
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
        decay_parameters_sha256=_canonical_sha256(_decay_parameters(modules)),
        layers=layers,
        measurements=validated_measurements,
    )


def validate_dasc_model_structure(model: nn.Module, policy: DASCPolicy) -> None:
    """Reject restoring a policy onto a different GDN module structure."""
    modules = _get_gdn_modules(model)
    actual = _canonical_sha256(_model_structure(modules))
    if actual != policy.model_structure_sha256:
        raise ApplyModeError("DASC policy does not match the model's GDN module structure")


def validate_dasc_decay_parameters(model: nn.Module, policy: DASCPolicy) -> None:
    """Reject exporting a policy for different GDN decay parameters."""
    modules = _get_gdn_modules(model)
    actual = _canonical_sha256(_decay_parameters(modules))
    if actual != policy.decay_parameters_sha256:
        raise ApplyModeError("DASC policy does not match the model's GDN decay parameters")
