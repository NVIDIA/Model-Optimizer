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

"""Content-addressed physical-versus-runtime width-slice equivalence."""

from __future__ import annotations

import copy
import hashlib
import importlib
import inspect
import json
import math
from contextlib import ExitStack, contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import torch
from safetensors.torch import load_file as load_safetensors
from safetensors.torch import save_file as save_safetensors

from ..block_config import BlockConfig, maybe_cast_block_configs
from ..candidates import _AXIS_TO_TARGET, _apply_axis_edits, _axis_base_value
from ..dataset import DataLayout, PuzzletronBatch, batch_from_automodel
from ..identity import canonicalize, stable_hash
from ..plugins.automodel.batch_adapter import canonicalize_position_ids, validated_forward_kwargs
from ..pruning.compact_runtime import (
    compact_gated_delta_net_forward,
    compact_grouped_attention_forward,
    resolve_compact_grouped_attention_target,
)
from ..pruning.materialize import materialize_hidden_width_checkpoint, materialize_model_from_sorted
from ..pruning.runtime_hidden_width import hidden_width_layer_context
from ..pruning.runtime_ple import ple_layer_context
from ..tools.checkpoint_utils import load_model_config
from ..tools.checkpoint_utils_hf import init_model_from_config

__all__ = [
    "WidthSliceCase",
    "build_width_slice_cases",
    "compare_width_slice_outputs",
    "evaluate_width_slice_equivalence",
    "normalize_width_slice_batch",
    "validate_width_slice_artifacts",
]

_SCHEMA_VERSION = 3
_IMPLEMENTATION_VERSION = "width-slice-equivalence-v3"
_DEFAULT_TOLERANCES = {
    "loss_atol": 1.0e-5,
    "loss_rtol": 1.0e-5,
    "output_atol": 1.0e-5,
    "output_rtol": 1.0e-5,
}
_REQUIRED_CASE_FIELDS = {
    "schema_version",
    "case_identity",
    "axis_id",
    "scope",
    "layers",
    "target_value",
    "source_value",
    "checkpoint_identity",
    "checkpoint_config_identity",
    "target_config_identity",
    "batch_fingerprint",
    "implementation_provenance",
    "case_contract",
    "tolerances",
    "lineage",
    "structural_evidence",
    "runtime_hook_count",
    "runtime_hook_executions",
    "runtime_axis_evidence",
    "comparison_evidence",
    "target_applied",
    "metrics",
    "passed",
    "record_hash",
}


def _replace_block_scoring_recipe():
    """Load the AutoModel-backed recipe only when width-slice execution needs it."""

    try:
        from ..plugins.automodel.solution_recipe import ReplaceBlockScoringRecipe
    except ImportError as error:
        missing_name = getattr(error, "name", "") or ""
        if missing_name == "nemo_automodel" or missing_name.startswith("nemo_automodel."):
            raise ImportError(
                "Puzzletron width-slice equivalence requires a compatible NeMo AutoModel "
                "installation; follow examples/puzzletron/README.md."
            ) from error
        raise
    return ReplaceBlockScoringRecipe


@dataclass(frozen=True)
class WidthSliceCase:
    """One explicit descriptor-derived axis target from an authoritative checkpoint."""

    case_identity: str
    axis_id: str
    axis_class: str
    scope: str
    layers: tuple[int, ...]
    source_value: int
    target_value: int
    subblock_kind: str
    field: str
    source_block_config: BlockConfig | None
    target_block_config: BlockConfig | None
    expected_structure: dict[str, Any]
    implementation_provenance: dict[str, Any]


def _block_payload(block: BlockConfig | None) -> dict[str, Any] | None:
    return None if block is None else canonicalize(block.to_dict())


def _case_contract(case: WidthSliceCase) -> dict[str, Any]:
    return {
        "case_identity": case.case_identity,
        "axis_id": case.axis_id,
        "axis_class": case.axis_class,
        "scope": case.scope,
        "layers": list(case.layers),
        "source_value": case.source_value,
        "target_value": case.target_value,
        "subblock_kind": case.subblock_kind,
        "field": case.field,
        "source_block_config": _block_payload(case.source_block_config),
        "target_block_config": _block_payload(case.target_block_config),
        "expected_structure": canonicalize(case.expected_structure),
        "implementation_provenance": canonicalize(case.implementation_provenance),
    }


def _json_hash(payload: Mapping[str, Any], *, prefix: str) -> str:
    return stable_hash(canonicalize(dict(payload)), prefix=prefix)


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return f"sha256:{digest.hexdigest()}"


def _checkpoint_identity(checkpoint_dir: Path) -> tuple[str, str, dict[str, Any]]:
    config_path = checkpoint_dir / "config.json"
    if not config_path.is_file():
        raise FileNotFoundError(f"sorted checkpoint missing config.json: {checkpoint_dir}")
    config_payload = json.loads(config_path.read_text(encoding="utf-8"))
    config_identity = _json_hash(config_payload, prefix="checkpoint_config")
    files = []
    for path in sorted(checkpoint_dir.rglob("*")):
        if not path.is_file() or path.name == "puzzletron_realization.json":
            continue
        if path.suffix not in {".json", ".safetensors", ".bin", ".pth"}:
            continue
        files.append(
            {
                "path": path.relative_to(checkpoint_dir).as_posix(),
                "size": path.stat().st_size,
                "sha256": _file_sha256(path),
            }
        )
    if not any(item["path"].endswith((".safetensors", ".bin", ".pth")) for item in files):
        raise FileNotFoundError(f"sorted checkpoint has no weight files: {checkpoint_dir}")
    identity = _json_hash(
        {"config_identity": config_identity, "files": files},
        prefix="sorted_checkpoint_content",
    )
    return identity, config_identity, {"config": config_payload, "files": files}


def _callable_source(fn: Any) -> dict[str, str]:
    fn = inspect.unwrap(fn)
    path = f"{fn.__module__}.{fn.__qualname__}"
    return {
        "path": path,
        "source_hash": stable_hash(inspect.getsource(fn), prefix="python_source"),
    }


def _implementation_provenance(axis: Any, descriptor: Any) -> dict[str, Any]:
    recipe = _replace_block_scoring_recipe()
    functions = (
        build_width_slice_cases,
        _case_identity_payload,
        _reduced_value,
        _block_configs,
        _axis_base_value,
        _apply_axis_edits,
        materialize_model_from_sorted,
        materialize_hidden_width_checkpoint,
        init_model_from_config,
        _child_config,
        _prune_target,
        recipe.prune_block_context,
        recipe.architecture_context,
        recipe._typed_subblock_runtime_hooks,
        resolve_compact_grouped_attention_target,
        compact_grouped_attention_forward,
        compact_gated_delta_net_forward,
        hidden_width_layer_context,
        ple_layer_context,
        _runtime_context,
        _axis_specific_changed_shapes,
        batch_from_automodel,
        validated_forward_kwargs,
        _forward,
        compare_width_slice_outputs,
        _validate_case_record,
        validate_width_slice_artifacts,
    )
    descriptor_functions = tuple(
        getattr(descriptor, name)
        for name in (
            "puzzletron_capabilities",
            "get_language_model_config",
            "set_block_configs",
            "layer_block_name",
            "embedding_pruning_spec",
        )
        if callable(getattr(descriptor, name, None))
    )
    sources = [_callable_source(fn) for fn in (*functions, *descriptor_functions)]
    return {
        "version": _IMPLEMENTATION_VERSION,
        "materialize_impl": str(axis.materialize_impl),
        "runtime_slice_impl": str(axis.runtime_slice_impl),
        "descriptor": f"{descriptor.__module__}.{descriptor.__qualname__}",
        "functions": sources,
        "source_hash": stable_hash(sources, prefix="width_slice_implementation"),
    }


def _reduced_value(base_value: int, *, field: str, alignment: int, subblock: Any) -> int:
    if int(base_value) < 2:
        raise ValueError(f"cannot build a reduced {field} target from {base_value}")
    target = int(base_value) // 2
    if (
        field
        in {
            "intermediate_size",
            "expert_intermediate_size",
            "shared_expert_intermediate_size",
            "latent_dim",
            "hidden_size",
        }
        and alignment > 1
    ):
        target = max(alignment, (target // alignment) * alignment)
    if field == "num_experts":
        target = max(target, int(getattr(subblock, "top_k", 1) or 1))
    if not 0 < target < int(base_value):
        raise ValueError(
            f"axis field {field!r} has no legal reduced target: source={base_value} target={target}"
        )
    return target


def _block_configs(config: Any) -> list[BlockConfig]:
    values = getattr(config, "block_configs", None)
    if values is None:
        language = getattr(config, "text_config", None)
        values = getattr(language, "block_configs", None)
    if values is None:
        raise ValueError("width-slice equivalence requires checkpoint block_configs")
    return maybe_cast_block_configs(values)


def _case_identity_payload(
    *,
    checkpoint_identity: str,
    axis_id: str,
    scope: str,
    layers: tuple[int, ...],
    source_value: int,
    target_value: int,
    provenance: Mapping[str, Any],
) -> dict[str, Any]:
    return {
        "checkpoint_identity": checkpoint_identity,
        "axis_id": axis_id,
        "scope": scope,
        "layers": layers,
        "source_value": source_value,
        "target_value": target_value,
        "implementation_provenance": provenance,
    }


def build_width_slice_cases(
    descriptor: Any,
    checkpoint_config: Any,
    sorted_checkpoint_dir: str | Path,
    *,
    alignment: int = 1,
    sampled_layers: Sequence[int] | None = None,
) -> dict[str, WidthSliceCase]:
    """Build every capability-declared materialize/runtime case from HF config."""

    checkpoint_dir = Path(sorted_checkpoint_dir)
    checkpoint_identity, _, _ = _checkpoint_identity(checkpoint_dir)
    capabilities = descriptor.puzzletron_capabilities(checkpoint_config)
    blocks = _block_configs(checkpoint_config)
    num_layers = len(blocks)
    if sampled_layers is not None:
        requested = tuple(dict.fromkeys(int(layer) for layer in sampled_layers))
        if len(requested) != 2 or any(layer < 0 or layer >= num_layers for layer in requested):
            raise ValueError(
                "width-slice sampled_layers must contain exactly two distinct valid layers"
            )
    else:
        requested = None

    cases: dict[str, WidthSliceCase] = {}
    for axis_id, axis in sorted(capabilities.axes.items()):
        if not axis.materialize_impl or not axis.runtime_slice_impl:
            continue
        provenance = _implementation_provenance(axis, descriptor)
        axis_class = f"{axis.materialize_impl}::{axis.runtime_slice_impl}"
        if axis_id == "hidden_width":
            language = descriptor.get_language_model_config(checkpoint_config)
            source = int(getattr(language, axis.field))
            target = _reduced_value(
                source,
                field=axis.field,
                alignment=alignment,
                subblock=None,
            )
            layers = tuple(range(num_layers))
            payload = _case_identity_payload(
                checkpoint_identity=checkpoint_identity,
                axis_id=axis_id,
                scope="global",
                layers=layers,
                source_value=source,
                target_value=target,
                provenance=provenance,
            )
            case_id = _json_hash(payload, prefix="width_slice_case")
            cases[case_id] = WidthSliceCase(
                case_identity=case_id,
                axis_id=axis_id,
                axis_class=axis_class,
                scope="global",
                layers=layers,
                source_value=source,
                target_value=target,
                subblock_kind=axis.subblock_kind,
                field=axis.field,
                source_block_config=None,
                target_block_config=None,
                expected_structure={
                    "field": axis.field,
                    "before": source,
                    "after": target,
                    "requires_tensor_shape_change": True,
                },
                implementation_provenance=provenance,
            )
            continue
        if axis_id == "ple_width":
            language = descriptor.get_language_model_config(checkpoint_config)
            source = int(getattr(language, axis.field))
            target = _reduced_value(
                source,
                field=axis.field,
                alignment=alignment,
                subblock=None,
            )
            layers = tuple(range(num_layers))
            payload = _case_identity_payload(
                checkpoint_identity=checkpoint_identity,
                axis_id=axis_id,
                scope="global",
                layers=layers,
                source_value=source,
                target_value=target,
                provenance=provenance,
            )
            case_id = _json_hash(payload, prefix="width_slice_case")
            cases[case_id] = WidthSliceCase(
                case_identity=case_id,
                axis_id=axis_id,
                axis_class=axis_class,
                scope="global",
                layers=layers,
                source_value=source,
                target_value=target,
                subblock_kind=axis.subblock_kind,
                field=axis.field,
                source_block_config=None,
                target_block_config=None,
                expected_structure={
                    "field": axis.field,
                    "before": source,
                    "after": target,
                    "requires_tensor_shape_change": True,
                },
                implementation_provenance=provenance,
            )
            continue

        target_spec = _AXIS_TO_TARGET.get(axis_id)
        if target_spec is None:
            target_spec = (axis.subblock_kind, axis.field)
        subblock_kind, edit_field = target_spec
        applicable = [
            layer_idx
            for layer_idx, block in enumerate(blocks)
            if block.get_subblock(subblock_kind) is not None
            and _axis_base_value(block.require_subblock(subblock_kind), edit_field) is not None
        ]
        if requested is not None:
            selected = [layer for layer in requested if layer in applicable]
        elif len(applicable) >= 2:
            selected = [applicable[0], applicable[-1]]
        else:
            selected = applicable
        if len(selected) != 2:
            raise ValueError(
                f"axis {axis_id!r} requires exactly two applicable sampled layers; "
                f"applicable={applicable} requested={requested}"
            )
        for layer_idx in selected:
            source_block = blocks[layer_idx]
            subblock = source_block.require_subblock(subblock_kind)
            source = int(_axis_base_value(subblock, edit_field))
            target = _reduced_value(
                source,
                field=edit_field,
                alignment=alignment,
                subblock=subblock,
            )
            target_block, changed = _apply_axis_edits(
                source_block,
                ((axis_id, subblock_kind, edit_field, target),),
            )
            if changed.get(axis_id) != target or target_block == source_block:
                raise RuntimeError(
                    f"descriptor target was not applied for axis={axis_id} layer={layer_idx}"
                )
            layers = (layer_idx,)
            payload = _case_identity_payload(
                checkpoint_identity=checkpoint_identity,
                axis_id=axis_id,
                scope="layer",
                layers=layers,
                source_value=source,
                target_value=target,
                provenance=provenance,
            )
            case_id = _json_hash(payload, prefix="width_slice_case")
            cases[case_id] = WidthSliceCase(
                case_identity=case_id,
                axis_id=axis_id,
                axis_class=axis_class,
                scope="layer",
                layers=layers,
                source_value=source,
                target_value=target,
                subblock_kind=subblock_kind,
                field=edit_field,
                source_block_config=source_block,
                target_block_config=target_block,
                expected_structure={
                    "field": edit_field,
                    "before": source,
                    "after": target,
                    "requires_tensor_shape_change": True,
                },
                implementation_provenance=provenance,
            )
    if not cases:
        raise ValueError("descriptor has no materializable runtime width axes")
    return cases


def normalize_width_slice_batch(
    collated: Mapping[str, Any] | PuzzletronBatch,
    *,
    descriptor: Any,
    checkpoint_config: Any,
    layout: DataLayout | str,
    sample_ids: Sequence[str],
    source_metadata: Mapping[str, Any],
) -> PuzzletronBatch:
    """Normalize one scorer/collator batch through the shared AutoModel adapter."""

    batch = (
        collated
        if isinstance(collated, PuzzletronBatch)
        else batch_from_automodel(
            collated,
            sample_ids=sample_ids,
            source_metadata=source_metadata,
            layout=layout,
        )
    )
    return canonicalize_position_ids(
        batch,
        descriptor=descriptor,
        config=checkpoint_config,
    )


def _normalized_tolerances(values: Mapping[str, float] | None) -> dict[str, float]:
    result = dict(_DEFAULT_TOLERANCES)
    supplied = dict(values or {})
    unknown = set(supplied) - set(result)
    if unknown:
        raise ValueError(f"unknown width-slice tolerances: {sorted(unknown)}")
    result.update({key: float(value) for key, value in supplied.items()})
    if any(not math.isfinite(value) or value < 0.0 for value in result.values()):
        raise ValueError(f"width-slice tolerances must be finite and non-negative: {result}")
    return result


def compare_width_slice_outputs(
    *,
    physical_loss: torch.Tensor,
    runtime_loss: torch.Tensor,
    physical_output: torch.Tensor,
    runtime_output: torch.Tensor,
    tolerances: Mapping[str, float],
) -> dict[str, Any]:
    """Apply explicit atol/rtol gates to loss and output tensors."""

    gates = _normalized_tolerances(tolerances)
    p_loss = float(physical_loss.detach().float().cpu())
    r_loss = float(runtime_loss.detach().float().cpu())
    if not math.isfinite(p_loss) or not math.isfinite(r_loss):
        raise RuntimeError("width-slice losses must be finite")
    loss_delta = abs(p_loss - r_loss)
    loss_allowed = gates["loss_atol"] + gates["loss_rtol"] * max(abs(p_loss), abs(r_loss))
    shapes_match = tuple(physical_output.shape) == tuple(runtime_output.shape)
    if shapes_match:
        physical_float = physical_output.detach().float()
        runtime_float = runtime_output.detach().float()
        if not bool(torch.isfinite(physical_float).all()) or not bool(
            torch.isfinite(runtime_float).all()
        ):
            raise RuntimeError("width-slice outputs must be finite")
        delta = (physical_float - runtime_float).abs()
        max_delta = float(delta.max().cpu()) if delta.numel() else 0.0
        mean_delta = float(delta.mean().cpu()) if delta.numel() else 0.0
        allowed = gates["output_atol"] + gates["output_rtol"] * runtime_float.abs()
        max_excess = (
            float((delta - allowed).max().cpu()) if delta.numel() else -float(gates["output_atol"])
        )
        output_close = max_excess <= 0.0
    else:
        max_delta = None
        mean_delta = None
        max_excess = None
        output_close = False
    loss_close = loss_delta <= loss_allowed
    return {
        "physical_loss": p_loss,
        "runtime_loss": r_loss,
        "loss_delta": loss_delta,
        "loss_allowed_delta": loss_allowed,
        "loss_close": loss_close,
        "physical_output_shape": list(physical_output.shape),
        "runtime_output_shape": list(runtime_output.shape),
        "output_max_abs_delta": max_delta,
        "output_mean_abs_delta": mean_delta,
        "output_max_tolerance_excess": max_excess,
        "output_close": output_close,
        "passed": loss_close and output_close,
    }


def _config_identity(config: Any, *, prefix: str) -> str:
    payload = config.to_dict() if hasattr(config, "to_dict") else config
    return _json_hash(payload, prefix=prefix)


def _descriptor_reference(descriptor: Any) -> dict[str, str]:
    return {
        "module": descriptor.__module__,
        "qualname": descriptor.__qualname__,
    }


def _resolve_descriptor(reference: Mapping[str, Any]) -> Any:
    module_name = reference.get("module")
    qualname = reference.get("qualname")
    if not isinstance(module_name, str) or not isinstance(qualname, str) or "<locals>" in qualname:
        raise RuntimeError("width-slice descriptor reference is not independently resolvable")
    value = importlib.import_module(module_name)
    for component in qualname.split("."):
        value = getattr(value, component)
    return value


def _child_config(descriptor: Any, config: Any, case: WidthSliceCase) -> Any:
    child = copy.deepcopy(config)
    if case.axis_id == "hidden_width":
        spec = descriptor.embedding_pruning_spec(
            config,
            widths=(case.source_value, case.target_value),
            alignment=1,
        )
        return spec.update_config_object(child, case.target_value)
    if case.axis_id == "ple_width":
        language = descriptor.get_language_model_config(child)
        setattr(language, case.field, case.target_value)
        return child
    blocks = _block_configs(child)
    blocks[case.layers[0]] = case.target_block_config
    descriptor.set_block_configs(child, blocks)
    return child


def _layer(model: torch.nn.Module, descriptor: Any, layer_idx: int) -> torch.nn.Module:
    canonical = descriptor.layer_block_name(layer_idx)
    adapted = descriptor.adapt_module_name_for_model(canonical, model)
    for name in dict.fromkeys((adapted, canonical)):
        try:
            return model.get_submodule(name)
        except AttributeError:
            pass
    raise AttributeError(f"runtime model has no descriptor layer {canonical!r}")


class _RuntimeRecipeAdapter:
    def prune_block_context(self, *args, **kwargs):
        return _replace_block_scoring_recipe().prune_block_context(self, *args, **kwargs)

    def architecture_context(self, *args, **kwargs):
        return _replace_block_scoring_recipe().architecture_context(self, *args, **kwargs)

    @staticmethod
    def _maybe_submodule(*args, **kwargs):
        return _replace_block_scoring_recipe()._maybe_submodule(*args, **kwargs)

    def __init__(self, model: torch.nn.Module, descriptor: Any):
        self.model_parts = [model]
        self._descriptor = descriptor

    def _descriptor_cls(self):
        return self._descriptor

    def _find_decoder_layer(self, layer_idx: int):
        return _layer(self.model_parts[0], self._descriptor, layer_idx)


def _prune_target(case: WidthSliceCase, config: Any, descriptor: Any) -> dict[str, Any]:
    language = descriptor.get_language_model_config(config)
    attention = case.source_block_config.get_subblock("attention")
    child_attention = case.target_block_config.get_subblock("attention")
    ffn = case.source_block_config.get_subblock("ffn")
    child_ffn = case.target_block_config.get_subblock("ffn")
    return {
        "layer_idx": case.layers[0],
        "teacher_block_config": case.source_block_config,
        "child_block_config": case.target_block_config,
        "orig_intermediate": getattr(ffn, "intermediate_size", None),
        "target_intermediate": getattr(child_ffn, "intermediate_size", None),
        "orig_num_q": getattr(attention, "num_query_heads", None),
        "orig_num_kv": getattr(attention, "num_kv_heads", None),
        "target_num_q": getattr(child_attention, "num_query_heads", None),
        "target_num_kv": getattr(child_attention, "num_kv_heads", None),
        "head_dim": int(
            getattr(language, "head_dim", 0)
            or int(language.hidden_size) // int(language.num_attention_heads)
        ),
    }


def _hook_snapshot(module: torch.nn.Module) -> dict[tuple[str, str, int], tuple[Any, Any]]:
    snapshot = {}
    for module_name, item in module.named_modules():
        for hook_kind, attribute in (
            ("forward", "_forward_hooks"),
            ("forward_pre", "_forward_pre_hooks"),
        ):
            hooks = getattr(item, attribute, {})
            snapshot.update(
                {
                    (module_name, hook_kind, int(hook_id)): (hooks, callback)
                    for hook_id, callback in hooks.items()
                }
            )
    return snapshot


def _tensor_shapes(value: Any) -> list[list[int]]:
    if torch.is_tensor(value):
        return [list(value.shape)]
    if isinstance(value, Mapping):
        return [shape for item in value.values() for shape in _tensor_shapes(item)]
    if isinstance(value, tuple | list):
        return [shape for item in value for shape in _tensor_shapes(item)]
    return []


@contextmanager
def _runtime_context(
    model: torch.nn.Module,
    descriptor: Any,
    config: Any,
    case: WidthSliceCase,
):
    before = _hook_snapshot(model)
    evidence = {"count": 0, "executions": []}
    with ExitStack() as stack:
        if case.axis_id == "hidden_width":
            spec = descriptor.embedding_pruning_spec(
                config,
                widths=(case.source_value, case.target_value),
                alignment=1,
            )
            for layer_idx in case.layers:
                stack.enter_context(
                    hidden_width_layer_context(
                        _layer(model, descriptor, layer_idx),
                        canonical_layer_name=descriptor.layer_block_name(layer_idx),
                        spec=spec,
                        width=case.target_value,
                    )
                )
        elif case.axis_id == "ple_width":
            spec = descriptor.ple_pruning_spec(config)
            if spec is None:
                raise RuntimeError("PLE capability declared without a pruning spec")
            for layer_idx in case.layers:
                stack.enter_context(
                    ple_layer_context(
                        _layer(model, descriptor, layer_idx),
                        spec=spec,
                        width=case.target_value,
                    )
                )
        else:
            adapter = _RuntimeRecipeAdapter(model, descriptor)
            stack.enter_context(
                adapter.architecture_context((_prune_target(case, config, descriptor),))
            )
        after = _hook_snapshot(model)
        installed_hooks = {key: value for key, value in after.items() if key not in before}
        installed = len(installed_hooks)
        if installed <= 0:
            raise RuntimeError(f"runtime hook missing for axis={case.axis_id} layers={case.layers}")
        for (module_name, hook_kind, hook_id), (hooks, callback) in installed_hooks.items():

            def monitored(
                *args, __callback=callback, __module=module_name, __kind=hook_kind, **kwargs
            ):
                result = __callback(*args, **kwargs)
                evidence["count"] += 1
                evidence["executions"].append(
                    {
                        "axis_id": case.axis_id,
                        "module": __module,
                        "hook_kind": __kind,
                        "input_shapes": _tensor_shapes(args[1:] if len(args) > 1 else args),
                        "output_shapes": _tensor_shapes(result),
                    }
                )
                return result

            hooks[hook_id] = monitored
        completed = False
        try:
            yield installed, evidence
            completed = True
        finally:
            if completed and evidence["count"] <= 0:
                raise RuntimeError(
                    f"axis target hook did not execute for axis={case.axis_id} layers={case.layers}"
                )


def _forward(model: torch.nn.Module, batch: PuzzletronBatch) -> tuple[torch.Tensor, torch.Tensor]:
    kwargs = validated_forward_kwargs(model, batch)
    signature = inspect.signature(model.forward)
    accepts_kwargs = any(
        parameter.kind is inspect.Parameter.VAR_KEYWORD
        for parameter in signature.parameters.values()
    )
    if batch.labels is not None and (accepts_kwargs or "labels" in signature.parameters):
        kwargs["labels"] = batch.labels
    result = model(**kwargs)
    loss = result.get("loss") if isinstance(result, Mapping) else getattr(result, "loss", None)
    output = (
        result.get("logits") if isinstance(result, Mapping) else getattr(result, "logits", None)
    )
    if not torch.is_tensor(loss) or not torch.is_tensor(output):
        raise TypeError("equivalence models must return tensor loss and logits")
    return loss.detach(), output.detach()


def _shapes(model: torch.nn.Module, descriptor: Any, case: WidthSliceCase) -> dict[str, list[int]]:
    state = model.state_dict()
    prefixes = (
        tuple(descriptor.layer_block_name(layer) + "." for layer in case.layers)
        if case.scope == "layer"
        else ()
    )
    return {
        key: list(value.shape)
        for key, value in state.items()
        if not prefixes or key.startswith(prefixes)
    }


def _changed_shapes(
    before: Mapping[str, list[int]], after: Mapping[str, list[int]]
) -> dict[str, dict[str, list[int] | None]]:
    return {
        key: {"before": before.get(key), "after": after.get(key)}
        for key in sorted(set(before) | set(after))
        if before.get(key) != after.get(key)
    }


def _gdn_projection_width(block_config: BlockConfig | None) -> int | None:
    if block_config is None:
        return None
    mamba = block_config.get_subblock("mamba")
    if mamba is None:
        return None
    values = (
        getattr(mamba, "num_groups", None),
        getattr(mamba, "state_dim", None),
        getattr(mamba, "num_heads", None),
        getattr(mamba, "head_dim", None),
    )
    if any(value is None for value in values):
        return None
    num_groups, state_dim, num_heads, head_dim = (int(value) for value in values)
    return 2 * num_groups * state_dim + num_heads * head_dim


def _axis_specific_changed_shapes(
    before: Mapping[str, list[int]],
    after: Mapping[str, list[int]],
    *,
    descriptor: Any,
    case: WidthSliceCase,
    num_layers: int,
) -> dict[str, dict[str, list[int] | None]]:
    """Return descriptor-owned changes with the requested ratio or coupled geometry."""

    changed = _changed_shapes(before, after)
    pattern = None
    if case.scope == "layer":
        category = "ffn" if case.subblock_kind in {"ffn", "moe"} else "attention"
        pattern = descriptor.layer_name_predicates(num_layers).get(
            f"block_{case.layers[0]}_{category}"
        )
    gdn_projection_widths = (
        _gdn_projection_width(case.source_block_config),
        _gdn_projection_width(case.target_block_config),
    )

    def has_expected_geometry(item: Mapping[str, list[int] | None]) -> bool:
        source_shape = item.get("before")
        target_shape = item.get("after")
        if source_shape is None or target_shape is None:
            return True
        if any(
            int(source_dim) * case.target_value == int(target_dim) * case.source_value
            for source_dim, target_dim in zip(source_shape, target_shape)
            if int(source_dim) > 0 and int(target_dim) > 0 and source_dim != target_dim
        ):
            return True
        source_projection, target_projection = gdn_projection_widths
        return (
            case.axis_id in {"gdn_key_head_dim", "gdn_value_head_dim", "gdn_value_heads_per_group"}
            and source_projection is not None
            and target_projection is not None
            and any(
                source_dim != target_dim
                and int(source_dim) == source_projection
                and int(target_dim) == target_projection
                for source_dim, target_dim in zip(source_shape, target_shape)
            )
        )

    return {
        key: value
        for key, value in changed.items()
        if (pattern is None or pattern.fullmatch(key)) and has_expected_geometry(value)
    }


def _case_path(artifact_dir: Path, case: WidthSliceCase) -> Path:
    layer = "global" if case.scope == "global" else f"layer_{case.layers[0]}"
    return artifact_dir / "cases" / case.axis_id / f"{layer}-{case.case_identity}.json"


def _comparison_path(artifact_dir: Path, case: WidthSliceCase) -> Path:
    return artifact_dir / "comparisons" / f"{case.case_identity}.safetensors"


def _write_comparison_tensors(
    path: Path,
    *,
    physical_loss: torch.Tensor,
    runtime_loss: torch.Tensor,
    physical_output: torch.Tensor,
    runtime_output: torch.Tensor,
) -> dict[str, Any]:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    save_safetensors(
        {
            "physical_loss": physical_loss.detach().float().cpu().contiguous().reshape(()),
            "runtime_loss": runtime_loss.detach().float().cpu().contiguous().reshape(()),
            "physical_output": physical_output.detach().float().cpu().contiguous(),
            "runtime_output": runtime_output.detach().float().cpu().contiguous(),
        },
        str(temporary),
    )
    temporary.replace(path)
    return {
        "path": path.name,
        "sha256": _file_sha256(path),
    }


def _atomic_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def _validate_case_record(
    record: Any,
    *,
    artifact_dir: Path | None = None,
    expected_case: WidthSliceCase | None = None,
    descriptor: Any = None,
    num_layers: int | None = None,
) -> dict[str, Any]:
    if not isinstance(record, dict) or not _REQUIRED_CASE_FIELDS.issubset(record):
        raise RuntimeError("width-slice case has incomplete schema")
    expected_hash = _json_hash(
        {key: value for key, value in record.items() if key != "record_hash"},
        prefix="width_slice_record",
    )
    if record["record_hash"] != expected_hash:
        raise RuntimeError(f"width-slice case hash mismatch: {record.get('case_identity')}")
    if expected_case is not None and record.get("case_contract") != _case_contract(expected_case):
        raise RuntimeError(
            f"width-slice case target/provenance differs from current implementation: "
            f"{record.get('case_identity')}"
        )
    if expected_case is not None:
        expected_declarations = {
            "case_identity": expected_case.case_identity,
            "axis_id": expected_case.axis_id,
            "axis_class": expected_case.axis_class,
            "scope": expected_case.scope,
            "layers": list(expected_case.layers),
            "source_value": expected_case.source_value,
            "target_value": expected_case.target_value,
            "expected_structure": expected_case.expected_structure,
            "implementation_provenance": expected_case.implementation_provenance,
        }
        if any(record.get(key) != value for key, value in expected_declarations.items()):
            raise RuntimeError("width-slice case declarations differ from regenerated target")
    lineage = record.get("lineage")
    if not isinstance(lineage, dict) or any(
        lineage.get(key) != expected
        for key, expected in (
            ("physical_source_identity", record["checkpoint_identity"]),
            ("runtime_source_identity", record["checkpoint_identity"]),
            ("physical_source_config_identity", record["checkpoint_config_identity"]),
            ("runtime_source_config_identity", record["checkpoint_config_identity"]),
        )
    ):
        raise RuntimeError("width-slice case lineage identities differ")
    provenance = record.get("implementation_provenance")
    if (
        not isinstance(provenance, dict)
        or provenance.get("version") != _IMPLEMENTATION_VERSION
        or not isinstance(provenance.get("source_hash"), str)
    ):
        raise RuntimeError("width-slice case implementation provenance is incomplete")
    structure = record.get("structural_evidence")
    if (
        not isinstance(structure, dict)
        or not isinstance(structure.get("before"), dict)
        or not isinstance(structure.get("after"), dict)
        or not isinstance(structure.get("changed_tensors"), dict)
        or not isinstance(structure.get("axis_changed_tensors"), dict)
        or not structure["axis_changed_tensors"]
    ):
        raise RuntimeError("width-slice case structural evidence is missing")
    if expected_case is not None and descriptor is not None and num_layers is not None:
        recomputed_axis_shapes = _axis_specific_changed_shapes(
            structure["before"],
            structure["after"],
            descriptor=descriptor,
            case=expected_case,
            num_layers=num_layers,
        )
        if structure["axis_changed_tensors"] != recomputed_axis_shapes:
            raise RuntimeError("width-slice axis-specific structural evidence differs")
    runtime_axis_evidence = record.get("runtime_axis_evidence")
    if (
        not isinstance(runtime_axis_evidence, list)
        or len(runtime_axis_evidence) != record.get("runtime_hook_executions")
        or any(item.get("axis_id") != record.get("axis_id") for item in runtime_axis_evidence)
    ):
        raise RuntimeError("width-slice runtime axis hook evidence is incomplete")
    metrics = record.get("metrics")
    if not isinstance(metrics, dict) or not all(
        key in metrics
        for key in (
            "loss_delta",
            "loss_allowed_delta",
            "loss_close",
            "output_close",
            "passed",
        )
    ):
        raise RuntimeError("width-slice case has incomplete metrics schema")
    numeric_metrics = (
        "physical_loss",
        "runtime_loss",
        "loss_delta",
        "loss_allowed_delta",
        "output_max_abs_delta",
        "output_mean_abs_delta",
        "output_max_tolerance_excess",
    )
    if any(
        value is not None
        and (not isinstance(value, int | float) or not math.isfinite(float(value)))
        for value in (metrics.get(key) for key in numeric_metrics)
    ):
        raise RuntimeError("width-slice case metrics must be finite")
    physical_loss = float(metrics.get("physical_loss"))
    runtime_loss = float(metrics.get("runtime_loss"))
    tolerances = _normalized_tolerances(record.get("tolerances"))
    expected_loss_delta = abs(physical_loss - runtime_loss)
    expected_loss_allowed = tolerances["loss_atol"] + tolerances["loss_rtol"] * max(
        abs(physical_loss), abs(runtime_loss)
    )
    shapes_match = metrics.get("physical_output_shape") == metrics.get("runtime_output_shape")
    output_excess = metrics.get("output_max_tolerance_excess")
    expected_output_close = bool(
        shapes_match and output_excess is not None and float(output_excess) <= 0.0
    )
    expected_loss_close = expected_loss_delta <= expected_loss_allowed
    if (
        not math.isclose(
            float(metrics["loss_delta"]), expected_loss_delta, rel_tol=0.0, abs_tol=1e-12
        )
        or not math.isclose(
            float(metrics["loss_allowed_delta"]),
            expected_loss_allowed,
            rel_tol=0.0,
            abs_tol=1e-12,
        )
        or metrics["loss_close"] is not expected_loss_close
        or metrics["output_close"] is not expected_output_close
        or metrics["passed"] is not (expected_loss_close and expected_output_close)
    ):
        raise RuntimeError("width-slice derived metrics/verdicts are inconsistent")
    if artifact_dir is not None:
        comparison = record.get("comparison_evidence")
        if not isinstance(comparison, dict) or not isinstance(comparison.get("path"), str):
            raise RuntimeError("width-slice raw comparison evidence is missing")
        comparison_path = artifact_dir / "comparisons" / Path(comparison["path"]).name
        if not comparison_path.is_file() or comparison.get("sha256") != _file_sha256(
            comparison_path
        ):
            raise RuntimeError("width-slice raw comparison evidence hash mismatch")
        tensors = load_safetensors(str(comparison_path))
        if set(tensors) != {
            "physical_loss",
            "runtime_loss",
            "physical_output",
            "runtime_output",
        }:
            raise RuntimeError("width-slice raw comparison tensor schema is incomplete")
        raw_metrics = compare_width_slice_outputs(
            physical_loss=tensors["physical_loss"],
            runtime_loss=tensors["runtime_loss"],
            physical_output=tensors["physical_output"],
            runtime_output=tensors["runtime_output"],
            tolerances=tolerances,
        )
        if canonicalize(metrics) != canonicalize(raw_metrics):
            raise RuntimeError("width-slice metrics differ from raw comparison tensors")
    recomputed_passed = bool(
        record["target_applied"]
        and record["runtime_hook_count"] > 0
        and record["runtime_hook_executions"] > 0
        and metrics["loss_close"]
        and metrics["output_close"]
        and metrics["passed"]
    )
    if record["passed"] is not recomputed_passed:
        raise RuntimeError("width-slice case passed field does not match semantic evidence")
    return record


def _existing_case(
    path: Path,
    expected: Mapping[str, Any],
    *,
    artifact_dir: Path,
    case: WidthSliceCase,
    descriptor: Any,
    num_layers: int,
) -> dict[str, Any] | None:
    if not path.is_file():
        return None
    try:
        record = _validate_case_record(
            json.loads(path.read_text(encoding="utf-8")),
            artifact_dir=artifact_dir,
            expected_case=case,
            descriptor=descriptor,
            num_layers=num_layers,
        )
    except (json.JSONDecodeError, RuntimeError, TypeError, ValueError):
        return None
    if any(record.get(key) != value for key, value in expected.items()):
        return None
    return record


def _evaluate_case(
    *,
    descriptor: Any,
    checkpoint_dir: Path,
    checkpoint_config: Any,
    checkpoint_identity: str,
    checkpoint_config_identity: str,
    batch: PuzzletronBatch,
    case: WidthSliceCase,
    artifact_dir: Path,
    tolerances: Mapping[str, float],
) -> dict[str, Any]:
    child_config = _child_config(descriptor, checkpoint_config, case)
    target_config_identity = _config_identity(child_config, prefix="width_slice_target_config")
    replacements = []
    physical_source = checkpoint_dir
    if case.axis_id == "hidden_width":
        physical_source = artifact_dir / "materialized" / case.case_identity
        materialize_hidden_width_checkpoint(
            checkpoint_dir,
            descriptor,
            case.target_value,
            physical_source,
            alignment=1,
            overwrite=False,
        )
        physical_config = load_model_config(
            physical_source,
            trust_remote_code=descriptor.requires_trust_remote_code(),
        )
        physical_model = materialize_model_from_sorted(
            physical_source,
            [],
            descriptor,
            physical_config,
        )
    else:
        if case.scope == "layer":
            replacements = [
                {
                    "parent_layer_indices": [case.layers[0]],
                    "child_block_configs": [case.target_block_config],
                }
            ]
        physical_model = materialize_model_from_sorted(
            checkpoint_dir,
            replacements,
            descriptor,
            child_config,
        )
    runtime_model = materialize_model_from_sorted(
        checkpoint_dir,
        [],
        descriptor,
        copy.deepcopy(checkpoint_config),
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    physical_model.to(device).eval()
    runtime_model.to(device).eval()
    device_batch = batch.to(device)
    runtime_shapes = _shapes(runtime_model, descriptor, case)
    physical_shapes = _shapes(physical_model, descriptor, case)
    changed = _changed_shapes(runtime_shapes, physical_shapes)
    axis_changed = _axis_specific_changed_shapes(
        runtime_shapes,
        physical_shapes,
        descriptor=descriptor,
        case=case,
        num_layers=len(_block_configs(checkpoint_config)),
    )
    if case.expected_structure.get("requires_tensor_shape_change") and not axis_changed:
        raise RuntimeError(
            "vacuous or unrelated physical shape change: "
            f"target not applied for {case.axis_id} {case.layers}"
        )
    with torch.inference_mode():
        physical_loss, physical_output = _forward(physical_model, device_batch)
        with _runtime_context(runtime_model, descriptor, checkpoint_config, case) as (
            hook_count,
            runtime_evidence,
        ):
            runtime_loss, runtime_output = _forward(runtime_model, device_batch)
    if runtime_evidence["count"] <= 0:
        raise RuntimeError(f"runtime axis hook did not execute: {case.case_identity}")
    metrics = compare_width_slice_outputs(
        physical_loss=physical_loss,
        runtime_loss=runtime_loss,
        physical_output=physical_output,
        runtime_output=runtime_output,
        tolerances=tolerances,
    )
    target_applied = bool(
        case.source_value != case.target_value and axis_changed and hook_count > 0
    )
    if not target_applied:
        raise RuntimeError(f"target was not applied for case {case.case_identity}")
    comparison_path = _comparison_path(artifact_dir, case)
    comparison_evidence = _write_comparison_tensors(
        comparison_path,
        physical_loss=physical_loss,
        runtime_loss=runtime_loss,
        physical_output=physical_output,
        runtime_output=runtime_output,
    )
    record = {
        "schema_version": _SCHEMA_VERSION,
        "case_identity": case.case_identity,
        "case_contract": _case_contract(case),
        "axis_id": case.axis_id,
        "axis_class": case.axis_class,
        "scope": case.scope,
        "layers": list(case.layers),
        "layer_idx": case.layers[0] if case.scope == "layer" else None,
        "source_value": case.source_value,
        "target_value": case.target_value,
        "expected_structure": case.expected_structure,
        "checkpoint_identity": checkpoint_identity,
        "checkpoint_config_identity": checkpoint_config_identity,
        "target_config_identity": target_config_identity,
        "batch_fingerprint": batch.fingerprint,
        "implementation_provenance": case.implementation_provenance,
        "tolerances": dict(tolerances),
        "lineage": {
            "physical_source_identity": checkpoint_identity,
            "runtime_source_identity": checkpoint_identity,
            "physical_source_checkpoint": str(physical_source),
            "runtime_source_checkpoint": str(checkpoint_dir),
            "physical_source_config_identity": checkpoint_config_identity,
            "runtime_source_config_identity": checkpoint_config_identity,
        },
        "structural_evidence": {
            "before": runtime_shapes,
            "after": physical_shapes,
            "changed_tensors": changed,
            "axis_changed_tensors": axis_changed,
        },
        "runtime_hook_count": hook_count,
        "runtime_hook_executions": runtime_evidence["count"],
        "runtime_axis_evidence": runtime_evidence["executions"],
        "comparison_evidence": comparison_evidence,
        "target_applied": target_applied,
        "metrics": metrics,
        "passed": bool(target_applied and metrics["passed"]),
    }
    record["record_hash"] = _json_hash(record, prefix="width_slice_record")
    del physical_model, runtime_model
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return record


def evaluate_width_slice_equivalence(
    *,
    descriptor: Any,
    sorted_checkpoint_dir: str | Path,
    batch: PuzzletronBatch,
    artifact_dir: str | Path,
    tolerances: Mapping[str, float] | None = None,
    alignment: int = 1,
    sampled_layers: Sequence[int] | None = None,
) -> dict[str, Any]:
    """Execute all declarative cases with strict lineage and resumable evidence."""

    if not isinstance(batch, PuzzletronBatch):
        raise TypeError(f"batch must be PuzzletronBatch, got {type(batch).__name__}")
    checkpoint_dir = Path(sorted_checkpoint_dir)
    output_dir = Path(artifact_dir)
    gates = _normalized_tolerances(tolerances)
    checkpoint_identity, config_identity, inventory = _checkpoint_identity(checkpoint_dir)
    config = load_model_config(
        checkpoint_dir,
        trust_remote_code=descriptor.requires_trust_remote_code(),
    )
    cases = descriptor.width_slice_equivalence_operations(
        config,
        checkpoint_dir,
        alignment=alignment,
        sampled_layers=sampled_layers,
    )
    descriptor_reference = _descriptor_reference(descriptor)
    case_generation = {
        "alignment": int(alignment),
        "sampled_layers": None
        if sampled_layers is None
        else [int(layer) for layer in sampled_layers],
    }
    run_identity = _json_hash(
        {
            "schema_version": _SCHEMA_VERSION,
            "checkpoint_identity": checkpoint_identity,
            "checkpoint_config_identity": config_identity,
            "case_identities": sorted(cases),
            "batch_fingerprint": batch.fingerprint,
            "tolerances": gates,
            "descriptor": descriptor_reference,
            "case_generation": case_generation,
        },
        prefix="width_slice_run",
    )
    expected_case_paths = {_case_path(output_dir, case) for case in cases.values()}
    expected_comparison_paths = {_comparison_path(output_dir, case) for case in cases.values()}
    for stale_path in (output_dir / "cases").rglob("*.json"):
        if stale_path not in expected_case_paths:
            stale_path.unlink()
    for stale_path in (output_dir / "comparisons").glob("*.safetensors"):
        if stale_path not in expected_comparison_paths:
            stale_path.unlink()
    records = []
    for case_id, case in sorted(cases.items()):
        expected = {
            "schema_version": _SCHEMA_VERSION,
            "case_identity": case_id,
            "checkpoint_identity": checkpoint_identity,
            "checkpoint_config_identity": config_identity,
            "batch_fingerprint": batch.fingerprint,
            "implementation_provenance": case.implementation_provenance,
            "tolerances": gates,
        }
        path = _case_path(output_dir, case)
        record = _existing_case(
            path,
            expected,
            artifact_dir=output_dir,
            case=case,
            descriptor=descriptor,
            num_layers=len(_block_configs(config)),
        )
        if record is None:
            record = _evaluate_case(
                descriptor=descriptor,
                checkpoint_dir=checkpoint_dir,
                checkpoint_config=config,
                checkpoint_identity=checkpoint_identity,
                checkpoint_config_identity=config_identity,
                batch=batch,
                case=case,
                artifact_dir=output_dir,
                tolerances=gates,
            )
            _atomic_json(path, record)
        records.append(record)

    summary = {
        "schema_version": _SCHEMA_VERSION,
        "status": "complete",
        "run_identity": run_identity,
        "checkpoint_identity": checkpoint_identity,
        "checkpoint_dir": str(checkpoint_dir.resolve()),
        "checkpoint_config_identity": config_identity,
        "checkpoint_inventory": inventory,
        "batch_fingerprint": batch.fingerprint,
        "batch_layout": batch.layout.value,
        "batch_modality": batch.modality.value,
        "tolerances": gates,
        "descriptor": descriptor_reference,
        "case_generation": case_generation,
        "case_contracts": {case_id: _case_contract(case) for case_id, case in cases.items()},
        "case_hashes": {record["case_identity"]: record["record_hash"] for record in records},
        "cases": records,
        "passed": all(record["passed"] for record in records),
    }
    summary["artifact_identity"] = _json_hash(summary, prefix="width_slice_summary")
    _atomic_json(output_dir / "summary.json", summary)
    manifest = {
        "schema_version": _SCHEMA_VERSION,
        "status": "complete",
        "run_identity": run_identity,
        "checkpoint_identity": checkpoint_identity,
        "checkpoint_dir": str(checkpoint_dir.resolve()),
        "checkpoint_config_identity": config_identity,
        "batch_fingerprint": batch.fingerprint,
        "tolerances": gates,
        "descriptor": descriptor_reference,
        "case_generation": case_generation,
        "case_contracts": summary["case_contracts"],
        "expected_cases": sorted(cases),
        "completed_cases": sorted(record["case_identity"] for record in records),
        "case_hashes": summary["case_hashes"],
        "summary_path": str(output_dir / "summary.json"),
        "summary_identity": summary["artifact_identity"],
        "passed": summary["passed"],
        "artifact_identity": summary["artifact_identity"],
    }
    manifest["manifest_hash"] = _json_hash(manifest, prefix="width_slice_manifest")
    _atomic_json(output_dir / "manifest.json", manifest)
    validate_width_slice_artifacts(output_dir, descriptor=descriptor)
    return summary


def validate_width_slice_artifacts(
    artifact_dir: str | Path,
    *,
    descriptor: Any = None,
) -> dict[str, Any]:
    """Validate complete case, summary, and manifest schemas and content hashes."""

    root = Path(artifact_dir)
    manifest_path = root / "manifest.json"
    summary_path = root / "summary.json"
    if not manifest_path.is_file() or not summary_path.is_file():
        raise RuntimeError(f"missing width-slice manifest or summary under {root}")
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as error:
        raise RuntimeError(f"invalid width-slice JSON: {error}") from error
    manifest_hash = _json_hash(
        {key: value for key, value in manifest.items() if key != "manifest_hash"},
        prefix="width_slice_manifest",
    )
    if manifest.get("manifest_hash") != manifest_hash:
        raise RuntimeError("width-slice manifest hash mismatch")
    summary_identity = _json_hash(
        {key: value for key, value in summary.items() if key != "artifact_identity"},
        prefix="width_slice_summary",
    )
    if summary.get("artifact_identity") != summary_identity:
        raise RuntimeError("width-slice summary hash mismatch")
    if manifest.get("summary_identity") != summary_identity:
        raise RuntimeError("width-slice manifest/summary identity mismatch")
    if manifest.get("artifact_identity") != summary_identity:
        raise RuntimeError("width-slice manifest artifact identity mismatch")
    checkpoint_dir = manifest.get("checkpoint_dir")
    if not isinstance(checkpoint_dir, str) or summary.get("checkpoint_dir") != checkpoint_dir:
        raise RuntimeError("width-slice checkpoint lineage path is missing")
    descriptor_reference = manifest.get("descriptor")
    if summary.get("descriptor") != descriptor_reference:
        raise RuntimeError("width-slice descriptor identity differs")
    current_descriptor = descriptor or _resolve_descriptor(descriptor_reference)
    if _descriptor_reference(current_descriptor) != descriptor_reference:
        raise RuntimeError("width-slice current descriptor identity differs")
    checkpoint_path = Path(checkpoint_dir)
    current_checkpoint_identity, current_config_identity, _ = _checkpoint_identity(checkpoint_path)
    if (
        manifest.get("checkpoint_identity") != current_checkpoint_identity
        or summary.get("checkpoint_identity") != current_checkpoint_identity
        or manifest.get("checkpoint_config_identity") != current_config_identity
        or summary.get("checkpoint_config_identity") != current_config_identity
    ):
        raise RuntimeError("width-slice checkpoint content identity changed")
    generation = manifest.get("case_generation")
    if not isinstance(generation, dict) or summary.get("case_generation") != generation:
        raise RuntimeError("width-slice case generation settings are missing")
    current_config = load_model_config(
        checkpoint_path,
        trust_remote_code=current_descriptor.requires_trust_remote_code(),
    )
    regenerated = current_descriptor.width_slice_equivalence_operations(
        current_config,
        checkpoint_path,
        alignment=int(generation.get("alignment", 1)),
        sampled_layers=generation.get("sampled_layers"),
    )
    regenerated_contracts = {case_id: _case_contract(case) for case_id, case in regenerated.items()}
    if (
        manifest.get("case_contracts") != regenerated_contracts
        or summary.get("case_contracts") != regenerated_contracts
    ):
        raise RuntimeError(
            "width-slice cases/targets/provenance differ from current implementation"
        )
    recomputed_run_identity = _json_hash(
        {
            "schema_version": _SCHEMA_VERSION,
            "checkpoint_identity": current_checkpoint_identity,
            "checkpoint_config_identity": current_config_identity,
            "case_identities": sorted(regenerated),
            "batch_fingerprint": summary.get("batch_fingerprint"),
            "tolerances": summary.get("tolerances"),
            "descriptor": descriptor_reference,
            "case_generation": generation,
        },
        prefix="width_slice_run",
    )
    if (
        summary.get("run_identity") != recomputed_run_identity
        or manifest.get("run_identity") != recomputed_run_identity
        or manifest.get("batch_fingerprint") != summary.get("batch_fingerprint")
        or manifest.get("tolerances") != summary.get("tolerances")
    ):
        raise RuntimeError("width-slice run identity differs from regenerated inputs")
    expected = manifest.get("expected_cases")
    completed = manifest.get("completed_cases")
    if not isinstance(expected, list) or completed != expected or expected != sorted(regenerated):
        raise RuntimeError("width-slice manifest has incomplete case coverage")
    records_by_id = {}
    for path in sorted((root / "cases").rglob("*.json")):
        raw_record = json.loads(path.read_text(encoding="utf-8"))
        case_id = raw_record.get("case_identity")
        if case_id not in regenerated:
            raise RuntimeError(f"unexpected width-slice case identity: {case_id}")
        record = _validate_case_record(
            raw_record,
            artifact_dir=root,
            expected_case=regenerated[case_id],
            descriptor=current_descriptor,
            num_layers=len(_block_configs(current_config)),
        )
        records_by_id[record["case_identity"]] = record
    missing = sorted(set(expected) - set(records_by_id))
    if missing:
        raise RuntimeError(f"missing width-slice case artifacts: {missing}")
    if set(records_by_id) != set(expected):
        raise RuntimeError("width-slice case artifacts have unexpected identities")
    case_hashes = {case_id: records_by_id[case_id]["record_hash"] for case_id in expected}
    if manifest.get("case_hashes") != case_hashes or summary.get("case_hashes") != case_hashes:
        raise RuntimeError("width-slice case hash inventory mismatch")
    summary_cases = summary.get("cases")
    if (
        not isinstance(summary_cases, list)
        or {record.get("case_identity"): record.get("record_hash") for record in summary_cases}
        != case_hashes
    ):
        raise RuntimeError("width-slice summary case schema is incomplete")
    if summary_cases != [records_by_id[case_id] for case_id in expected]:
        raise RuntimeError("width-slice summary cases differ from validated case artifacts")
    comparison_names = {
        Path(records_by_id[case_id]["comparison_evidence"]["path"]).name for case_id in expected
    }
    actual_comparisons = {path.name for path in (root / "comparisons").glob("*.safetensors")}
    if actual_comparisons != comparison_names:
        raise RuntimeError("width-slice raw comparison artifact inventory differs")
    passed = all(records_by_id[case_id]["passed"] for case_id in expected)
    if summary.get("passed") is not passed or manifest.get("passed") is not passed:
        raise RuntimeError("width-slice aggregate passed field is inconsistent")
    if manifest.get("status") != "complete" or summary.get("status") != "complete":
        raise RuntimeError("width-slice artifacts are not semantically complete")
    return summary
