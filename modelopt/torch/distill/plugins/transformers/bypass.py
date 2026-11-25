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

"""ModelSpec-driven text-only Bypass on stable Transformers 5.23.x."""

import importlib
import inspect
import weakref
from contextlib import ExitStack
from dataclasses import dataclass, replace
from importlib.metadata import PackageNotFoundError, version

import torch
from packaging.version import Version
from torch import nn

from modelopt.torch.models.specs import BypassSpec, get_spec
from modelopt.torch.quantization.nn import QuantModule, TensorQuantizer

from ..._bypass_capture import (
    BypassUnit,
    ModuleBypassAdapter,
    _valid_token_mask,
    has_iq_weights,
    validate_unit_ownership,
)
from .output_head import bypass_head_context
from .packing import batch_documents, document_batches, empty_document_batch


def _require_transformers():
    try:
        installed = Version(version("transformers"))
    except PackageNotFoundError as exc:
        raise ImportError("Transformers Bypass requires transformers~=5.23.0") from exc
    if installed.release[:2] != (5, 23) or installed.is_prerelease or installed.is_devrelease:
        raise ImportError(f"Transformers Bypass requires stable 5.23.x; found {installed}")


def _model_spec(model):
    config = getattr(model, "config", None)
    candidates = [getattr(config, "model_type", None)]
    getter = getattr(config, "get_text_config", None)
    if callable(getter):
        candidates.append(getattr(getter(), "model_type", None))
    for name in candidates:
        spec = get_spec(name) if isinstance(name, str) else None
        if spec is not None and spec.bypass_spec is not None:
            return spec.bypass_spec
    return BypassSpec()


def _document_dependent_positions(model):
    config = getattr(model, "config", None)
    getter = getattr(config, "get_text_config", None)
    config = getter() if callable(getter) else config
    pending = [getattr(config, name, None) for name in ("rope_scaling", "rope_parameters")]
    while pending:
        value = pending.pop()
        if isinstance(value, dict):
            if value.get("rope_type", value.get("type")) in ("dynamic", "longrope"):
                return True
            pending.extend(value.values())
    return False


@dataclass(frozen=True)
class _ReplayFacts:
    spec: BypassSpec
    modules: tuple[weakref.ReferenceType[nn.Module], ...]
    head_ref: weakref.ReferenceType[nn.Module] | None
    head_paths: tuple[str, ...]

    @property
    def head(self):
        return None if self.head_ref is None else self.head_ref()


class TransformersBypassAdapter(ModuleBypassAdapter):
    """Adopt native text decoders and caller-owned placement, without a family registry."""

    def __init__(self, device_mesh=None):
        """Require the tested integration version only when this adapter is used."""
        _require_transformers()
        self.device_mesh = device_mesh
        self.context = None
        if device_mesh is not None:
            from ..._bypass_parallel import torch_mesh_context
            from .parallel import native_mesh_context

            self.context = (
                native_mesh_context(device_mesh)
                if callable(getattr(device_mesh, "get_mesh", None))
                else torch_mesh_context(device_mesh)
            )
        self.student = None
        self._native_context = False
        self._providers = weakref.WeakKeyDictionary()
        self._model_facts = weakref.WeakKeyDictionary()

    def _facts(self, model):
        if model is None:
            return _ReplayFacts(BypassSpec(), (), None, ())
        task = getattr(model, "thinker", model)
        getter = getattr(task, "get_output_embeddings", None)
        try:
            head = getter() if callable(getter) else None
        except NotImplementedError:
            head = None
        facts = self._model_facts.get(model)
        if facts is None or facts.head is not head:
            modules = tuple(model.named_modules())
            facts = _ReplayFacts(
                _model_spec(model),
                tuple(weakref.ref(child) for _, child in modules),
                None if head is None else weakref.ref(head),
                tuple(name for name, child in modules if child is head),
            )
            self._model_facts[model] = facts
        return facts

    def _remember_tp_outputs(self, model, plan=None):
        if (
            self.context is None
            or self.context.tensor_mesh is None
            or self.context.tensor_mesh.size() == 1
        ):
            return
        from .parallel import _tp_output_layouts

        metadata = self.context.parallel_metadata.get(id(model))
        if metadata is None:
            metadata = getattr(model, "_modelopt_bypass_placement", {})
        if "tp_outputs" in metadata:
            return
        outputs = getattr(model, "_modelopt_bypass_tp_outputs", None)
        if outputs is None:
            if plan is None:
                plan = metadata.get("plans", {}).get("tp")
            if plan is None:
                if hasattr(model, "_modelopt_bypass_torch_mesh"):
                    raise ValueError(
                        "Prepared native HF model lacks retained TP output declarations"
                    )
                return  # Unplaced teachers have no applied native output declaration.
            outputs = _tp_output_layouts(model, plan)
        # Runtime-only facts describe the applied plan, never subsequently edited rules.
        self.context.parallel_metadata[id(model)] = {**metadata, "tp_outputs": outputs}
        if self.context.mesh_manager is None:
            model._modelopt_bypass_tp_outputs = outputs

    def parallelize(self, model, *, initialize=None):
        """Stage native fields before caller-owned HF TP/EP/FSDP, or validate adoption."""
        self._model_facts.pop(model, None)
        if self.context is None:
            raise ValueError("parallelize requires the caller's actual device_mesh")
        if self.context.mesh_manager is not None:
            from .parallel import parallelize_native_transformers

            parallelize_native_transformers(model, self.context, initialize=initialize)
            self._remember_tp_outputs(model)
            return model
        from ..._bypass_parallel import _size, prepare_quantizer_partitions
        from .parallel import (
            _synchronize_placement_state,
            _synchronize_validation,
            apply_hf_tensor_parallelism,
            validate_hf_tensor_parallelism,
        )

        prepared = hasattr(model, "_modelopt_bypass_torch_mesh")
        _synchronize_placement_state(self.context, prepared)
        error, tp_plan = None, None
        try:
            if initialize is not None:
                raise ValueError("Native initialization callbacks require TransformersDeviceMesh")
            if prepared and model._modelopt_bypass_torch_mesh is not self.device_mesh:
                raise ValueError("Prepared Torch models must use their exact caller-owned mesh")
            if _size(self.context.pipeline_mesh) > 1:
                raise ValueError("Native PP requires a validated family transport")
            if _size(self.context.tensor_mesh) > 1:
                if prepared:
                    if not hasattr(model, "_modelopt_bypass_tp_outputs"):
                        raise ValueError(
                            "Prepared native HF model lacks retained TP output declarations"
                        )
                else:
                    tp_rules = dict(model.tp_plan)
                    tp_plan = model._expand_parallel_plan(tp_rules)
                    validate_hf_tensor_parallelism(
                        model, self.context.tensor_mesh, tp_plan, rules=tp_rules
                    )
        except Exception as exc:
            error = f"{type(exc).__name__}: {exc}"
        _synchronize_validation(self.context, error)
        if _size(self.context.tensor_mesh) > 1 and not prepared:
            from modelopt.torch.quantization.repair_partition import prepare_packed_weights_for_tp

            prepare_packed_weights_for_tp(model)
            apply_hf_tensor_parallelism(model, self.context.tensor_mesh, tp_plan)
            prepare_quantizer_partitions(model, self.context)
        model._modelopt_bypass_torch_mesh = self.device_mesh
        self._remember_tp_outputs(model, tp_plan)
        return model

    def adopt(self, model):
        """Collectively validate realized placement without installing another set of hooks."""
        if self.context is None:
            return
        from .parallel import _synchronize_validation

        error = None
        if self.context.mesh_manager is not None:
            if not getattr(model, "_mesh_manager", None) and self.context.device_mesh.size() > 1:
                error = "Prepare IQ placement with TransformersBypassAdapter.parallelize first"
        elif getattr(model, "_modelopt_bypass_torch_mesh", None) is not self.device_mesh and (
            hasattr(model, "_modelopt_bypass_torch_mesh") or self.context.device_mesh.size() > 1
        ):
            error = "Already placed Torch models must use their exact caller-owned mesh"
        _synchronize_validation(self.context, error)
        if self.context.mesh_manager is not None:
            self.parallelize(model)

    def _provider(self, model):
        spec = _model_spec(model)
        if spec.adapter_path is None:
            return None
        if model not in self._providers:
            module, _, attribute = spec.adapter_path.rpartition(".")
            self._providers[model] = getattr(importlib.import_module(module), attribute)()
        return self._providers[model]

    def _decoder_layers(self, model):
        provider = self._provider(model)
        if provider is not None and hasattr(provider, "decoder_layers"):
            return provider.decoder_layers(model)
        spec = _model_spec(model)
        for path in spec.decoder_paths:
            try:
                layers = model.get_submodule(path)
            except AttributeError:
                continue
            if isinstance(layers, (nn.ModuleList, nn.ModuleDict)):
                return layers
        if spec.decoder_paths:
            raise ValueError("No text decoder matches this model's BypassSpec.decoder_paths")
        # Unknown ordinary models use public decoder roots and conservative document isolation.
        queue, seen = [model], set()
        getter = getattr(model, "get_decoder", None)
        if callable(getter):
            try:
                decoder = getter()
                if isinstance(decoder, nn.Module):
                    queue.insert(0, decoder)
            except (AttributeError, NotImplementedError):
                pass
        while queue:
            root = queue.pop(0)
            if id(root) in seen:
                continue
            seen.add(id(root))
            for name in ("layers", "h", "block"):
                layers = getattr(root, name, None)
                if isinstance(layers, (nn.ModuleList, nn.ModuleDict)):
                    return layers
            queue.extend(
                child
                for name in (
                    "thinker",
                    "language_model",
                    "text_model",
                    "decoder",
                    "model",
                    "backbone",
                    "transformer",
                )
                if isinstance(child := getattr(root, name, None), nn.Module)
                and id(child) not in seen
            )
        raise ValueError(
            "No text decoder found; provide exact modules or a ModelSpec Bypass adapter"
        )

    def _layers(self, model):
        layers = self._decoder_layers(model)
        names = {id(module): name for name, module in model.named_modules()}
        items = layers.items() if isinstance(layers, nn.ModuleDict) else enumerate(layers)
        return [(int(index), names[id(module)], module) for index, module in items]

    def _selected_paths(self, model, config, *, quantized_only=True):
        try:
            layers = self._layers(model)
        except ValueError:
            if config.modules is None or config.layer_indices is not None:
                raise
            layers = None
        selected = config.layer_indices
        if selected is not None:
            available = {index for index, _, _ in layers}
            if (
                not selected
                or len(selected) != len(set(selected))
                or not set(selected).issubset(available)
            ):
                raise ValueError(
                    "layer_indices must select distinct existing logical decoder positions"
                )
        paths = config.modules
        if paths is None:
            roles, provider = _model_spec(model).feedforward_names, self._provider(model)
            paths = []
            for index, path, block in layers:
                if selected is not None and index not in selected:
                    continue
                attributes = (
                    provider.unit_attributes(block)
                    if provider is not None and hasattr(provider, "unit_attributes")
                    else roles
                )
                for name in attributes:
                    child = block._modules.get(name)
                    if child is not None and (not quantized_only or has_iq_weights(child)):
                        paths.append(path + "." + name)
            paths = [
                path
                for path in paths
                if not any(path.startswith(parent + ".") for parent in paths if parent != path)
            ]
        elif selected is not None:
            positions = {
                path: next(
                    (
                        index
                        for index, block, _ in layers
                        if path == block or path.startswith(block + ".")
                    ),
                    None,
                )
                for path in paths
            }
            if any(index is None for index in positions.values()):
                raise ValueError(
                    "Every module intersected with layer_indices must belong to one decoder position"
                )
            paths = [path for path in paths if positions[path] in selected]
        if not paths:
            raise ValueError("Bypass selectors produced no eligible text boundaries")
        return layers, paths

    def weight_patterns(self, model, config=None):
        """Derive ordinary ModelOpt PTQ weight globs from the same selected boundaries.

        Call before PTQ to configure the weights that Bypass will later repair.
        ``config`` accepts the same module/layer selectors as :func:`bypass`;
        routing gates are excluded and weights need not already be quantized.
        Each returned glob can be used as a ``quantizer_name`` in the ordinary
        ModelOpt ``quant_cfg`` list. The stacked tutorial uses this contract.
        """
        from modelopt.torch.quantization.repair import _router_paths

        from ...config import BypassConfig

        config = config if isinstance(config, BypassConfig) else BypassConfig(**(config or {}))
        _, paths = self._selected_paths(model, config, quantized_only=False)
        patterns = []
        for path in paths:
            owner = model.get_submodule(path)
            routers = _router_paths(owner)
            # fnmatch's '*' crosses module separators; ancestor globs can re-enable routers.
            for relative, module in owner.named_modules():
                if any(
                    not relative
                    or relative == router
                    or relative.startswith(router + ".")
                    or router.startswith(relative + ".")
                    for router in routers
                ):
                    continue
                if any(parameter.ndim >= 2 for parameter in module.parameters(recurse=False)):
                    patterns.append(".".join(filter(None, (path, relative, "*weight_quantizer*"))))
        return list(dict.fromkeys(patterns))

    def resolve_units(self, student, teacher, config):
        """Select dense/MoE owners and map their exact logical positions to the teacher."""
        for model in (student, teacher):
            self._model_facts.pop(model, None)
            self._remember_tp_outputs(model)
        self.student = student
        layers, paths = self._selected_paths(student, config)
        targets = (
            {index: path for index, path, _ in self._layers(teacher)} if layers is not None else {}
        )
        units = []
        for path in paths:
            target = path
            for index, block, _ in layers or ():
                if path == block or path.startswith(block + "."):
                    if index not in targets:
                        raise ValueError("Teacher is missing a selected logical decoder position")
                    target = targets[index] + path[len(block) :]
                    break
            units.append(BypassUnit(path, path, target))
        validate_unit_ownership(student, teacher, units)
        return units

    def _batch_with_validity(self, batch):
        documents = batch.get("_packed_seq_ids") if isinstance(batch, dict) else None
        if documents is None:
            return batch
        tokens = batch.get("input_ids")
        if (
            not isinstance(documents, torch.Tensor)
            or not isinstance(tokens, torch.Tensor)
            or documents.shape != tokens.shape
            or documents.ndim != 2
            or documents.is_floating_point()
            or documents.is_complex()
            or bool((documents < 0).any())
        ):
            raise ValueError("Packed document IDs must be nonnegative integers matching input_ids")
        if batch.get("valid_mask") is not None or batch.get("loss_mask") is not None:
            return batch
        # This HF-owned convention defines zero as padding, independently of 4D attention.
        return {**batch, "valid_mask": documents.ne(0)}

    def validate_batch(self, batch, config):
        """Establish HF-owned packed validity before generic budget/mask preflight."""
        if not isinstance(batch, dict):
            raise ValueError("Transformers Bypass requires dict batches")
        self._validate_text_batch(batch)
        super().validate_batch(self._batch_with_validity(batch), config)

    def microbatches(self, batch, size):
        """Preserve [RoPE axes,batch,sequence] wrapper positions when slicing samples."""
        batch = self._batch_with_validity(batch)
        positions = batch.get("position_ids") if isinstance(batch, dict) else None
        if not isinstance(positions, torch.Tensor) or positions.ndim != 3:
            yield from super().microbatches(batch, size)
            return
        inputs = batch.get("input_ids", batch.get("inputs_embeds"))
        if not isinstance(inputs, torch.Tensor) or positions.shape[1:] != inputs.shape[:2]:
            raise ValueError("mRoPE positions must have [axes,batch,sequence] shape")
        rest = {k: v for k, v in batch.items() if k != "position_ids"}
        for index, microbatch in enumerate(super().microbatches(rest, size)):
            yield {**microbatch, "position_ids": positions[:, index * size : (index + 1) * size]}

    def snapshot_unit_state(self, unit, module, batch):
        """Delegate exceptional side inputs to the model-owned adapter."""
        provider = self._provider(self.student)
        return None if provider is None else provider.snapshot_unit_state(unit, module, batch)

    def replay(self, module, record):
        """Replay the actual selected module with model-owned side state if needed."""
        provider = self._provider(self.student)
        return (
            super().replay(module, record) if provider is None else provider.replay(module, record)
        )

    def capture(self, unit, args, kwargs, output, batch):
        """Align token masks for a native expert container that flattens token axes."""
        record = super().capture(unit, args, kwargs, output, batch)
        mask = record.valid_mask
        if mask is not None and record.target.ndim == 2 and record.target.shape[0] == mask.numel():
            record = replace(record, valid_mask=mask.flatten())
        return record

    def _validate_text_batch(self, batch):
        attention = batch.get("attention_mask")
        if (
            isinstance(attention, torch.Tensor)
            and attention.ndim == 2
            and attention.is_floating_point()
        ):
            raise ValueError(
                "A 2D attention mask must be Boolean or integer; use 4D for floating additive bias."
            )
        if any(
            batch.get(name) is not None
            for name in ("past_key_values", "past_key_value", "cache_params")
        ):
            raise ValueError("Independent Bypass batches cannot consume generation caches")
        if any(
            batch.get(name) is not None
            for name in ("pixel_values", "pixel_values_videos", "input_features", "audio_values")
        ):
            raise ValueError("Transformers Bypass currently supports text-only inputs")

    def prepare_batch(self, model, batch):
        """Convert compact document IDs into an HF block-causal attention mask."""
        self._validate_text_batch(batch)
        result = {
            key: value
            for key, value in batch.items()
            if key
            not in (
                "valid_mask",
                "_packed_seq_ids",
                "labels",
                "loss_mask",
                "sample_ids",
                "document_ids",
            )
        }
        attention = result.get("attention_mask")
        if (
            isinstance(attention, torch.Tensor)
            and attention.ndim == 4
            and not (attention.dtype == torch.bool or attention.is_floating_point())
        ):
            raise ValueError("A 4D attention mask must be Boolean or floating additive bias.")
        config = getattr(model, "config", None)
        if config is not None and callable(getattr(config, "get_text_config", None)):
            config = config.get_text_config()
        if (
            isinstance(attention, torch.Tensor)
            and attention.ndim == 4
            and attention.dtype == torch.bool
            and getattr(config, "_attn_implementation", None) == "eager"
        ):
            if not bool(attention.any(dim=-1).all()):
                raise ValueError(
                    "Fully masked Boolean query rows require an SDPA attention backend."
                )
            dtype = next(model.parameters()).dtype
            dtype = dtype if dtype.is_floating_point else torch.float32
            attention = torch.zeros_like(attention, dtype=dtype).masked_fill_(
                ~attention, torch.finfo(dtype).min
            )
            result["attention_mask"] = attention
        documents = batch.get("_packed_seq_ids")
        if isinstance(documents, torch.Tensor):
            if documents.ndim != 2 or documents.shape != batch["input_ids"].shape:
                raise ValueError("Packed document IDs must match [batch, sequence] token IDs.")
            attention = result.get("attention_mask")
            if bool((documents > 1).any()) or (
                isinstance(attention, torch.Tensor) and attention.ndim == 4
            ):
                if getattr(config, "_attn_implementation", None) not in (None, "eager", "sdpa"):
                    raise ValueError("This attention backend cannot consume packed block masks.")
                allowed = documents[:, None, :, None] == documents[:, None, None, :]
                allowed.tril_()
                allowed.logical_and_(documents[:, None, :, None].gt(0))
                if isinstance(attention, torch.Tensor):
                    if attention.ndim == 2:
                        allowed.logical_and_(attention[:, None, None, :].bool())
                    elif attention.ndim == 4 and attention.dtype == torch.bool:
                        if torch.broadcast_shapes(allowed.shape, attention.shape) == allowed.shape:
                            allowed.logical_and_(attention)
                        else:
                            allowed = allowed & attention
                    elif attention.ndim != 4:
                        raise ValueError("Packed attention masks must have two or four dimensions.")
                # HF eager attention adds its mask to scores; a Boolean mask
                # would add zeros/ones rather than exclude document boundaries.
                if (
                    isinstance(attention, torch.Tensor)
                    and attention.ndim == 4
                    and attention.is_floating_point()
                ):
                    shape = torch.broadcast_shapes(attention.shape, allowed.shape)
                    additive = attention.expand(shape).clone()
                    disallowed = (
                        allowed.logical_not_() if shape == allowed.shape else ~allowed.expand(shape)
                    )
                    # Retain stronger -inf masks and NaN propagation from the supplied bias.
                    disallowed.logical_and_(additive >= torch.finfo(additive.dtype).min)
                else:
                    dtype = next(model.parameters()).dtype
                    dtype = dtype if dtype.is_floating_point else torch.float32
                    additive = torch.zeros(allowed.shape, device=documents.device, dtype=dtype)
                    disallowed = allowed.logical_not_()
                    if isinstance(attention, torch.Tensor) and attention.ndim == 4:
                        additive.masked_fill_(~attention, float("-inf"))
                        disallowed.logical_and_(attention)
                result["attention_mask"] = additive.masked_fill_(
                    disallowed, torch.finfo(additive.dtype).min
                )
            else:
                result["attention_mask"] = documents.ne(0) & (
                    attention.bool() if isinstance(attention, torch.Tensor) else True
                )
        return result

    def _batch_documents(self, model, spec=None, modules=None):
        spec = _model_spec(model) if spec is None else spec
        if not spec.batch_packed_documents:
            return False
        # Unknown activation transforms or dynamic position scaling may couple rows.
        # PTQ converts modules in place; inspect live quantizer state on the cached owners.
        owners = tuple(
            model.modules() if modules is None else (reference() for reference in modules)
        )
        quantizers = {
            id(module): module for module in owners if isinstance(module, TensorQuantizer)
        }
        weights = set()
        for module in owners:
            if isinstance(module, QuantModule):
                weights.update(
                    id(quantizer)
                    for container in module.iter_weight_quantizers_for_calibration()
                    for quantizer in container.modules()
                )
                # In-place PTQ can add quantizers after the owner scan was cached.
                quantizers.update(
                    (id(child), child)
                    for child in module.modules()
                    if isinstance(child, TensorQuantizer)
                )
        if any(
            identity not in weights and quantizer.is_enabled
            for identity, quantizer in quantizers.items()
        ):
            return False
        return not _document_dependent_positions(model)

    def forward_teacher(self, teacher, args, kwargs):
        """Run the native text task, retaining outer hooks for wrapped text models."""
        target = getattr(teacher, "thinker", teacher)
        if target is teacher:
            return super().forward_teacher(teacher, args, kwargs)
        had_override, previous = "forward" in teacher.__dict__, teacher.__dict__.get("forward")
        if "use_cache" in inspect.signature(target.forward).parameters:
            kwargs = {**kwargs, "use_cache": False}
        teacher.forward = lambda *args, **kwargs: target(*args, **kwargs)
        try:
            return super().forward_teacher(teacher, args, kwargs)
        finally:
            if had_override:
                teacher.forward = previous
            else:
                delattr(teacher, "forward")

    def _synchronize_error(self, error):
        from ..._bypass_parallel import _synchronize_error

        _synchronize_error(self.context, error)

    def _cp_inputs(self, teacher, batch):
        from modelopt.torch.quantization.repair_partition import requires_full_context_activations

        models = [model for model in (teacher, self.student) if model is not None]
        for model in models:
            config = model.config.get_text_config()
            if (
                not _model_spec(model).causal_context_parallel
                or not model.supports_context_parallel
                or config._attn_implementation != "sdpa"
                or next(model.parameters()).dtype not in (torch.float16, torch.bfloat16)
                or requires_full_context_activations(model)
                or _document_dependent_positions(model)
            ):
                raise ValueError("This model requires faithful full-context execution")
        supported = {
            "input_ids",
            "attention_mask",
            "position_ids",
            "valid_mask",
            "_packed_seq_ids",
            "use_cache",
            "labels",
            "loss_mask",
            "sample_ids",
            "document_ids",
        }
        if any(key not in supported and value is not None for key, value in batch.items()):
            raise ValueError("Native CP cannot discard model-specific inputs")
        tokens = batch["input_ids"]
        if tokens.ndim != 2:
            raise ValueError("Native CP needs [batch, sequence] tokens")
        positions = batch.get("position_ids")
        for name in ("attention_mask", "valid_mask", "_packed_seq_ids"):
            value = batch.get(name)
            if value is not None and (value.shape != tokens.shape or not bool(value.eq(1).all())):
                raise ValueError("Native CP requires unpadded single-document causal rows")
        if positions is not None and (
            positions.shape != tokens.shape
            or bool((positions[:, 1:] - positions[:, :-1] != 1).any())
        ):
            raise ValueError("Native CP cannot discard position resets or mRoPE axes")
        if positions is None:
            positions = torch.arange(tokens.shape[1], device=tokens.device).expand_as(tokens)
        return tokens, positions

    @staticmethod
    def _causal_attention(module, args, kwargs):
        return args, {**kwargs, "attention_mask": None, "is_causal": True}

    def _document_teacher(self, teacher, units, document, consume):
        from ..._bypass_parallel import _all_rank_values, _size

        cp = None if self.context is None else self.context.context_mesh
        native, inputs = False, None
        if _size(cp) > 1:
            try:
                native_api = importlib.import_module(
                    "torch.distributed.tensor.experimental._context_parallel._attention"
                )
                if hasattr(native_api, "_ContextParallel"):
                    inputs = self._cp_inputs(teacher, document)
                    native = True
            except Exception:
                pass
            native = all(_all_rank_values(self.context, native))
        if not native:
            prepared, error = None, None
            try:
                prepared = {
                    **self.prepare_batch(teacher, document),
                    "valid_mask": _valid_token_mask(document),
                }
            except Exception as exc:
                if self.context is None or self.context.device_mesh.size() == 1:
                    raise
                error = f"{type(exc).__name__}: {exc}"
            self._synchronize_error(error)
            return super().teacher_microbatch(teacher, units, prepared, consume)
        from .context_parallel import CPSequenceLayout

        assert inputs is not None
        tokens, positions = inputs
        layout = CPSequenceLayout(cp, tokens.shape[1], device=tokens.device)
        batch = {
            "input_ids": layout.shard(tokens, 1),
            "position_ids": layout.shard(positions, 1),
            "valid_mask": layout.shard(torch.ones_like(tokens, dtype=torch.bool), 1),
        }
        handles = [
            module.register_forward_pre_hook(self._causal_attention, with_kwargs=True)
            for model in (teacher, self.student)
            if model is not None
            for name, module in model.named_modules()
            if name.endswith("self_attn")
        ]
        self._native_context = True
        try:
            with layout.attention():
                return super().teacher_microbatch(teacher, units, batch, consume)
        finally:
            self._native_context = False
            for handle in handles:
                handle.remove()

    def teacher_microbatch(self, teacher, units, batch, consume):
        """Keep document recurrence faithful and match every rank's forward schedule."""
        from ..._bypass_parallel import _all_rank_max

        documents, error = [], None
        try:
            batch = self._batch_with_validity(batch)
            self._validate_text_batch(batch)
            facts = self._facts(teacher)
            spec = facts.spec
            documents = document_batches(batch) if spec.isolated_documents else [batch]
            if (
                "input_ids" in batch
                and spec.isolated_documents
                and self._batch_documents(teacher, spec, facts.modules)
            ):
                student_facts = self._facts(self.student)
                if self._batch_documents(self.student, student_facts.spec, student_facts.modules):
                    documents = batch_documents(documents, batch["input_ids"].numel())
        except Exception as exc:
            if self.context is None or self.context.device_mesh.size() == 1:
                raise
            error = f"{type(exc).__name__}: {exc}"
        self._synchronize_error(error)
        slots = _all_rank_max(self.context, len(documents))
        with ExitStack() as stack:
            error = None
            try:
                stack.enter_context(
                    bypass_head_context(teacher, units, facts.head, head_paths=facts.head_paths)
                )
            except Exception as exc:
                if self.context is None or self.context.device_mesh.size() == 1:
                    raise
                error = f"{type(exc).__name__}: {exc}"
            self._synchronize_error(error)
            for index in range(slots):
                document, error = None, None
                try:
                    document = (
                        documents[index]
                        if index < len(documents)
                        else empty_document_batch(documents[0] if documents else batch)
                    )
                except Exception as exc:
                    if self.context is None or self.context.device_mesh.size() == 1:
                        raise
                    error = f"{type(exc).__name__}: {exc}"
                self._synchronize_error(error)
                self._document_teacher(teacher, units, document, consume)
