# SPDX-FileCopyrightText: Copyright (c) 2023-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
"""Quantization support for accelerate modified models."""

import inspect
import os
import warnings
from collections import OrderedDict, UserDict
from contextlib import contextmanager
from itertools import chain
from typing import Any
from weakref import WeakKeyDictionary, ref

import torch
import transformers
from accelerate import init_empty_weights, load_checkpoint_and_dispatch
from accelerate.hooks import AlignDevicesHook, SequentialHook
from accelerate.utils import get_max_memory, infer_auto_device_map
from accelerate.utils.dataclasses import CustomDtype
from accelerate.utils.offload import OffloadedWeightsLoader, PrefixedDataset

import modelopt.torch.quantization as mtq

from ..nn import TensorQuantizer
from ..utils.core_utils import _weight_access_guard

__all__ = ["init_quantized_weights"]

_ROLLBACK_RECEIPTS = WeakKeyDictionary()


def _get_offload_hook(hook):
    if isinstance(hook, AlignDevicesHook) and hook.offload and hook.weights_map is not None:
        return hook
    elif isinstance(hook, SequentialHook):
        for h in hook.hooks:
            align_hook = _get_offload_hook(h)
            if align_hook is not None:
                return align_hook
    return None


def _weights_map_and_prefix(align_hook):
    weights_map = align_hook.weights_map
    prefix = ""
    while type(weights_map) is PrefixedDataset:
        prefix = weights_map.prefix + prefix
        weights_map = weights_map.dataset
    return weights_map, prefix


def _resident_weights_map(weights_map):
    if type(weights_map) is OffloadedWeightsLoader:
        weights_map = weights_map.state_dict
    if type(weights_map) is UserDict:
        weights_map = weights_map.data
    if type(weights_map) not in (dict, OrderedDict):
        raise ValueError("Accelerate weight access cannot safely inspect this offload mapping")
    return weights_map


def _managed_tensors(module, include_buffers=True, recurse=False):
    """Inspect persistence at each registering owner, without Accelerate 1.6's set mutation."""
    yield from module.named_parameters(recurse=recurse, remove_duplicate=False)
    if include_buffers:
        for name, tensor in module.named_buffers(recurse=recurse, remove_duplicate=False):
            path, _, local_name = name.rpartition(".")
            owner = module.get_submodule(path)
            if local_name not in owner._non_persistent_buffers_set:
                yield name, tensor


def _needs_materialize(module, hook):
    parameters = list(module.parameters(recurse=hook.place_submodules))
    # A resident parameter may contain caller edits; native onload would overwrite them.
    return any(tensor.is_meta for tensor in parameters) or (
        not parameters
        and hook.offload_buffers
        and any(
            tensor.is_meta for _, tensor in _managed_tensors(module, recurse=hook.place_submodules)
        )
    )


def _rollback_scope(module, materialized):
    scope = {("module", id(mod)) for mod in module.modules()}
    tensors = list(chain(module.parameters(), module.buffers()))
    for mod, hook, _ in materialized:
        dataset, prefix = _weights_map_and_prefix(hook)
        if type(dataset) is OffloadedWeightsLoader:
            dataset = dataset.state_dict
        if type(dataset) is UserDict:
            dataset = dataset.data
        scope.add(("map", id(dataset)))
        if hook.tied_params_map is not None:
            scope.add(("ties", id(hook.tied_params_map)))
        if type(dataset) in (dict, OrderedDict):
            tensors.extend(
                dict.get(dataset, prefix + name)
                for name, _ in _managed_tensors(mod, recurse=hook.place_submodules)
            )
        else:
            scope.add(("opaque_map", 0))
    for tensor in tensors:
        if isinstance(tensor, torch.Tensor):
            scope.add(("tensor", id(tensor)))
            if isinstance(tensor, torch.distributed.tensor.DTensor):
                tensor = tensor.to_local()
            if (
                tensor.device.type == "cpu"
                and tensor.layout == torch.strided
                and tensor.numel() > 0
            ):
                scope.add(("storage", tensor.untyped_storage().data_ptr()))
    return scope


def _invalidate_rollbacks(scope):
    for receipt, affected in list(_ROLLBACK_RECEIPTS.items()):
        if (
            ("opaque_map", 0) in scope
            or ("opaque_map", 0) in affected
            or not scope.isdisjoint(affected)
        ):
            _ROLLBACK_RECEIPTS.pop(receipt, None)


def _expire_overlapping_rollbacks(module, materialized=(), *, capture_scope=False, strict=True):
    if not capture_scope and not _ROLLBACK_RECEIPTS:
        return None
    try:
        scope = _rollback_scope(module, materialized)
        _invalidate_rollbacks(scope)
        return scope
    except BaseException as exc:
        _ROLLBACK_RECEIPTS.clear()
        if strict or not isinstance(exc, Exception):
            raise


def _snapshot_aliased_weights(entered, retained):
    """Copy backing tensors that CPU weights can mutate during or after onload."""
    tensors = [
        (hook, name, tensor)
        for mod, hook, _ in entered
        for name, tensor in _managed_tensors(mod, recurse=True)
    ]
    live = (tensor for mod, _, _ in entered for tensor in chain(mod.parameters(), mod.buffers()))
    live_storages = set()
    for tensor in chain(live, retained):
        if isinstance(tensor, torch.distributed.tensor.DTensor):
            tensor = tensor.to_local()
        if tensor.device.type == "cpu" and tensor.layout == torch.strided and tensor.numel() > 0:
            live_storages.add(tensor.untyped_storage().data_ptr())
    snapshots = {}
    for hook, name, managed in tensors:
        weights_map, prefix = _weights_map_and_prefix(hook)
        tensor = _resident_weights_map(weights_map).get(prefix + name)
        if tensor is not None and tensor.layout != torch.strided:
            raise ValueError("Accelerate writeback requires strided backing tensors")
        if (
            tensor is not None
            and tensor.device.type == "cpu"
            and (
                tensor.untyped_storage().data_ptr() in live_storages
                or (managed.is_meta and str(hook.execution_device).split(":")[0] == "cpu")
            )
        ):
            key = (
                tensor.untyped_storage().data_ptr(),
                tensor.storage_offset(),
                tensor.shape,
                tensor.stride(),
                tensor.dtype,
            )
            # Prove disjoint strides without a full-size offset tensor; ambiguous
            # interleaved layouts remain unsupported, while ordinary stepped slices work.
            span = 1
            for stride, size in sorted(zip(tensor.stride(), tensor.shape, strict=True)):
                if size <= 1:
                    continue
                if stride < span:
                    raise ValueError(
                        "Accelerate rollback cannot restore overlapping or unproven backing views"
                    )
                span += (size - 1) * stride
            if key not in snapshots:
                snapshots[key] = tensor, tensor.detach().clone()
    return snapshots.values()


def _writeback_params_to_weights_map(module, align_hook, journal):
    """Write all non-meta parameters and buffers back to the hook's CPU weights_map."""
    buffer_names = {name for name, _ in module.named_buffers()}
    dataset, prefix = _weights_map_and_prefix(align_hook)
    w_map = _resident_weights_map(dataset)
    for name, tensor in module.state_dict(keep_vars=True).items():
        if tensor.device.type == "meta":
            continue
        key = prefix + name
        if key in w_map:
            value = tensor.detach().to(w_map[key].device, dtype=w_map[key].dtype)
        elif isinstance(dataset, OffloadedWeightsLoader) and key in dataset.index:
            # Disk-offloaded weight: promote into state_dict so the next
            # pre_forward picks up the modified tensor instead of the stale
            # on-disk version.  OffloadedWeightsLoader.__getitem__ gives
            # state_dict priority over index, so this is sufficient.
            value = tensor.detach().cpu()
            warnings.warn(
                "Accelerate disk-offload writeback is currently kept in CPU state_dict; "
                "writing updates back to disk offload files is TODO.",
                RuntimeWarning,
                stacklevel=2,
            )
        elif align_hook.offload_buffers and name in buffer_names:
            # A buffer first registered inside the context, such as a GPTQ payload pin: without a
            # backing value the hook cannot reload it after offloading the module again.
            value = tensor.detach().cpu()
        else:
            continue
        journal.append((w_map, key, w_map.get(key)))
        w_map[key] = value


def _own_quantizers(module):
    """Quantizers whose state belongs with ``module``'s weights, not with a hooked descendant's."""
    for child in module.children():
        if isinstance(child, TensorQuantizer):
            yield child
        elif not hasattr(child, "_hf_hook"):
            yield from _own_quantizers(child)


# A quantizer's calibrators: its amax calibrator, and its bias calibrator under static bias.
_CALIBRATORS = ("_calibrator", "_bias_calibrator")


def _quantizer_tensors(module):
    """``(key, tensor)`` for the state of ``module``'s own quantizers and their calibrators."""
    for quantizer in _own_quantizers(module):
        sources = {"_parameters": quantizer._parameters, "_buffers": quantizer._buffers}
        for source in _CALIBRATORS:
            calibrator = getattr(quantizer, source, None)
            sources[source] = vars(calibrator) if calibrator is not None else {}
        for source, tensors in sources.items():
            for name, tensor in tensors.items():
                if isinstance(tensor, torch.Tensor):
                    yield (id(quantizer), source, name), tensor


@contextmanager
def _offload_buffer_persistence(module, hook):
    """Give recursive native hooks qualified exclusions without changing caller metadata."""
    if not (hook.offload and hook.offload_buffers and hook.place_submodules):
        yield
        return
    modules = list(module.named_modules())
    original = [(owner, owner._non_persistent_buffers_set) for _, owner in modules]
    names = {
        f"{path}.{name}" if path else name
        for path, owner in modules
        for name in owner._non_persistent_buffers_set
    }
    if not names:
        yield
        return
    try:
        for owner, _ in original:
            owner._non_persistent_buffers_set = set()
        module._non_persistent_buffers_set = names
        yield
    finally:
        for owner, saved in original:
            owner._non_persistent_buffers_set = saved


def _carry_quantizers_with_offload(hook):
    """Make an offload ``hook`` move its module's quantizer state on and off the GPU with the weights.

    Accelerate only manages the tensors a module had when it was dispatched, so calibration state
    such as NVFP4's per-block ``_amax``, and its calibrator's copy, stayed on the execution device
    for every offloaded layer and added up over the model. The CPU tensors stay the master copy:
    a forward gets GPU copies, and after it the CPU tensors come back. A quantizer's own tensors
    are always copied back, since its setters write through ``.data``, which leaves ``_version``
    unchanged; a calibrator's tensor is copied back only if the forward replaced or modified it.
    The packed-weight cache is keyed to the weight just released, so it is dropped.
    """
    if getattr(hook, "_modelopt_carries_quantizers", False):
        return
    pre_forward, post_forward = hook.pre_forward, hook.post_forward
    onloaded: dict[tuple, tuple[torch.Tensor, torch.Tensor, int]] = {}

    def carrying_pre_forward(module, *args, **kwargs):
        with _offload_buffer_persistence(module, hook):
            result = pre_forward(module, *args, **kwargs)
        if hook.offload:
            for key, tensor in _quantizer_tensors(module):
                if tensor.device.type == "cpu":
                    master = tensor.data
                    tensor.data = master.to(hook.execution_device)
                    onloaded[key] = (tensor, master, tensor._version)
        return result

    def carrying_post_forward(module, output):
        with _offload_buffer_persistence(module, hook):
            output = post_forward(module, output)
        if hook.offload:
            for key, tensor in _quantizer_tensors(module):
                # meta: a hook that places submodules offloaded it already.
                if tensor.device.type in ("cpu", "meta"):
                    continue
                entry = onloaded.pop(key, None)
                if entry is None or entry[0] is not tensor:
                    tensor.data = tensor.data.cpu()
                    continue
                master = entry[1]
                if key[1] not in _CALIBRATORS or entry[2] != tensor._version:
                    if master.shape != tensor.shape or master.dtype != tensor.dtype:
                        tensor.data = tensor.data.cpu()
                        continue
                    # Into the existing master: a new CPU tensor costs 8x as much in page faults.
                    master.copy_(tensor.data)
                tensor.data = master
            for quantizer in _own_quantizers(module):
                quantizer._quantizer_cache = None
        return output

    hook.pre_forward, hook.post_forward = carrying_pre_forward, carrying_post_forward
    hook._modelopt_carries_quantizers = True


@contextmanager
def weight_access_and_writeback_context(module, writeback: bool = True, *, rollback: bool = False):
    """Context manager for weight access and writeback for modules managed by accelerate.

    Handles CPU-offloaded and disk-offloaded models. Iterates over the module and all
    its descendants, materializing weights from any offload hook found. If ``writeback``
    is True, writes materialized tensors back on exit. ``pre_forward`` is skipped on
    modules whose weights are already
    materialized (not on meta) to avoid overwriting them with stale CPU copies.
    Failed windows restore backing maps and pre-window registrations and placement. With
    ``rollback=True``, aliased CPU values are also snapshotted for restoration.
    The ``writeback=False`` flag only skips publication: in-place CPU alias mutations require
    ``rollback=True`` to be restored after failure, even in a read-only window.
    With ``rollback=True``, the yielded callback restores offload backing maps, saved registrations
    and placement, and snapshotted CPU backing aliases after a local success or failure. It does
    not restore in-place edits to already-resident execution tensors. Callers must restore those
    values from their own backups. Backing maps that hold execution-device tensors instead of CPU
    tensors are also not snapshotted; callers own their restoration. This includes aliases created
    on onload when a backing is already on the execution device. Inside an outer materialization
    window, restore caller-owned values before that
    outer scope republishes them. Successful callback restoration is idempotent; a failed attempt
    reports its error and remains retryable so the caller can finish restoration.
    A callback is valid only after its window exits and until another overlapping access enters
    or publishes, including preflight rejections, read-only or failed accesses and enclosing publication.
    Overlap includes shared owners, registered tensors, canonical backing maps, tied-parameter
    metadata, or native CPU storage. Read-only mappings outside the supported exact types,
    including custom dict subclasses, are opaque and conservatively expire all pending callbacks.
    An expired callback raises before restoring anything;
    a completed callback stays idempotent. Tracking holds weak callbacks and metadata,
    never global tensor/model references.
    CPU tensors sharing live, retained master, or CPU-onload backing storage require host snapshots;
    backing views must have provably disjoint strides (ordinary stepped slices are supported).
    Other rollback state holds references and metadata, without copying execution-device weights.
    Only rollback windows refuse mixed meta/resident owners; ordinary windows retain native onload.
    Windows and undo calls require per-process caller serialization; concurrent calls raise before
    mutation. Same-thread nested windows are supported.
    """
    with (
        _weight_access_guard(),
        _weight_access_and_writeback_context(module, writeback, rollback=rollback) as undo,
    ):
        yield undo


@contextmanager
def _weight_access_and_writeback_context(module, writeback: bool = True, *, rollback: bool = False):
    assert any(hasattr(child, "_hf_hook") for child in module.modules())

    materialized: list[tuple[torch.nn.Module, AlignDevicesHook, bool]] = []
    for mod in module.modules():
        if not hasattr(mod, "_hf_hook"):
            continue
        hook = _get_offload_hook(mod._hf_hook)
        if hook is None:
            continue
        needs_materialize = _needs_materialize(mod, hook)
        materialized.append((mod, hook, needs_materialize))

    scope = _expire_overlapping_rollbacks(module, materialized, capture_scope=rollback)

    for mod, hook, _ in materialized:
        assert len(hook.weights_map) > 0
        if writeback or rollback:
            _resident_weights_map(_weights_map_and_prefix(hook)[0])
        if rollback:
            # Rollback cannot recover resident updates overwritten during a mixed-state onload.
            offloaded = [
                tensor.device.type == "meta"
                for _, tensor in _managed_tensors(
                    mod,
                    include_buffers=hook.offload_buffers,
                    recurse=hook.place_submodules,
                )
            ]
            if any(offloaded) and not all(offloaded):
                raise ValueError(
                    "Accelerate weight access requires all managed tensors on meta or resident"
                )

    entered = []
    ready = False
    error = None
    propagating = False
    snapshots = ()
    journal = []
    registrations = [
        (
            mod,
            mod._parameters,
            mod._parameters.copy(),
            mod._buffers,
            mod._buffers.copy(),
            mod._non_persistent_buffers_set,
            mod._non_persistent_buffers_set.copy(),
        )
        for mod in module.modules()
    ]
    originals = {
        id(tensor): (tensor, tensor.data)
        for _, _, parameters, _, buffers, _, _ in registrations
        for tensor in chain(parameters.values(), buffers.values())
        if tensor is not None
    }
    ties = {
        id(hook.tied_params_map): (
            hook.tied_params_map,
            {pointer: values.copy() for pointer, values in hook.tied_params_map.items()},
        )
        for _, hook, _ in materialized
        if hook.tied_params_map is not None
    }
    rolled_back = False
    scope_failed = False

    def expire_rollbacks():
        nonlocal scope, scope_failed, error
        try:
            scope = (scope or set()) | _rollback_scope(module, materialized)
            _invalidate_rollbacks(scope)
        except BaseException as exc:
            scope_failed = True
            _ROLLBACK_RECEIPTS.clear()
            if error is None:
                error = exc
            else:
                error.add_note(f"Accelerate rollback receipt inspection failed: {exc!r}")

    def restore():
        nonlocal rolled_back
        if rolled_back:
            return
        rollback_error = None

        def attempt(action, *args):
            nonlocal rollback_error
            try:
                action(*args)
            except BaseException as exc:
                if rollback_error is None:
                    rollback_error = exc
                else:
                    rollback_error.add_note(f"Accelerate weight-access rollback failed: {exc!r}")

        for weights_map, key, original in reversed(journal):
            if original is None:
                attempt(weights_map.pop, key, None)
            else:
                attempt(weights_map.__setitem__, key, original)
        with torch.no_grad():
            for original, snapshot in snapshots:
                attempt(original.copy_, snapshot)
            for tensor, data in originals.values():
                attempt(setattr, tensor, "data", data)
        for (
            mod,
            registry,
            parameters,
            buffer_registry,
            buffers,
            nonpersistent,
            exclusions,
        ) in registrations:
            # Bypass custom setters while restoring the original registry and its metadata.
            for target, values in ((registry, parameters), (buffer_registry, buffers)):
                if isinstance(target, OrderedDict):
                    OrderedDict.clear(target)
                    for name, value in values.items():
                        OrderedDict.__setitem__(target, name, value)
                else:
                    dict.clear(target)
                    dict.update(target, values)
            mod._parameters = registry
            mod._buffers = buffer_registry
            set.clear(nonpersistent)
            set.update(nonpersistent, exclusions)
            mod._non_persistent_buffers_set = nonpersistent
        for mapping, values in ties.values():
            mapping.clear()
            mapping.update({pointer: entries.copy() for pointer, entries in values.items()})
        if rollback_error is not None:
            raise rollback_error
        rolled_back = True

    def undo():
        with _weight_access_guard():
            if rolled_back:
                return
            receipt = receipt_ref()
            if receipt not in _ROLLBACK_RECEIPTS:
                raise RuntimeError(
                    "Accelerate rollback callback is not valid before exit or after overlapping access"
                )
            restore()
            _ROLLBACK_RECEIPTS.pop(receipt, None)

    receipt_ref = ref(undo)

    try:
        if rollback:
            # CPU onload can alias and mutate the backing even before the hook returns.
            snapshots = _snapshot_aliased_weights(
                materialized, (data for _, data in originals.values())
            )
        for mod, hook, needs_materialize in materialized:
            if needs_materialize or (hook.offload_buffers and hook.place_submodules):
                _carry_quantizers_with_offload(hook)
            # An enclosing hook may already have loaded this owner's parameters.
            loads = _needs_materialize(mod, hook)
            # pre_forward can fail after loading only part of this owner's tensors.
            entered.append((mod, hook, loads))
            if loads:
                hook.pre_forward(mod)
            hook.offload = False
        ready = True
        yield undo if rollback else None
    except BaseException as exc:
        error = exc
        propagating = True
        raise
    finally:
        if _ROLLBACK_RECEIPTS:
            expire_rollbacks()
        if ready and writeback and error is None:
            # Publish every backing slot before a nested hook can replace shared tensors by meta.
            for mod, hook, _ in reversed(entered):
                try:
                    _writeback_params_to_weights_map(mod, hook, journal)
                except BaseException as exc:
                    if error is None:
                        error = exc
                    else:
                        error.add_note(f"Accelerate weight-access cleanup failed: {exc!r}")
        for mod, hook, was_materialized in reversed(entered):
            hook.offload = True
            try:
                if was_materialized:
                    hook.post_forward(mod, None)
            except BaseException as exc:
                if error is None:
                    error = exc
                else:
                    error.add_note(f"Accelerate weight-access cleanup failed: {exc!r}")
        if rollback or _ROLLBACK_RECEIPTS:
            expire_rollbacks()
        if error is not None:
            try:
                restore()
            except BaseException as exc:
                error.add_note(f"Accelerate weight-access rollback failed: {exc!r}")
        if ready and rollback and not rolled_back:
            _ROLLBACK_RECEIPTS[undo] = {("opaque_map", 0)} if scope_failed else scope
        if error is not None and not propagating:
            raise error


@contextmanager
def init_quantized_weights(
    quant_cfg: dict[str, Any], gpu_mem_percentage: float = 0.8, quant_gemm: bool = False
):
    """Context manager for initializing and loading HuggingFace models with quantized and compressed weights.

    This context manager patches `from_pretrained` to automatically:
    1. Initialize the model with an empty config
    2. Apply quantization configuration
    3. Compress the weights
    4. Load and dispatch the model to appropriate devices

    Args:
        quant_cfg: The quantization config to apply to the model
        gpu_mem_percentage: Percentage of GPU memory to use (0.0-1.0)
        quant_gemm: Whether to enable quantized GEMM

    Example:
        ```python
        from transformers import AutoModelForCausalLM
        from modelopt.torch.quantization.plugins.huggingface import init_quantized_weights
        import modelopt.torch as mtq

        # Load quantized and compressed model in one step
        with init_quantized_weights(mtq.NVFP4_DEFAULT_CFG):
            model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-2-70b-hf")


        # The model is already quantized, compressed, and loaded to appropriate devices

        # calibrate model
        mtq.calibrate(model, "max", forward_loop)
        ```
    """

    def get_no_split_module_classes(model):
        """Get no-split module classes for device mapping."""
        no_split_classes = set()

        # Get all named modules
        for name, module in model.named_modules():
            # Look for first layer patterns
            if name.endswith((".layers.0", ".layer.0", ".h.0", ".blocks.0")):
                no_split_classes.add(module.__class__.__name__)

        return list(no_split_classes)

    def get_model_device_map(model, gpu_mem_percentage):
        """Create optimized device map for the quantized and compressed model."""
        # Get compression ratio to adjust memory usage
        max_memory = get_max_memory()
        no_split_classes = get_no_split_module_classes(model)
        if len(no_split_classes) == 0:
            warnings.warn(
                "No no-split module classes found for the model. Default device map might be incorrect."
            )
        max_memory = {k: v * gpu_mem_percentage for k, v in max_memory.items()}

        def _get_special_dtype_map(model):
            special_dtype_map = {}

            def _get_byte_size(module):
                weight_num_bit = module.weight_quantizer.num_bits
                if isinstance(weight_num_bit, tuple):
                    assert len(weight_num_bit) == 2, (
                        "Weight num bit must be a tuple of two integers."
                    )
                    weight_num_bit = weight_num_bit[0] + weight_num_bit[1] + 1
                if weight_num_bit == 8:
                    return CustomDtype.FP8
                elif weight_num_bit == 4:
                    return CustomDtype.INT4
                else:
                    raise ValueError(f"Unsupported weight num bit: {weight_num_bit}")

            for name, module in model.named_modules():
                if (
                    hasattr(module, "weight")
                    and hasattr(module, "weight_quantizer")
                    and not module.weight_quantizer.fake_quant
                ):
                    special_dtype_map[name + ".weight"] = _get_byte_size(module)
            return special_dtype_map

        special_dtype_map = _get_special_dtype_map(model)
        inferred_device_map = infer_auto_device_map(
            model,
            max_memory=max_memory,
            no_split_module_classes=no_split_classes,
            special_dtypes=special_dtype_map,
        )
        return inferred_device_map

    # Store original from_pretrained methods that we'll override
    original_auto_causal_lm_from_pretrained = transformers.AutoModelForCausalLM.from_pretrained

    # Create patched from_pretrained that applies quantization and compression
    def patched_from_pretrained(cls, /, pretrained_model_name_or_path, *args, **kwargs):
        """Patched from_pretrained that handles quantization, compression and device mapping."""
        # This loader quantizes with quant_cfg and places the model itself.
        if kwargs.pop("quantization_config", None) is not None:
            raise ValueError(
                "init_quantized_weights() quantizes with quant_cfg; drop quantization_config."
            )
        if kwargs.pop("device_map", None) is not None:
            raise ValueError(
                "init_quantized_weights() computes the device map; drop device_map and set "
                "gpu_mem_percentage instead."
            )
        # As in load_pretrained(), the subfolder is part of the checkpoint path: the config and
        # the weights are both read from it.
        subfolder = kwargs.pop("subfolder", None)
        if subfolder:
            pretrained_model_name_or_path = os.path.join(pretrained_model_name_or_path, subfolder)
        # Initialize with empty config first to avoid loading weights twice
        config = kwargs.pop("config", None)
        trust_remote_code = kwargs.pop("trust_remote_code", False)
        legacy_dtype = kwargs.pop("torch_dtype", None)
        if kwargs.get("dtype") is None and legacy_dtype is not None:
            kwargs["dtype"] = legacy_dtype
        # load_checkpoint_and_dispatch() gets only the kwargs it takes; the rest build the model.
        dispatch_params = inspect.signature(load_checkpoint_and_dispatch).parameters
        dispatch_kwargs = {k: kwargs.pop(k) for k in list(kwargs) if k in dispatch_params}
        if config is None:
            # Get config from pretrained model
            from transformers import AutoConfig

            # As in from_pretrained(), the config load takes the Hub options (revision, token, ...)
            # and any of the config's own attributes; the weights are still read from the path.
            config, kwargs = AutoConfig.from_pretrained(
                pretrained_model_name_or_path,
                trust_remote_code=trust_remote_code,
                return_unused_kwargs=True,
                **kwargs,
            )

        with init_empty_weights():
            # config.dtype is None, not absent, when unset, so a getattr default never applies.
            dtype = dispatch_kwargs.get("dtype") or getattr(config, "dtype", None) or torch.float16
            # from_config() raises on any kwarg the model doesn't take.
            model = cls.from_config(config, dtype=dtype, **kwargs)
        # init_empty_weights re-wraps every registered parameter, so the tie made inside
        # from_config is lost; without it the absent tied checkpoint key (e.g. lm_head) stays meta.
        model.tie_weights()

        mtq.quantize(model, quant_cfg)
        mtq.compress(model, config=mtq.CompressConfig(quant_gemm=quant_gemm))
        _device_map = get_model_device_map(model, gpu_mem_percentage)

        return load_checkpoint_and_dispatch(
            model,
            checkpoint=pretrained_model_name_or_path,
            device_map=_device_map,
            *args,
            **dispatch_kwargs,
        )

    # A sharded load_pretrained builds the model without from_pretrained, so it reads this mark.
    patched_from_pretrained._modelopt_init_quantized_weights = True  # type: ignore[attr-defined]
    try:
        # Apply our patches
        transformers.AutoModelForCausalLM.from_pretrained = classmethod(patched_from_pretrained)
        yield
    finally:
        # Restore original methods
        transformers.AutoModelForCausalLM.from_pretrained = original_auto_causal_lm_from_pretrained
