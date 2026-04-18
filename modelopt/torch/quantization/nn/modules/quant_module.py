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

"""Base class for quantization modules."""

import contextlib
import json
import threading
import warnings
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn

from modelopt.torch.opt.dynamic import DynamicModule, _DMRegistryCls
from modelopt.torch.utils.distributed import ParallelState

from ...tensor_quant import QUANT_DESC_8BIT_PER_TENSOR
from ...utils import is_quantized_linear, is_torch_export_mode
from ...utils.core_utils import quantizer_attr_names, weight_attr_names
from .tensor_quantizer import NVFP4StaticQuantizer, SequentialQuantizer, TensorQuantizer

__all__ = [
    "QuantInputBase",
    "QuantLinearConvBase",
    "QuantModule",
    "QuantModuleRegistry",
]


class QuantModule(DynamicModule):
    """A base class for quantized modules.

    In addition, the class also provides ``parallel_state`` attribute that can be used to access
    the parallel state of the module.
    """

    _parallel_state: ParallelState

    @classmethod
    @torch.no_grad()
    def convert(cls, module: nn.Module, **setup_kwargs: Any) -> "QuantModule":
        """Convert the module to a dynamic module."""
        module = super().convert(module, **setup_kwargs)

        # setup parallel state now that the module is converted
        if module.parallel_state is None:
            module._initialize_parallel_state()

        return module

    @property
    def parallel_state(self) -> ParallelState | None:
        """Return the parallel state of the quant module."""
        return getattr(self, "_parallel_state", None)

    @parallel_state.setter
    def parallel_state(self, parallel_state: ParallelState):
        """Set the parallel state of the dynamic module."""
        assert isinstance(parallel_state, ParallelState), (
            "parallel_state must be a ParallelState object!"
        )
        self._parallel_state = parallel_state

    def _initialize_parallel_state(self):
        """Initialize the parallel state of the dynamic module.

        This method is called only if the `QuantModule` does not have a `parallel_state` attribute
        after `_setup` is called.
        """
        if torch.distributed.is_initialized():
            warnings.warn(
                f"Distributed training is initialized but no parallel_state is set for {type(self)}. "
                "Using default parallel_state which has data_parallel_group set to the default process group and "
                "tensor_parallel_group is unspecified. "
                "If you are using tensor parallelism for this module, you should set the parallel_state "
                "in its `_setup` method."
            )

        self.parallel_state = ParallelState(data_parallel_group=None)

    def modelopt_post_restore(self, prefix: str = ""):
        """Post-restore to correctly configure the TensorQuantizer states.

        TensorQuantizer states are restored to their shape before saving. Now we need to further configure them.
            1. For non-sharded modules this simply involves moving the TensorQuantizer states to the right device.
                This applies for regular Pytorch models and HuggingFace models.
            2. For sharded modules the restored states of TensorQuantizer could be incorrect. This is because
                parallelism such as TP might have been changed between saving and resoring. So we need to re-calculate
                the state shapes. Hence such modules should override this and implement their own logic.
        """
        # Get a parameter or buffer that does not belong to a TensorQuantizer
        non_tq_param_or_buffer = None
        for name, param_or_buffer in self.state_dict().items():
            parent = self.get_submodule(name.rsplit(".", 1)[0]) if "." in name else self
            if not isinstance(parent, TensorQuantizer):
                non_tq_param_or_buffer = param_or_buffer
                break

        if non_tq_param_or_buffer is None:
            warnings.warn(
                f"Could not identify the device for TensorQuantizer states of {prefix}. "
                "Please move the model to the right device now. This can be done by calling "
                "`model.to(device)`."
            )
            return

        # Move the TensorQuantizer states to the right device (dtype should have been restored).
        for module in self.modules():
            if isinstance(module, TensorQuantizer):
                module.to(non_tq_param_or_buffer.device)

    def iter_weights_for_calibration(self):
        """Yield ``(weight, weight_quantizer)`` pairs for weight-only calibration."""
        from modelopt.torch.quantization.utils import quantizer_attr_names, weight_attr_names

        for weight_name in weight_attr_names(self):
            weight_quantizer = getattr(self, quantizer_attr_names(weight_name).weight_quantizer)
            yield getattr(self, weight_name), weight_quantizer

    def fold_weight(self, keep_attrs: bool = False):
        """Fold the weight for faster eval."""
        # Handle all attributes that end with _weight_quantizer
        for name in dir(self):
            attr = getattr(self, name)
            if (
                name.endswith("weight_quantizer")
                and isinstance(attr, TensorQuantizer)
                and attr.fake_quant
            ):
                # Get the corresponding weight name by removing _weight_quantizer suffix
                weight_name = name[:-10]

                assert hasattr(self, weight_name), (
                    f"{name} doesn't have a corresponding {weight_name} in {self.__class__.__name__}"
                )
                weight = getattr(self, weight_name)
                weight.data.copy_(attr(weight.float()).to(weight.dtype))
                attr.disable()
                if not keep_attrs:
                    _attrs = [
                        "_pre_quant_scale",
                        "_amax",
                    ]
                    for attr_name in _attrs:
                        if hasattr(attr, attr_name):
                            delattr(attr, attr_name)


QuantModuleRegistry = _DMRegistryCls("Quant", QuantModule)


class QuantInputBase(QuantModule):
    """Base class for modules where the input is quantized."""

    input_quantizer: TensorQuantizer
    output_quantizer: TensorQuantizer
    default_quant_desc_input = QUANT_DESC_8BIT_PER_TENSOR
    default_quant_desc_output = QUANT_DESC_8BIT_PER_TENSOR

    def forward(self, input, *args, **kwargs):
        """Quantize the input before calling the original forward method."""
        input = self.input_quantizer(input)
        # Check MR: https://github.com/NVIDIA/Model-Optimizer/pull/824
        if hasattr(self, "_forward_pre_dm"):
            pre_fwd = getattr(self, "_forward_pre_dm")

            def _is_forward_in_mro(bound_or_func) -> bool:
                # If this is a bound method, compare its underlying function to any `forward`
                # implementation in the current MRO. If it matches, it's not an external monkey-patch.
                if hasattr(bound_or_func, "__func__"):
                    fn = bound_or_func.__func__
                    for cls in type(self).mro():
                        if cls.__dict__.get("forward") is fn:
                            return True
                return False

            if pre_fwd is getattr(self, "forward") or _is_forward_in_mro(pre_fwd):
                output = super().forward(input, *args, **kwargs)
            else:
                output = pre_fwd(input, *args, **kwargs)
        else:
            output = super().forward(input, *args, **kwargs)
        if isinstance(output, tuple):
            return (self.output_quantizer(output[0]), *output[1:])
        return self.output_quantizer(output)

    def _setup(self):
        """Patch the module's forward method to quantize the input."""
        self._register_temp_attribute(
            "input_quantizer", TensorQuantizer(self.default_quant_desc_input)
        )
        self._register_temp_attribute(
            "output_quantizer", TensorQuantizer(self.default_quant_desc_output)
        )
        self.output_quantizer.disable()


def _last_axis_block_size_from_dict(bs: dict | None) -> int | None:
    if not bs:
        return None
    v = bs.get(-1)
    if v is None:
        v = bs.get("-1")
    return int(v) if v is not None else None


def _tensor_quantizer_is_nvfp4_packed_storage(tq: TensorQuantizer) -> bool:
    if isinstance(tq, NVFP4StaticQuantizer):
        return True
    if getattr(tq, "_is_nvfp4_static_quantizer", False):
        return True
    bs = tq.block_sizes
    return bool(
        getattr(tq, "_num_bits", None) == (2, 1)
        and bs
        and bs.get("scale_bits") == (4, 3)
        and _last_axis_block_size_from_dict(bs) is not None
    )


def _nvfp4_tensor_quantizer_for_weight_quantizer(
    wq: TensorQuantizer | SequentialQuantizer,
) -> TensorQuantizer | None:
    if isinstance(wq, SequentialQuantizer):
        for q in wq:
            if isinstance(q, TensorQuantizer) and _tensor_quantizer_is_nvfp4_packed_storage(q):
                return q
        return None
    if isinstance(wq, TensorQuantizer) and _tensor_quantizer_is_nvfp4_packed_storage(wq):
        return wq
    return None


def _compute_hf_export_input_scale_tensor(
    module: nn.Module,
    attrs,
    iq: TensorQuantizer | SequentialQuantizer,
    weight_nvfp4_tq: TensorQuantizer | None,
) -> torch.Tensor | None:
    """Tensor matching ``unified_export_hf._export_quantized_weight`` ``input_scale``.

    Export uses :func:`modelopt.torch.export.quant_utils.get_activation_scaling_factor`
    (NVFP4 path) which returns ``None`` when the quantizer is **disabled**. After HF restore,
    input quantizers are often disabled while ``amax`` is still restored, so no buffer is
    registered and checkpoint ``input_scale`` keys remain unused.
    """
    if "disabled" in repr(iq):
        return None

    def _first_input_tq_with_amax(obj) -> TensorQuantizer | None:
        if isinstance(obj, TensorQuantizer) and getattr(obj, "amax", None) is not None:
            return obj
        if isinstance(obj, SequentialQuantizer):
            for q in obj:
                if isinstance(q, TensorQuantizer) and getattr(q, "amax", None) is not None:
                    return q
        return None

    tqi = _first_input_tq_with_amax(iq)
    if tqi is None:
        return None

    from modelopt.torch.quantization.qtensor.nvfp4_tensor import NVFP4QTensor as _NV4

    act = None
    try:
        from modelopt.torch.export.quant_utils import (
            get_activation_scaling_factor as _export_act_scale,
        )

        act = _export_act_scale(module, input_quantizer_name=attrs.input_quantizer)
    except Exception:
        act = None
    if act is None:
        try:
            act = _NV4.get_activation_scaling_factor(tqi)
        except Exception:
            act = None
    if act is None and weight_nvfp4_tq is not None and getattr(tqi, "amax", None) is not None:
        try:
            act = tqi.amax.float() / (tqi.maxbound * 448.0)
        except Exception:
            act = None
    return act


def _quant_linear_raw_weight_parameter(module: nn.Module) -> nn.Parameter | None:
    """Raw ``weight`` :class:`nn.Parameter` (avoids DynamicModule ``weight`` getter returning a tensor)."""
    w = module._parameters.get("weight")
    return w if isinstance(w, nn.Parameter) else None


def _hf_export_scale_buffer_leaf_names() -> frozenset[str]:
    """Buffer leaf names written by :mod:`modelopt.torch.export.unified_export_hf` (per weight slot)."""
    out: list[str] = []
    for wn in ("weight", "weight0"):
        a = quantizer_attr_names(wn)
        out.extend(
            [
                a.weight_scale,
                a.weight_scale_2,
                a.input_scale,
                a.output_scale,
            ]
        )
    # :func:`postprocess_state_dict` maps ``input_quantizer._pre_quant_scale`` → ``pre_quant_scale``.
    out.append("pre_quant_scale")
    return frozenset(out)


_hf_scale_buffer_checkpoint_reg: set[tuple[int, str]] = set()
_hf_scale_buffer_checkpoint_lock = threading.Lock()


def _read_hf_checkpoint_weight_map(ckpt_dir: Path) -> dict[str, str]:
    """Map tensor key → shard filename (same layout as :mod:`modelopt.torch.opt.plugins.huggingface`)."""
    index = ckpt_dir / "model.safetensors.index.json"
    if index.is_file():
        return json.loads(index.read_text(encoding="utf-8")).get("weight_map", {})
    shards = sorted(ckpt_dir.glob("*.safetensors"))
    if not shards:
        return {}
    weight_map: dict[str, str] = {}
    try:
        from safetensors import safe_open

        for sp in shards:
            with safe_open(str(sp), framework="pt", device="cpu") as f:
                for k in f.keys():
                    weight_map[k] = sp.name
    except Exception:
        from safetensors.torch import load_file

        preferred = [p for p in shards if p.name == "model.safetensors"]
        pick = preferred[0] if preferred else max(shards, key=lambda p: p.stat().st_size)
        weight_map = {k: pick.name for k in load_file(str(pick), device="cpu").keys()}
    return weight_map


def _get_submodule_for_hf_export_scale(model: nn.Module, head: str) -> nn.Module | None:
    """Resolve ``head`` for checkpoints that use ``model.…`` or a stripped prefix."""
    candidates = [head]
    if head.startswith("model."):
        candidates.append(head.removeprefix("model."))
    else:
        candidates.append(f"model.{head}")
    tried: set[str] = set()
    for h in candidates:
        if not h or h in tried:
            continue
        tried.add(h)
        try:
            return model.get_submodule(h)
        except (AttributeError, ValueError, KeyError, TypeError):
            continue
    return None


def register_hf_export_scale_buffers_from_checkpoint_dir(model: nn.Module, ckpt_dir: Path | str) -> None:
    """Create export scale buffers from the full safetensors ``weight_map`` (shapes from disk).

    Needed when Hugging Face drops scale keys before ``_load_state_dict_into_meta_model`` because
    they are missing from ``key_renaming_mapping``; the per-shard meta ``state_dict`` then never
    lists ``weight_scale`` / ``input_scale`` / ``pre_quant_scale``, so shard-only registration
    cannot attach buffers and those checkpoint keys stay unused.
    """
    ckpt_dir = Path(ckpt_dir).resolve()
    weight_map = _read_hf_checkpoint_weight_map(ckpt_dir)
    if not weight_map:
        return
    leaves = _hf_export_scale_buffer_leaf_names()
    from collections import defaultdict

    by_shard: dict[str, list[str]] = defaultdict(list)
    for key in weight_map:
        if not isinstance(key, str):
            continue
        head, dot, leaf = key.rpartition(".")
        if dot == "" or leaf not in leaves:
            continue
        by_shard[weight_map[key]].append(key)

    try:
        from safetensors import safe_open
    except ImportError:
        return

    for shard_name, keys in by_shard.items():
        spath = ckpt_dir / shard_name
        if not spath.is_file():
            continue
        with safe_open(str(spath), framework="pt", device="cpu") as f:
            fkeys = set(f.keys())
            for key in keys:
                if key not in fkeys:
                    continue
                head, _, leaf = key.rpartition(".")
                try:
                    meta_t = f.get_tensor(key)
                except Exception:
                    continue
                mod = _get_submodule_for_hf_export_scale(model, head)
                if mod is None or not _is_hf_export_scale_parent_module(mod):
                    continue
                shape = tuple(meta_t.shape)
                dt = meta_t.dtype
                if not (hasattr(dt, "is_floating_point") and dt.is_floating_point):
                    dt = torch.float32
                placeholder = torch.empty(shape, dtype=dt, device="cpu")
                existing = mod._buffers.get(leaf)
                if existing is None or tuple(existing.shape) != shape:
                    mod.register_buffer(leaf, placeholder)


def register_hf_export_scale_buffers_from_checkpoint_dir_once(model: nn.Module, ckpt_dir: Path | str) -> None:
    """Same as :func:`register_hf_export_scale_buffers_from_checkpoint_dir` but once per (model, dir).

    Parallel HF shard loaders may invoke the meta load hook many times; this avoids repeated I/O.
    """
    path = str(Path(ckpt_dir).resolve())
    reg_key = (id(model), path)
    with _hf_scale_buffer_checkpoint_lock:
        if reg_key in _hf_scale_buffer_checkpoint_reg:
            return
        register_hf_export_scale_buffers_from_checkpoint_dir(model, Path(path))
        _hf_scale_buffer_checkpoint_reg.add(reg_key)


def _is_hf_export_scale_parent_module(module: nn.Module) -> bool:
    """True for ModelOpt quant linears that may own HF export buffers (incl. ``SequentialQuantizer`` input).

    :func:`is_quantized_linear` requires ``input_quantizer`` to be a :class:`TensorQuantizer` only.
    W4A8 / AWQ-style stacks use :class:`SequentialQuantizer` on the input path; without this, shard
    metadata never registers ``weight_scale`` / ``input_scale`` / ``pre_quant_scale`` and safetensors
    keys show as unused.
    """
    iq = getattr(module, "input_quantizer", None)
    if not isinstance(module, QuantModule):
        return False
    if not isinstance(iq, (TensorQuantizer, SequentialQuantizer)):
        return False
    if not hasattr(module, "weight_quantizer"):
        return False
    wp = _quant_linear_raw_weight_parameter(module)
    if wp is not None and wp.dim() == 2:
        return True
    w0 = module._parameters.get("weight0")
    return isinstance(w0, nn.Parameter) and w0.dim() == 2


def _sync_hf_export_scale_buffer_into_input_quantizers(
    iq: TensorQuantizer | SequentialQuantizer,
    buf: torch.Tensor,
) -> None:
    """Copy a loaded root scale buffer into the first suitable ``_pre_quant_scale`` slot."""
    if buf is None or buf.numel() == 0:
        return
    candidates: list[TensorQuantizer] = []
    if isinstance(iq, TensorQuantizer):
        candidates = [iq]
    elif isinstance(iq, SequentialQuantizer):
        candidates = [m for m in iq if isinstance(m, TensorQuantizer)]

    b = buf.detach()
    for tq in candidates:
        if not getattr(tq, "_enable_pre_quant_scale", True):
            continue
        if hasattr(tq, "_pre_quant_scale") and tq._pre_quant_scale is not None:
            ps = tq._pre_quant_scale
            src = b.to(device=ps.device, dtype=ps.dtype)
            if tuple(ps.shape) == tuple(src.shape):
                ps.data.copy_(src)
                return
            if src.numel() == 1 and ps.numel() >= 1:
                ps.data.copy_(src.expand_as(ps))
                return
            if ps.numel() == 1 and src.numel() >= 1:
                ps.data.copy_(src.reshape(-1)[0].expand_as(ps))
                return
            continue
        tq.register_buffer("_pre_quant_scale", b.clone().to(dtype=torch.float32))
        return


def sync_hf_export_pre_quant_scale_buffers(model: nn.Module) -> None:
    """Copy loaded root scale buffers into ``input_quantizer._pre_quant_scale``.

    Unified HF export may store smoothquant scale as ``pre_quant_scale`` (postprocessed from
    ``input_quantizer._pre_quant_scale``) and/or activation scale as ``input_scale`` from
    :func:`modelopt.torch.export.quant_utils.get_activation_scaling_factor`. Runtime forward only
    multiplies activations by ``TensorQuantizer._pre_quant_scale`` (not the root ``input_scale``
    buffer), so checkpoints that only populate ``input_scale`` must still be synced here.

    Per-weight slots (``weight`` / ``weight0``) use :func:`weight_attr_names` so expert linears
    pick up ``weight0_input_scale`` when appropriate.
    """
    for module in model.modules():
        if not _is_hf_export_scale_parent_module(module):
            continue
        for wn in weight_attr_names(module):
            a = quantizer_attr_names(wn)
            iq = getattr(module, a.input_quantizer, None)
            if iq is None or not isinstance(iq, (TensorQuantizer, SequentialQuantizer)):
                continue
            buf = module._buffers.get("pre_quant_scale")
            if wn != "weight":
                alt = module._buffers.get(f"{wn}_pre_quant_scale")
                if alt is not None and alt.numel() > 0:
                    buf = alt
            if buf is None or buf.numel() == 0:
                buf = module._buffers.get(a.input_scale)
            if buf is None or buf.numel() == 0:
                continue
            _sync_hf_export_scale_buffer_into_input_quantizers(iq, buf)


def register_hf_export_scale_buffers_from_meta_state_dict(
    model: nn.Module,
    state_dict: dict[str, Any],
) -> None:
    """Resize/create export scale buffers to match the **current safetensors shard** metadata.

    Hugging Face passes a ``state_dict`` of **meta** tensors (shape/dtype only) into
    ``_load_state_dict_into_meta_model``. Using those shapes avoids mismatches when quantizer
    metadata is incomplete (auto-mixed quant, missing ``_amax``, or DynamicModule ``weight`` access).
    """
    leaves = _hf_export_scale_buffer_leaf_names()
    for key, meta_t in state_dict.items():
        if not isinstance(key, str):
            continue
        head, dot, leaf = key.rpartition(".")
        if dot == "" or leaf not in leaves:
            continue
        if not hasattr(meta_t, "shape"):
            continue
        mod = _get_submodule_for_hf_export_scale(model, head)
        if mod is None:
            continue
        if not _is_hf_export_scale_parent_module(mod):
            continue
        shape = tuple(meta_t.shape)
        dt = meta_t.dtype
        if not (hasattr(dt, "is_floating_point") and dt.is_floating_point):
            dt = torch.float32
        placeholder = torch.empty(shape, dtype=dt, device="cpu")
        existing = mod._buffers.get(leaf)
        if existing is None or tuple(existing.shape) != shape:
            mod.register_buffer(leaf, placeholder)


def _nvfp4_scale_buffer_tensor(t_cpu: torch.Tensor, param_device: torch.device) -> torch.Tensor:
    """Materialize export scale buffers so ``accelerate`` / ``device_map`` can place them.

    Do **not** register ``torch.empty(..., device="meta")`` placeholders: if a safetensors key
    does not assign (e.g. ``input_scale`` shape mismatch), the buffer stays meta and
    ``dispatch_model`` raises ``we need a `value` to put in on {device}``.

    Keep tensors concrete (clone on CPU when the owning parameter is still meta).
    """
    if t_cpu.device.type == "meta":
        return torch.tensor(1.0, dtype=torch.float32)
    out = t_cpu.detach().clone()
    if param_device.type != "meta":
        out = out.to(device=param_device)
    return out


def _is_nvfp4_export_buffer_module(module: nn.Module) -> bool:
    """Like ``is_quantized_linear`` but allow ``SequentialQuantizer`` input paths."""
    if _is_hf_export_scale_parent_module(module):
        return True
    wq = getattr(module, "weight_quantizer", None)
    w = _quant_linear_raw_weight_parameter(module)
    if (
        wq is None
        or w is None
        or w.dim() != 2
        or not hasattr(module, "out_features")
        or not hasattr(module, "in_features")
    ):
        return False
    if not isinstance(wq, (TensorQuantizer, SequentialQuantizer)):
        return False
    return _nvfp4_tensor_quantizer_for_weight_quantizer(wq) is not None


def register_hf_nvfp4_export_scale_buffer_placeholders(model: nn.Module) -> None:
    """Register ``weight_scale`` / ``weight_scale_2`` / ``input_scale`` on NVFP4 linears.

    :func:`modelopt.torch.export.unified_export_hf._export_quantized_weight` saves these
    buffers in safetensors. Without matching module attributes, HuggingFace leaves them
    unused and packed-weight dequant must guess scales from the quantizer alone, which can
    diverge from export and corrupt logits.

    Buffers are initialized from the restored quantizers (same formulas as export) and are
    overwritten by ``load_state_dict`` when checkpoint keys match.

    Runs both after quantizer restore (logical weights) and again immediately before HF
    meta ``load_state_dict`` (packed ``uint8`` weights): the strict ``(out, in)`` shape check
    would otherwise skip every layer once weights are resized.
    """
    from modelopt.torch.quantization.qtensor.nvfp4_tensor import NVFP4QTensor

    for module in model.modules():
        if not _is_nvfp4_export_buffer_module(module):
            continue
        tq = _nvfp4_tensor_quantizer_for_weight_quantizer(module.weight_quantizer)
        if tq is None:
            continue
        attrs = quantizer_attr_names("weight")
        w = _quant_linear_raw_weight_parameter(module)
        if w is None or w.dim() != 2:
            continue
        out_f, in_f = int(module.out_features), int(module.in_features)
        if int(w.shape[0]) != out_f:
            continue
        if w.dtype == torch.uint8:
            if int(w.shape[1]) * 2 < in_f:
                continue
        elif tuple(w.shape) != (out_f, in_f):
            continue

        # Compute scales on CPU so meta / empty tensors in the model do not break block reducers.
        logical_dummy = torch.randn(out_f, in_f, device="cpu", dtype=torch.float32).mul_(0.01)
        dev = w.device

        # HF ``from_pretrained`` often builds the graph on ``meta`` first. Do not read quantizer
        # buffers or call NVFP4 reducers on meta tensors (``Cannot copy out of meta tensor``).
        if dev.type == "meta":
            if attrs.weight_scale not in module._buffers:
                module.register_buffer(attrs.weight_scale, torch.tensor(1.0, dtype=torch.float32))
            if attrs.weight_scale_2 not in module._buffers:
                module.register_buffer(attrs.weight_scale_2, torch.tensor(1.0, dtype=torch.float32))
            iq = getattr(module, attrs.input_quantizer, None)
            if iq is not None and attrs.input_scale not in module._buffers:
                module.register_buffer(attrs.input_scale, torch.tensor(1.0, dtype=torch.float32))
            continue

        try:
            w_scale, w_scale_2 = NVFP4QTensor.get_weights_scaling_factor_from_quantizer(
                tq, logical_dummy, weights_scaling_factor_2=None, keep_high_precision=False
            )
        except Exception:
            continue

        if attrs.weight_scale not in module._buffers:
            module.register_buffer(
                attrs.weight_scale,
                _nvfp4_scale_buffer_tensor(w_scale.detach().cpu(), dev),
            )
        if attrs.weight_scale_2 not in module._buffers:
            w2 = w_scale_2.squeeze()
            module.register_buffer(
                attrs.weight_scale_2,
                _nvfp4_scale_buffer_tensor(w2.detach().cpu(), dev),
            )

        iq = getattr(module, attrs.input_quantizer, None)
        if iq is not None and attrs.input_scale not in module._buffers:
            act = _compute_hf_export_input_scale_tensor(module, attrs, iq, tq)
            a_cpu: torch.Tensor | None = None
            if act is not None and act.device.type != "meta":
                try:
                    # Match ``unified_export_hf._export_quantized_weight`` (``.squeeze()`` only).
                    a_cpu = act.squeeze().detach().float().cpu()
                except (NotImplementedError, RuntimeError):
                    a_cpu = None
            if a_cpu is not None:
                module.register_buffer(attrs.input_scale, _nvfp4_scale_buffer_tensor(a_cpu, dev))
            else:
                module.register_buffer(attrs.input_scale, torch.tensor(1.0, dtype=torch.float32))


def _fp8_leaf_weight_quantizer(
    wq: TensorQuantizer | SequentialQuantizer | None,
) -> TensorQuantizer | None:
    """Return the ``TensorQuantizer`` that carries FP8 E4M3 weight state (``num_bits == (4, 3)``)."""
    if wq is None:
        return None
    if isinstance(wq, SequentialQuantizer):
        for q in wq:
            if isinstance(q, TensorQuantizer) and q.num_bits == (4, 3):
                return q
        return None
    if isinstance(wq, TensorQuantizer) and wq.num_bits == (4, 3):
        return wq
    return None


def _fp8_effective_block_size_for_placeholder(wq: TensorQuantizer | None) -> int:
    """Block size along input features for block-wise FP8 scales (export / HF shards)."""
    bs = _last_axis_block_size_from_dict(getattr(wq, "block_sizes", None) if wq is not None else None)
    if bs is not None and bs > 0:
        return bs
    return 8


def _infer_fp8_weight_scale_placeholder_from_linear(
    module: nn.Module,
    wq: TensorQuantizer | None,
) -> torch.Tensor:
    """Ones matching common FP8 shard layouts when ``_amax`` is missing."""
    out_f = int(module.out_features)
    in_f = int(module.in_features)
    bs = _fp8_effective_block_size_for_placeholder(wq)
    n_blocks = (in_f + bs - 1) // bs
    # Single block along input → export often uses a 1D per-output-row scale ``(out_features,)``.
    if n_blocks <= 1:
        return torch.ones(out_f, dtype=torch.float32)
    return torch.ones(out_f, n_blocks, dtype=torch.float32)


def _fp8_default_e4m3_weight_scale_placeholder_cpu(
    wq: TensorQuantizer | None,
    module: nn.Module | None = None,
) -> torch.Tensor:
    """Tensor layout for ``weight_scale`` matching ``unified_export_hf`` FP8 export.

    Export uses ``amax / maxbound`` with the same rank as ``_amax`` (scalar, per-row, or 2D block).
    Hugging Face meta ``load_state_dict`` must see a buffer with a **matching shape**; registering
    a scalar while the shard holds e.g. ``[out_features, n_blocks]`` raises a size mismatch.

    When ``_amax`` is absent (common right before safetensors assign), infer a 2D block scale layout
    from ``out_features``, ``in_features``, and the quantizer's last-axis block size (default 8).
    """
    if wq is not None and hasattr(wq, "_amax") and wq._amax is not None:
        try:
            amax_f = wq._amax.to(torch.float32)
            mb = float(wq.maxbound)
        except (NotImplementedError, RuntimeError):
            amax_f = None
        else:
            if amax_f.device.type == "meta":
                return torch.ones(amax_f.shape, dtype=torch.float32)

            try:
                if amax_f.dim() == 0:
                    return torch.tensor((amax_f / mb).item(), dtype=torch.float32)
                if amax_f.dim() == 1 and amax_f.numel() == 1:
                    return torch.tensor((amax_f / mb).item(), dtype=torch.float32)
                return (amax_f / mb).float().detach().cpu()
            except (NotImplementedError, RuntimeError):
                pass

    if module is not None and hasattr(module, "out_features") and hasattr(module, "in_features"):
        try:
            return _infer_fp8_weight_scale_placeholder_from_linear(module, wq)
        except (AttributeError, TypeError, ValueError):
            pass

    return torch.tensor(1.0, dtype=torch.float32)


def register_hf_fp8_export_scale_buffer_placeholders(model: nn.Module) -> None:
    """Register ``weight_scale`` / ``input_scale`` on default FP8 (E4M3) linears.

    :func:`modelopt.torch.export.unified_export_hf._export_quantized_weight` registers these
    buffers for ``QUANTIZATION_FP8``. Without them, HuggingFace reports safetensors keys as unused
    and :meth:`QuantLinearConvBase._get_quantized_weight` must fall back to quantizer ``_amax``,
    which is often **not** loaded from shards (only ``modelopt_state.pth`` metadata) — producing
    random scales and garbled generation.

    Placeholders match the export formulas; ``load_state_dict`` overwrites them from the checkpoint.

    ``huggingface`` may call this once after ``restore_from_modelopt_state`` (sometimes leaving a
    scalar ``weight_scale``) and again before meta shard load; we **replace** buffers whose shape
    does not match the current placeholder so the second call can correct an earlier scalar.
    """
    for module in model.modules():
        if not _is_hf_export_scale_parent_module(module):
            continue
        raw_wq = getattr(module, "weight_quantizer", None)
        wq = _fp8_leaf_weight_quantizer(raw_wq)
        if wq is None:
            continue

        w = _quant_linear_raw_weight_parameter(module)
        if w is None:
            continue
        dev = w.device
        attrs = quantizer_attr_names("weight")

        desired_cpu = _fp8_default_e4m3_weight_scale_placeholder_cpu(wq, module)
        if dev.type == "meta":
            placeholder = desired_cpu
        else:
            placeholder = _nvfp4_scale_buffer_tensor(desired_cpu, dev)

        existing_ws = module._buffers.get(attrs.weight_scale)
        if existing_ws is None or tuple(existing_ws.shape) != tuple(placeholder.shape):
            module.register_buffer(attrs.weight_scale, placeholder)

        if dev.type == "meta":
            iq = getattr(module, attrs.input_quantizer, None)
            if iq is not None and attrs.input_scale not in module._buffers:
                module.register_buffer(attrs.input_scale, torch.tensor(1.0, dtype=torch.float32))
            continue

        iq = getattr(module, attrs.input_quantizer, None)
        if iq is not None and attrs.input_scale not in module._buffers:
            act: torch.Tensor | None = None
            if isinstance(iq, TensorQuantizer) and hasattr(iq, "_amax") and iq._amax is not None:
                if iq._amax.device.type != "meta":
                    try:
                        from modelopt.torch.export.quant_utils import get_scaling_factor

                        act = get_scaling_factor(iq)
                    except Exception:
                        act = None
                    if act is None:
                        try:
                            ea = iq.export_amax()
                            if ea is not None and ea.device.type != "meta":
                                act = ea.float() / iq.maxbound
                        except (NotImplementedError, RuntimeError):
                            act = None
            if act is not None and act.device.type != "meta":
                try:
                    a = act.squeeze().detach().float().cpu()
                except (NotImplementedError, RuntimeError):
                    a = torch.tensor(1.0, dtype=torch.float32)
            else:
                a = torch.tensor(1.0, dtype=torch.float32)
            module.register_buffer(attrs.input_scale, _nvfp4_scale_buffer_tensor(a, dev))


def _try_dequant_packed_nvfp4_linear_weight(
    module: "QuantLinearConvBase", packed: torch.Tensor
) -> torch.Tensor | None:
    """Dequantize HF-exported packed NVFP4 ``uint8`` weights for fake-quant ``F.linear``.

    Unified HF export stores linear weights as packed ``uint8``; fake-quant kernels expect
    floating-point tensors. Prefer ``weight_scale`` / ``weight_scale_2`` loaded from the
    HF checkpoint when present; otherwise fall back to the restored weight quantizer.
    """
    tq = _nvfp4_tensor_quantizer_for_weight_quantizer(module.weight_quantizer)
    if tq is None or packed.dtype != torch.uint8 or packed.dim() != 2:
        return None
    bs = tq.block_sizes
    block_size = _last_axis_block_size_from_dict(bs)
    if block_size is None:
        return None

    out_f, in_f = int(module.out_features), int(module.in_features)
    if packed.shape[0] != out_f:
        return None
    padded_in = packed.shape[1] * 2
    if padded_in < in_f:
        return None

    dt = (
        module.bias.dtype
        if module.bias is not None and module.bias.dtype.is_floating_point
        else torch.bfloat16
    )
    logical_dummy = torch.empty(out_f, in_f, device=packed.device, dtype=dt)

    from modelopt.torch.quantization.qtensor.nvfp4_tensor import NVFP4QTensor

    attrs = quantizer_attr_names("weight")
    buf_ws = getattr(module, attrs.weight_scale, None)
    buf_w2 = getattr(module, attrs.weight_scale_2, None)
    if (
        isinstance(buf_ws, torch.Tensor)
        and buf_ws.numel() > 0
        and isinstance(buf_w2, torch.Tensor)
        and buf_w2.numel() > 0
    ):
        w_scale = buf_ws.to(device=packed.device)
        w_scale_2 = buf_w2.to(device=packed.device)
    else:
        try:
            w_scale, w_scale_2 = NVFP4QTensor.get_weights_scaling_factor_from_quantizer(
                tq, logical_dummy, weights_scaling_factor_2=None, keep_high_precision=False
            )
        except Exception:
            return None

    qtensor = NVFP4QTensor(torch.Size([out_f, padded_in]), dt, packed)
    try:
        dq = qtensor.dequantize(
            dtype=dt,
            scale=w_scale,
            double_scale=w_scale_2,
            block_sizes=tq.block_sizes,
            fast=False,
        )
    except Exception:
        return None

    if dq.shape[-1] > in_f:
        dq = dq[..., :in_f]
    elif dq.shape[-1] < in_f:
        return None
    return dq


def _is_int4_awq_or_w4a8_awq_weight_layer(module: nn.Module) -> bool:
    """Whether ``module`` matches INT4_AWQ / W4A8_AWQ (same as export format checks).

    Do **not** call :func:`modelopt.torch.export.quant_utils.get_quantization_format` here:
    it does ``getattr(module, "weight")``, which re-enters the dynamic ``weight`` accessor
    and causes infinite recursion during :meth:`QuantLinearConvBase._get_quantized_weight`.
    """
    wq = getattr(module, "weight_quantizer", None)
    if wq is None:
        return False
    # Do not gate on ``is_enabled``: HF reload may disable quantizers while ``weight`` stays
    # packed ``uint8``; we still must unpack for ``F.linear`` (fake_quant does not support Byte).
    if isinstance(wq, SequentialQuantizer):
        if len(wq) != 2:
            return False
        q0, q1 = wq[0], wq[1]
        if getattr(q0, "num_bits", None) != 4 or getattr(q1, "num_bits", None) != (4, 3):
            return False
        if _tensor_quantizer_is_nvfp4_packed_storage(q0):
            return False
        bs = getattr(q0, "block_sizes", None) or {}
        return bool(bs.get(-1) or bs.get("-1"))
    if not isinstance(wq, TensorQuantizer):
        return False
    if getattr(wq, "num_bits", None) != 4:
        return False
    if _tensor_quantizer_is_nvfp4_packed_storage(wq):
        return False
    bs = getattr(wq, "block_sizes", None) or {}
    return bool(bs.get(-1) or bs.get("-1"))


def _try_dequant_packed_int4_awq_hf_linear_weight(
    module: "QuantLinearConvBase", packed: torch.Tensor
) -> torch.Tensor | None:
    """Dequantize HF-exported INT4 AWQ / W4A8-AWQ first-stage packed ``uint8`` weights.

    Layout matches :func:`modelopt.torch.export.quant_utils.pack_int4_in_uint8` (2D):
    ``(out_features // 2, in_features)`` with low/high nibbles for adjacent output rows.
    """
    if not _is_int4_awq_or_w4a8_awq_weight_layer(module):
        return None
    if packed.dtype != torch.uint8 or packed.dim() != 2:
        return None
    out_f, in_f = int(module.out_features), int(module.in_features)
    if out_f % 2 != 0 or tuple(packed.shape) != (out_f // 2, in_f):
        return None

    wq = module.weight_quantizer
    wq0 = wq[0] if isinstance(wq, SequentialQuantizer) else wq
    bs = getattr(wq0, "block_sizes", None)
    if not isinstance(bs, dict):
        return None
    block_size = bs.get(-1)
    if block_size is None:
        block_size = bs.get(len(packed.shape) - 1)
    if block_size is None:
        return None
    block_size = int(block_size)
    if block_size <= 0 or in_f % block_size != 0:
        return None

    attrs = quantizer_attr_names("weight")
    n_blocks = in_f // block_size
    scale_buf = getattr(module, attrs.weight_scale, None)
    scales: torch.Tensor | None = None
    if isinstance(scale_buf, torch.Tensor) and scale_buf.numel() > 0:
        scales = scale_buf.to(device=packed.device, dtype=torch.float32).reshape(-1)
    if scales is None or scales.numel() == 0:
        amax = getattr(wq0, "amax", None)
        if isinstance(amax, torch.Tensor) and amax.numel() > 0:
            mb = float(getattr(wq0, "maxbound", 7.0))
            scales = (amax.float().reshape(-1) / mb).to(device=packed.device)
    if scales is None or scales.numel() == 0:
        return None
    if scales.numel() == 1 and n_blocks > 1:
        scales = scales.expand(n_blocks)
    elif scales.numel() > n_blocks:
        scales = scales[:n_blocks]
    elif scales.numel() != n_blocks:
        return None

    # Inverse of pack_int4_in_uint8 2D branch: packed_byte = val0 | (val1 << 4)
    v_lo = (packed & 0x0F).to(torch.float32) - 8.0
    v_hi = ((packed >> 4) & 0x0F).to(torch.float32) - 8.0
    int4_rows = torch.stack((v_lo, v_hi), dim=0).permute(1, 2, 0).reshape(out_f, in_f)

    col_idx = torch.arange(in_f, device=packed.device, dtype=torch.long)
    scale_per_col = scales[col_idx // block_size].unsqueeze(0)
    dq = int4_rows * scale_per_col

    out_dt = (
        module.bias.dtype
        if module.bias is not None and module.bias.dtype.is_floating_point
        else torch.bfloat16
    )
    return dq.to(out_dt)


def _try_dequant_int8_hf_linear_weight(
    module: "QuantLinearConvBase", weight: torch.Tensor
) -> torch.Tensor | None:
    """Dequantize HF-exported INT8_SQ / INT8_WO ``int8`` weights for ``F.linear``.

    Export uses ``(weight / scale[:, None]).round().clamp(-128, 127).to(torch.int8)``; inverse is
    ``int8.float() * scale[:, None]`` (see :func:`modelopt.torch.export.quant_utils.from_quantized_weight`).
    Required when ``TensorQuantizer`` modules are disabled (they would otherwise pass int8 through).
    """
    if weight.dtype != torch.int8 or weight.dim() != 2:
        return None
    wq = _weight_quantizer_leaf(module)
    if wq is None or getattr(wq, "num_bits", None) != 8:
        return None

    out_f, in_f = int(module.out_features), int(module.in_features)
    if tuple(weight.shape) != (out_f, in_f):
        return None

    attrs = quantizer_attr_names("weight")
    scale_buf = getattr(module, attrs.weight_scale, None)
    scales: torch.Tensor | None = None
    if isinstance(scale_buf, torch.Tensor) and scale_buf.numel() > 0:
        scales = scale_buf.to(device=weight.device, dtype=torch.float32).reshape(-1)
    if scales is None or scales.numel() == 0:
        amax = getattr(wq, "amax", None)
        if isinstance(amax, torch.Tensor) and amax.numel() > 0:
            mb = float(getattr(wq, "maxbound", 127.0))
            scales = (amax.float().reshape(-1) / mb).to(device=weight.device)
    if scales is None or scales.numel() == 0:
        return None

    w_f = weight.to(torch.float32)
    if scales.numel() == 1:
        dq = w_f * scales.item()
    elif scales.numel() == out_f:
        dq = w_f * scales[:, None]
    else:
        return None

    out_dt = (
        module.bias.dtype
        if module.bias is not None and module.bias.dtype.is_floating_point
        else torch.bfloat16
    )
    return dq.to(out_dt)


def _tensor_is_fp8_e4m3fn(weight: torch.Tensor) -> bool:
    """Match FP8 E4M3 dtypes even if ``torch.float8_e4m3fn`` is missing from the ``torch`` module."""
    dt = weight.dtype
    ref = getattr(torch, "float8_e4m3fn", None)
    if ref is not None and dt == ref:
        return True
    ds = str(dt)
    return "float8" in ds.lower() and "e4m3" in ds.lower()


def _weight_quantizer_leaf(
    module: "QuantLinearConvBase",
) -> TensorQuantizer | None:
    """First ``TensorQuantizer`` used for export scaling (handles ``SequentialQuantizer``)."""
    wq = getattr(module, "weight_quantizer", None)
    if wq is None:
        return None
    if isinstance(wq, SequentialQuantizer):
        return wq[0] if len(wq) > 0 else None
    return wq


def _is_fp8_e4m3_default_hf_linear(module: "QuantLinearConvBase") -> bool:
    """Whether this layer matches default FP8 export (``QUANTIZATION_FP8``, not block/MX formats).

    Used to treat ``uint8`` weight tensors as E4M3 bytes (safetensors) without colliding with
    NVFP4 / INT4 packed layouts.
    """
    wq = _weight_quantizer_leaf(module)
    if wq is None or wq.num_bits != (4, 3):
        return False
    if wq.block_sizes:
        return False
    return True


def _is_fp8_pb_wo_hf_linear(module: "QuantLinearConvBase") -> bool:
    """Match :func:`modelopt.torch.export.quant_utils.get_quantization_format` ``QUANTIZATION_FP8_PB_WO``."""
    wq = _weight_quantizer_leaf(module)
    if wq is None or getattr(wq, "num_bits", None) != (4, 3):
        return False
    if not getattr(wq, "fake_quant", False):
        return False
    bs = getattr(wq, "block_sizes", None)
    return isinstance(bs, dict) and len(bs) > 0


def _fp8_pb_wo_dequant_block_sizes(bs_raw: dict) -> dict[int, int] | None:
    """Keep only axis index → block size entries for :meth:`FP8QTensor.dequantize`.

    Quantizer ``block_sizes`` also carries ``type``, ``scale_bits``, etc.; those must not be passed
    through ``int(key)`` — doing so made the whole PB_WO reload path return ``None`` and weights
    dequantize incorrectly (garbled generation on auto-mixed FP8 exports).
    """
    out: dict[int, int] = {}
    skip = frozenset({"type", "scale_bits", "scale_block_sizes"})
    for k, v in bs_raw.items():
        if k in skip:
            continue
        try:
            dim = int(k)
        except (TypeError, ValueError):
            continue
        try:
            out[dim] = int(v)
        except (TypeError, ValueError):
            return None
    return out if out else None


def _try_dequant_fp8_pb_wo_hf_linear_weight(
    module: "QuantLinearConvBase", weight: torch.Tensor
) -> torch.Tensor | None:
    """Dequantize HF-exported block fake-quant FP8 (``QUANTIZATION_FP8_PB_WO``) weights.

    Export uses :meth:`modelopt.torch.quantization.qtensor.fp8_tensor.FP8QTensor.quantize` with
    2D block ``block_sizes``; reload must use :meth:`FP8QTensor.dequantize`, not
    :func:`modelopt.torch.export.quant_utils.from_quantized_weight` (which only handles 1D scales).
    Without this path, ``uint8`` / E4M3 weights skip :func:`_is_fp8_e4m3_default_hf_linear` and fall
    through to a wrong multiply → garbage logits (common in auto-mixed FP8 exports).
    """
    if not _is_fp8_pb_wo_hf_linear(module) or weight.dim() != 2:
        return None

    out_f, in_f = int(module.out_features), int(module.in_features)
    if tuple(weight.shape) != (out_f, in_f):
        return None

    wq = _weight_quantizer_leaf(module)
    assert wq is not None
    bs_raw = getattr(wq, "block_sizes", None)
    if not isinstance(bs_raw, dict):
        return None
    block_sizes = _fp8_pb_wo_dequant_block_sizes(bs_raw)
    if not block_sizes:
        return None

    attrs = quantizer_attr_names("weight")
    scales = getattr(module, attrs.weight_scale, None)
    if not isinstance(scales, torch.Tensor) or scales.numel() == 0:
        return None

    qd = weight.contiguous()
    if qd.dtype == torch.uint8:
        qd = qd.view(torch.float8_e4m3fn)
    elif not _tensor_is_fp8_e4m3fn(qd):
        return None

    out_dt = (
        module.bias.dtype
        if module.bias is not None and module.bias.dtype.is_floating_point
        else torch.bfloat16
    )

    from modelopt.torch.quantization.qtensor.fp8_tensor import FP8QTensor

    try:
        qt = FP8QTensor(torch.Size([out_f, in_f]), out_dt, qd)
        return qt.dequantize(
            dtype=out_dt,
            scale=scales.to(device=qd.device),
            block_sizes=block_sizes,
        )
    except Exception:
        return None


def _try_dequant_fp8_e4m3_hf_linear_weight(
    module: "QuantLinearConvBase", weight: torch.Tensor
) -> torch.Tensor | None:
    """Dequantize HF-exported FP8 E4M3 ``float8_e4m3fn`` weights for ``F.linear``.

    Must match :func:`modelopt.torch.export.quant_utils.from_quantized_weight` for
    ``QUANTIZATION_FP8`` — manual multiply/broadcast easily drifts from export (garbled logits).
    """
    if weight.dim() != 2:
        return None

    out_f, in_f = int(module.out_features), int(module.in_features)
    if tuple(weight.shape) != (out_f, in_f):
        return None

    # E4M3 weights with block fake-quant (PB_WO) must use FP8QTensor.dequantize (handles 2D scales).
    if weight.dtype == torch.uint8 or _tensor_is_fp8_e4m3fn(weight):
        dq_pb = _try_dequant_fp8_pb_wo_hf_linear_weight(module, weight)
        if dq_pb is not None:
            return dq_pb

    if weight.dtype == torch.uint8:
        if not _is_fp8_e4m3_default_hf_linear(module):
            return None
    elif _tensor_is_fp8_e4m3fn(weight):
        pass
    else:
        return None

    out_dt = (
        module.bias.dtype
        if module.bias is not None and module.bias.dtype.is_floating_point
        else torch.bfloat16
    )

    from modelopt.torch.export.quant_utils import QUANTIZATION_FP8, from_quantized_weight

    attrs = quantizer_attr_names("weight")
    scale_buf = getattr(module, attrs.weight_scale, None)
    if isinstance(scale_buf, torch.Tensor) and scale_buf.numel() > 0:
        # Do not fall back to quantizer amax on failure: checkpoint ``weight_scale`` must match
        # safetensors weights; a silent fallback (e.g. after a failed non-contiguous ``view``)
        # produces plausible logits that decode as garbage.
        return from_quantized_weight(
            weight,
            scale_buf.to(device=weight.device),
            QUANTIZATION_FP8,
            out_dt,
        )

    wq = _weight_quantizer_leaf(module)
    if wq is not None:
        amax = getattr(wq, "amax", None)
        if isinstance(amax, torch.Tensor) and amax.numel() > 0:
            mb = float(getattr(wq, "maxbound", 448.0))
            sf = (amax.float() / mb).to(device=weight.device)
            return from_quantized_weight(weight, sf, QUANTIZATION_FP8, out_dt)
    return None


class QuantLinearConvBase(QuantInputBase):
    """Base class for quantized linear modules.

    Quantized linear modules are modules where both the input and the weight are quantized.
    """

    weight_quantizer: TensorQuantizer | SequentialQuantizer
    _enable_weight_quantization: bool
    default_quant_desc_weight = QUANT_DESC_8BIT_PER_TENSOR

    @contextlib.contextmanager
    def quantize_weight(self):
        """Context in which `self.weight` is quantized."""
        self._enable_weight_quantization = True
        try:
            yield
        finally:
            self._enable_weight_quantization = False

    @staticmethod
    def _get_quantized_weight(module: "QuantLinearConvBase", weight: torch.Tensor) -> torch.Tensor:
        # Materialize HF checkpoint dtypes to float for ``F.linear`` **before** gating on
        # ``_enable_weight_quantization``: ``self.weight`` can be read without the
        # ``quantize_weight()`` context (e.g. accelerator hooks), which would otherwise return
        # raw uint8 / int8 / float8 and crash in ``nn.Linear``.
        if isinstance(weight, torch.Tensor) and weight.dim() == 2:
            if weight.dtype == torch.uint8:
                dq = _try_dequant_packed_nvfp4_linear_weight(module, weight)
                if dq is not None:
                    return dq
                dq = _try_dequant_fp8_pb_wo_hf_linear_weight(module, weight)
                if dq is not None:
                    return dq
                dq = _try_dequant_packed_int4_awq_hf_linear_weight(module, weight)
                if dq is not None:
                    return dq
            elif weight.dtype == torch.int8:
                dq = _try_dequant_int8_hf_linear_weight(module, weight)
                if dq is not None:
                    return dq

            dq = _try_dequant_fp8_e4m3_hf_linear_weight(module, weight)
            if dq is not None:
                return dq

        if isinstance(weight, torch.Tensor) and weight.dtype == torch.uint8:
            raise RuntimeError(
                "Packed uint8 weight could not be dequantized for this layer (NVFP4/INT4 AWQ); "
                "refusing to run fake_quant on Byte. Check weight_scale buffers and quantizer state."
            )
        if isinstance(weight, torch.Tensor) and weight.dtype == torch.int8:
            raise RuntimeError(
                "INT8 weight could not be dequantized for this layer; refusing ``F.linear`` with "
                "Char weights. Check ``weight_scale`` buffers and 8-bit weight_quantizer state."
            )
        if isinstance(weight, torch.Tensor) and _tensor_is_fp8_e4m3fn(weight):
            raise RuntimeError(
                "FP8 E4M3 weight could not be dequantized; refusing ``F.linear`` with float8 "
                "weights. Check ``weight_scale`` buffers and FP8 weight_quantizer state."
            )

        if module._enable_weight_quantization or is_torch_export_mode():
            return module.weight_quantizer(weight)
        return weight

    def forward(self, input, *args, **kwargs):
        """Quantize the input and the weight before calling the original forward method."""
        # self.quntize_weight() setting attributes is not allowed for torch.export.
        if is_torch_export_mode():
            return super().forward(input, *args, **kwargs)

        with self.quantize_weight():
            return super().forward(input, *args, **kwargs)

    def _setup(self):
        super()._setup()
        self._register_temp_attribute(
            "weight_quantizer", TensorQuantizer(self.default_quant_desc_weight)
        )
        self._register_temp_attribute("_enable_weight_quantization", False)
        self._register_dynamic_attribute("weight", self._get_quantized_weight)


class _LegacyQuantInputBaseMixin:
    """A mixin to support legacy quantized modules which needs to have an __init__ method."""

    _quantized_cls = QuantInputBase
    default_quant_desc_input = QUANT_DESC_8BIT_PER_TENSOR
    default_quant_desc_output = QUANT_DESC_8BIT_PER_TENSOR

    def __init__(self, *args, quant_desc_input=None, **kwargs):
        """Initialize the module with its original __init__ and patch its forward."""
        self.default_quant_desc_input = quant_desc_input or self.default_quant_desc_input
        super().__init__(*args, **kwargs)
        QuantModuleRegistry.convert(self)


class _LegacyQuantLinearConvBaseMixin(_LegacyQuantInputBaseMixin):
    """A mixin to support legacy quantized modules which needs to have an __init__ method."""

    _quantized_cls = QuantLinearConvBase
    default_quant_desc_weight = QUANT_DESC_8BIT_PER_TENSOR

    def __init__(self, *args, quant_desc_input=None, quant_desc_weight=None, **kwargs):
        """Initialize the module with its original __init__ and patch its forward."""
        self.default_quant_desc_weight = quant_desc_weight or self.default_quant_desc_weight
        super().__init__(*args, quant_desc_input=quant_desc_input, **kwargs)
