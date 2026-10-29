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

"""Regionwise Gumbel-Softmax refinement with ordinary ModelOpt quantizer state."""

import hashlib
import math
from contextlib import contextmanager
from copy import deepcopy
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.func import functional_call
from torch.utils._pytree import tree_map
from torch.utils.checkpoint import checkpoint

from .conversion import set_quantizer_by_cfg_context
from .nn import SequentialQuantizer, TensorQuantizer
from .qtensor.nvfp4_tensor import NVFP4QTensor
from .utils import is_quantized_linear
from .utils.layerwise_calib import LayerActivationCollector
from .utils.shared_input import SHARED_PATTERNS, find_shared_input_groups

__all__ = ["gsq"]

_REGIONS = {"self_attn", "linear_attn", "attention", "mlp", "feed_forward"}


class _Assignment(nn.Module):
    def __init__(self, module, shared_scales, scale_key=None):
        super().__init__()
        self.shape, self.dtype = module.weight.shape, module.weight.dtype
        q = module.weight_quantizer
        self.nvfp4 = q.is_nvfp4_static
        self.amax_dtype = q.amax.dtype
        hard = q(module.weight).detach().float()
        if self.nvfp4:
            block, global_scale = NVFP4QTensor.get_weights_scaling_factor_from_quantizer(
                q, module.weight
            )
            global_scale = global_scale.clamp_min(torch.finfo(torch.float32).tiny)
            scale = block.float() * global_scale
            self.log_block = nn.Parameter(block.float().log())
            codebook = torch.tensor(
                [-6, -4, -3, -2, -1.5, -1, -0.5, 0, 0.5, 1, 1.5, 2, 3, 4, 6],
                device=hard.device,
            )
            source_scale, key = global_scale, id(q.global_amax)
            normalized = hard.reshape(*scale.shape, 16) / scale.unsqueeze(-1)
        else:
            source_scale, key = q.amax.float() / 448, id(q.amax)
            scale = source_scale.clamp_min(torch.finfo(torch.float32).tiny)
            raw = torch.arange(256, device=hard.device, dtype=torch.uint8)
            values = raw.view(torch.float8_e4m3fn).float()
            codebook = values[torch.isfinite(values)].unique(sorted=True)
            normalized = hard / scale
        self.log_scale = shared_scales.setdefault(
            key if self.nvfp4 or scale_key is None else scale_key,
            nn.Parameter(source_scale.clamp_min(torch.finfo(torch.float32).tiny).log()),
        )
        self.register_buffer("codebook", codebook)
        # Searchsorted avoids a weight-size-by-codebook temporary.
        upper = torch.searchsorted(codebook, normalized.contiguous()).clamp_max(len(codebook) - 1)
        lower = (upper - 1).clamp_min(0)
        center = torch.where(
            (normalized - codebook[lower]).abs() <= (normalized - codebook[upper]).abs(),
            lower,
            upper,
        )
        self.register_buffer("center", center.reshape(self.shape).to(torch.uint8))
        logits = torch.zeros((*self.shape, 5), device=hard.device, dtype=self.dtype)
        logits[..., 2] = 2
        self.logits = nn.Parameter(logits)

    def scales(self, hard):
        global_scale = self.log_scale.exp()
        if not self.nvfp4:
            projected = (global_scale * 448).to(self.amax_dtype).float() / 448
            return (projected if hard else global_scale + (projected - global_scale).detach()), None
        block = self.log_block.exp().clamp(2**-9, 448)
        projected = block.to(torch.float8_e4m3fn).float()
        block = projected if hard else block + (projected - block).detach()
        return block * global_scale, global_scale

    def forward(self, temperature=1.0, hard=False):
        indices = self.center.to(torch.int16).unsqueeze(-1) + torch.arange(
            -2, 3, device=self.center.device, dtype=torch.int16
        )
        valid = (indices >= 0) & (indices < len(self.codebook))
        values = self.codebook[indices.clamp(0, len(self.codebook) - 1).long()]
        logits = self.logits.masked_fill(~valid, -torch.inf)
        if hard:
            codes = values.gather(-1, logits.argmax(-1, keepdim=True)).squeeze(-1)
        else:
            probabilities = F.gumbel_softmax(logits, tau=temperature, dim=-1)
            codes = (values * probabilities).sum(-1)
        scale, _ = self.scales(hard)
        if self.nvfp4:
            codes = codes.reshape(*scale.shape, 16)
            return (codes * scale.unsqueeze(-1)).reshape(self.shape).to(self.dtype)
        return (codes * scale).to(self.dtype)

    @torch.no_grad()
    def install(self, module, weight):
        module.weight.copy_(weight)
        scale, global_scale = self.scales(True)
        q = module.weight_quantizer
        if self.nvfp4:
            q.amax = (scale * 6).reshape_as(q.amax)
            q.global_amax = (global_scale * (6 * 448)).reshape_as(q.global_amax)
        else:
            q.amax = (scale * 448).reshape_as(q.amax)


def _tensor_output(output):
    return output if isinstance(output, torch.Tensor) else output[0]


def _fingerprint(model, settings):
    digest = hashlib.sha256(repr((type(model), settings)).encode())
    if hasattr(getattr(model, "config", None), "to_json_string"):
        digest.update(model.config.to_json_string().encode())
    for name, tensor in model.state_dict().items():
        digest.update(str((name, tensor.shape, tensor.dtype)).encode())
        digest.update(
            tensor.detach().reshape(-1).contiguous().cpu().view(torch.uint8).numpy().tobytes()
        )
    return digest.hexdigest()


def _rng_state():
    return {"cpu": torch.get_rng_state(), "cuda": torch.cuda.get_rng_state_all()}


def _set_rng(state):
    torch.set_rng_state(state["cpu"].cpu())
    if state["cuda"]:
        torch.cuda.set_rng_state_all([s.cpu() for s in state["cuda"]])


def _save(path, state):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp")
    torch.save(state, temporary)
    temporary.replace(path)


def _enabled_weight_modules(model):
    modules = [
        (n, m)
        for n, m in model.named_modules()
        if is_quantized_linear(m) and m.weight_quantizer.is_enabled
    ]
    allowed = {id(m.weight_quantizer) for _, m in modules}
    if any(
        isinstance(m, TensorQuantizer) and m.is_enabled and id(m) not in allowed
        for m in model.modules()
    ):
        raise ValueError(
            "GSQ requires only eligible linear weight quantizers; disable embeddings and activations"
        )
    return modules


@contextmanager
def _gsq_calibration(model, config):
    """Validate whole-model resume identity and recover weights if a later layer fails."""
    enabled = _enabled_weight_modules(model)
    if any(not _REGIONS.intersection(n.split(".")) for n, _ in enabled):
        raise ValueError("GSQ quantized weights must belong to attention/MLP regions")
    if len({m.weight.device for _, m in enabled}) > 1 or any(m.weight.is_meta for _, m in enabled):
        raise ValueError("GSQ requires resident weights on one device")
    if torch.distributed.is_initialized() and torch.distributed.get_world_size() > 1:
        raise ValueError("GSQ supports single-process calibration")
    if config.optimizer_checkpoint_dir:
        settings = config.model_dump(exclude={"resume", "optimizer_checkpoint_dir"})
        settings["layerwise"].pop("checkpoint_dir", None)
        path = Path(config.optimizer_checkpoint_dir) / "manifest.pt"
        fingerprint = _fingerprint(model, settings)
        if config.resume:
            manifest = torch.load(path, weights_only=True) if path.exists() else {}
            if manifest.get("schema") != 1 or manifest.get("fingerprint") != fingerprint:
                raise ValueError("Incompatible GSQ model, data, or optimization settings")
        else:
            _save(path, {"schema": 1, "fingerprint": fingerprint})
    originals = [
        (m.weight, m.weight.detach().cpu().clone())
        for m in model.modules()
        if is_quantized_linear(m) and m.weight_quantizer.is_enabled
    ]
    children = {m: dict(m._modules) for m in model.modules()}
    quantizers = [m for m in model.modules() if isinstance(m, TensorQuantizer)]
    tensors = [t for m in model.modules() for t in m.buffers(recurse=False)]
    old_values = [(t, t.detach().clone()) for t in tensors]
    old_quantizers = deepcopy(quantizers, {id(t): t for t in tensors})
    try:
        yield
    except BaseException:
        with torch.no_grad():
            for parameter, original in [*originals, *old_values]:
                parameter.copy_(original)
        for module, original_children in children.items():
            module._modules = original_children
        for quantizer, original in zip(quantizers, old_quantizers):
            quantizer.__class__ = original.__class__
            quantizer.__dict__.clear()
            quantizer.__dict__.update(original.__dict__)
        raise


def gsq(
    model,
    forward_loop,
    initializer="max",
    steps=100,
    assignment_lr=0.01,
    scale_lr=0.001,
    temperature=1.0,
    final_temperature=0.1,
    eval_every=10,
    seed=0,
    optimizer_checkpoint_dir=None,
    resume=False,
    data_fingerprint="",
):
    """Refine attention/MLP weights with fixed five-value floating-point neighborhoods.

    The normal calibration dispatcher supplies decoder layers and replays their real
    arguments. Final weights and legal scales remain in the ordinary model state.
    """
    # Calibration/export dispatch imports this module; defer the reverse dependencies.
    from modelopt.torch.export.quant_utils import preprocess_linear_fusion

    from .model_calib import gptq, layerwise_calibrate, max_calibrate

    if forward_loop is None:
        raise ValueError("GSQ requires a replayable forward_loop")
    options = {
        "initializer": initializer,
        "steps": steps,
        "assignment_lr": assignment_lr,
        "scale_lr": scale_lr,
        "temperature": temperature,
        "final_temperature": final_temperature,
        "eval_every": eval_every,
        "seed": seed,
        "optimizer_checkpoint_dir": optimizer_checkpoint_dir,
        "resume": resume,
        "data_fingerprint": data_fingerprint,
    }
    linears = _enabled_weight_modules(model)
    if not linears:
        return
    if LayerActivationCollector.is_supported(model):
        return layerwise_calibrate(model, forward_loop, gsq, **options)
    regions = [(n, m) for n, m in model.named_modules() if n.split(".")[-1] in _REGIONS]
    regions = [(n, m) for n, m in regions if not any(n.startswith(p + ".") for p, _ in regions)]
    for name, module in linears:
        q = module.weight_quantizer
        if isinstance(q, SequentialQuantizer):
            raise ValueError("GSQ requires one floating-point weight quantizer")
        fp8 = q.is_fp8
        nvfp4 = (
            q.is_nvfp4_static
            and q.block_sizes[-1] == 16
            and not q.block_sizes.get("four_over_six", False)
        )
        if not (fp8 or nvfp4) or q.pre_quant_scale is not None or q.backend is not None:
            raise ValueError("GSQ supports per-tensor E4M3 and static block-16 NVFP4 weights")
        if not any(name == region or name.startswith(region + ".") for region, _ in regions):
            raise ValueError(f"GSQ requires an attention/MLP reconstruction region for {name}")
    weights = {id(m.weight) for _, m in linears}
    if len(weights) != len(linears):
        raise ValueError("GSQ does not support tied quantized weights")
    if any(
        isinstance(m, TensorQuantizer) and m.is_enabled
        for n, m in model.named_modules()
        if not n.endswith("weight_quantizer")
    ):
        raise ValueError("GSQ requires disabled activation and KV-cache quantizers")
    if initializer not in {"max", "gptq"}:
        raise ValueError("GSQ initializer must be max or gptq")
    identity = getattr(getattr(model, "_layerwise_calib", None), "name", "decoder")
    path = (
        Path(optimizer_checkpoint_dir) / (identity.replace(".", "_") + ".pt")
        if optimizer_checkpoint_dir
        else None
    )
    settings = {k: v for k, v in options.items() if k not in {"resume", "optimizer_checkpoint_dir"}}
    fingerprint = _fingerprint(model, settings) if path else None
    saved = (
        torch.load(path, map_location="cpu", weights_only=True)
        if resume and path and path.exists()
        else None
    )
    if saved and saved["fingerprint"] != fingerprint:
        raise ValueError("Incompatible GSQ optimizer checkpoint")
    original = {n: p.detach().clone() for n, p in model.named_parameters()}
    quantizers = [m for m in model.modules() if isinstance(m, TensorQuantizer)]
    original_quantizers = deepcopy(quantizers)
    flags = {p: p.requires_grad for p in model.parameters()}
    training = {m: m.training for m in model.modules()}
    caller_rng = _rng_state()
    model.gsq = {}
    try:
        model.eval()
        for p in flags:
            p.requires_grad_(False)
        torch.manual_seed(seed)
        # Capture the original targets before any mutating initializer.
        targets = {}
        with set_quantizer_by_cfg_context(model, [{"quantizer_name": "*", "enable": False}]):
            for name, region in regions:
                samples = []

                def capture(m, args, kwargs, output):
                    def detach(x):
                        return x.detach() if isinstance(x, torch.Tensor) else x

                    samples.append(
                        (
                            tree_map(detach, args),
                            tree_map(detach, kwargs),
                            _tensor_output(output).detach(),
                        )
                    )

                with torch.no_grad():
                    handle = region.register_forward_hook(capture, with_kwargs=True)
                    try:
                        forward_loop(model)
                    finally:
                        handle.remove()
                targets[name] = samples
        with torch.no_grad():
            (max_calibrate if initializer == "max" else gptq)(model, forward_loop)
        scale_keys = {}
        for _, members in find_shared_input_groups(model, SHARED_PATTERNS):
            preprocess_linear_fusion(members)
            for module in members:
                scale_keys[id(module)] = id(members[0])
        for index, (name, region) in enumerate(regions):
            selected = [
                (n, m)
                for n, m in region.named_modules()
                if is_quantized_linear(m) and m.weight_quantizer.is_enabled
            ]
            if not selected:
                continue
            if not targets[name]:
                raise ValueError(f"GSQ region {name} was never called")
            shared = {}
            assignments = nn.ModuleDict(
                {
                    str(i): _Assignment(m, shared, scale_keys.get(id(m)))
                    for i, (_, m) in enumerate(selected)
                }
            )
            with (
                torch.enable_grad(),
                set_quantizer_by_cfg_context(region, [{"quantizer_name": "*", "enable": False}]),
            ):
                result, state = _refine(
                    region,
                    selected,
                    assignments,
                    targets[name],
                    options,
                    saved if saved and saved["region"] == index else None,
                    lambda state: (
                        _save(path, {**state, "fingerprint": fingerprint, "region": index})
                        if path
                        else None
                    ),
                )
            for (_, module), weight, assignment in zip(selected, result, assignments.values()):
                assignment.install(module, weight)
            model.gsq[name] = {**settings, **state}
    except BaseException:
        with torch.no_grad():
            for n, p in model.named_parameters():
                p.copy_(original[n])
        # Restore quantizer buffers as well as execution flags after a failed initializer.
        for quantizer, original_quantizer in zip(quantizers, original_quantizers):
            quantizer.__class__ = original_quantizer.__class__
            quantizer.__dict__.clear()
            quantizer.__dict__.update(original_quantizer.__dict__)
        raise
    finally:
        for p, flag in flags.items():
            p.requires_grad_(flag)
        for m, flag in training.items():
            m.training = flag
        _set_rng(caller_rng)


def _refine(region, selected, assignments, samples, options, saved, save):
    scales = [p for n, p in assignments.named_parameters() if not n.endswith("logits")]
    optimizer = torch.optim.Adam(
        [
            {"params": [a.logits for a in assignments.values()], "lr": options["assignment_lr"]},
            {"params": scales, "lr": options["scale_lr"]},
        ]
    )

    def predict(args, kwargs, hard, tau=1.0):
        weights = {
            (n + "." if n else "") + "weight": (
                a(hard=True) if hard else checkpoint(a, tau, False, use_reentrant=False)
            )
            for (n, _), a in zip(selected, assignments.values())
        }
        return _tensor_output(functional_call(region, weights, args, kwargs))

    @torch.no_grad()
    def evaluate():
        losses = [
            F.mse_loss(predict(args, kwargs, True).float(), target.float())
            for args, kwargs, target in samples
        ]
        loss = torch.stack(losses).mean().item()
        if not math.isfinite(loss):
            raise FloatingPointError("Nonfinite GSQ reconstruction loss")
        return loss

    initial_loss = evaluate()
    best_loss, best = (
        initial_loss,
        {k: v.detach().clone() for k, v in assignments.state_dict().items()},
    )
    start = 0
    if saved:
        assignments.load_state_dict(saved["assignments"])
        optimizer.load_state_dict(saved["optimizer"])
        _set_rng(saved["rng"])
        start, best, best_loss, initial_loss = (
            saved["step"],
            saved["best"],
            saved["best_loss"],
            saved["initial_loss"],
        )
    for step in range(start, options["steps"]):
        args, kwargs, target = samples[step % len(samples)]
        tau = options["temperature"] * (options["final_temperature"] / options["temperature"]) ** (
            step / max(options["steps"] - 1, 1)
        )
        optimizer.zero_grad(set_to_none=True)
        loss = F.mse_loss(predict(args, kwargs, False, tau).float(), target.float())
        if not torch.isfinite(loss):
            raise FloatingPointError("Nonfinite GSQ reconstruction loss")
        loss.backward()
        optimizer.step()
        if (step + 1) % options["eval_every"] == 0 or step + 1 == options["steps"]:
            hard_loss = evaluate()
            if hard_loss < best_loss:
                best_loss, best = (
                    hard_loss,
                    {k: v.detach().clone() for k, v in assignments.state_dict().items()},
                )
            save(
                {
                    "assignments": assignments.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "rng": _rng_state(),
                    "step": step + 1,
                    "best": best,
                    "best_loss": best_loss,
                    "initial_loss": initial_loss,
                }
            )
    assignments.load_state_dict(best)
    return [a(hard=True).detach() for a in assignments.values()], {
        "initial_loss": initial_loss,
        "best_loss": best_loss,
        "steps": options["steps"],
    }
