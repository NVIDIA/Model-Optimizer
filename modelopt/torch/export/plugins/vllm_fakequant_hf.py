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
"""Export HuggingFace model to vLLM fakequant checkpoint."""

import logging
import time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path

import torch
import torch.nn as nn

import modelopt.torch.opt as mto
from modelopt.torch.quantization.config import RotateConfig
from modelopt.torch.quantization.conversion import quantizer_state
from modelopt.torch.quantization.nn import QuantModule, TensorQuantizer
from modelopt.torch.quantization.utils import get_quantizer_state_dict
from modelopt.torch.utils import get_unwrapped_name

__all__ = ["export_hf_vllm_fq_checkpoint"]

logger = logging.getLogger(__name__)


@dataclass
class _WeightQuantWork:
    """A single weight tensor to be fake-quantized during export."""

    sd_key: str
    quantizer: TensorQuantizer
    weight: torch.Tensor
    # For optional pre_quant_scale folding:
    inp_q: TensorQuantizer | None
    inp_q_key: str | None


def _collect_quant_work(
    model: nn.Module, state_dict: dict[str, torch.Tensor]
) -> list[_WeightQuantWork]:
    """Collect all weight quantization work items from the model."""
    work_items = []
    seen_keys: set[str] = set()
    for module_name, module in model.named_modules():
        if not isinstance(module, QuantModule):
            continue
        for attr_name, quantizer in module.named_children():
            if not (
                attr_name.endswith("weight_quantizer")
                and isinstance(quantizer, TensorQuantizer)
                and quantizer.fake_quant
                and quantizer.is_enabled
            ):
                continue
            weight_name = attr_name.removesuffix("_quantizer")
            prefix = f"{module_name}." if module_name else ""
            sd_key = f"{prefix}{weight_name}"
            assert sd_key not in seen_keys, f"Weight {sd_key} has already been fakequantized"
            seen_keys.add(sd_key)
            if sd_key not in state_dict:
                continue
            # Check for pre_quant_scale folding eligibility.
            inp_q = None
            inp_q_key = None
            inp_attr = attr_name.replace("weight_quantizer", "input_quantizer")
            if hasattr(module, inp_attr):
                candidate = getattr(module, inp_attr)
                if (
                    hasattr(candidate, "_pre_quant_scale")
                    and candidate._pre_quant_scale is not None
                    and candidate._disabled
                    and getattr(candidate, "_enable_pre_quant_scale", True)
                ):
                    inp_q = candidate
                    inp_q_key = get_unwrapped_name(
                        f"{module_name}.{inp_attr}" if module_name else inp_attr, model
                    )
            work_items.append(
                _WeightQuantWork(
                    sd_key=sd_key,
                    quantizer=quantizer,
                    weight=state_dict[sd_key],
                    inp_q=inp_q,
                    inp_q_key=inp_q_key,
                )
            )
    return work_items


def _process_weight(item: _WeightQuantWork) -> tuple[str, torch.Tensor, str | None]:
    """Fake-quantize a single weight tensor and optionally fold pre_quant_scale.

    Returns (sd_key, quantized_weight_on_cpu, inp_q_key_or_None).
    """
    w = item.weight
    w_quant = item.quantizer(w.float()).to(w.dtype)
    if item.inp_q is not None:
        scale = item.inp_q._pre_quant_scale.squeeze().to(device=w_quant.device)
        w_quant = (w_quant * scale[None, :]).to(w_quant.dtype)
    return item.sd_key, w_quant.cpu(), item.inp_q_key


def _process_device_batch(items: list[_WeightQuantWork], device: torch.device):
    """Process all weight items on a single GPU. Runs in a dedicated thread."""
    with torch.inference_mode(), torch.cuda.device(device):
        results = [_process_weight(item) for item in items]
        torch.cuda.synchronize(device)
    return results


def disable_rotate(quantizer: TensorQuantizer):
    """Return a disabled copy of the quantizer's ``_rotate`` field, preserving its type."""
    if isinstance(quantizer._rotate, RotateConfig):
        return RotateConfig(enable=False)
    if isinstance(quantizer._rotate, dict):  # backward compat: old checkpoints stored a dict
        return dict(quantizer._rotate, enable=False)
    return False


def export_hf_vllm_fq_checkpoint(
    model: nn.Module,
    export_dir: Path | str,
    parallel: bool = True,
):
    """Export quantized HF weights + ``vllm_fq_modelopt_state.pth`` for vLLM fake-quant reload.

    Folds fake-quant weights into a ``state_dict()`` copy (optional
    ``pre_quant_scale`` into weight when input fake-quant is off), drops quantizer
    keys from the HF save, briefly disables weight quantizers to snapshot
    ModelOpt/quantizer state, then re-enables them. Writes ``export_dir`` via
    ``save_pretrained(..., save_modelopt_state=False)``.

    Args:
        model: In-memory quantized model.
        export_dir: Output dir for HF files and ``vllm_fq_modelopt_state.pth``.
        parallel: If True, fake-quantize weights across GPUs concurrently using
            one thread per GPU device. Falls back to sequential when all weights
            are on the same device or on CPU. Default True.
    """
    export_dir = Path(export_dir)
    export_dir.mkdir(parents=True, exist_ok=True)

    # Step 1: Build the folded HF state dict.
    # model.state_dict() returns detached copies of all tensors, so model
    # parameters are never modified. Apply each weight quantizer's fake-quant
    # to the corresponding weight tensor in the copy.
    state_dict = model.state_dict()
    fakequant_weights: set[str] = set()
    input_quantizers_folded_pqs: set[str] = set()

    work_items = _collect_quant_work(model, state_dict)

    # Group work items by device for parallel dispatch.
    device_groups: dict[torch.device, list[_WeightQuantWork]] = defaultdict(list)
    for item in work_items:
        device_groups[item.weight.device].append(item)

    num_cuda_devices = sum(1 for d in device_groups if d.type == "cuda")
    use_parallel = parallel and num_cuda_devices > 1

    t0 = time.monotonic()
    with torch.inference_mode():
        if use_parallel:
            logger.info(
                "Parallel export: %d weights across %d GPUs (%s)",
                len(work_items),
                num_cuda_devices,
                ", ".join(f"{d}: {len(items)} weights" for d, items in device_groups.items()),
            )
            with ThreadPoolExecutor(max_workers=num_cuda_devices) as pool:
                # Submit GPU batches first (non-blocking)
                futures = [
                    pool.submit(_process_device_batch, items, device)
                    for device, items in device_groups.items()
                    if device.type == "cuda"
                ]
                # Process CPU weights inline while GPU futures run
                for device, items in device_groups.items():
                    if device.type != "cuda":
                        for sd_key, w_quant, inp_q_key in map(_process_weight, items):
                            state_dict[sd_key] = w_quant
                            fakequant_weights.add(sd_key)
                            if inp_q_key is not None:
                                input_quantizers_folded_pqs.add(inp_q_key)
                # Collect GPU results
                for future in futures:
                    for sd_key, w_quant, inp_q_key in future.result():
                        state_dict[sd_key] = w_quant
                        fakequant_weights.add(sd_key)
                        if inp_q_key is not None:
                            input_quantizers_folded_pqs.add(inp_q_key)
        else:
            # Sequential fallback (single GPU, CPU, or parallel=False).
            for item in work_items:
                sd_key, w_quant, inp_q_key = _process_weight(item)
                state_dict[sd_key] = w_quant
                fakequant_weights.add(sd_key)
                if inp_q_key is not None:
                    input_quantizers_folded_pqs.add(inp_q_key)

    elapsed = time.monotonic() - t0
    logger.info(
        "Export step 1 (%s): %d weights fake-quantized in %.1fs",
        "parallel" if use_parallel else "sequential",
        len(fakequant_weights),
        elapsed,
    )

    # Filter quantizer tensors out for a clean HF checkpoint.
    clean_sd = {k: v for k, v in state_dict.items() if "quantizer" not in k}

    # Step 2: Disable weight quantizers, save modelopt state + quantizer state
    # dict, then re-enable. The _disabled=True flag is captured in modelopt_state
    # so that on vLLM reload weight quantizers stay off while input/output/
    # attention quantizers remain active.
    # Rotation is also cleared: the weight was already folded with rotation applied,
    # so if fold_weight is called on reload it must not re-rotate the exported weight.
    wqs_to_restore = []
    for _, module in model.named_modules():
        if isinstance(module, QuantModule):
            for attr_name, quantizer in module.named_children():
                if (
                    attr_name.endswith("weight_quantizer")
                    and isinstance(quantizer, TensorQuantizer)
                    and quantizer.is_enabled
                ):
                    quantizer.disable()
                    orig_rotate = quantizer._rotate
                    if quantizer.rotate_is_enabled:
                        quantizer._rotate = disable_rotate(quantizer)
                    wqs_to_restore.append((quantizer, orig_rotate))

    quantizer_state_dict = get_quantizer_state_dict(model)
    for key in list(quantizer_state_dict):
        if key.endswith("weight_quantizer"):
            # Fakequant amax is folded into HF weights; do not reload weight quantizer tensors.
            quantizer_state_dict.pop(key)
        elif key in input_quantizers_folded_pqs:
            # pre_quant_scale was folded into the weight; keep the buffer for strict load but
            # save identity so activations are not scaled twice.
            qstate_val = quantizer_state_dict[key]
            if isinstance(qstate_val, dict) and "_pre_quant_scale" in qstate_val:
                quantizer_state_dict[key]["_pre_quant_scale"] = torch.ones_like(
                    qstate_val["_pre_quant_scale"]
                )
    modelopt_state = mto.modelopt_state(model)
    # ``modelopt_state`` may be stale if another mode (e.g. calibrate) ran last. Rebuild
    # ``quantizer_state`` and drop disabled weight quantizer entries (weights already folded).
    qstate = quantizer_state(model)
    for key in list(qstate):
        if key.endswith("weight_quantizer") and qstate[key].get("_disabled"):
            qstate.pop(key)

    for mode_str, m_state in modelopt_state.get("modelopt_state_dict", []):
        if mode_str == "quantize" and "metadata" in m_state:
            m_state["metadata"]["quantizer_state"] = qstate
            break

    # Per-quantizer tensor dict loaded alongside metadata on reload.
    modelopt_state["modelopt_state_weights"] = quantizer_state_dict
    torch.save(modelopt_state, export_dir / "vllm_fq_modelopt_state.pth")

    # Step 3: Save HF weights using the pre-built folded state dict.
    model.save_pretrained(export_dir, state_dict=clean_sd, save_modelopt_state=False)

    for wq, orig_rotate in wqs_to_restore:
        wq.enable()
        if orig_rotate is not None:
            wq._rotate = orig_rotate
