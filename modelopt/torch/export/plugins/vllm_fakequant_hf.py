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
from pathlib import Path

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)

import modelopt.torch.opt as mto
from modelopt.torch.quantization.config import RotateConfig
from modelopt.torch.quantization.conversion import quantizer_state
from modelopt.torch.quantization.nn import QuantModule, TensorQuantizer
from modelopt.torch.quantization.utils import get_quantizer_state_dict
from modelopt.torch.utils import get_unwrapped_name

__all__ = ["export_hf_vllm_fq_checkpoint"]


def disable_rotate(quantizer: TensorQuantizer):
    """Return a disabled copy of the quantizer's ``_rotate`` field, preserving its type."""
    if isinstance(quantizer._rotate, RotateConfig):
        return RotateConfig(enable=False)
    if isinstance(quantizer._rotate, dict):  # backward compat: old checkpoints stored a dict
        return dict(quantizer._rotate, enable=False)
    return False


def _materialize_offloaded_weights(
    model: nn.Module,
    state_dict: dict[str, torch.Tensor],
    meta_keys: list[str],
) -> None:
    """Replace meta tensors in state_dict with actual data from accelerate offload hooks.

    When a model is loaded with ``device_map="auto"`` and some layers are offloaded
    to CPU or disk, ``model.state_dict()`` returns meta tensors (no data) for those
    layers. This function walks the model's accelerate hooks to retrieve the actual
    weight data and updates state_dict in-place.
    """
    hook_map: dict[str, tuple] = {}
    for name, module in model.named_modules():
        hook = getattr(module, "_hf_hook", None)
        if hook is None:
            continue
        hooks = [hook]
        if hasattr(hook, "hooks"):
            hooks = hook.hooks
        for h in hooks:
            if hasattr(h, "weights_map") and h.weights_map is not None:
                prefix = f"{name}." if name else ""
                hook_map[prefix] = (module, h)
                break

    materialized = 0
    for key in meta_keys:
        for prefix, (module, hook) in hook_map.items():
            if not key.startswith(prefix):
                continue
            local_key = key[len(prefix):]
            wmap = hook.weights_map
            if hasattr(wmap, "dataset"):
                lookup_key = wmap.prefix + local_key
                actual_sd = wmap.dataset.state_dict
            else:
                lookup_key = local_key
                actual_sd = wmap
            if lookup_key in actual_sd:
                state_dict[key] = actual_sd[lookup_key].detach().clone()
                materialized += 1
                break
        else:
            logger.warning("Could not materialize meta tensor for key: %s", key)

    logger.info("Materialized %d/%d offloaded weights to CPU", materialized, len(meta_keys))


def _save_clean_checkpoint(
    model: nn.Module,
    clean_sd: dict[str, torch.Tensor],
    export_dir: Path,
) -> None:
    """Save clean weights + config directly, bypassing model.save_pretrained().

    For accelerate-offloaded models, ``save_pretrained(state_dict=clean_sd)``
    ignores the provided state_dict and saves from internal state, leaking
    quantizer keys. This function saves ``clean_sd`` directly via safetensors
    API, guaranteeing only the intended keys are written.
    """
    import json

    from huggingface_hub import split_torch_state_dict_into_shards
    from safetensors.torch import save_file

    cpu_sd = {k: v.cpu() if v.device.type != "cpu" else v for k, v in clean_sd.items()}

    state_dict_split = split_torch_state_dict_into_shards(cpu_sd, max_shard_size="5GB")
    for shard_file, tensor_keys in state_dict_split.filename_to_tensors.items():
        shard = {k: cpu_sd[k] for k in tensor_keys}
        save_file(shard, str(export_dir / shard_file))
        logger.info("Saved shard: %s (%d tensors)", shard_file, len(shard))

    if state_dict_split.is_sharded:
        index = {
            "metadata": state_dict_split.metadata,
            "weight_map": state_dict_split.tensor_to_filename,
        }
        (export_dir / "model.safetensors.index.json").write_text(
            json.dumps(index, indent=2)
        )

    if hasattr(model, "config"):
        model.config.save_pretrained(export_dir)
        config_path = export_dir / "config.json"
        if config_path.exists():
            config = json.loads(config_path.read_text())
            if config.pop("auto_map", None):
                config_path.write_text(json.dumps(config, indent=2))
                logger.info("Saved config.json (auto_map stripped)")

    logger.info(
        "Checkpoint saved: %d weights in %d shard(s)",
        len(cpu_sd),
        len(state_dict_split.filename_to_tensors),
    )


def export_hf_vllm_fq_checkpoint(
    model: nn.Module,
    export_dir: Path | str,
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
    """
    export_dir = Path(export_dir)
    export_dir.mkdir(parents=True, exist_ok=True)

    # Step 1: Build the folded HF state dict.
    # model.state_dict() returns detached copies of all tensors, so model
    # parameters are never modified. Apply each weight quantizer's fake-quant
    # to the corresponding weight tensor in the copy.
    state_dict = model.state_dict()

    # Handle accelerate-offloaded models: state_dict() returns meta tensors
    # for CPU/disk-offloaded layers. Materialize them from the offload hooks.
    meta_keys = [k for k, v in state_dict.items() if v.is_meta]
    if meta_keys:
        logger.info(
            "Found %d meta tensors in state_dict (accelerate offloading). "
            "Materializing from offload hooks...",
            len(meta_keys),
        )
        _materialize_offloaded_weights(model, state_dict, meta_keys)

    fakequant_weights = set()
    input_quantizers_folded_pqs = (
        set()
    )  # keys for input_quantizers where pre_quant_scale was folded
    with torch.inference_mode():
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
                assert sd_key not in fakequant_weights, (
                    f"Weight {sd_key} has already been fakequantized"
                )
                if sd_key in state_dict:
                    w = state_dict[sd_key]
                    # Quantizer kernels (e.g., fp4_fake_quant_block) require CUDA.
                    # Offloaded weights materialized to CPU need a GPU hop.
                    if not w.is_cuda:
                        qtensors = list(quantizer.parameters()) or list(quantizer.buffers())
                        if qtensors and qtensors[0].is_cuda:
                            w = w.to(qtensors[0].device)
                    w_quant = quantizer(w.float()).to(w.dtype).cpu()
                    # Fold pre_quant_scale: (x*s)@fake_quant(W) = x@(fake_quant(W)*s)
                    # Only valid when input_quantizer does NOT fake-quant activations. If it does
                    # fake_quant(x*s), the non-linearity prevents folding s into W.
                    inp_attr = attr_name.replace("weight_quantizer", "input_quantizer")
                    if hasattr(module, inp_attr):
                        inp_q = getattr(module, inp_attr)
                        if (
                            hasattr(inp_q, "_pre_quant_scale")
                            and inp_q._pre_quant_scale is not None
                            and inp_q._disabled
                        ):
                            scale = inp_q._pre_quant_scale.squeeze().to(device=w_quant.device)
                            w_quant = (w_quant * scale[None, :]).to(w_quant.dtype)
                            inp_q_key = get_unwrapped_name(
                                f"{module_name}.{inp_attr}" if module_name else inp_attr, model
                            )
                            input_quantizers_folded_pqs.add(inp_q_key)
                    state_dict[sd_key] = w_quant
                    fakequant_weights.add(sd_key)

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

    # Step 3: Save HF weights directly from clean_sd.
    # Bypass model.save_pretrained() because accelerate-offloaded models
    # ignore the state_dict= argument, leaking quantizer keys into safetensors.
    _save_clean_checkpoint(model, clean_sd, export_dir)

    for wq, orig_rotate in wqs_to_restore:
        wq.enable()
        wq._rotate = orig_rotate
