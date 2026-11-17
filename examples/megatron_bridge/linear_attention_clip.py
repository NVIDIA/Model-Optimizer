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

"""Calibrate and save experimental NVFP4 state caps through the normal QAD loop."""

import json
import os
from pathlib import Path

import torch
from megatron.core import parallel_state

from modelopt.torch.quantization.linear_attention.state_clip import (
    NVFP4StateClip,
    ReferenceK128StateClip,
)


def attach_state_clips(model, initial_path=None, calibrating=False):
    initial = json.loads(Path(initial_path).read_text()) if initial_path else None
    tp_rank = parallel_state.get_tensor_model_parallel_rank()
    tp_size = parallel_state.get_tensor_model_parallel_world_size()
    attached = []
    for layer in model.modules():
        quantizer = getattr(layer, "gdn_state_quantizer", None)
        if quantizer is None or not quantizer.is_enabled:
            continue
        if not quantizer.is_nvfp4_dynamic or layer.linear_attention_config.precision != "vllm":
            raise ValueError("State clipping requires the native GDN NVFP4 V16 recipe")
        reference = layer.linear_attention_config.experimental_state_format
        clip_type = ReferenceK128StateClip if reference else NVFP4StateClip
        if initial is not None and initial["format"] != clip_type.format:
            raise ValueError("Incompatible state-clip calibration policy")
        if reference and (initial is None or calibrating):
            raise ValueError("Reference K128 requires the peer's calibrated caps")
        index = str(layer.layer_number - 1)
        heads = layer.A_log.numel()
        if initial is None:
            values = torch.ones(heads, device=layer.A_log.device)
        else:
            values = torch.tensor(initial["layers"][index], device=layer.A_log.device)
            if values.numel() != heads * tp_size:
                raise ValueError("Calibration head count differs from model")
            values = values[tp_rank * heads : (tp_rank + 1) * heads]
        layer.state_clip = (
            ReferenceK128StateClip(values)
            if reference
            else NVFP4StateClip(values, calibrating=calibrating)
        )
        # Bridge destroys process groups before the training context exits.
        layer.state_clip._export_rank = (
            tp_rank,
            torch.distributed.get_rank(),
            torch.distributed.get_world_size(),
        )
        # Keep a single registered owner for optimizer/checkpoint traversal.
        object.__setattr__(quantizer, "_state_clip", layer.state_clip)
        attached.append((index, layer.state_clip))
    if not attached:
        raise ValueError("No enabled GDN state quantizers found for clipping")
    return attached


def save_state_clips(attached, output, *, calibrated):
    """Write rank-local caps after Bridge teardown; merge outside the distributed workers."""
    tp_rank, rank, world_size = attached[0][1]._export_rank
    local = {}
    for index, clip in attached:
        values = clip.observed_amax if calibrated else clip.caps().detach()
        if not torch.isfinite(values).all() or (values <= 0).any():
            raise ValueError("Nonfinite/unobserved state caps; refuse to export")
        local[index] = values.float().cpu().tolist()
    job = os.environ.get("SLURM_JOB_ID", "local")
    payload = {
        "format": attached[0][1].format,
        "calibrated": calibrated,
        "learning_rate_multiplier": 1000.0,
        "softmin_power": 32,
        "parameterization": "bounded_exp_bf16_normalized_grad"
        if isinstance(attached[0][1], ReferenceK128StateClip)
        else "exp_fp32",
        "layers": local,
        "tp_rank": tp_rank,
        "rank": rank,
        "world_size": world_size,
        "job": job,
    }
    destination = Path(output)
    destination = destination.with_name(f"{destination.stem}.{job}.rank{rank:05d}.json")
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(".tmp")
    temporary.write_text(json.dumps(payload, indent=2) + "\n")
    temporary.replace(destination)
