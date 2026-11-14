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

"""Calibration inputs and shape profiles for exported decoder-only and hybrid LLMs."""

import json
import os
from pathlib import Path

import numpy as np
import onnx
import torch
from huggingface_hub import hf_hub_download
from huggingface_hub.errors import EntryNotFoundError
from safetensors import safe_open

__all__ = [
    "calibration_feed",
    "embed_table",
    "ep_profiles",
    "needs_signature_inputs",
    "read_signature",
    "save_quantized_model",
    "text_config_of",
]

_DTYPES = {
    "tensor(float16)": np.float16,
    "tensor(float)": np.float32,
    "tensor(double)": np.float64,
    "tensor(int64)": np.int64,
    "tensor(int32)": np.int32,
    "tensor(bool)": np.bool_,
}
_BATCH, _SEQ, _PAST, _TOTAL = "batch", "seq", "past", "total"
_ONNX_ELEM = {
    "FLOAT16": "tensor(float16)",
    "FLOAT": "tensor(float)",
    "DOUBLE": "tensor(double)",
    "INT64": "tensor(int64)",
    "INT32": "tensor(int32)",
    "BOOL": "tensor(bool)",
}


def text_config_of(config):
    """Return decoder configuration without changing ordinary decoder-only configs."""
    text_config = getattr(config, "text_config", None)
    return config if text_config is None else text_config


def save_quantized_model(model, output_path):
    """Export weights externally while retaining small shape-inference constants inline."""
    onnx.save_model(
        model,
        output_path,
        save_as_external_data=True,
        location=os.path.basename(output_path) + "_data",
        size_threshold=1024,
    )


def np_dtype(ort_type):
    try:
        return _DTYPES[ort_type]
    except KeyError:
        raise ValueError(f"unsupported input element type {ort_type!r}") from None


def _role(dim):
    """Classify the conventional batch/sequence/cache dimension names used by exports."""
    n = str(dim).lower()
    if "batch" in n:
        return _BATCH
    if "past" in n:
        return _PAST
    if "total" in n:
        return _TOTAL
    if "seq" in n:
        return _SEQ
    return None


def resolve_shape(shape, batch, seq, past):
    """Resolve named dynamic dimensions, rejecting unknown names instead of guessing."""
    sizes = {_BATCH: batch, _SEQ: seq, _PAST: past, _TOTAL: past + seq}
    out = []
    for dim in shape:
        if isinstance(dim, int):
            out.append(dim)
            continue
        role = _role(dim)
        if role is None:
            raise ValueError(
                f"cannot size the dimension {dim!r} of {tuple(shape)}: not a batch / sequence / past / total dimension"
            )
        out.append(sizes[role])
    return tuple(out)


def kind_of(name):
    """'cache' for past_key_values.*, else the input's own name."""
    return "cache" if name.startswith("past_key_values.") else name


def has_embeds_input(inputs):
    names = {i[0] for i in inputs}
    return "inputs_embeds" in names and not names & {"input_ids", "token_ids", "tokens"}


def shapes_profile(inputs, batch, seq, past):
    """Build an EP profile string naming every declared model input."""
    parts = []
    for name, shape, _dtype in inputs:
        if None in shape:
            raise ValueError(f"Unspecified dimension in ONNX input {name!r}")
        parts.append(f"{name}:{'x'.join(str(d) for d in resolve_shape(shape, batch, seq, past))}")
    return ",".join(parts)


def read_signature(model_path):
    """(name, shape, type) of the real inputs of an ONNX file (initializers excluded); external data is not loaded."""
    model = onnx.load(model_path, load_external_data=False)
    initializers = {i.name for i in model.graph.initializer}
    out = []
    for i in model.graph.input:
        if i.name in initializers:
            continue
        dims = [
            d.dim_param
            if d.HasField("dim_param")
            else d.dim_value
            if d.HasField("dim_value")
            else None
            for d in i.type.tensor_type.shape.dim
        ]
        elem = onnx.TensorProto.DataType.Name(i.type.tensor_type.elem_type)
        out.append((i.name, dims, _ONNX_ELEM.get(elem, f"tensor({elem.lower()})")))
    return out


def needs_signature_inputs(inputs):
    """Identify embedding-only inputs or cache states beyond standard attention key/value."""
    return has_embeds_input(inputs) or any(
        kind_of(n) == "cache" and not n.endswith((".key", ".value")) for n, _s, _t in inputs
    )


def ep_profiles(inputs, calibration_eps, batch_size=1):
    """Return min/opt/max options in calibration EP order, using the ONNX signature."""
    shapes = {
        "min": shapes_profile(inputs, 1, 1, 0),
        "opt": shapes_profile(inputs, batch_size, 512, 512),
        "max": shapes_profile(inputs, batch_size, 1024, 1024),
    }
    profiles = []
    for ep in calibration_eps:
        prefix = "nv" if "NvTensorRtRtx" in ep else "trt" if ep == "trt" else None
        profiles.append(
            {f"{prefix}_profile_{k}_shapes": v for k, v in shapes.items()} if prefix else {}
        )
    return profiles


def calibration_feed(inputs, input_ids, attention_mask, table=None, zero_cache=None):
    """Build a prefill calibration feed from the declared inputs.

    Attention and linear-attention caches start at zero. Embedding-only models
    require the checkpoint's token embedding table; unknown inputs are rejected.
    A caller-owned zero_cache shares read-only state buffers across calibration samples.
    """
    ids = np.asarray(input_ids)
    mask = np.asarray(attention_mask)
    batch, seq = ids.shape
    feed = {}
    if zero_cache is None:
        zero_cache = {}
    for name, shape, dtype in inputs:
        if None in shape:
            raise ValueError(f"Unspecified dimension in ONNX input {name!r}")
        if kind_of(name) == "cache":
            key = (resolve_shape(shape, batch, seq=0, past=0), np_dtype(dtype))
            if key not in zero_cache:
                zero_cache[key] = np.zeros(key[0], dtype=key[1])
            feed[name] = zero_cache[key]
        elif name == "input_ids":
            feed[name] = ids.astype(np_dtype(dtype))
        elif name == "inputs_embeds":
            if table is None:
                raise ValueError("the model declares inputs_embeds: pass the token-embedding table")
            with np.errstate(over="ignore", invalid="ignore"):
                values = table[torch.as_tensor(ids)].float().numpy().astype(np_dtype(dtype))
            if not np.isfinite(values).all():
                raise ValueError("Embedding values must be finite in the model's input dtype")
            feed[name] = values
        elif name == "attention_mask":
            feed[name] = np.broadcast_to(mask, resolve_shape(shape, batch, seq, 0)).astype(
                np_dtype(dtype)
            )
        elif name == "position_ids":
            pos = mask.astype(np.int64).cumsum(-1) - 1
            pos[mask == 0] = 1
            shp = resolve_shape(shape, batch, seq, 0)
            feed[name] = np.broadcast_to(pos, shp).astype(np_dtype(dtype)).copy()
        else:
            raise ValueError(f"no calibration value for the model input {name!r}")
    return feed


def embed_table(model_name, cache_dir=None):
    """Memory-map token embeddings from a local checkpoint or the Hugging Face cache."""

    def fetch(filename):
        # Shard names originate in the checkpoint index, not in trusted source code.
        if Path(filename).name != filename or "\\" in filename or filename in {".", ".."}:
            raise ValueError("Expected a checkpoint filename without directory components")
        if Path(model_name).is_dir():
            path = Path(model_name) / filename
            if not path.is_file():
                raise FileNotFoundError(
                    f"Missing checkpoint file {path}. Embedding-only models need the Hub ID "
                    "or the original safetensors checkpoint directory, not the ONNX export directory."
                )
            return path
        return Path(hf_hub_download(model_name, filename, cache_dir=cache_dir))

    try:
        index = json.loads(fetch("model.safetensors.index.json").read_text())
    except (FileNotFoundError, EntryNotFoundError):
        shards = {"model.safetensors"}
    else:
        shards = {s for n, s in index["weight_map"].items() if n.endswith("embed_tokens.weight")}
    if len(shards) != 1:
        raise ValueError(f"Expected one embedding shard, found {sorted(shards)}")
    with safe_open(fetch(shards.pop()), framework="pt", device="cpu") as checkpoint:
        names = checkpoint.keys()
        names = [n for n in names if n.endswith("embed_tokens.weight")]
        if len(names) != 1:
            raise ValueError(f"Expected one token embedding tensor, found {names}")
        table = checkpoint.get_tensor(names[0])
    if table.ndim != 2 or any(dim <= 0 for dim in table.shape):
        raise ValueError("Expected a two-dimensional embedding tensor")
    if table.dtype not in (torch.float16, torch.float32, torch.bfloat16):
        raise ValueError(f"Unsupported embedding dtype: {table.dtype}")
    return table
