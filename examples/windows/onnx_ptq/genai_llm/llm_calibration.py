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

import hashlib
import json
import math
import os
import re
import struct
import tempfile
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

import numpy as np
import onnx
from huggingface_hub import HfApi, get_token, hf_hub_url

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


def calibration_feed(inputs, input_ids, attention_mask, table=None):
    """Build a prefill calibration feed from the declared inputs.

    Attention and linear-attention caches start at zero. Embedding-only models
    require the checkpoint's token embedding table; unknown inputs are rejected.
    """
    ids = np.asarray(input_ids)
    mask = np.asarray(attention_mask)
    batch, seq = ids.shape
    feed = {}
    for name, shape, dtype in inputs:
        if None in shape:
            raise ValueError(f"Unspecified dimension in ONNX input {name!r}")
        if kind_of(name) == "cache":
            feed[name] = np.zeros(resolve_shape(shape, batch, seq=0, past=0), dtype=np_dtype(dtype))
        elif name == "input_ids":
            feed[name] = ids.astype(np_dtype(dtype))
        elif name == "inputs_embeds":
            if table is None:
                raise ValueError("the model declares inputs_embeds: pass the token-embedding table")
            feed[name] = table[ids].astype(np_dtype(dtype))
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


class _HubRedirectHandler(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        before, after = urllib.parse.urlsplit(req.full_url), urllib.parse.urlsplit(newurl)
        if after.scheme != "https":
            raise ValueError("Checkpoint redirects must use HTTPS")
        redirected = super().redirect_request(req, fp, code, msg, headers, newurl)
        if redirected is not None and (before.scheme, before.netloc) != (
            after.scheme,
            after.netloc,
        ):
            redirected.remove_header("Authorization")
        return redirected


def _read_range(opener, url, first, size, token):
    request = urllib.request.Request(url, headers={"Range": f"bytes={first}-{first + size - 1}"})
    if token:
        request.add_header("Authorization", f"Bearer {token}")
    with opener.open(request, timeout=120) as response:
        expected = f"bytes {first}-{first + size - 1}/"
        if response.status != 206 or not response.headers.get("Content-Range", "").startswith(
            expected
        ):
            raise ValueError("Checkpoint server did not honor the requested byte range")
        data = response.read(size + 1)
    if len(data) != size:
        raise ValueError("Incomplete checkpoint byte range")
    return data


def _cached_embedding(path, shape, chunk_size):
    try:
        table = np.load(path, mmap_mode="r", allow_pickle=False)
        if (
            not isinstance(table, np.memmap)
            or table.shape != tuple(shape)
            or table.dtype != np.float16
        ):
            return None
        # Scan through bounded reads without faulting every mapped cache page into process RSS.
        with Path(path).open("rb") as stream:
            stream.seek(table.offset)
            remaining = table.size
            while remaining:
                count = min(remaining, chunk_size // table.dtype.itemsize)
                values = np.fromfile(stream, dtype=table.dtype, count=count)
                if values.size != count or not np.isfinite(values).all():
                    return None
                remaining -= count
        return table
    except (OSError, ValueError, EOFError):
        return None


def embed_table(model_name, cache_file, max_output_bytes=8 * 1024**3):
    """Load only the checkpoint's embedding tensor, caching float16 rows on disk.

    Local cache identities hash the embedding bytes; Hub identities pin an immutable commit.
    Hub weights are read with bounded byte ranges, never as an entire model shard.
    The float16 cache is limited to 8 GiB by default, configurable via max_output_bytes.
    """
    local = Path(model_name).is_dir()
    opener = urllib.request.build_opener(_HubRedirectHandler)
    token = get_token()
    revision = None
    if not local:
        api = HfApi()
        if urllib.parse.urlsplit(api.endpoint).scheme != "https":
            raise ValueError("Checkpoint downloads must use HTTPS")
        revision = api.model_info(model_name, expand=["sha"], timeout=60, token=token).sha
        if not isinstance(revision, str) or not re.fullmatch(r"[0-9a-f]{40}", revision):
            raise ValueError("Expected an immutable Hub commit")

    def read(filename, first, size):
        # Shard names originate in the checkpoint index, not in trusted source code.
        if Path(filename).name != filename or "\\" in filename or filename in {".", ".."}:
            raise ValueError("Expected a checkpoint filename without directory components")
        if local:
            with (Path(model_name) / filename).open("rb") as stream:
                stream.seek(first)
                data = stream.read(size)
            if len(data) != size:
                raise ValueError("Incomplete local checkpoint tensor")
            return data
        url = hf_hub_url(model_name, filename, revision=revision)
        if urllib.parse.urlsplit(url).scheme != "https":
            raise ValueError("Checkpoint downloads must use HTTPS")
        return _read_range(opener, url, first, size, token)

    try:
        filename = "model.safetensors.index.json"
        if local:
            with (Path(model_name) / filename).open() as stream:
                index = json.load(stream)
        else:
            request = urllib.request.Request(hf_hub_url(model_name, filename, revision=revision))
            if urllib.parse.urlsplit(request.full_url).scheme != "https":
                raise ValueError("Checkpoint downloads must use HTTPS")
            if token:
                request.add_header("Authorization", f"Bearer {token}")
            with opener.open(request, timeout=60) as response:
                index = json.loads(response.read(16 * 1024 * 1024))
        names = [name for name in index["weight_map"] if name.endswith("embed_tokens.weight")]
        if len(names) != 1:
            raise ValueError(f"Expected one token embedding tensor, found {names}")
        name = names[0]
        shard = index["weight_map"][name]
    except (FileNotFoundError, urllib.error.HTTPError) as error:
        if isinstance(error, urllib.error.HTTPError) and error.code != 404:
            raise
        name, shard = None, "model.safetensors"

    header_size = struct.unpack("<Q", read(shard, 0, 8))[0]
    if not 0 < header_size <= 100 * 1024 * 1024:
        raise ValueError("Invalid safetensors header size")
    header_data = read(shard, 8, header_size)
    header = json.loads(header_data)
    if name is None:
        names = [key for key in header if key.endswith("embed_tokens.weight")]
        if len(names) != 1:
            raise ValueError(f"Expected one token embedding tensor, found {names}")
        name = names[0]
    tensor = header[name]
    shape, dtype = tensor["shape"], tensor["dtype"]
    if len(shape) != 2 or any(type(dim) is not int or dim <= 0 for dim in shape):
        raise ValueError("Expected a two-dimensional embedding tensor")
    source_dtype = {"F16": "<f2", "F32": "<f4", "BF16": "<u2"}.get(dtype)
    if source_dtype is None:
        raise ValueError(f"Unsupported embedding dtype: {dtype}")
    itemsize = np.dtype(source_dtype).itemsize
    first, last = tensor["data_offsets"]
    if (
        not isinstance(first, int)
        or not isinstance(last, int)
        or first < 0
        or last - first != math.prod(shape) * itemsize
    ):
        raise ValueError("Invalid embedding tensor byte offsets")
    if math.prod(shape) * np.dtype(np.float16).itemsize > max_output_bytes:
        raise ValueError("Embedding tensor exceeds the configured maximum cache size")

    identity = {
        "model": str(Path(model_name).resolve()) if local else model_name,
        "revision": revision,
        "name": name,
        "tensor": tensor,
    }
    fingerprint = hashlib.sha256(json.dumps(identity, sort_keys=True).encode())
    chunk_size = 8 * 1024 * 1024
    if local:
        # Content, rather than timestamps, detects checkpoints copied or rewritten in place.
        for offset in range(first, last, chunk_size):
            fingerprint.update(
                read(shard, 8 + header_size + offset, min(chunk_size, last - offset))
            )
    digest = fingerprint.hexdigest()
    cache_path = Path(cache_file)
    cache_path = cache_path.with_name(f"{cache_path.stem}_{digest}{cache_path.suffix}")
    if cache_path.is_file():
        cached = _cached_embedding(cache_path, shape, chunk_size)
        if cached is not None:
            return cached
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    # Disk-backed output avoids full-size byte buffers and conversion copies for large vocabularies.
    with tempfile.NamedTemporaryFile(dir=cache_path.parent, suffix=".npy", delete=False) as stream:
        temporary = Path(stream.name)
    mapped = flat = None
    try:
        mapped = np.lib.format.open_memmap(
            temporary, mode="w+", dtype=np.float16, shape=tuple(shape)
        )
        flat = mapped.reshape(-1)
        converted_fingerprint = hashlib.sha256(json.dumps(identity, sort_keys=True).encode())
        for offset in range(first, last, chunk_size):
            raw = read(shard, 8 + header_size + offset, min(chunk_size, last - offset))
            if local:
                converted_fingerprint.update(raw)
            values = np.frombuffer(raw, dtype=source_dtype)
            if dtype == "BF16":
                values = (values.astype(np.uint32) << 16).view(np.float32)
            with np.errstate(over="ignore", invalid="ignore"):
                values = values.astype(np.float16)
            if not np.isfinite(values).all():
                raise ValueError("Embedding values must be finite and representable in float16")
            start = (offset - first) // itemsize
            flat[start : start + len(values)] = values
        if local and (
            converted_fingerprint.hexdigest() != digest
            or read(shard, 8, header_size) != header_data
        ):
            raise ValueError("Checkpoint changed while loading the embedding tensor")
        mapped.flush()
        flat = mapped = None
        try:
            os.replace(temporary, cache_path)
        except OSError:
            # Another reader can already have the same immutable cache mapped on Windows.
            cached = _cached_embedding(cache_path, shape, chunk_size)
            if cached is None:
                raise
            return cached
    finally:
        # Close memory maps before rename/unlink on Windows, including failed downloads.
        if mapped is not None:
            flat = mapped = None
        temporary.unlink(missing_ok=True)
    return np.load(cache_path, mmap_mode="r", allow_pickle=False)
