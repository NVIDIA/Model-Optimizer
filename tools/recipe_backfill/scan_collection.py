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

"""Scan a Hugging Face collection of quantized checkpoints into a module-format snapshot.

For every model in the collection this reads ``config.json``,
``hf_quant_config.json`` and ``model.safetensors.index.json`` from the Hub and
writes ``published_checkpoints.json``: the per-module quantization format of each
published checkpoint, in the compact form described in ``checkpoint_scan.py``.

That snapshot is what ``verify_recipes.py`` and the recipe unit tests check the
shipped recipes against, so the check runs offline and does not depend on the Hub
staying reachable (or on a gated repo staying readable).

Usage::

    python tools/recipe_backfill/scan_collection.py \
        --collection nvidia/inference-optimized-checkpoints-with-model-optimizer-66aa84f7966b3150262481a4 \
        --out tools/recipe_backfill/published_checkpoints.json

Set ``HF_TOKEN`` for gated repos (the Llama-4 mirrors need it).
"""

from __future__ import annotations

import argparse
import gzip
import json
import os
import struct
import sys
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from checkpoint_scan import BF16, drop_unbuilt_layers, module_map_from_quant_config, pack_module_map

HUB = "https://huggingface.co"
DEFAULT_COLLECTION = (
    "nvidia/inference-optimized-checkpoints-with-model-optimizer-66aa84f7966b3150262481a4"
)


#: Optional on-disk cache of raw Hub responses, so a re-run costs no bandwidth.
CACHE_DIR: Path | None = None


def _open(url: str, headers: dict[str, str], timeout: int):
    """Open an https:// URL, rejecting any other scheme."""
    if not url.startswith("https://"):
        raise ValueError(f"refusing to fetch a non-https URL: {url!r}")
    return urllib.request.urlopen(  # nosec B310 - scheme checked above
        urllib.request.Request(url, headers=headers), timeout=timeout
    )


def _get(url: str) -> bytes:
    cached = None
    if CACHE_DIR is not None:
        cached = CACHE_DIR / (url.replace("https://", "").replace("/", "_") + ".gz")
        if cached.is_file():
            return gzip.decompress(cached.read_bytes())
    headers = {"User-Agent": "modelopt-recipe-backfill", "Accept-Encoding": "gzip"}
    token = os.environ.get("HF_TOKEN")
    if token:
        headers["Authorization"] = f"Bearer {token}"
    with _open(url, headers, timeout=300) as r:
        body = r.read()
        body = gzip.decompress(body) if r.headers.get("Content-Encoding") == "gzip" else body
    if cached is not None:
        cached.parent.mkdir(parents=True, exist_ok=True)
        cached.write_bytes(gzip.compress(body))
    return body


def _get_json(url: str) -> dict:
    return json.loads(_get(url))


def _file(model_id: str, name: str) -> dict:
    return _get_json(f"{HUB}/{model_id}/resolve/main/{name}")


def _range(url: str, start: int, end: int) -> bytes:
    headers = {"User-Agent": "modelopt-recipe-backfill", "Range": f"bytes={start}-{end}"}
    token = os.environ.get("HF_TOKEN")
    if token:
        headers["Authorization"] = f"Bearer {token}"
    with _open(url, headers, timeout=120) as r:
        return r.read()


_SCALAR_DTYPES = {"F32": ("<f", 4), "F16": ("<e", 2), "F64": ("<d", 8)}

#: safetensors dtypes that mean "not quantized".
_HIGH_PRECISION_DTYPES = {"BF16", "F16", "F32", "F64"}


def read_shard_header(model_id: str, shard: str) -> dict:
    """Read one safetensors shard's JSON header (dtypes + shapes) with two range requests."""
    url = f"{HUB}/{model_id}/resolve/main/{shard}"
    header_len = struct.unpack("<Q", _range(url, 0, 7))[0]
    return json.loads(_range(url, 8, 8 + header_len - 1))


def probe_weight_dtypes(model_id: str, weight_map: dict[str, str]) -> dict[str, str]:
    """Per-module weight dtype, for checkpoints that export no scale tensors.

    The oldest ModelOpt exports (``nvidia/DeepSeek-R1-NVFP4-v2`` and friends) ship an
    index listing only ``.weight``, so there is no scale tensor to tell a quantized
    module from an untouched one -- and their ``exclude_modules`` lists are not
    exhaustive (the ``mlp.gate`` routers are missing from them). The shard *headers*
    still carry each tensor's dtype, which settles it: a packed low-precision weight
    is not BF16.
    """
    shards = sorted({shard for name, shard in weight_map.items() if name.endswith(".weight")})
    dtypes: dict[str, str] = {}
    for shard in shards:
        try:
            header = read_shard_header(model_id, shard)
        except (urllib.error.HTTPError, urllib.error.URLError, struct.error, ValueError):
            continue
        for tensor, info in header.items():
            if tensor.endswith(".weight") and isinstance(info, dict):
                dtypes[tensor.removesuffix(".weight")] = info.get("dtype", "")
    return dtypes


def read_scalar(model_id: str, shard: str, tensor: str) -> float | None:
    """Read one scalar tensor out of a remote safetensors shard with range requests.

    Two small ranges (the header, then the four bytes of the scalar) instead of the
    whole multi-GB shard. Used to tell a *cast* KV cache -- ``use_constant_amax``
    pins amax to the E4M3 max, so the exported ``k_scale`` is exactly ``1.0`` -- from
    a calibrated one, which the exported ``hf_quant_config.json`` cannot express.
    """
    url = f"{HUB}/{model_id}/resolve/main/{shard}"
    try:
        header_len = struct.unpack("<Q", _range(url, 0, 7))[0]
        header = json.loads(_range(url, 8, 8 + header_len - 1))
        info = header[tensor]
        fmt = _SCALAR_DTYPES.get(info["dtype"])
        if fmt is None or (info["shape"] not in ([], [1])):
            return None
        start, end = info["data_offsets"]
        raw = _range(url, 8 + header_len + start, 8 + header_len + end - 1)
        return struct.unpack(fmt[0], raw[: fmt[1]])[0]
    except (urllib.error.HTTPError, urllib.error.URLError, KeyError, struct.error, ValueError):
        return None


def _scale_formats(tensor_names: list[str]) -> dict[str, str]:
    """Per-module format implied by the exported scale tensors."""
    leaves: dict[str, set[str]] = {}
    for n in tensor_names:
        mod, _, leaf = n.rpartition(".")
        leaves.setdefault(mod, set()).add(leaf)
    out = {}
    for mod, present in leaves.items():
        if "weight_scale_2" in present:
            out[mod] = "NVFP4" if "input_scale" in present else "W4A16_NVFP4"
        elif "weight_scale_inv" in present:
            out[mod] = "FP8_BLOCK_SCALES"
        elif "weight_scale" in present:
            out[mod] = "FP8" if "input_scale" in present else "FP8_WO"
        elif "weight" in present:
            out[mod] = BF16
    return out


def scan_model(model_id: str) -> dict:
    """Scan one published checkpoint into a snapshot entry."""
    entry: dict = {"id": model_id}
    info = _get_json(f"{HUB}/api/models/{model_id}")
    siblings = [s["rfilename"] for s in info.get("siblings", [])]
    card = info.get("cardData") or {}
    base = card.get("base_model")
    entry["base_model"] = base[0] if isinstance(base, list) else base

    if "hf_quant_config.json" not in siblings:
        entry["skipped"] = "no hf_quant_config.json (not a ModelOpt HF checkpoint)"
        return entry
    quant = _file(model_id, "hf_quant_config.json").get("quantization", {})
    entry["quant_algo"] = quant.get("quant_algo")
    entry["kv_cache_quant_algo"] = quant.get("kv_cache_quant_algo")
    if quant.get("group_size"):
        entry["group_size"] = quant["group_size"]

    text_cfg: dict = {}
    if "config.json" in siblings:
        cfg = _file(model_id, "config.json")
        text_cfg = cfg.get("text_config") or cfg
        entry["model_type"] = cfg.get("model_type") or text_cfg.get("model_type")

    indexes = [f for f in siblings if f.endswith("index.json") and f != "model_index.json"]
    if not indexes:
        entry["skipped"] = "no safetensors index (ONNX / non-safetensors export)"
        return entry
    names: list[str] = []
    weight_map: dict[str, str] = {}
    for ix in indexes:
        prefix = ix.rsplit("/", 1)[0] + "/" if "/" in ix else ""
        for k, shard in _file(model_id, ix)["weight_map"].items():
            names.append(prefix + k)
            weight_map[prefix + k] = prefix + shard

    module_names = [n.rpartition(".")[0] for n in names if n.endswith(".weight")]
    scales = _scale_formats(names)
    has_scales = any(fmt != BF16 for fmt in scales.values())
    if not has_scales:
        # No scale tensors: fall back to the shard headers' weight dtypes.
        dtypes = probe_weight_dtypes(model_id, weight_map)
        scales = {
            m: ("QUANTIZED" if d and d not in _HIGH_PRECISION_DTYPES else BF16)
            for m, d in dtypes.items()
        }
        has_scales = any(fmt != BF16 for fmt in scales.values())
        entry["weight_dtype_probe"] = has_scales
    module_map = module_map_from_quant_config(module_names, quant, scales if has_scales else None)
    module_map, unbuilt = drop_unbuilt_layers(
        module_map,
        text_cfg.get("num_hidden_layers"),
        text_cfg.get("num_nextn_predict_layers") or 0,
    )
    if unbuilt:
        entry["unbuilt_layer_modules"] = len(unbuilt)
        entry["num_hidden_layers"] = text_cfg.get("num_hidden_layers")
    entry["modules"] = pack_module_map(module_map)
    kv = sorted({n.rpartition(".")[0] for n in names if n.endswith((".k_scale", ".v_scale"))})
    entry["kv_scale_modules"] = pack_module_map(dict.fromkeys(kv, "KV")) if kv else {}

    # Sample one exported scale per kind so the snapshot records what
    # hf_quant_config.json cannot: whether KV / expert activations were calibrated
    # or pinned to a constant amax (which exports as exactly 1.0).
    samples: dict[str, float] = {}
    k_scales = sorted(n for n in names if n.endswith(".k_scale"))
    if k_scales:
        value = read_scalar(model_id, weight_map[k_scales[0]], k_scales[0])
        if value is not None:
            samples["k_scale"] = value
    expert_inputs = sorted(n for n in names if ".experts." in n and n.endswith(".input_scale"))
    if expert_inputs:
        value = read_scalar(model_id, weight_map[expert_inputs[0]], expert_inputs[0])
        if value is not None:
            samples["expert_input_scale"] = value
    if samples:
        entry["scale_samples"] = samples
    return entry


def main() -> int:
    """CLI entry point."""
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--collection", default=DEFAULT_COLLECTION)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--jobs", type=int, default=6)
    ap.add_argument("--cache-dir", type=Path, help="cache raw Hub responses here")
    args = ap.parse_args()

    global CACHE_DIR
    CACHE_DIR = args.cache_dir

    coll = _get_json(f"{HUB}/api/collections/{args.collection}")
    model_ids = [i["id"] for i in coll["items"] if i.get("type") == "model"]
    print(f"{len(model_ids)} models in {args.collection}", file=sys.stderr)

    def safe(model_id: str) -> dict:
        try:
            return scan_model(model_id)
        except (urllib.error.HTTPError, urllib.error.URLError, KeyError, ValueError) as exc:
            return {"id": model_id, "skipped": f"{type(exc).__name__}: {exc}"}

    with ThreadPoolExecutor(max_workers=args.jobs) as pool:
        entries = list(pool.map(safe, model_ids))
    for e in entries:
        note = e.get("skipped", f"{len(e.get('modules', {}))} module patterns")
        print(f"  {e['id']:60} {note}", file=sys.stderr)

    args.out.write_text(
        json.dumps(
            {"collection": args.collection, "checkpoints": sorted(entries, key=lambda e: e["id"])},
            indent=1,
        )
        + "\n"
    )
    print(f"wrote {args.out}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
