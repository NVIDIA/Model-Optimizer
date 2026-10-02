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

"""Hugging Face checkpoint utility.

General-purpose logic about the on-disk shape of an HF checkpoint (index, shards, sidecars) --
not export-specific, so it lives under ``modelopt.torch.utils.plugins`` alongside
``model_load_utils`` rather than under ``modelopt.torch.export``. Deliberately independent of
that module's ``transformers``/``accelerate`` module-scope imports; only needs
``huggingface_hub`` + ``safetensors``.
"""

import contextlib
import fnmatch
import json
import os
import re
import shutil
import warnings
from collections.abc import Callable, Iterable
from pathlib import Path
from typing import Any

import torch
from huggingface_hub import snapshot_download
from huggingface_hub.errors import LocalEntryNotFoundError
from safetensors.torch import safe_open
from tqdm import tqdm

_HF_HUB_OFFLINE_TRUE_VALUES = {"1", "ON", "YES", "TRUE"}


def _as_nonnegative_int(value: Any) -> int | None:
    """Return ``value`` as an int when it is a non-negative integer."""
    if isinstance(value, bool):
        return None
    if isinstance(value, int) and value >= 0:
        return value
    return None


def _get_num_nextn_predict_layers(config_data: dict[str, Any], model: Any) -> int | None:
    """Get the number of next-token-prediction layers from config metadata."""
    num_nextn_predict_layers = _as_nonnegative_int(config_data.get("num_nextn_predict_layers"))
    if num_nextn_predict_layers is not None:
        return num_nextn_predict_layers

    model_config = getattr(model, "config", None)
    if model_config is not None:
        num_nextn_predict_layers = _as_nonnegative_int(
            getattr(model_config, "num_nextn_predict_layers", None)
        )
        if num_nextn_predict_layers is not None:
            return num_nextn_predict_layers

    return None


def _get_rope_theta(config_data: dict[str, Any], model: Any) -> Any:
    """Return rope_theta from exported config data or the in-memory model config."""
    rope_theta = config_data.get("rope_theta")
    if rope_theta is not None:
        return rope_theta

    model_config = getattr(model, "config", None)
    if model_config is None:
        return None

    return getattr(model_config, "rope_theta", None)


def _sanitize_llama3_rope_config(config_data: dict[str, Any], model: Any) -> None:
    """Fill missing llama3 rope_theta in rope config metadata when available."""
    rope_theta = _get_rope_theta(config_data, model)
    if rope_theta is None:
        return

    for key in ("rope_parameters", "rope_scaling"):
        rope_config = config_data.get(key)
        if not isinstance(rope_config, dict):
            continue

        rope_type = rope_config.get("rope_type", rope_config.get("type"))
        if rope_type == "llama3" and "rope_theta" not in rope_config:
            rope_config["rope_theta"] = rope_theta


def sanitize_hf_config_for_deployment(config_data: dict[str, Any], model: Any) -> None:
    """Sanitize exported Hugging Face config metadata for deployment runtimes.

    Fix conservative deployment-only config incompatibilities:

    * add missing llama3 ``rope_theta`` metadata when available;
    * trim trailing MTP/next-token-prediction ``layer_types`` entries only when
      the mismatch is exactly explained by next-token-prediction metadata.
    """
    _sanitize_llama3_rope_config(config_data, model)

    num_hidden_layers = _as_nonnegative_int(config_data.get("num_hidden_layers"))
    layer_types = config_data.get("layer_types")
    if num_hidden_layers is None or not isinstance(layer_types, list):
        return

    num_layer_types = len(layer_types)
    if num_layer_types == num_hidden_layers:
        return

    num_nextn_predict_layers = _get_num_nextn_predict_layers(config_data, model)
    if (
        num_layer_types > num_hidden_layers
        and num_nextn_predict_layers == num_layer_types - num_hidden_layers
    ):
        warnings.warn(
            "Trimming config.layer_types from "
            f"{num_layer_types} to {num_hidden_layers} entries so it matches "
            "num_hidden_layers; the removed entries correspond to "
            "num_nextn_predict_layers.",
            stacklevel=2,
        )
        config_data["layer_types"] = layer_types[:num_hidden_layers]


def _is_hf_hub_offline() -> bool:
    return os.environ.get("HF_HUB_OFFLINE", "").strip().upper() in _HF_HUB_OFFLINE_TRUE_VALUES


def _copy_python_files(source_dir: Path, save_dir: Path) -> None:
    for py_file in source_dir.glob("*.py"):
        shutil.copy2(py_file, save_dir / py_file.name)


def copy_hf_ckpt_remote_code(
    pretrained_model_path: str | os.PathLike, save_directory: str | os.PathLike
):
    """Copy remote code from pretrained model to save directory.

    For models that keep configuration and modeling files as part of the checkpoint,
    we need to copy them to the export directory for seamless integration with inference
    frameworks.

    If ``pretrained_model_path`` is a local directory, Python files are copied directly.
    If it's a HF Hub model ID (e.g. ``nvidia/NVIDIA-Nemotron-Nano-12B-v2``), the Hub
    snapshot is resolved first and Python files are copied from that snapshot. When
    ``HF_HUB_OFFLINE`` is set, the snapshot must already be available in the local
    Hugging Face cache.

    Args:
        pretrained_model_path: Local path to the pretrained model or HuggingFace Hub model ID.
        save_directory: Path to the save directory.
    """
    hf_checkpoint_path = Path(pretrained_model_path)
    save_dir = Path(save_directory)
    save_dir.mkdir(parents=True, exist_ok=True)

    if hf_checkpoint_path.is_dir():
        _copy_python_files(hf_checkpoint_path, save_dir)
    else:
        local_files_only = _is_hf_hub_offline()
        try:
            source_dir = Path(
                snapshot_download(
                    repo_id=str(pretrained_model_path),
                    allow_patterns=["*.py"],
                    local_files_only=local_files_only,
                )
            )
        except LocalEntryNotFoundError as exc:
            if local_files_only:
                raise RuntimeError(
                    f"Could not copy Python sidecar files for {pretrained_model_path!r} because "
                    "HF_HUB_OFFLINE is enabled and the files are not available in the local "
                    "Hugging Face cache. Populate the cache with the model's *.py files or pass "
                    "a local pretrained model directory."
                ) from exc
            raise

        _copy_python_files(source_dir, save_dir)


def load_multimodal_components(
    pretrained_model_path: str | os.PathLike,
    prefixes: tuple[str, ...] = ("multi_modal_projector", "vision_model"),
) -> dict[str, torch.Tensor]:
    """Load multimodal components from safetensors file.

    Args:
        pretrained_model_path: Directory or HuggingFace repo id of the pretrained model.
        prefixes: Tensor key prefixes to select.  Defaults to the LLaVA-style
            ``multi_modal_projector`` / ``vision_model`` prefixes.  Pass
            ``("model.visual.",)`` for Qwen3-VL checkpoints.

    Returns:
        A dictionary of multimodal components.
    """
    hf_checkpoint_path = Path(pretrained_model_path)
    if not hf_checkpoint_path.is_dir():
        # Also accept a repo id, which is what the example scripts pass to quantize.py.
        # Fetched in two stages: the vision tower is a small fraction of a VLM checkpoint, so
        # pulling every shard to keep a few would waste tens of GB.
        local_files_only = _is_hf_hub_offline()
        repo_id = str(pretrained_model_path)
        try:
            index_dir = Path(
                snapshot_download(
                    repo_id=repo_id,
                    allow_patterns=["model.safetensors.index.json"],
                    local_files_only=local_files_only,
                )
            )
        except (LocalEntryNotFoundError, OSError, ValueError) as exc:
            raise ValueError(
                f"Invalid pretrained model path: {pretrained_model_path}. It should be a "
                "directory or an available HuggingFace repo id."
            ) from exc

        index_file = index_dir / "model.safetensors.index.json"
        if index_file.is_file():
            try:
                weight_map = json.loads(index_file.read_text())["weight_map"]
            except (json.JSONDecodeError, KeyError) as exc:
                raise ValueError(f"Malformed safetensors index in {repo_id}.") from exc
            wanted = sorted(
                {shard for key, shard in weight_map.items() if key.startswith(prefixes)}
            )
        else:
            wanted = ["model.safetensors"]  # unsharded checkpoint
        # Kept separate from the resolution failure above: a hub outage or a full disk here is
        # retryable, not a bad path.
        hf_checkpoint_path = Path(
            snapshot_download(
                repo_id=repo_id,
                allow_patterns=["model.safetensors.index.json", *wanted],
                local_files_only=local_files_only,
            )
        )

    safetensors_file = Path(hf_checkpoint_path) / "model.safetensors"
    safetensors_index_file = Path(hf_checkpoint_path) / "model.safetensors.index.json"

    multimodal_state_dict = {}

    if safetensors_file.is_file():
        print(f"Loading multimodal components from single file: {safetensors_file}")
        with safe_open(safetensors_file, framework="pt") as f:
            multimodal_keys = [
                key
                for key in f.keys()  # noqa: SIM118
                if key.startswith(prefixes)
            ]
            for key in tqdm(multimodal_keys, desc="Loading multimodal tensors"):
                multimodal_state_dict[key] = f.get_tensor(key)

    elif safetensors_index_file.is_file():
        print(f"Loading multimodal components from sharded model: {hf_checkpoint_path}")
        with open(safetensors_index_file) as f:
            safetensors_index = json.load(f)

        all_shard_files = sorted(
            {
                shard
                for key, shard in safetensors_index["weight_map"].items()
                if key.startswith(prefixes)
            }
        )
        for shard_file in all_shard_files:
            safetensors_filepath = Path(hf_checkpoint_path) / shard_file
            with safe_open(safetensors_filepath, framework="pt") as f:
                for key in f.keys():  # noqa: SIM118
                    if key.startswith(prefixes):
                        multimodal_state_dict[key] = f.get_tensor(key)

    else:
        print(f"Warning: No safetensors files found in {hf_checkpoint_path}")

    if not multimodal_state_dict:
        raise ValueError(
            f"No tensors under {prefixes} in {pretrained_model_path}; the vision tower would be "
            "missing from the export. The checkpoint's prefixes have likely changed."
        )

    print(f"Successfully loaded {len(multimodal_state_dict)} multimodal tensors")
    return multimodal_state_dict


def _matches_any_pattern(file_name: str, patterns: tuple[str, ...]) -> bool:
    return any(fnmatch.fnmatchcase(file_name, pattern) for pattern in patterns)


# Standard HF weight-file names: ``model.safetensors`` or ``model-00001-of-00005.safetensors``.
_IS_MAIN_WEIGHT_SHARD = re.compile(r"model(-\d{5}-of-\d{5})?\.safetensors")

# Off-index files that re-ship weights rather than add new ones. ``consolidated*.safetensors``
# is Mistral's second full copy of the model (vLLM's mistral load-format looks for it BY NAME,
# so copying it into an export is not inert -- it can be served in place of the quantized
# weights). ``adapter_model.safetensors`` is a PEFT adapter, whose tensor names do not overlap
# the index, so only a name rule catches it.
_IS_WEIGHT_DUPLICATE = re.compile(r"(consolidated[^/]*|adapter_model)\.safetensors")


# --- What reaches the export without passing through quantization --------------------------
# Two disjoint sets, distinguished by what the LOADER did with the file. That difference decides
# both how we find them and how we move them, so it is worth keeping straight:
#
#   1. UNPLACED weights. The loader opened the file and read the tensor, but the model had no
#      parameter for it, so transformers reports it in ``unexpected_keys`` -- an MTP head the
#      recipe did not quantize is the common case. Moved as TENSORS: located in whichever shard
#      holds them and merged into the exporter's ``extra_state_dict``.
#      Found by: ``read_unplaced_weights`` / ``carryable_unplaced_keys`` / ``locate_source_keys``.
#
#   2. OFF-INDEX sidecars. The index never names the file, so the loader never opened it and never
#      had the chance to call anything unexpected -- GLM-4.7 keeps its MTP head in a standalone
#      ``mtp.safetensors`` exactly this way. Moved as FILES: copied byte for byte, so no host
#      memory is spent re-serialising tensors the export does not otherwise touch.
#      Found by: ``off_index_safetensors_files`` / ``off_index_tensor_names``.
#
# The index is what separates them, and it is NOT an inventory of the checkpoint: a tensor missing
# from ``weight_map`` but sitting in a shard the index names for other tensors is set 1, not set 2.
# See ``record_unplaced_source_keys`` for the loading behaviour this rests on.
#
# Both sets must reach ``quantization_config.ignore``, or a deployment framework reads the
# top-level ``quant_algo`` and tries to load an original-precision weight as a quantized one
# (NVBug 5718750). ``_modelopt_carried_over_names`` is their union, recorded by the export once it
# knows what it actually wrote.

# --- HF checkpoint layout: what counts as a file inside a checkpoint -------------------
# A hub snapshot stores every entry as a symlink into a sibling ``blobs/`` directory, so
# "inside the checkpoint" has to mean the snapshot dir OR that blob root. Getting this wrong
# in either direction is costly: reject links and no hub checkpoint works, follow them blindly
# and a checkpoint can name any file on the host.
_MAX_CHECKPOINT_METADATA_BYTES = 128 * 1024 * 1024


def _is_relative_to(path: Path, root: Path) -> bool:
    return path == root or root in path.parents


def _snapshot_blob_root(source_root: Path) -> Path | None:
    if source_root.parent.name != "snapshots":
        return None
    blob_root = source_root.parent.parent / "blobs"
    return blob_root.resolve(strict=True) if blob_root.is_dir() else None


def _allowed_source_roots(src_dir: Path) -> list[Path]:
    source_root = src_dir.resolve(strict=True)
    allowed_roots = [source_root]
    if blob_root := _snapshot_blob_root(source_root):
        allowed_roots.append(blob_root)
    return allowed_roots


def resolve_checkpoint_file(
    src_dir: Path,
    relative_path: str | Path,
    *,
    max_bytes: int | None = _MAX_CHECKPOINT_METADATA_BYTES,
) -> Path:
    """Resolve a contained regular checkpoint file and optionally bound its size."""
    src = src_dir / relative_path
    try:
        resolved_src = src.resolve(strict=True)
    except OSError as exc:
        raise ValueError(f"checkpoint source is not a readable regular file: {src}") from exc
    if not resolved_src.is_file():
        raise ValueError(f"checkpoint source must resolve to a regular file: {src}")
    if not any(_is_relative_to(resolved_src, root) for root in _allowed_source_roots(src_dir)):
        raise ValueError(f"checkpoint source is outside the checkpoint directory: {src}")
    if max_bytes is not None and resolved_src.stat().st_size > max_bytes:
        raise ValueError(f"checkpoint source exceeds the {max_bytes}-byte size limit: {src}")
    return resolved_src


def off_index_safetensors_files(src: "str | os.PathLike") -> list[str]:
    """Safetensors files in a checkpoint that model loading never opens.

    Transformers reads the shards named in ``model.safetensors.index.json`` -- or the single
    ``model.safetensors`` when there is no index -- and nothing else. A checkpoint may ship more:
    GLM-4.7 keeps its MTP head in a standalone ``mtp.safetensors``. Those tensors are never loaded,
    never quantized, and never reported as ``unexpected_keys`` (the loader did not see them to call
    them unexpected), so they are not "the unquantized source weights" the export must avoid
    re-emitting -- they are untouched sidecars that happen to be in safetensors format. See
    :func:`~modelopt.torch.utils.plugins.model_load_utils.record_unplaced_source_keys` for why
    "never opened" is a property of the FILE rather than of the individual tensor: a tensor the
    index omits from a file it DOES open is reported, and is carried rather than copied.

    Files named like a main weight shard are excluded whatever the index says. An index that is
    empty, partial or malformed would otherwise make the source weights look off-index, and
    copying those into an export would leave unquantized weights beside the quantized ones.

    Files that re-ship the indexed weights are excluded too, by name for the conventions we know
    (``consolidated.safetensors``, ``adapter_model.safetensors``) and by tensor-name overlap for
    the ones we do not. See :func:`_without_reshipped_weights`.
    """
    d = Path(src)
    if not d.is_dir():
        return []
    index_file = d / "model.safetensors.index.json"
    indexed_tensors: set[str] = set()
    if index_file.exists():
        with open(index_file) as f:
            weight_map = json.load(f).get("weight_map", {})
        read_by_loader = set(weight_map.values())
        indexed_tensors = set(weight_map)
    else:
        read_by_loader = {"model.safetensors"}
        single = d / "model.safetensors"
        if single.exists():
            # Same reason the indexed branch fills this in: without the loaded tensor names,
            # _without_reshipped_weights has nothing to compare against and silently becomes a
            # no-op, leaving a second full copy under an unrecognised name to be copied verbatim.
            with contextlib.suppress(Exception), safe_open(str(single), framework="pt") as f:
                indexed_tensors = set(f.keys())

    candidates = [
        f.name
        for f in d.glob("*.safetensors")
        if f.name not in read_by_loader
        and not _IS_MAIN_WEIGHT_SHARD.fullmatch(f.name)
        and not _IS_WEIGHT_DUPLICATE.fullmatch(f.name)
    ]
    return sorted(_without_reshipped_weights(d, candidates, indexed_tensors))


def _without_reshipped_weights(
    d: Path, candidates: list[str], indexed_tensors: set[str]
) -> list[str]:
    """Drop candidates that re-ship weights the index already covers.

    The name rules above only catch conventions we know. A checkpoint free to invent its own
    filename can still carry a second copy of the indexed weights, and copying that into an
    export puts unquantized tensors beside the quantized ones. Overlapping tensor names are the
    general signal: a genuine sidecar (an MTP head) holds names the index does not have, which
    is exactly why the loader never placed them.

    Best-effort. Reads safetensors headers, never tensor data, and keeps any candidate whose
    header cannot be read: refusing to copy a real sidecar because of an unreadable header would
    silently drop weights from the export, which is the failure this whole path exists to avoid.
    """
    if not candidates or not indexed_tensors:
        return candidates

    kept = []
    for name in candidates:
        try:
            with safe_open(str(d / name), framework="pt") as f:
                names = set(f.keys())
        except Exception:
            kept.append(name)
            continue
        if names and names <= indexed_tensors:
            warnings.warn(
                f"Skipping {name}: it re-ships {len(names)} weight(s) the checkpoint index "
                "already covers, so copying it would duplicate unquantized weights."
            )
            continue
        kept.append(name)
    return kept


def indexed_weight_map(ckpt: "str | Path") -> dict[str, str]:
    """``param name -> shard file`` as the checkpoint INDEX declares it.

    Named for what it returns rather than for the question callers want answered. For a sharded
    checkpoint this is ``weight_map`` verbatim: it describes what the loader will look for, not
    what the shards physically contain, and the two differ -- a tensor present in a shard but
    absent from the index is invisible here. :func:`locate_source_keys` exists to cover that gap
    and should be preferred by anything asking "where does this key actually live". Only the
    single-file case is exhaustive, because there is no index for it to disagree with.

    Independent of the loader's dependencies, deliberately -- only stdlib + safetensors, not
    transformers/accelerate, so it stays answerable in the partial-install environments where
    those are absent.

    Returns ``{}``, not an exception, when neither an index nor a single-file checkpoint exists:
    right for callers that treat "nothing recorded" as legitimate (e.g. :func:`locate_source_keys`
    below). Callers for whom that is a genuine error (a missing checkpoint, not merely nothing
    recorded) should check for the empty result and raise themselves.

    The indexed case, which is every sharded checkpoint, is a stdlib JSON read and needs nothing.
    Only a single-file ``model.safetensors`` needs safetensors, and only to list its keys.
    """
    index = Path(ckpt) / "model.safetensors.index.json"
    if index.exists():
        with open(index) as f:
            return json.load(f).get("weight_map", {})
    single = Path(ckpt) / "model.safetensors"
    if single.exists():
        with safe_open(str(single), framework="pt") as f:
            return dict.fromkeys(f.keys(), "model.safetensors")
    return {}


def read_safetensors_subset(
    ckpt_path: "str | Path",
    weight_map: dict,
    select: Callable[[str], bool],
) -> dict:
    """Read tensors whose name satisfies ``select`` from safetensors files.

    Groups param names by file to avoid re-opening. Returns CPU tensors.
    Uses ``safe_open`` so only the requested tensors' bytes are read.

    ``get_tensor`` returns a zero-copy view into the mmap'd file; the bytes are
    not actually read from disk until first touched. We ``clone()`` here to force
    the read eagerly, while this function runs (each rank reading its own layers
    in parallel, for FSDP2 callers). Without it the read is deferred to a later
    per-source broadcast, which is serialized across ranks and silently destroys
    the read parallelism such callers exist to provide.
    """
    by_file: dict[str, list[str]] = {}
    for name, file in weight_map.items():
        if select(name):
            by_file.setdefault(file, []).append(name)

    state: dict[str, torch.Tensor] = {}
    for file, names in by_file.items():
        with safe_open(os.path.join(str(ckpt_path), file), framework="pt", device="cpu") as f:
            for name in names:
                state[name] = f.get_tensor(name).clone()
    return state


def locate_source_keys(ckpt: "str | Path", keys: list[str]) -> dict[str, str]:
    """Map each key to the safetensors file holding it, for keys the index may not list.

    ``model.safetensors.index.json`` is not a complete inventory of the checkpoint. Transformers
    enumerates the CONTENTS of each shard it opens, so it reports a tensor the index omits -- an
    MTP head stored inside a main shard is exactly that case, and it reaches
    ``_modelopt_unplaced_source_keys`` like any other unplaced key. Resolving purely through
    ``weight_map`` then finds no shard for it and drops it silently, which is the failure the
    carry-over exists to prevent.

    The index answers for everything it lists, at no cost. Only the leftovers trigger a header
    scan -- names, never tensor data -- and only when there are any, which is the rare case.
    """
    # indexed_weight_map, not anything from model_load_utils: that module imports transformers
    # and accelerate at module scope, and reaching for it here would make this answer "nothing to
    # carry" wherever they are absent -- the partial-install environments -- silencing the
    # --vllm_fakequant_export guard in exactly the case it exists for. Pinned by
    # test_carryable_unplaced_keys_works_without_the_loader_dependencies, which caught this.
    if not Path(ckpt).is_dir():
        # A checkpoint that is not there is a different failure from a key that is not in it, and
        # the caller's handler already says the right thing about the first ("could not copy ...
        # the checkpoint will be missing them"). Collapsing both into "in no safetensors file"
        # would report a missing directory as a missing tensor.
        raise ValueError(f"source checkpoint is not a directory: {ckpt}")

    weight_map = indexed_weight_map(ckpt)
    located = {k: weight_map[k] for k in keys if k in weight_map}
    remaining = {k for k in keys if k not in located}
    if not remaining:
        return located

    for shard in sorted(Path(ckpt).glob("*.safetensors")):
        if not remaining:
            break
        try:
            with safe_open(str(shard), framework="pt") as f:
                found = remaining.intersection(f.keys())
        except Exception:
            continue
        located.update(dict.fromkeys(found, shard.name))
        remaining -= found

    if remaining:
        warnings.warn(
            f"{len(remaining)} checkpoint key(s) the loader reported are in no safetensors file "
            f"of {ckpt} (e.g. {min(remaining)}); they cannot be carried into the export."
        )
    return located


def copy_off_index_safetensors(src: "str | os.PathLike", dst: "str | os.PathLike") -> list[str]:
    """Copy the safetensors files model loading never reads, verbatim.

    Copying beats reading them into a state dict and re-serialising: no host memory is spent on
    tensors the export does not touch, the bytes and the file layout are preserved exactly, and a
    consumer that finds them by filename (vLLM looks for the MTP sidecar) sees what it saw in the
    source. See :func:`off_index_safetensors_files` for how "never reads" is decided.
    """
    names = off_index_safetensors_files(src)
    copied = []
    for name in names:
        target = Path(dst) / name
        if target.exists():
            continue
        # copy2 follows symlinks, so a checkpoint shipping ``x.safetensors -> /etc/passwd`` would
        # copy whatever that names into the export under an approved-looking name. The question
        # is where the link LANDS, not whether it is a link: a Hugging Face snapshot stores every
        # file as a symlink into ``../../blobs/<sha>``, so refusing links outright drops the
        # sidecar of every hub-downloaded checkpoint -- the GLM-4.7 ``mtp.safetensors`` this path
        # exists to carry included.
        #
        # resolve_checkpoint_file already draws that line and is tested against both shapes: it
        # resolves strictly, demands a regular file, and demands the target sit under the
        # checkpoint dir or its sibling ``blobs/``. max_bytes=None because its default bounds
        # metadata, and these are weight files.
        try:
            source = resolve_checkpoint_file(Path(src), name, max_bytes=None)
        except ValueError as exc:
            warnings.warn(f"Skipping {name}: {exc}")
            continue
        shutil.copy2(source, target)
        copied.append(name)
    return copied


def copy_non_safetensor_files_from_ckpt(
    src: str | os.PathLike,
    dst: str | os.PathLike,
    *,
    exclude_files: Iterable[str] | None = None,
    exclude_patterns: Iterable[str] | None = None,
) -> list[str]:
    """Copy every non-safetensors file from a local HF checkpoint dir verbatim.

    Use as a baseline so tokenizer files, remote_code ``*.py``, README, LICENSE, etc.
    are preserved from the source. Callers can exclude additional files or patterns when
    copying after export-owned metadata has already been written.

    Args:
        src: Source HF checkpoint directory. Must be a local path.
        dst: Destination directory; created if missing.
        exclude_files: Exact file names to skip.
        exclude_patterns: Glob patterns for additional files to skip.

    Returns:
        File names copied into ``dst``.
    """
    if not os.path.isdir(src):
        raise ValueError(f"Invalid source path: {src}. It should be a directory.")
    exclude_files = set(exclude_files or ())
    exclude_patterns = tuple(exclude_patterns or ())
    copied_files = []
    os.makedirs(dst, exist_ok=True)
    for entry in sorted(os.listdir(src)):
        if entry in exclude_files or _matches_any_pattern(entry, exclude_patterns):
            continue
        sp = os.path.join(src, entry)
        if not os.path.isfile(sp):
            continue
        if entry.endswith(".safetensors") or entry == "model.safetensors.index.json":
            continue
        try:
            shutil.copy2(sp, dst)
        except OSError as error:
            warnings.warn(f"Failed to copy checkpoint sidecar {entry}: {error}")
            continue
        copied_files.append(entry)
    return copied_files
