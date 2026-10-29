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

"""Assignment-bound recovery over the existing layerwise checkpoint format."""

import hashlib
import json
import re
import shutil
from contextlib import contextmanager
from pathlib import Path

from modelopt.torch.utils import safe_save

from ._calibration_store import _CalibrationStore, _LimitedWriter
from .utils.layerwise_calib import _CheckpointState

__all__ = []


def _fingerprint(path):
    if path.is_symlink() or not path.is_file():
        raise ValueError("Layerwise recovery requires regular managed files")
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(2**20), b""):
            digest.update(chunk)
    return {"bytes": path.stat().st_size, "sha256": digest.hexdigest()}


@contextmanager
def _assignment_checkpoint(
    directory,
    *,
    assignment_digest,
    provenance,
    num_layers,
    max_bytes,
    save_every=1,
    calib_mutates_weights=False,
):
    """Lease final-calibration recovery for one already-validated frozen assignment."""
    if not isinstance(assignment_digest, str) or not re.fullmatch(
        r"[0-9a-f]{64}", assignment_digest
    ):
        raise ValueError("Final calibration requires a frozen assignment digest")
    if any(
        not isinstance(provenance.get(key), str) or not provenance[key]
        for key in ("source", "inputs", "code")
    ):
        raise ValueError("Final calibration requires source, input and code provenance")
    if any(type(value) is not int or value < 1 for value in (num_layers, save_every)):
        raise ValueError("Final calibration requires positive layer and checkpoint intervals")
    if num_layers > 10000 or type(calib_mutates_weights) is not bool:
        raise ValueError("Invalid final-calibration checkpoint layout")
    if type(max_bytes) is not int or max_bytes < 65536:
        raise ValueError("Final calibration requires at least 64 KiB of checkpoint storage")
    binding = json.loads(
        json.dumps(
            {
                "version": 1,
                "assignment": assignment_digest,
                "provenance": provenance,
                "num_layers": num_layers,
                "save_every": save_every,
                "calib_mutates_weights": calib_mutates_weights,
            },
            sort_keys=True,
            allow_nan=False,
        )
    )
    store = _CalibrationStore(directory)
    if store.directory.is_symlink():
        raise ValueError("Final-calibration checkpoint root cannot be a symlink")
    if store.world_size != 1:
        raise ValueError("Assignment-bound recovery currently requires one process")
    store.max_artifact_bytes = min(2**20, max_bytes // 4)
    boundary = store.directory / "boundary"
    with store.exclusive(boundary):
        yield _AssignmentCheckpoint(store, boundary, binding, max_bytes)


class _AssignmentCheckpoint(_CheckpointState):
    def __init__(self, store, boundary, binding, max_bytes):
        super().__init__(
            str(store.directory),
            binding["num_layers"],
            save_every=binding["save_every"],
            calib_mutates_weights=binding["calib_mutates_weights"],
        )
        self.store, self.boundary, self.binding = store, boundary, binding
        # Reserve two atomic metadata generations and the small legacy manifest. Tensor files
        # are append-only until publication, so the previous boundary survives a failed write.
        self.data_limit = max_bytes - 2 * store.max_artifact_bytes - 4096
        self.records = []
        self._validate_tree()
        self._check_size()
        if boundary.exists():
            state = store.load(boundary / "rank0.pth", store.max_artifact_bytes)
            if state["binding"] != binding:
                raise ValueError("Final-calibration assignment or provenance differs")
            self.records = state["layers"]
            self.start_layer = len(self.records)
            if self.start_layer > self.num_layers:
                raise ValueError("Final-calibration prefix exceeds the decoder")
            expected = {
                "quantizer_state.pt",
                "output_meta.pt",
                "weights.pt" if self.calib_mutates_weights else "quantizer_buffers.pt",
            }
            for index, record in enumerate(self.records):
                if set(record) != expected:
                    raise ValueError("Incomplete final-calibration layer state")
                for name, fingerprint in record.items():
                    self._verify(self._layer(index) / name, fingerprint)
            inputs = state["next_inputs"]
            if 0 < self.start_layer < self.num_layers:
                self._verify(self._layer(self.start_layer - 1) / "next_inputs.pt", inputs)
            elif inputs is not None:
                raise ValueError("Unexpected final-calibration boundary inputs")
        else:
            if any(store.directory.glob("layer_*")) or (store.directory / "manifest.json").exists():
                raise ValueError("Cannot adopt an unbound layerwise checkpoint")
            store._publish(boundary, {"binding": binding, "layers": [], "next_inputs": None})
        # Only after validating the committed prefix may abandoned work be reclaimed.
        for path in store.directory.glob("layer_*"):
            if int(path.name[6:]) >= self.start_layer:
                shutil.rmtree(path)
        self._prune_stale_next_inputs(keep=self.start_layer - 1)
        self._last_saved_layer = self.start_layer - 1
        self._next_to_save = self.start_layer
        self._boundary_layer = self.start_layer - 1
        self._data_bytes = self._check_size()

    def _layer(self, index):
        return self.store.directory / f"layer_{index:04d}"

    def _validate_tree(self):
        files = {
            "weights.pt",
            "quantizer_buffers.pt",
            "quantizer_state.pt",
            "output_meta.pt",
            "next_inputs.pt",
        }
        for path in self.store.directory.iterdir():
            if path.name == "boundary":
                target = path.resolve()
                if (
                    not path.is_symlink()
                    or target.parent != self.store.directory.resolve()
                    or not re.fullmatch(r"\.boundary-[^/]{8}", target.name)
                    or not target.is_dir()
                    or {p.name for p in target.iterdir()} != {"rank0.pth"}
                    or (target / "rank0.pth").is_symlink()
                    or not (target / "rank0.pth").is_file()
                ):
                    raise ValueError("Unmanaged final-calibration boundary")
                continue
            if re.fullmatch(r"\.boundary-[^/]{8}", path.name):
                continue  # The shared atomic store validates its own namespace.
            if path.name in {".boundary.lock", "manifest.json", "manifest.json.tmp"}:
                if path.is_symlink() or not path.is_file():
                    raise ValueError("Unmanaged layerwise checkpoint metadata")
                continue
            if (
                not re.fullmatch(r"layer_\d{4}", path.name)
                or path.is_symlink()
                or not path.is_dir()
                or int(path.name[6:]) >= self.num_layers
            ):
                raise ValueError("Unmanaged layerwise checkpoint entry")
            if any(
                p.name not in files or p.is_symlink() or not p.is_file() for p in path.iterdir()
            ):
                raise ValueError("Unmanaged layerwise checkpoint files")

    def _verify(self, path, fingerprint):
        if not isinstance(fingerprint, dict) or _fingerprint(path) != fingerprint:
            raise ValueError("Final-calibration prefix or activation boundary is corrupt")

    def _check_size(self):
        size = sum(
            p.stat().st_size for d in self.store.directory.glob("layer_*") for p in d.iterdir()
        )
        if size > self.data_limit:
            raise ValueError("Final-calibration checkpoint exceeds its byte limit")
        return size

    def _save_file(self, payload, path):
        # The lease excludes other writers; count each append once, including failed writes.
        with Path(path).open("xb") as stream:
            try:
                with _LimitedWriter(stream, self.data_limit - self._data_bytes) as writer:
                    safe_save(payload, writer)
            finally:
                self._data_bytes += Path(path).stat().st_size

    def save(self, layer_idx, model, layers, next_layer_inputs=None):
        if layer_idx != self._next_to_save or self._layer(layer_idx).exists():
            raise ValueError("Final calibration must append layers in order")
        if (next_layer_inputs is None) != (layer_idx + 1 == self.num_layers):
            raise ValueError("Final-calibration checkpoint requires its next activation boundary")
        super().save(layer_idx, model, layers, next_layer_inputs)
        record = {
            p.name: _fingerprint(p)
            for p in self._layer(layer_idx).iterdir()
            if p.name != "next_inputs.pt"
        }
        self.records.append(record)
        self._next_to_save += 1
        if self._last_saved_layer != layer_idx:
            return
        inputs = self._layer(layer_idx) / "next_inputs.pt"
        fingerprint = _fingerprint(inputs) if next_layer_inputs is not None else None
        self.store._publish(
            self.boundary,
            {
                "binding": self.binding,
                "layers": self.records,
                "next_inputs": fingerprint,
            },
        )
        previous = self._layer(self._boundary_layer) / "next_inputs.pt"
        if self._boundary_layer >= 0 and previous.exists():
            size = previous.stat().st_size
            previous.unlink()
            self._data_bytes -= size
        self._boundary_layer = layer_idx
