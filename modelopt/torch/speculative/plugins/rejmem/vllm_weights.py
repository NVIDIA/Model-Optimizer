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

"""C3 draft weight updates inside the vLLM worker: the ``weight_version`` counter and DFlash's fused weights.

The trainer pushes draft weights through vLLM's built-in RL weight transfer (see ``weight_push.py``).
vLLM reloads the draft layerwise into the original parameter storage, which keeps CUDA graphs valid, but in
0.33.0 ``DFlashQwen3ForCausalLM.load_weights`` then rebuilds its fused context-K/V tensors as new
allocations that the FULL draft graph never reads, and does so mid-reload while the RoPE cache is a meta
placeholder. Here that rebuild runs once the update has finished, into the tensors the graph captured, and
every completed draft update increments ``weight_version``. vLLM does not roll back a failed update (the
draft and the shared target embedding stay mid-reload), so later draft updates are refused until restart.
Always installed; no environment variable.
"""

from collections.abc import Callable, Iterable, Iterator

import torch
import vllm
from vllm.model_executor.models.qwen3_dflash import DFlashQwen3ForCausalLM, DFlashQwen3Model
from vllm.v1.worker.gpu_worker import Worker

__all__ = ["current_weight_version", "draft_manifest", "install"]

# Tensors that DFlashQwen3Model._build_fused_kv_buffers derives from the draft weights.
_FUSED_TENSORS = (
    "_hidden_norm_weight",
    "_fused_kv_weight",
    "_fused_kv_bias",
    "_k_norm_weights",
    "_rope_cos_sin_cache",
)

_weight_version = 0
_draft_manifest: dict[str, tuple[str, list[int]]] = {}
_draft_update_failed = False
_installed = False


def current_weight_version() -> int:
    """Completed draft weight updates in this worker process; 0 means the launch checkpoint."""
    return _weight_version


def draft_manifest() -> dict[str, tuple[str, list[int]]]:
    """``name -> (dtype, shape)`` of the draft checkpoint this worker was launched with, in load order."""
    return dict(_draft_manifest)


def install() -> None:
    """Patch vLLM 0.33.0's DFlash draft loading and the worker's draft weight-update calls; idempotent."""
    global _installed
    if _installed:
        return
    if vllm.__version__ != "0.33.0":
        raise RuntimeError(f"rejmem patches vLLM 0.33.0 internals, found vLLM {vllm.__version__}")
    load_weights = DFlashQwen3ForCausalLM.load_weights
    build = DFlashQwen3Model._build_fused_kv_buffers
    start = Worker.start_draft_weight_update
    update = Worker.update_weights
    finish = Worker.finish_weight_update

    def patched_load_weights(self, weights):
        if not hasattr(self.model, "_num_attn_layers"):  # the launch checkpoint, not an update
            weights = _record_manifest(weights)
        return load_weights(self, weights)

    def patched_build(self):
        # During an update this runs mid-reload; finish_weight_update refreshes the tensors instead.
        if not hasattr(self, "_num_attn_layers"):
            build(self)

    def patched_start(self):
        if _draft_update_failed:
            # A new session would skip the modules the failed one left mid-reload and mix in its loads.
            raise RuntimeError("a draft weight update failed earlier; restart the vLLM server")
        start(self)

    def patched_update(self, update_info):
        global _draft_update_failed
        # vLLM keeps _weight_update_is_draft after a session ends; a call outside a session changes nothing.
        is_draft = self._weight_update_active and self._weight_update_is_draft
        try:
            update(self, update_info)
        except BaseException:
            _draft_update_failed |= is_draft
            raise

    def patched_finish(self):
        global _weight_version, _draft_update_failed
        is_draft = self._weight_update_active and self._weight_update_is_draft
        try:
            finish(self)
            if is_draft:
                _refresh_fused_tensors(self.get_draft_model().model, build)
        except BaseException:
            _draft_update_failed |= is_draft
            raise
        if is_draft:
            _weight_version += 1

    DFlashQwen3ForCausalLM.load_weights = patched_load_weights
    DFlashQwen3Model._build_fused_kv_buffers = patched_build
    Worker.start_draft_weight_update = patched_start
    Worker.update_weights = patched_update
    Worker.finish_weight_update = patched_finish
    _installed = True


def _record_manifest(
    weights: Iterable[tuple[str, torch.Tensor]],
) -> Iterator[tuple[str, torch.Tensor]]:
    for name, tensor in weights:
        _draft_manifest[name] = (str(tensor.dtype), list(tensor.shape))
        yield name, tensor


@torch.no_grad()
def _refresh_fused_tensors(model: torch.nn.Module, build: Callable[..., None]) -> None:
    """Rebuild the fused tensors from the reloaded weights and copy them into the captured ones."""
    if type(model) is not DFlashQwen3Model:
        raise NotImplementedError(
            f"rejmem draft weight updates support DFlashQwen3Model only, not {type(model).__name__}"
        )
    captured = {name: getattr(model, name) for name in _FUSED_TENSORS}
    build(model)
    for name, old in captured.items():
        new = getattr(model, name)
        if old is None or new is None:
            if old is not new:
                raise RuntimeError(f"DFlash {name} changed between None and a tensor on reload")
            continue
        if new.shape != old.shape or new.dtype != old.dtype:
            raise RuntimeError(f"DFlash {name} changed shape or dtype on reload")
        if new.data_ptr() != old.data_ptr():
            old.copy_(new)
        setattr(model, name, old)
