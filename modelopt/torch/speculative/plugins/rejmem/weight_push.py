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

r"""Push the trainer's drafter weights into its local vLLM servers (C3, trainer side).

The servers run vLLM's built-in RL weight transfer with the CUDA IPC backend and the rejmem worker extension,
bound to loopback because these dev-mode endpoints unpickle request data::

    VLLM_SERVER_DEV_MODE=1 VLLM_ALLOW_INSECURE_SERIALIZATION=1 vllm serve ... --host 127.0.0.1 \
        --weight-transfer-config '{"backend": "ipc"}' \
        --worker-extension-cls modelopt.torch.speculative.plugins.rejmem.vllm_worker.RejMemWorkerExtension

Each export tensor is shared through a CUDA IPC handle and every server copies it into its draft
(``/start_draft_weight_update``, ``/update_weights``, ``/finish_weight_update``), so the trainer and the
servers must share a node; they may use different GPUs of it (peer access). A push first pauses every
server (``/pause?mode=wait``): running requests finish on the old weights, queued ones start on the new
weights, and new ones are refused (HTTP 503) until the push completes. This module must not import vLLM.
"""

import base64
import contextlib
import pickle  # nosec B403 - dumps our own IPC handles; vLLM's HTTP IPC protocol takes them pickled
from collections.abc import Mapping, Sequence
from typing import Any

import httpx
import torch
from torch.multiprocessing.reductions import reduce_tensor

__all__ = ["DraftWeightPusher"]


class DraftWeightPusher:
    """Push drafter weights to local TP1 vLLM servers and confirm that every server applied them.

    ``weight_version`` is the number of draft updates the servers have completed since launch, i.e. the
    version their rollouts are stamped with. Version 0 is whatever export the servers were launched from:
    launch them from the trainer's current weights, or push once before the first rollout. Example::

        pusher = DraftWeightPusher(["http://127.0.0.1:65001", "http://127.0.0.1:65002"])
        pusher.push(model)  # after each optimizer step
    """

    def __init__(self, server_urls: Sequence[str], timeout: float = 600.0) -> None:
        """Connect to the servers; they must agree on the weight version and the draft checkpoint layout."""
        if not server_urls:
            raise ValueError("no vLLM server to push to")
        self._http = httpx.Client(timeout=timeout)
        self._urls = [url.rstrip("/") for url in server_urls]
        infos = {url: self._worker_info(url) for url in self._urls}
        first = infos[self._urls[0]]
        for url, info in infos.items():
            if (info["weight_version"], info["draft_manifest"]) != (
                first["weight_version"],
                first["draft_manifest"],
            ):
                raise RuntimeError(
                    f"{url} serves another draft version or layout than {self._urls[0]}"
                )
        if not first["draft_manifest"]:
            raise RuntimeError(f"{self._urls[0]} reports no draft checkpoint")
        self.weight_version: int = first["weight_version"]
        self._manifest: dict[str, list] = first["draft_manifest"]
        self._uuids = {url: info["device_uuid"] for url, info in infos.items()}
        self._failed = False
        for url in self._urls:
            self._post(url, "init_weight_transfer_engine", {"init_info": {"packed": False}})

    def push(self, weights: torch.nn.Module | Mapping[str, torch.Tensor]) -> int:
        """Push a DFlash-family model's export tensors, or an export-format state dict; return the new version.

        Raises before changing any server if the tensors do not match the servers' launch checkpoint or a
        server cannot be paused. A push that fails later leaves the servers paused with undefined drafts:
        restart them.
        """
        if self._failed:
            raise RuntimeError(
                "a previous push failed; restart the vLLM servers before pushing again"
            )
        tensors = self._export_tensors(weights)
        try:
            self._pause_all()
            self._failed = True
            for url in self._urls:
                self._push_to(url, tensors)
        finally:
            tensors.clear()  # the servers have copied them; let ipc_collect reclaim the shared blocks
            torch.cuda.ipc_collect()
        for url in self._urls:
            self._post(url, "resume")
        self._failed = False
        self.weight_version += 1
        return self.weight_version

    def close(self) -> None:
        """Close the HTTP connections."""
        self._http.close()

    def _export_tensors(
        self, weights: torch.nn.Module | Mapping[str, torch.Tensor]
    ) -> dict[str, torch.Tensor]:
        # A model goes through the exporter's own extraction, as the training-time export callback does.
        tensors: Mapping[str, torch.Tensor] = (
            weights.get_exporter()._extract_state_dict(weights.state_dict())
            if isinstance(weights, torch.nn.Module)
            else weights
        )
        got = {name: [str(t.dtype), list(t.shape)] for name, t in tensors.items()}
        if got != self._manifest:
            missing = sorted(self._manifest.keys() - got.keys())
            extra = sorted(got.keys() - self._manifest.keys())
            changed = sorted(
                n for n in got.keys() & self._manifest.keys() if got[n] != self._manifest[n]
            )
            raise ValueError(
                "tensors do not match the servers' draft checkpoint: "
                f"missing {missing[:8]}, unexpected {extra[:8]}, different dtype/shape {changed[:8]}"
            )
        # In the servers' load order, like a fresh launch.
        return {name: tensors[name].detach().to("cuda").contiguous() for name in self._manifest}

    def _pause_all(self) -> None:
        # Every server is paused before any is changed, so a refusal here leaves all of them as they were.
        # Servers paused already (a failed push leaves its servers paused) are refused, never resumed.
        if paused := [url for url in self._urls if self._get(url, "is_paused")["is_paused"]]:
            raise RuntimeError(f"{paused} are paused, e.g. after a failed push; restart them")
        try:
            for url in self._urls:
                self._post(url, "pause?mode=wait")
        except BaseException:
            for url in self._urls:
                with contextlib.suppress(Exception):
                    self._post(url, "resume")
            raise

    def _push_to(self, url: str, tensors: dict[str, torch.Tensor]) -> None:
        self._post(url, "start_draft_weight_update")
        self._post(
            url, "update_weights", {"update_info": _ipc_update_info(tensors, self._uuids[url])}
        )
        self._post(url, "finish_weight_update")
        # Cached prefix blocks hold draft K/V computed with the old weights.
        if not self._post(url, "reset_prefix_cache")["success"]:
            raise RuntimeError(f"{url} could not reset its prefix cache")
        version = self._worker_info(url)["weight_version"]
        if version != self.weight_version + 1:
            raise RuntimeError(
                f"{url} is at weight_version {version}, expected {self.weight_version + 1}"
            )

    def _worker_info(self, url: str) -> dict[str, Any]:
        results = self._post(url, "collective_rpc", {"method": "rejmem_weight_info"})["results"]
        if len(results) != 1:
            raise RuntimeError(f"{url} runs {len(results)} workers; only TP1 servers are supported")
        return results[0]

    def _get(self, url: str, path: str) -> Any:
        return _json(self._http.get(f"{url}/{path}"))

    def _post(self, url: str, path: str, body: dict[str, Any] | None = None) -> Any:
        return _json(self._http.post(f"{url}/{path}", json=body))


def _json(response: httpx.Response) -> Any:
    if response.is_error:
        request = response.request
        raise RuntimeError(
            f"{request.method} {request.url} failed ({response.status_code}): {response.text[:2000]}"
        )
    return response.json()


def _ipc_update_info(tensors: dict[str, torch.Tensor], device_uuid: str) -> dict[str, Any]:
    """The ``IPCWeightTransferUpdateInfo`` fields that vLLM's IPC engine loads on ``device_uuid``.

    Exported afresh for every server: torch pairs each IPC export with exactly one consumer release.
    """
    handles = [{device_uuid: reduce_tensor(t)[1]} for t in tensors.values()]
    return {
        "names": list(tensors),
        "dtype_names": [str(t.dtype).removeprefix("torch.") for t in tensors.values()],
        "shapes": [list(t.shape) for t in tensors.values()],
        "ipc_handles_pickled": base64.b64encode(pickle.dumps(handles)).decode(),
    }
