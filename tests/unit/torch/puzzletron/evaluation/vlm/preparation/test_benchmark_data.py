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

"""High-value safety and recovery tests for VLM benchmark data preparation."""

import hashlib
import io
import json
import tarfile
import zipfile
from pathlib import Path
from types import SimpleNamespace

import pytest

from examples.puzzletron.evaluation.vlm.preparation import benchmark_data as preparation

_EXPECTED_DATASETS = {
    "video_mmmu": (
        "lmms-lab/VideoMMMU",
        "d1c35ac933123d79e877b7f1b9506afb0309cf1b",
        "video_mmmu",
    ),
    "mmvu_val": ("lmms-lab/MMVU", "7537bc8a4b6716be5a9995e022c295679f4af616", "mmvu"),
    "mvbench": (
        "OpenGVLab/MVBench",
        "a776e554280b99b70f00cc3eacd69a65e0727efc",
        "mvbench_video",
    ),
    "videomme": (
        "lmms-lab/Video-MME",
        "ead1408f75b618502df9a1d8e0950166bf0a2a0b",
        "videomme",
    ),
    "longvideobench_val_v": (
        "longvideobench/LongVideoBench",
        "60d1c89c1919a198b73be39c2babb213b29d6a5c",
        "datasets/longvideobench",
    ),
    "mlvu_dev": ("sy1998/MLVU_dev", "96207eb9aa7101e2a495dd147684a7e618c79e12", "mlvu"),
    "perceptiontest_val_mc": (
        "lmms-lab/PerceptionTest_Val",
        "c5e520d8c4167fb1f135c36e9d6e67312b4f8e6b",
        "perceptiontest_val",
    ),
}


def _write_zip(path: Path, members: dict[str, bytes]) -> None:
    with zipfile.ZipFile(path, "w") as archive:
        for name, payload in members.items():
            archive.writestr(name, payload)


@pytest.mark.parametrize(
    ("task", "repository", "revision", "directory"),
    [(task, *values) for task, values in _EXPECTED_DATASETS.items()],
)
def test_every_preparation_contract_is_explicitly_pinned(task, repository, revision, directory):
    assert set(preparation.DATASETS) == set(_EXPECTED_DATASETS)
    item = preparation.DATASETS[task]
    assert (item.repository, item.revision, item.preparation_dir) == (
        repository,
        revision,
        directory,
    )


def test_zip_preparation_is_revision_bound_idempotent_and_byte_verified(tmp_path):
    hf_home = tmp_path / "hf-home"
    snapshot = preparation._hub_snapshot(hf_home, "mmvu_val")
    snapshot.mkdir(parents=True)
    _write_zip(snapshot / "videos.zip", {"videos/sample.mp4": b"video"})

    first = preparation._prepare(hf_home, "mmvu_val", snapshot)
    second = preparation._prepare(hf_home, "mmvu_val", snapshot)

    assert second == first
    assert first["status"] == "complete"
    assert first["files"] == 1
    marker = json.loads((hf_home / "mmvu" / preparation._MARKER_NAME).read_text())
    assert marker["revision"] == _EXPECTED_DATASETS["mmvu_val"][1]

    target = tmp_path / "target"
    target.mkdir()
    archive = tmp_path / "single.zip"
    _write_zip(archive, {"sample.mp4": b"expected"})
    preparation._extract_zip(archive, target)
    assert preparation._extract_zip(archive, target)["new_files"] == 0
    (target / "sample.mp4").write_bytes(b"differed")
    with pytest.raises(ValueError, match="differs from the archive"):
        preparation._extract_zip(archive, target)


@pytest.mark.parametrize("member", ["../escape.mp4", "/absolute.mp4"])
def test_archive_extraction_rejects_paths_outside_owned_root(tmp_path, member):
    archive = tmp_path / "unsafe.zip"
    target = tmp_path / "target"
    target.mkdir()
    _write_zip(archive, {member: b"unsafe"})

    with pytest.raises(ValueError, match="unsafe archive member"):
        preparation._extract_zip(archive, target)

    assert not (tmp_path / "escape.mp4").exists()


def test_archive_extraction_rejects_links_and_streams_multipart_tar(tmp_path):
    unsafe = io.BytesIO()
    with tarfile.open(fileobj=unsafe, mode="w") as archive:
        link = tarfile.TarInfo("videos/link.mp4")
        link.type = tarfile.SYMTYPE
        link.linkname = "../../outside"
        archive.addfile(link)
    unsafe.seek(0)
    target = tmp_path / "target"
    target.mkdir()
    with pytest.raises(ValueError, match="link or special file"):
        preparation._extract_tar(unsafe, target, label="unsafe")

    payload = io.BytesIO()
    with tarfile.open(fileobj=payload, mode="w") as archive:
        for name, value in (("videos/one.mp4", b"one"), ("videos/two.mp4", b"two")):
            member = tarfile.TarInfo(name)
            member.size = len(value)
            archive.addfile(member, io.BytesIO(value))
    data = payload.getvalue()
    midpoint = len(data) // 2
    parts = (tmp_path / "part-aa", tmp_path / "part-ab")
    parts[0].write_bytes(data[:midpoint])
    parts[1].write_bytes(data[midpoint:])
    stream = preparation._ConcatenatedReader(parts)
    try:
        report = preparation._extract_tar(stream, target, label="multipart")
    finally:
        stream.close()
    assert report["new_files"] == 2
    assert (target / "videos/two.mp4").read_bytes() == b"two"


def test_interrupted_initialization_leaves_target_retryable(monkeypatch, tmp_path):
    hf_home = tmp_path / "hf-home"
    hf_home.mkdir()
    target = hf_home / preparation.DATASETS["mmvu_val"].preparation_dir
    write_marker = preparation._write_marker

    def interrupt(staging, payload):
        write_marker(staging, payload)
        raise RuntimeError("interrupted")

    monkeypatch.setattr(preparation, "_write_marker", interrupt)
    with pytest.raises(RuntimeError, match="interrupted"):
        preparation._prepare_target(hf_home, "mmvu_val")
    assert not target.exists()

    monkeypatch.setattr(preparation, "_write_marker", write_marker)
    prepared, complete = preparation._prepare_target(hf_home, "mmvu_val")
    assert prepared == target
    assert complete is None
    assert json.loads((target / preparation._MARKER_NAME).read_text())["status"] == "in_progress"


def test_range_download_resumes_without_forwarding_credentials_and_verifies_hash(
    monkeypatch, tmp_path
):
    payload = b"pinned-range-download"
    entry = SimpleNamespace(
        path="archives/video.zip",
        size=len(payload),
        lfs={"sha256": hashlib.sha256(payload).hexdigest()},
    )
    destination = tmp_path / entry.path
    destination.parent.mkdir(parents=True)
    partial = destination.with_name(f".{destination.name}.modelopt-part")
    partial.write_bytes(payload[:7])
    observed = {}
    monkeypatch.setattr(
        preparation,
        "hf_hub_url",
        lambda *_args, **_kwargs: "https://huggingface.co/source",
    )

    def fake_metadata(*_args, **kwargs):
        observed["metadata_token"] = kwargs.get("token")
        return SimpleNamespace(
            commit_hash="revision",
            location="https://example/cdn",
            size=len(payload),
            xet_file_data=None,
        )

    def fake_headers(**kwargs):
        observed["headers_token"] = kwargs.get("token")
        return {"authorization": "secret", "user-agent": "test"}

    monkeypatch.setattr(preparation, "get_hf_file_metadata", fake_metadata)
    monkeypatch.setattr(preparation, "build_hf_headers", fake_headers)

    def fake_http_get(url, stream, **kwargs):
        observed.update(url=url, **kwargs)
        stream.write(payload[kwargs["resume_size"] :])

    monkeypatch.setattr(preparation, "http_get", fake_http_get)
    report = preparation._download_range_file(
        repository="owner/repository",
        revision="revision",
        entry=entry,
        snapshot=tmp_path,
        destination=destination,
    )

    assert report["status"] == "downloaded"
    assert destination.read_bytes() == payload
    assert observed["resume_size"] == 7
    assert observed["metadata_token"] is None
    assert observed["headers_token"] is None
    assert observed["headers"] == {"user-agent": "test"}
    assert not partial.exists()


def test_range_download_reuses_only_repository_cache_symlinks(tmp_path):
    payload = b"cached-range-download"
    entry = SimpleNamespace(
        path="archives/video.zip",
        size=len(payload),
        lfs={"sha256": hashlib.sha256(payload).hexdigest()},
    )
    repository_cache = tmp_path / "hub/datasets--owner--repository"
    snapshot = repository_cache / "snapshots/revision"
    destination = snapshot / entry.path
    destination.parent.mkdir(parents=True)
    blob = repository_cache / "blobs/pinned"
    blob.parent.mkdir()
    blob.write_bytes(payload)
    destination.symlink_to(blob)

    report = preparation._download_range_file(
        repository="owner/repository",
        revision="revision",
        entry=entry,
        snapshot=snapshot,
        destination=destination,
    )

    assert report["status"] == "reused"
    destination.unlink()
    outside = tmp_path / "outside"
    outside.write_bytes(payload)
    destination.symlink_to(outside)
    with pytest.raises(ValueError, match="escapes its repository cache"):
        preparation._download_range_file(
            repository="owner/repository",
            revision="revision",
            entry=entry,
            snapshot=snapshot,
            destination=destination,
        )


def test_range_snapshot_records_exact_revision_and_rejects_unsafe_repository_paths(
    monkeypatch, tmp_path
):
    item = preparation.DATASETS["mlvu_dev"]

    class SafeApi:
        def list_repo_tree(self, *args, **kwargs):
            assert args == (item.repository,)
            assert kwargs["revision"] == item.revision
            return [SimpleNamespace(path="data/file.bin", size=4)]

    monkeypatch.setattr(preparation, "HfApi", SafeApi)

    def fake_download(**kwargs):
        kwargs["destination"].parent.mkdir(parents=True, exist_ok=True)
        kwargs["destination"].write_bytes(b"data")
        return {"path": kwargs["entry"].path, "bytes": 4, "status": "downloaded"}

    monkeypatch.setattr(preparation, "_download_range_file", fake_download)
    snapshot = preparation._range_download(tmp_path, "mlvu_dev")
    marker = json.loads(preparation._range_download_marker(tmp_path, "mlvu_dev").read_text())
    assert (snapshot / "data/file.bin").read_bytes() == b"data"
    assert marker["schema"] == "modelopt.vlm-benchmark-range-download/v1"
    assert marker["revision"] == item.revision
    assert marker["status"] == "complete"

    class UnsafeApi:
        def list_repo_tree(self, *_args, **_kwargs):
            return [SimpleNamespace(path="../escape.bin", size=4)]

    monkeypatch.setattr(preparation, "HfApi", UnsafeApi)
    unsafe_root = tmp_path / "unsafe"
    unsafe_root.mkdir()
    with pytest.raises(ValueError, match="unsafe archive member"):
        preparation._range_download(unsafe_root, "mlvu_dev")
    assert not (tmp_path / "escape.bin").exists()
