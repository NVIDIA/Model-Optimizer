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
import os
import stat
import tarfile
import threading
import zipfile
from pathlib import Path
from types import SimpleNamespace

import pytest

from examples.puzzletron.evaluation.vlm.preparation import benchmark_data as preparation

_EXPECTED_DATASETS = {
    "realworldqa": (
        "lmms-lab/RealWorldQA",
        "907c4e5228fd1703c710ed937601cb5f89ab8d5c",
        None,
    ),
    "mmmu_val": (
        "lmms-lab/MMMU",
        "364f2e2eb107b36e07ff4c5a15f5947a759cef47",
        None,
    ),
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


def _emulate_atomic_exchange(first: Path, second: Path) -> bool:
    displaced = second.with_name(f".{second.name}.test-exchange")
    second.rename(displaced)
    first.rename(second)
    displaced.rename(first)
    return True


def _rewrite_with_distinct_mtime(path: Path, payload: bytes) -> None:
    """Make metadata-based invalidation deterministic on coarse-clock filesystems."""
    before = path.stat()
    path.write_bytes(payload)
    after = path.stat()
    if after.st_mtime_ns == before.st_mtime_ns:
        os.utime(path, ns=(after.st_atime_ns, before.st_mtime_ns + 1_000_000_000))


def test_atomic_exchange_directories_when_supported(tmp_path):
    first = tmp_path / "first"
    second = tmp_path / "second"
    first.mkdir()
    second.mkdir()
    (first / "identity").write_text("first")
    (second / "identity").write_text("second")

    if not preparation._atomic_exchange_directories(first, second):
        pytest.skip("atomic directory exchange is unavailable on this host")

    assert (first / "identity").read_text() == "second"
    assert (second / "identity").read_text() == "first"


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
    assert preparation.benchmark_catalog_contract((task,))[task] == {
        "repository": repository,
        "revision": revision,
        "requires_media": directory is not None,
        "preparation_dir": directory,
    }


def test_prepare_benchmark_datasets_dispatches_media_only_for_media_tasks(tmp_path, monkeypatch):
    hf_home = tmp_path / "hf-home"
    prepared = []

    def download(root, task, *, max_workers):
        del max_workers
        snapshot = preparation._hub_snapshot(root, task)
        snapshot.mkdir(parents=True)
        (snapshot / "dataset-info.json").write_text("{}")
        return snapshot

    def prepare(root, task, snapshot, *, verify_content=False):
        assert not verify_content
        prepared.append(task)
        media_root = root / preparation.DATASETS[task].preparation_dir
        media_root.mkdir(parents=True)
        (media_root / "sample.mp4").write_bytes(b"video")
        payload = {
            **preparation._marker_payload(task, status="complete"),
            "snapshot": str(snapshot),
            "media_root": str(media_root),
            "files": 1,
            "bytes": 5,
        }
        preparation._write_marker(media_root, payload)
        return payload

    monkeypatch.setattr(preparation, "_download", download)
    monkeypatch.setattr(preparation, "_prepare", prepare)

    reports = preparation.prepare_benchmark_datasets(
        hf_home, ["realworldqa", "mmmu_val", "mvbench"], max_workers=3
    )

    assert [report["task"] for report in reports] == ["realworldqa", "mmmu_val", "mvbench"]
    assert [report["requires_media"] for report in reports] == [False, False, True]
    assert prepared == ["mvbench"]


def test_prepare_benchmark_datasets_rejects_symlinked_hf_home(tmp_path):
    target = tmp_path / "target"
    target.mkdir()
    alias = tmp_path / "hf-home"
    alias.symlink_to(target, target_is_directory=True)

    with pytest.raises(ValueError, match="must not be a symlink"):
        preparation.prepare_benchmark_datasets(alias, ("realworldqa",))


def test_prepare_benchmark_datasets_rejects_dangling_symlinked_hf_home(tmp_path):
    alias = tmp_path / "hf-home"
    alias.symlink_to(tmp_path / "missing", target_is_directory=True)

    with pytest.raises(ValueError, match="must not be a symlink"):
        preparation.prepare_benchmark_datasets(alias, ("realworldqa",))


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
    assert stat.S_IMODE((hf_home / "mmvu").stat().st_mode) == 0o755
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


def test_prepared_media_reuses_metadata_unless_content_verification_is_requested(
    tmp_path, monkeypatch
):
    hf_home = tmp_path / "hf-home"
    snapshot = preparation._hub_snapshot(hf_home, "mmvu_val")
    snapshot.mkdir(parents=True)
    _write_zip(snapshot / "videos.zip", {"videos/sample.mp4": b"video"})
    expected = preparation._prepare(hf_home, "mmvu_val", snapshot)
    original_sha256 = preparation._sha256
    hashed = []

    def record_hash(path):
        hashed.append(path)
        return original_sha256(path)

    monkeypatch.setattr(preparation, "_sha256", record_hash)
    assert preparation._prepare(hf_home, "mmvu_val", snapshot) == expected
    assert not hashed

    assert preparation._prepare(hf_home, "mmvu_val", snapshot, verify_content=True) == expected
    assert hashed == [hf_home / "mmvu/videos/sample.mp4"]


def test_prepared_media_reuse_tolerates_distributed_filesystem_mtime_skew(tmp_path):
    root = tmp_path / "prepared"
    root.mkdir()
    (root / "sample.mp4").write_bytes(b"video")
    inventory = preparation._inventory(root)
    inventory[0]["mtime_ns"] += 1_000_000_000

    assert preparation._inventory_is_current(root, inventory)


def test_prepared_media_reuse_reports_paths_from_the_current_mount(tmp_path):
    hf_home = tmp_path / "hf-home"
    snapshot = preparation._hub_snapshot(hf_home, "mmvu_val")
    snapshot.mkdir(parents=True)
    _write_zip(snapshot / "videos.zip", {"videos/sample.mp4": b"video"})
    preparation._prepare(hf_home, "mmvu_val", snapshot)
    marker = hf_home / "mmvu" / preparation._MARKER_NAME
    payload = json.loads(marker.read_text())
    payload["snapshot"] = "/stale-mount/snapshot"
    payload["media_root"] = "/stale-mount/media"
    marker.write_text(json.dumps(payload))

    reused = preparation._prepare(hf_home, "mmvu_val", snapshot)

    assert reused["snapshot"] == str(snapshot)
    assert reused["media_root"] == str(hf_home / "mmvu")


@pytest.mark.parametrize("damage", ["missing", "corrupt", "unexpected"])
def test_complete_media_marker_repairs_owned_root_from_pinned_snapshot(
    tmp_path, monkeypatch, damage
):
    hf_home = tmp_path / "hf-home"
    snapshot = preparation._hub_snapshot(hf_home, "mmvu_val")
    snapshot.mkdir(parents=True)
    _write_zip(snapshot / "videos.zip", {"videos/sample.mp4": b"video"})
    preparation._prepare(hf_home, "mmvu_val", snapshot)
    monkeypatch.setattr(preparation, "_atomic_exchange_directories", _emulate_atomic_exchange)
    target = hf_home / "mmvu"
    media = target / "videos/sample.mp4"

    if damage == "missing":
        media.unlink()
    elif damage == "corrupt":
        _rewrite_with_distinct_mtime(media, b"wrong")
    elif damage == "unexpected":
        (target / "unexpected.bin").write_bytes(b"stale")
    report = preparation._prepare(hf_home, "mmvu_val", snapshot)

    assert report["status"] == "complete"
    assert media.read_bytes() == b"video"
    assert not (target / "unexpected.bin").exists()
    assert preparation._media_marker_is_current(
        target,
        "mmvu_val",
        json.loads((target / preparation._MARKER_NAME).read_text()),
    )


def test_repair_without_atomic_exchange_preserves_live_root(monkeypatch, tmp_path):
    hf_home = tmp_path / "hf-home"
    snapshot = preparation._hub_snapshot(hf_home, "mmvu_val")
    snapshot.mkdir(parents=True)
    _write_zip(snapshot / "videos.zip", {"videos/sample.mp4": b"video"})
    preparation._prepare(hf_home, "mmvu_val", snapshot)
    target = hf_home / "mmvu"
    media = target / "videos/sample.mp4"
    _rewrite_with_distinct_mtime(media, b"wrong")
    monkeypatch.setattr(preparation, "_atomic_exchange_directories", lambda *_args: False)

    with pytest.raises(RuntimeError, match="atomic media-directory exchange is unavailable"):
        preparation._prepare(hf_home, "mmvu_val", snapshot)

    assert target.is_dir()
    assert media.read_bytes() == b"wrong"
    assert not tuple(target.parent.glob(f".{target.name}.modelopt-staging.*"))
    assert not tuple(target.parent.glob(f".{target.name}.modelopt-replaced.*"))


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        (None, None, "readable ownership marker"),
        ("corrupt", "{", "readable ownership marker"),
        ("revision", "other-revision", "mismatched ownership: revision"),
        ("status", "unknown", "invalid ownership-marker status"),
    ],
)
def test_missing_or_mismatched_media_marker_preserves_unproven_root(
    tmp_path, field, value, message
):
    hf_home = tmp_path / "hf-home"
    snapshot = preparation._hub_snapshot(hf_home, "mmvu_val")
    snapshot.mkdir(parents=True)
    _write_zip(snapshot / "videos.zip", {"videos/sample.mp4": b"video"})
    preparation._prepare(hf_home, "mmvu_val", snapshot)
    target = hf_home / "mmvu"
    media = target / "videos/sample.mp4"
    marker = target / preparation._MARKER_NAME
    if field is None:
        marker.unlink()
    elif field == "corrupt":
        marker.write_text(value)
    else:
        payload = json.loads(marker.read_text())
        payload[field] = value
        marker.write_text(json.dumps(payload))

    with pytest.raises((FileExistsError, ValueError), match=message):
        preparation._prepare(hf_home, "mmvu_val", snapshot)

    assert target.is_dir()
    assert media.read_bytes() == b"video"


def test_media_repair_rejects_symlinks_in_owned_root(tmp_path):
    hf_home = tmp_path / "hf-home"
    snapshot = preparation._hub_snapshot(hf_home, "mmvu_val")
    snapshot.mkdir(parents=True)
    _write_zip(snapshot / "videos.zip", {"videos/sample.mp4": b"video"})
    preparation._prepare(hf_home, "mmvu_val", snapshot)
    target = hf_home / "mmvu"
    (target / "videos/sample.mp4").unlink()
    (target / "videos/sample.mp4").symlink_to(tmp_path / "outside")

    with pytest.raises(ValueError, match="repair refuses a symlink"):
        preparation._prepare(hf_home, "mmvu_val", snapshot)


def test_snapshot_inventory_rejects_partial_and_same_size_corruption(tmp_path):
    hf_home = tmp_path / "hf-home"
    snapshot = preparation._hub_snapshot(hf_home, "realworldqa")
    snapshot.mkdir(parents=True)
    first = snapshot / "first.json"
    second = snapshot / "second.json"
    first.write_bytes(b"one")
    second.write_bytes(b"two")
    report = preparation._snapshot_inventory_report(hf_home, "realworldqa", snapshot)

    assert preparation._snapshot_inventory_is_current(report)
    second.unlink()
    assert not preparation._snapshot_inventory_is_current(report)
    second.write_bytes(b"two")
    report = preparation._snapshot_inventory_report(hf_home, "realworldqa", snapshot)
    assert preparation._snapshot_inventory_is_current(report)
    _rewrite_with_distinct_mtime(first, b"bad")
    assert not preparation._snapshot_inventory_is_current(report)


def test_snapshot_inventory_reuses_metadata_unless_content_verification_is_requested(
    tmp_path, monkeypatch
):
    hf_home = tmp_path / "hf-home"
    snapshot = preparation._hub_snapshot(hf_home, "realworldqa")
    snapshot.mkdir(parents=True)
    sample = snapshot / "sample.json"
    sample.write_bytes(b"one")
    expected = preparation._snapshot_inventory_report(hf_home, "realworldqa", snapshot)
    original_sha256 = preparation._sha256
    hashed = []

    def record_hash(path):
        hashed.append(path)
        return original_sha256(path)

    monkeypatch.setattr(preparation, "_sha256", record_hash)
    assert preparation._snapshot_inventory_report(hf_home, "realworldqa", snapshot) == expected
    assert not hashed

    assert (
        preparation._snapshot_inventory_report(
            hf_home, "realworldqa", snapshot, verify_content=True
        )
        == expected
    )
    assert hashed == [sample]

    hashed.clear()
    _rewrite_with_distinct_mtime(sample, b"two")
    refreshed = preparation._snapshot_inventory_report(hf_home, "realworldqa", snapshot)
    assert hashed == [sample]
    assert refreshed["files"][0]["sha256"] == hashlib.sha256(b"two").hexdigest()


def test_snapshot_inventory_seals_and_validates_hub_blob_symlink(tmp_path):
    hf_home = tmp_path / "hf-home"
    snapshot = preparation._hub_snapshot(hf_home, "realworldqa")
    snapshot.mkdir(parents=True)
    payload = b"pinned blob"
    blob_sha256 = hashlib.sha256(payload).hexdigest()
    blob = snapshot.parent.parent / "blobs" / blob_sha256
    blob.parent.mkdir()
    blob.write_bytes(payload)
    (snapshot / "dataset.parquet").symlink_to(Path("../../blobs") / blob_sha256)

    report = preparation._snapshot_inventory_report(hf_home, "realworldqa", snapshot)

    assert preparation._snapshot_inventory_is_current(report)
    blob.write_bytes(b"broken blob")
    assert not preparation._snapshot_inventory_is_current(report)


def test_snapshot_inventory_rejects_blob_whose_content_differs_from_sha_name(tmp_path):
    hf_home = tmp_path / "hf-home"
    snapshot = preparation._hub_snapshot(hf_home, "realworldqa")
    snapshot.mkdir(parents=True)
    blob = snapshot.parent.parent / "blobs" / ("0" * 64)
    blob.parent.mkdir()
    blob.write_bytes(b"not the named content")
    (snapshot / "dataset.parquet").symlink_to(Path("../../blobs") / blob.name)

    with pytest.raises(ValueError, match="differs from its SHA-256 identity"):
        preparation._snapshot_inventory_report(hf_home, "realworldqa", snapshot)


def test_prepare_benchmark_datasets_rejects_catalog_pin_drift(tmp_path):
    catalog = preparation.benchmark_catalog_contract(("realworldqa",))
    catalog["realworldqa"]["revision"] = "stale"

    with pytest.raises(ValueError, match="differs from the authoritative catalog"):
        preparation.prepare_benchmark_datasets(
            tmp_path / "hf-home", ("realworldqa",), expected_catalog=catalog
        )


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
    snapshot = preparation._hub_snapshot(hf_home, "mmvu_val")
    snapshot.mkdir(parents=True)
    _write_zip(snapshot / "videos.zip", {"videos/sample.mp4": b"video"})
    target = hf_home / preparation.DATASETS["mmvu_val"].preparation_dir
    extract = preparation._extract

    def interrupt(*_args):
        raise RuntimeError("interrupted")

    monkeypatch.setattr(preparation, "_extract", interrupt)
    with pytest.raises(RuntimeError, match="interrupted"):
        preparation._prepare(hf_home, "mmvu_val", snapshot)
    assert not target.exists()
    assert not tuple(target.parent.glob(f".{target.name}.modelopt-staging.*"))

    monkeypatch.setattr(preparation, "_extract", extract)
    report = preparation._prepare(hf_home, "mmvu_val", snapshot)
    assert report["status"] == "complete"
    assert (target / "videos/sample.mp4").read_bytes() == b"video"


def test_task_lock_can_be_reacquired_after_body_failure(tmp_path):
    hf_home = tmp_path / "hf-home"
    hf_home.mkdir()

    with (
        pytest.raises(RuntimeError, match="failed while locked"),
        preparation._task_lock(hf_home, "mmvu_val"),
    ):
        raise RuntimeError("failed while locked")

    with preparation._task_lock(hf_home, "mmvu_val"):
        pass


def test_concurrent_media_preparation_is_task_locked_and_publishes_only_complete_root(
    monkeypatch, tmp_path
):
    hf_home = tmp_path / "hf-home"
    snapshot = preparation._hub_snapshot(hf_home, "mmvu_val")
    snapshot.mkdir(parents=True)
    target = hf_home / "mmvu"
    entered = threading.Event()
    release = threading.Event()
    second_done = threading.Event()
    extraction_count = 0
    results = []
    errors = []

    def extract(_task, _snapshot, staging):
        nonlocal extraction_count
        extraction_count += 1
        (staging / "videos").mkdir()
        (staging / "videos/sample.mp4").write_bytes(b"video")
        entered.set()
        assert release.wait(timeout=5)
        return []

    def run(*, second=False):
        try:
            results.append(preparation._prepare(hf_home, "mmvu_val", snapshot))
        except BaseException as error:  # pragma: no cover - asserted below
            errors.append(error)
        finally:
            if second:
                second_done.set()

    monkeypatch.setattr(preparation, "_extract", extract)
    first = threading.Thread(target=run)
    first.start()
    assert entered.wait(timeout=5)
    second = threading.Thread(target=run, kwargs={"second": True})
    second.start()

    assert not second_done.wait(timeout=0.1)
    assert not target.exists()
    release.set()
    first.join(timeout=5)
    second.join(timeout=5)

    assert not errors
    assert not first.is_alive() and not second.is_alive()
    assert extraction_count == 1
    assert len(results) == 2
    assert all(result["status"] == "complete" for result in results)
    assert (target / "videos/sample.mp4").read_bytes() == b"video"


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
