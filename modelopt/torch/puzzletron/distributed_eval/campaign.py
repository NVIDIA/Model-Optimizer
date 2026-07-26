"""Campaign initialization, loading, and coordinator ownership."""

from __future__ import annotations

import fcntl
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path

from .registry import WorkerRegistry
from .schema import CampaignManifest, EvaluationRequest
from .storage import CampaignStorage


class Campaign:
    def __init__(self, root: str | Path, manifest: CampaignManifest):
        self.storage = CampaignStorage(root)
        self.manifest = manifest
        self.registry = WorkerRegistry(self.storage.registry_dir)

    @classmethod
    def create(cls, root: str | Path, manifest: CampaignManifest) -> "Campaign":
        campaign = cls(root, manifest)
        campaign.storage.create(manifest)
        return campaign

    @classmethod
    def open(cls, root: str | Path) -> "Campaign":
        storage = CampaignStorage(root)
        return cls(root, storage.load_manifest())

    @property
    def root(self) -> Path:
        return self.storage.root

    @property
    def campaign_id(self) -> str:
        return self.manifest.campaign_id

    def validate_request(self, request: EvaluationRequest) -> None:
        if request.campaign_id != self.campaign_id:
            raise ValueError(
                f"Request belongs to {request.campaign_id}, campaign is {self.campaign_id}"
            )
        if request.evaluator_revision != self.manifest.evaluator_revision:
            raise ValueError(
                f"Request evaluator revision {request.evaluator_revision!r} does not match "
                f"campaign revision {self.manifest.evaluator_revision!r}"
            )
        if request.model != self.manifest.model:
            raise ValueError("Request model identity does not match campaign model identity")
        if request.data != self.manifest.data:
            raise ValueError("Request data identity does not match campaign data identity")
        if request.metrics != self.manifest.metrics:
            raise ValueError("Request metric settings do not match campaign metric settings")
        if request.precision != self.manifest.precision:
            raise ValueError("Request precision settings do not match campaign precision settings")

    @contextmanager
    def coordinator_lease(self) -> Iterator[None]:
        self.storage.coordinator_lock_path.parent.mkdir(parents=True, exist_ok=True)
        with self.storage.coordinator_lock_path.open("a+") as stream:
            try:
                fcntl.flock(stream.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError as error:
                raise RuntimeError(
                    f"Another coordinator holds {self.storage.coordinator_lock_path}"
                ) from error
            try:
                yield
            finally:
                fcntl.flock(stream.fileno(), fcntl.LOCK_UN)
