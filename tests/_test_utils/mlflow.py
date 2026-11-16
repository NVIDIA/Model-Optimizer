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

"""Shared stand-ins for the MLflow client, used by every suite that exercises tracking.

One fake rather than one per suite: four copies drifted, and a recording method that was a
no-op in one of them made a test pass while asserting nothing.
"""

import getpass
from pathlib import Path
from types import SimpleNamespace

import pytest


class FakeMlflow:
    """Stand-in for the ``mlflow`` module: no server, no dependency, records every call."""

    def __init__(self):
        self.tracking_uri = None
        self.experiment = None
        self.run_name = None
        self.status = None
        self.params = {}
        self.tags = {}
        self.texts = {}
        self.metrics = {}
        # {uploaded name: (artifact path, contents)}
        self.artifacts = {}
        # What the server says the run is called, which need not be what was requested.
        self.server_run_name = None
        # The run id a resumed run re-attached to, if one did.
        self.resumed = None
        # Runs the fluent API opened by itself, which is always a bug in the caller.
        self.strays = 0
        self._run = None

    def set_tracking_uri(self, uri):
        self.tracking_uri = uri

    def set_experiment(self, name):
        self.experiment = name

    def start_run(self, run_name=None, tags=None, description=None, run_id=None):
        self.resumed = run_id
        if run_id is None:
            self.run_name = run_name
            # Replaced, not merged: a run starts with only the tags it was opened with, so an
            # earlier run's cannot satisfy an assertion about this one. set_tags adds to these.
            self.tags = dict(tags or {})
        self._run = SimpleNamespace(
            info=SimpleNamespace(
                experiment_id="7",
                run_id=run_id or "deadbeef",
                run_name=self.server_run_name or self.run_name,
                status="RUNNING",
            )
        )
        return self._run

    def active_run(self):
        return self._run

    def get_run(self, run_id):
        return SimpleNamespace(info=SimpleNamespace(run_id=run_id, status=self.status))

    def _get_or_start_run(self):
        """What every fluent call does first: with nothing active, it opens a run of its own."""
        if self._run is None:
            self.strays += 1
            self._run = SimpleNamespace(
                info=SimpleNamespace(experiment_id="7", run_id="stray", run_name=None)
            )

    def log_params(self, params):
        self._get_or_start_run()
        self.params.update(params)

    def set_tags(self, tags):
        self._get_or_start_run()
        self.tags.update(tags)

    def log_text(self, text, artifact_file):
        self._get_or_start_run()
        self.texts[artifact_file] = text

    def log_artifact(self, local_path, artifact_path=None):
        self._get_or_start_run()
        self.artifacts[Path(local_path).name] = (artifact_path, Path(local_path).read_text())

    def log_metrics(self, metrics):
        self._get_or_start_run()
        self.metrics.update(metrics)

    def end_run(self, status=None):
        self.status = status
        self._run = None


def pin_tracking_env(monkeypatch):
    """Pin what the tracking reads from the environment; see :func:`clean_env`."""
    monkeypatch.setattr(getpass, "getuser", lambda: "tester")
    for name in ("MLFLOW_TRACKING_URI", "MLFLOW_TRACKING_USERNAME", "MLFLOW_TRACKING_PASSWORD"):
        # setenv before delenv: monkeypatch records nothing for a variable that was already
        # absent, so a test whose code *sets* one would otherwise leave it behind for the
        # rest of the session.
        monkeypatch.setenv(name, "")
        monkeypatch.delenv(name)


@pytest.fixture(autouse=True)
def clean_env(monkeypatch):
    """Pin what the tracking reads from the environment.

    A developer shell that exports $MLFLOW_TRACKING_URI -- exactly the population this feature
    is built for -- would otherwise flip the branch under test. Tests that want it set it.
    """
    pin_tracking_env(monkeypatch)
