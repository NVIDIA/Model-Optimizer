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

from pathlib import Path

from examples.puzzletron.acceptance_resume import build_payload, check_marker, write_marker
from modelopt.torch.puzzletron.manifest import StageManifest, write_stage_manifest


def test_resume_marker_detects_missing_or_mutated_outputs_and_upstreams(tmp_path: Path) -> None:
    root = tmp_path / "campaign"
    output = root / "artifacts" / "summary.json"
    output.parent.mkdir(parents=True)
    output.write_text('{"value": 1}\n')
    config = tmp_path / "config.yaml"
    config.write_text("force_hf: false\n")
    upstream_manifest = root / "manifests" / "convert.json"
    manifest = StageManifest(stage="convert", config={"convert": {}})
    manifest.complete(outputs={"teacher": "ready"})
    write_stage_manifest(upstream_manifest, manifest)
    upstream = write_marker(
        root,
        "convert",
        build_payload(
            root=root,
            config=config,
            mode="convert",
            width=None,
            depth=None,
            required_patterns=("manifests/convert.json",),
            stage_config={"convert": {}},
            source_roots=(),
        ),
    )

    kwargs = {
        "root": root,
        "config": config,
        "mode": "activation",
        "width": None,
        "depth": None,
        "required_patterns": ("artifacts/*.json",),
        "upstream_markers": (upstream,),
    }
    payload = build_payload(**kwargs)
    marker = write_marker(root, "activation", payload)

    assert check_marker(marker, **kwargs)

    output.write_text('{"value": 2}\n')
    assert not check_marker(marker, **kwargs)

    write_marker(root, "activation", build_payload(**kwargs))
    output.unlink()
    assert not check_marker(marker, **kwargs)

    output.write_text('{"value": 2}\n')
    write_marker(root, "activation", build_payload(**kwargs))
    upstream_payload = upstream_manifest.read_text().replace(
        '"status": "success"', '"status": "failed"'
    )
    upstream_manifest.write_text(upstream_payload)
    assert not check_marker(marker, **kwargs)


def test_resume_payload_rejects_required_pattern_without_matches(tmp_path: Path) -> None:
    config = tmp_path / "config.yaml"
    config.write_text("force_hf: false\n")

    try:
        build_payload(
            root=tmp_path,
            config=config,
            mode="convert",
            width=None,
            depth=None,
            required_patterns=("ckpts/teacher/config.json",),
            upstream_markers=(),
        )
    except FileNotFoundError as error:
        assert "ckpts/teacher/config.json" in str(error)
    else:
        raise AssertionError("missing required artifacts must invalidate a completion marker")


def _valid_activation_marker(tmp_path: Path):
    root = tmp_path / "campaign"
    config = tmp_path / "config.yaml"
    config.write_text("activation: {}\n")
    manifest = StageManifest(stage="activation", config={"activation": {}})
    manifest.complete(outputs={"activation": "ready"})
    write_stage_manifest(root / "manifests/activation.json", manifest)
    kwargs = {
        "root": root,
        "config": config,
        "mode": "activation",
        "width": None,
        "depth": None,
        "required_patterns": ("manifests/activation.json",),
        "stage_config": {"activation": {}},
        "source_roots": (),
    }
    marker = write_marker(root, "activation", build_payload(**kwargs))
    return marker, kwargs


def test_completion_marker_symlink_is_rejected(tmp_path: Path) -> None:
    marker, kwargs = _valid_activation_marker(tmp_path)
    external = tmp_path / "external-completion.json"
    marker.rename(external)
    marker.symlink_to(external)

    assert not check_marker(marker, **kwargs)


def test_completion_marker_symlinked_ancestor_is_rejected(tmp_path: Path) -> None:
    marker, kwargs = _valid_activation_marker(tmp_path)
    completion_dir = marker.parent
    external = tmp_path / "external-completions"
    completion_dir.rename(external)
    completion_dir.symlink_to(external, target_is_directory=True)

    assert not check_marker(marker, **kwargs)
