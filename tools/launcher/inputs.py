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

"""Hugging Face implementation of the launcher's provider-neutral input protocol."""

import argparse
import contextlib
import hashlib
import io
import json
import os
import re
import shlex
import sys
from pathlib import Path
from urllib.parse import unquote, urlsplit

__all__ = ["prepare_inputs", "resolve_local"]


def resolve_local(request):
    """Resolve revisions before downloading, and return only exact cache snapshots."""
    if request.get("schema_version") != 2 or not isinstance(request.get("inputs"), dict):
        raise ValueError("Expected a schema_version 2 input request")
    root = Path(request["hf_local"])
    if not root.is_absolute():
        raise ValueError("Input cache must be an absolute path")
    root = root.resolve()
    plans = []
    model_plans = []
    resolved = {}
    api = None
    for mount, declaration in request["inputs"].items():
        ref = declaration["ref"]
        model = re.fullmatch(r"models:/([\w./-]+)/(\d+)", ref)
        if model:
            # Registry support and filesystem copying are optional until requested.
            from mlflow.tracking import MlflowClient

            if not request.get("tracking_uri"):
                raise ValueError("Model references require the selected tracking URI")
            version = MlflowClient(tracking_uri=request["tracking_uri"]).get_model_version(
                model[1], model[2]
            )
            source = urlsplit(version.tags.get("modelopt.artifact_uri", version.source))
            if source.scheme != "file" or source.netloc not in ("", "localhost"):
                raise ValueError("Configure a site resolver for this model's artifact URI")
            model_plans.append((mount, ref, version, Path(unquote(source.path)), model[2]))
            continue
        # This optional dependency is needed on the host holding the HF input cache.
        from huggingface_hub import HfApi, snapshot_download

        if api is None:
            api = HfApi(token=os.environ.get("HF_TOKEN"))
        name, _, revision = declaration["ref"].partition("@")
        if not re.fullmatch(r"[\w.-]+/[\w.-]+", name) or any(
            part in (".", "..") for part in name.split("/")
        ):
            raise ValueError("Use org/name or org/name@revision")
        kind = declaration.get("kind", "model")
        if kind not in ("model", "dataset"):
            raise ValueError("Input kind must be model or dataset")
        info = api.repo_info(name, repo_type=kind, revision=revision or "main")
        if not re.fullmatch(r"[a-f0-9]{40}", info.sha):
            raise ValueError("Hugging Face did not return an exact commit")
        plans.append((mount, declaration, name, kind, info.sha))
    for mount, ref, version, source, number in model_plans:
        resolved[mount] = _model_snapshot(root, mount, ref, version, source, number)
    for mount, declaration, name, kind, sha in plans:
        path = Path(
            snapshot_download(
                name,
                repo_type=kind,
                revision=sha,
                # A separately mounted input cannot follow HF cache links into sibling blobs.
                local_dir=str(root / ".modelopt-hf" / kind / name / sha),
                token=os.environ.get("HF_TOKEN"),
            )
        ).resolve()
        if not path.is_dir() or not path.is_relative_to(root):
            raise ValueError("Snapshot must be inside the selected cache root")
        if any(item.is_symlink() for item in path.rglob("*")):
            raise ValueError("Input snapshot must be self-contained, without cache symlinks")
        resolved[mount] = {
            "mount": mount,
            "ref": declaration["ref"],
            "version": sha,
            "path": str(path),
            "backend": "huggingface",
            "uri": f"https://huggingface.co/{'datasets/' if kind == 'dataset' else ''}{name}/tree/{sha}",
        }
    return {
        "schema_version": 2,
        "status": "ready",
        "inputs": [resolved[mount] for mount in request["inputs"]],
    }


def _model_snapshot(root, mount, ref, version, source, number):
    # The publication helper is optional until a registry input is selected.
    from modelopt_launcher.common.workflow_artifacts import _inventory, publish_local

    digest = hashlib.sha256(json.dumps(_inventory(source), sort_keys=True).encode()).hexdigest()
    if version.tags.get("modelopt.artifact_digest", digest) != digest:
        raise ValueError("Registered model's artifact bytes changed")
    frozen = publish_local(
        source,
        {"ref": "models/" + hashlib.sha256(ref.encode()).hexdigest(), "version": number},
        root / ".modelopt-models",
    )
    return {
        "mount": mount,
        "ref": ref,
        "version": str(version.version),
        "backend": "filesystem",
        "path": unquote(urlsplit(frozen["uri"]).path),
        "model_uri": ref,
        "model_id": version.model_id or version.tags.get("modelopt.logged_model_id"),
        "run_id": version.run_id,
        "uri": version.tags.get("modelopt.artifact_uri", version.source),
    }


def prepare_inputs(request, *, user=None, identity=None):
    """Run on the cache host through NeMo's transport; pass HF credentials on stdin."""
    if request.get("host") in (None, "localhost", "127.0.0.1"):
        return resolve_local(request)
    # Only remote preparation needs the launcher transport and its optional MFA adapter.
    import nemo_run as run
    from modelopt_launcher.core import ControlMasterSSHTunnel

    if not user or not identity:
        raise ValueError("Remote input preparation requires profile user and identity")
    tunnel_type = (
        ControlMasterSSHTunnel if os.getenv("MODELOPT_LAUNCHER_SSH_CONTROL_PATH") else run.SSHTunnel
    )
    tunnel = tunnel_type(
        host=request["host"], user=user, identity=identity, job_dir=request["hf_local"]
    )
    bootstrap = (
        "import json,os,sys; p=json.load(sys.stdin); "
        "os.environ['HF_TOKEN']=p.pop('token'); "
        "ns={'__name__':'modelopt_input_worker'}; "
        "exec(compile(p.pop('code'),'modelopt_input_worker','exec'),ns); "
        "print(json.dumps(ns['resolve_local'](p['request'])))"
    )
    payload = {
        "request": request,
        "token": os.environ.get("HF_TOKEN", ""),
        "code": Path(__file__).read_text(),
    }
    try:
        tunnel.connect()
        result = tunnel.session.run(
            "python3 -c " + shlex.quote(bootstrap),
            in_stream=io.StringIO(json.dumps(payload)),
            hide=True,
            warn=True,
        )
        if result.exited:
            raise RuntimeError(
                "Input preparation failed on cache host; verify huggingface_hub setup"
            )
        return json.loads(result.stdout)
    finally:
        if tunnel.session:
            tunnel.session.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--user")
    parser.add_argument("--identity")
    args = parser.parse_args()
    try:
        with contextlib.redirect_stdout(sys.stderr):
            result = prepare_inputs(json.load(sys.stdin), user=args.user, identity=args.identity)
    except Exception as exc:
        result = {"schema_version": 2, "status": "blocked", "error": type(exc).__name__}
    print(json.dumps(result))
    return 0 if result["status"] == "ready" else 1


if __name__ == "__main__":
    raise SystemExit(main())
