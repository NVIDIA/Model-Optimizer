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

"""Run a workflow task with explicit artifact publication and MLflow model lineage."""

import hashlib
import json
import os
import shutil
import subprocess  # nosec B404
import sys
import tempfile
import time
from pathlib import Path

__all__ = ["publish_local", "run"]


def _save(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, name = tempfile.mkstemp(dir=path.parent)
    try:
        with os.fdopen(descriptor, "w") as stream:
            json.dump(value, stream, sort_keys=True, indent=2)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(name, path)
    finally:
        Path(name).unlink(missing_ok=True)


def _inventory(source):
    if source.is_symlink():
        raise ValueError("Publish a regular file or directory snapshot")
    paths = sorted(source.rglob("*")) if source.is_dir() else [source]
    files = {}
    for path in paths:
        if path.is_symlink():
            raise ValueError("Published outputs must contain regular files, without symlinks")
        if path.is_file():
            digest = hashlib.sha256()
            with path.open("rb") as stream:
                for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                    digest.update(chunk)
            files[str(path.relative_to(source) if source.is_dir() else ".")] = digest.hexdigest()
    if not files:
        raise ValueError("Publication requires a nonempty file or directory")
    return files


def publish_local(source, declaration, store):
    """Copy to a versioned filesystem store; refuse conflicts and verify reused bytes."""
    root = Path(store).resolve()
    destination = root / declaration["ref"] / declaration["version"]
    if not destination.resolve().is_relative_to(root):
        raise ValueError("Publication destination escapes artifact_store")
    files = _inventory(source)
    result = {
        "ref": declaration["ref"],
        "version": declaration["version"],
        "uri": (destination / "data").as_uri(),
        "backend": "filesystem",
        "files": files,
        "verified": True,
        "metadata": declaration.get("metadata", {}),
    }
    if destination.exists():
        if json.loads((destination / "manifest.json").read_text()) != result:
            raise ValueError("Artifact version already exists with different contents")
        if _inventory(destination / "data") != files:
            raise ValueError("Published artifact contents changed")
        return result
    destination.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=destination.parent) as temporary:
        staged = Path(temporary) / "artifact"
        staged.mkdir()
        if source.is_dir():
            shutil.copytree(source, staged / "data")
        else:
            shutil.copyfile(source, staged / "data")
        if _inventory(staged / "data") != files:
            raise ValueError("Source changed during publication")
        _save(staged / "manifest.json", result)
        staged.rename(destination)
    return result


def _publish(source, declaration, request, run_id):
    if not request.get("publisher"):
        return publish_local(source, declaration, request["artifact_store"])
    payload = {
        "schema_version": 1,
        "source_path": str(source),
        **declaration,
        "run_id": run_id,
        "task": request["task"],
        "execution_id": os.environ.get("MODELOPT_WORKFLOW_ID")
        or Path(request["workspace"]).parent.name,
    }
    # Execute the explicitly configured publisher argv without a shell.
    result = subprocess.run(  # nosec B603
        request["publisher"],
        input=json.dumps(payload),
        text=True,
        capture_output=True,
        timeout=3600,
        check=False,
    )
    if result.returncode:
        raise RuntimeError("Artifact publisher failed; inspect its saved publication receipt")
    published = json.loads(result.stdout)
    if (
        published.get("status") != "ready"
        or published.get("verified") is not True
        or published.get("ref") != declaration["ref"]
        or published.get("version") != declaration["version"]
        or not published.get("uri")
    ):
        raise ValueError("Publisher did not return the exact verified artifact identity")
    return published


def _register(client, run_id, key, artifact, name, state, save):
    # Import only for explicitly enabled MLflow tracking.
    from mlflow.entities import LoggedModelOutput
    from mlflow.exceptions import MlflowException

    entry = state.setdefault("models", {}).setdefault(key, {})
    if not entry.get("model_id"):
        if entry.get("creating_logged_model"):
            raise RuntimeError("Logged model creation is uncertain; reconcile this task receipt")
        entry["creating_logged_model"] = True
        save()
        run = client.get_run(run_id)
        model = client.create_logged_model(
            experiment_id=run.info.experiment_id,
            name=key.replace(".", "-"),
            source_run_id=run_id,
            model_type="checkpoint",
            tags={"modelopt.artifact_uri": artifact["uri"]},
        )
        entry["model_id"] = model.model_id
        save()
    identifier = entry["model_id"]
    if not entry.get("model_uri"):
        # Registry servers validate source schemes. Keep external bytes in their store
        # and register a small, readable manifest in MLflow's supported artifact store.
        with tempfile.TemporaryDirectory() as temporary:
            manifest = Path(temporary) / "artifact.json"
            _save(manifest, {"schema_version": 1, **artifact})
            client.log_model_artifact(identifier, str(manifest))
        model_source = client.get_logged_model(identifier).artifact_location
        versions = client.search_model_versions(f"name = '{name}' AND run_id = '{run_id}'")
        versions = [version for version in versions if version.tags.get("modelopt.output") == key]
        if len(versions) > 1:
            raise ValueError("Multiple model versions match the same task output")
        if versions:
            version = versions[0]
        else:
            if entry.get("creating_version"):
                raise RuntimeError("Model version creation is uncertain; reconcile before retry")
            try:
                client.create_registered_model(name)
            except MlflowException as exc:
                if exc.error_code != "RESOURCE_ALREADY_EXISTS":
                    raise
            entry["creating_version"] = True
            save()
            version = client.create_model_version(
                name,
                source=model_source,
                run_id=run_id,
                model_id=identifier,
                tags={
                    "modelopt.output": key,
                    "modelopt.artifact_version": artifact["version"],
                    "modelopt.artifact_uri": artifact["uri"],
                    "modelopt.logged_model_id": identifier,
                    **(
                        {
                            "modelopt.artifact_digest": hashlib.sha256(
                                json.dumps(artifact["files"], sort_keys=True).encode()
                            ).hexdigest()
                        }
                        if "files" in artifact
                        else {}
                    ),
                },
            )
        entry["model_uri"] = f"models:/{name}/{version.version}"
        save()
    client.finalize_logged_model(identifier, "READY")
    client.log_outputs(run_id, [LoggedModelOutput(identifier, step=0)])
    return {**artifact, "model_uri": entry["model_uri"], "model_id": identifier, "run_id": run_id}


def _wait(path, failure, timeout=3600):
    deadline = time.monotonic() + timeout
    while not path.exists():
        if failure.exists() or time.monotonic() >= deadline:
            raise RuntimeError("Artifact rank barrier failed")
        time.sleep(0.2)
    return json.loads(path.read_text())


def _payload(request, environment):
    root = Path(request["workspace"]) / ".artifacts"
    task = request["task"]
    rank = int(os.getenv("SLURM_PROCID", "0"))
    context = root / (task + ".context.json")
    failure = root / (task + ".failed.json")
    if rank == 0:
        _save(
            context,
            {
                key: value
                for key, value in environment.items()
                if key
                in {
                    "MLFLOW_TRACKING_URI",
                    "MLFLOW_RUN_ID",
                    "MLFLOW_EXPERIMENT_ID",
                    "MODELOPT_MODEL_URI",
                    "MODELOPT_INPUTS_MANIFEST",
                }
            },
        )
    else:
        environment.update(_wait(context, failure))
    # Executing the workflow author's command is the launcher's intended interface.
    code = subprocess.run(  # nosec B603, B607
        ["bash", "-e", "-c", request["command"]], env=environment, check=False
    ).returncode
    _save(root / f"{task}.rank-{rank}.json", {"exit_code": code})
    if rank == 0:
        for index in range(request.get("workers", 1)):
            value = _wait(root / f"{task}.rank-{index}.json", failure)
            if value["exit_code"]:
                raise RuntimeError("Task rank failed")
    else:
        _wait(root / (task + ".json"), failure)
    return code


def _leader(request):
    """Publish before task success; bind exact producer identities to each consumer run."""
    workspace = Path(request["workspace"])
    task = request["task"]
    root = workspace / ".artifacts"
    manifest_path = root / (task + ".json")
    state_path = root / (task + ".state.json")
    state = json.loads(state_path.read_text()) if state_path.exists() else {}
    if state.get("finished"):
        raise ValueError("Task already finished; use its receipt rather than running it again")
    inputs = list(request.get("assets", []))
    for mount, declaration in request.get("inputs", {}).items():
        producer, output = declaration["from"].split(".")
        producer_manifest = root / (producer + ".json")
        artifact = json.loads(producer_manifest.read_text())["outputs"][output]
        inputs.append({**artifact, "mount": mount, "role": declaration.get("role", output)})
    _save(root / (task + ".inputs.json"), inputs)
    environment = dict(os.environ, MODELOPT_INPUTS_MANIFEST=str(root / (task + ".inputs.json")))
    models = [item for item in inputs if item.get("model_uri")]
    if len(models) == 1:
        environment["MODELOPT_MODEL_URI"] = models[0]["model_uri"]
    client = None
    run_id = state.get("run_id")

    def save():
        _save(state_path, state)

    if request.get("tracking_uri"):
        from mlflow.entities import LoggedModelInput
        from mlflow.tracking import MlflowClient

        client = MlflowClient(tracking_uri=request["tracking_uri"])
        if not run_id:
            if state.get("creating_run"):
                raise RuntimeError("Task run creation is uncertain; reconcile its receipt")
            state["creating_run"] = True
            save()
            experiment_id = os.environ.get("MODELOPT_EXPERIMENT_ID")
            if not experiment_id:
                experiment_name = os.environ.get("MLFLOW_EXPERIMENT_NAME", "ModelOpt workflows")
                experiment = client.get_experiment_by_name(experiment_name)
                experiment_id = (
                    experiment.experiment_id
                    if experiment
                    else client.create_experiment(experiment_name)
                )
            tags = {"mlflow.runName": task}
            if os.environ.get("MODELOPT_PARENT_RUN_ID"):
                tags["mlflow.parentRunId"] = os.environ["MODELOPT_PARENT_RUN_ID"]
            created = client.create_run(experiment_id, tags=tags)
            run_id = state["run_id"] = created.info.run_id
            save()
        client.log_inputs(
            run_id,
            models=[LoggedModelInput(item["model_id"]) for item in models if item.get("model_id")],
        )
        for index, item in enumerate(models):
            client.set_tag(run_id, f"modelopt.input.{index}.model_uri", item["model_uri"])
        environment.update(
            MLFLOW_TRACKING_URI=request["tracking_uri"],
            MLFLOW_RUN_ID=run_id,
            MLFLOW_EXPERIMENT_ID=client.get_run(run_id).info.experiment_id,
        )
    try:
        if not state.get("payload_completed"):
            code = _payload(request, environment)
            if code:
                raise RuntimeError(f"Task payload failed with exit code {code}")
            state["payload_completed"] = True
            save()
        outputs = {}
        for name, declaration in request.get("outputs", {}).items():
            source = workspace / "outputs" / task / declaration["path"]
            if not source.exists() or not source.resolve().is_relative_to(workspace.resolve()):
                raise ValueError("Declared output is missing or escapes the workspace")
            artifact = {"path": str(source)}
            if "publish" in declaration:
                publication = declaration["publish"]
                artifact = _publish(source, publication, request, run_id)
                if publication.get("model_name"):
                    artifact = _register(
                        client,
                        run_id,
                        f"{task}.{name}",
                        artifact,
                        publication["model_name"],
                        state,
                        save,
                    )
            outputs[name] = artifact
        _save(
            manifest_path,
            {
                "schema_version": 1,
                "task": task,
                "run_id": run_id,
                "inputs": inputs,
                "outputs": outputs,
            },
        )
        state["finished"] = True
        save()
    except BaseException:
        if client:
            client.set_terminated(run_id, "FAILED")
        raise
    if client:
        client.set_terminated(run_id, "FINISHED")
    return 0


def run(request):
    """Publish once on rank zero after all payload ranks succeed."""
    try:
        if int(os.getenv("SLURM_PROCID", "0")):
            return _payload(request, dict(os.environ))
        return _leader(request)
    except BaseException:
        _save(
            Path(request["workspace"]) / ".artifacts" / (request["task"] + ".failed.json"),
            {"failed": True},
        )
        raise


if __name__ == "__main__":
    try:
        raise SystemExit(run(json.loads(sys.argv[1])))
    except Exception as error:
        # Publisher/HTTP exception text can contain credentials; receipts retain execution IDs.
        print(
            f"Workflow artifact step failed ({type(error).__name__}); inspect its receipt",
            file=sys.stderr,
        )
        raise SystemExit(1) from None
