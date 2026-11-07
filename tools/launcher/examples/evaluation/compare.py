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

"""Compare two staged checkpoints with lm-eval and the native AIPerf client."""

import argparse
import csv
import hashlib
import json
import os
import shlex
import signal
import subprocess
import sys
import tempfile
import time
import urllib.error
import urllib.request
from datetime import UTC, datetime
from pathlib import Path

TOKENIZER_FILES = (
    "config.json",
    "tokenizer.json",
    "tokenizer_config.json",
    "vocab.json",
    "merges.txt",
)
TOKENIZER_CHECK = """
import importlib.metadata, json, sys
from pathlib import Path
from aiperf.common.tokenizer import Tokenizer
import lm_eval
records = json.loads(Path(sys.argv[1]).read_text())
for record in records.values():
    tok = Tokenizer.from_pretrained(record['repository'], revision=record['revision'], resolve_alias=False)
    for text in ['Quantization uses four-bit weights.', 'Hello, 你好.\\nTest 123.']:
        ids = tok.encode(text)
        assert ids and tok.decode(ids) == text, 'Tokenizer round trip failed'
print(json.dumps({
    'status': 'passed',
    'versions': {p: importlib.metadata.version(p) for p in
                 ('aiperf', 'lm_eval', 'transformers', 'tokenizers', 'datasets')},
    'gsm8k_task': str(Path(lm_eval.__file__).parent / 'tasks/gsm8k/gsm8k-cot.yaml'),
}))
"""


def save(path, value):
    """Write a reviewable JSON receipt."""
    path.write_text(json.dumps(value, indent=2) + "\n")


def run(command, log, *, env=None, timeout=600):
    """Record and run a client without replacing its measurement implementation."""
    print(shlex.join(map(str, command)), flush=True)
    with log.open("w") as stream:
        subprocess.run(
            command, env=env, stdout=stream, stderr=subprocess.STDOUT, check=True, timeout=timeout
        )


def cache_tokenizer(checkpoint, role, cache):
    """Map exact checkpoint files into the cache required by offline AIPerf."""
    files = {
        name: hashlib.sha256((checkpoint / name).read_bytes()).hexdigest()
        for name in TOKENIZER_FILES
        if (checkpoint / name).is_file()
    }
    if not {"tokenizer.json", "tokenizer_config.json"}.issubset(files):
        raise ValueError(f"Missing tokenizer files: {checkpoint}")
    revision = hashlib.sha256(json.dumps(files, sort_keys=True).encode()).hexdigest()[:40]
    repository = f"modelopt-workflow/{role}"
    snapshot = cache / ("models--" + repository.replace("/", "--")) / "snapshots" / revision
    snapshot.mkdir(parents=True)
    for name in files:
        (snapshot / name).symlink_to(checkpoint / name)
    return {
        "source": str(checkpoint),
        "repository": repository,
        "revision": revision,
        "files": files,
    }


def accuracy_command(python, model, dataset_task, output, samples, url):
    """Use the canonical GSM8K eight-shot CoT task with a bounded sample count."""
    return [
        str(python),
        "-m",
        "lm_eval",
        "run",
        "--model",
        "local-completions",
        "--model_args",
        (
            f"model={model},base_url={url}/v1/completions,tokenizer_backend=None,"
            "tokenized_requests=False,num_concurrent=4,max_length=40960,max_gen_toks=512,max_retries=0"
        ),
        "--tasks",
        str(dataset_task),
        "--num_fewshot",
        "8",
        "--limit",
        str(samples),
        "--gen_kwargs",
        "temperature=0,max_gen_toks=512",
        "--seed",
        "42",
        "--log_samples",
        "--output_path",
        str(output),
    ]


def performance_command(python, model, tokenizer, concurrency, output, url):
    """Use fixed-length cached-prefix traffic, matching the AIB workload recipe."""
    return [
        str(python.parent / "aiperf"),
        "profile",
        "--model",
        model,
        "--url",
        url,
        "--endpoint-type",
        "completions",
        "--streaming",
        "--ui-type",
        "None",
        "--tokenizer",
        tokenizer["repository"],
        "--tokenizer-revision",
        tokenizer["revision"],
        "--num-prefix-prompts",
        "1",
        "--num-dataset-entries",
        "32",
        "--prefix-prompt-length",
        "32768",
        "--synthetic-input-tokens-mean",
        "2048",
        "--synthetic-input-tokens-stddev",
        "0",
        "--output-tokens-mean",
        "1024",
        "--output-tokens-stddev",
        "0",
        "--concurrency",
        str(concurrency),
        "--request-count",
        str(max(8, concurrency * 4)),
        "--warmup-request-count",
        "3",
        "--random-seed",
        "42",
        "--use-server-token-count",
        "--extra-inputs",
        "ignore_eos:true",
        "--extra-inputs",
        "min_tokens:1024",
        "--extra-inputs",
        "temperature:0",
        "--artifact-dir",
        str(output),
    ]


def wait_for_server(process, url):
    """Fail before clients run if the server exits or misses its startup deadline."""
    deadline = time.monotonic() + 600
    while time.monotonic() < deadline:
        if process.poll() is not None:
            raise RuntimeError("vLLM exited during startup; inspect server.log")
        try:
            with urllib.request.urlopen(url + "/health", timeout=2) as response:
                if response.status == 200:
                    return
        except (urllib.error.URLError, TimeoutError):
            time.sleep(1)
    raise TimeoutError("vLLM was not ready within 600 seconds")


def stop_server(process):
    """Release every serving worker before loading the other checkpoint."""
    if process.poll() is None:
        os.killpg(process.pid, signal.SIGTERM)
        try:
            process.wait(timeout=30)
        except subprocess.TimeoutExpired:
            os.killpg(process.pid, signal.SIGKILL)
            process.wait(timeout=10)


def collect_metrics(output, role, concurrencies):
    """Reject partial client exports; retain original accuracy filters and units."""
    metrics = {}
    results = list((output / "accuracy").rglob("results*.json"))
    if len(results) != 1:
        raise ValueError(f"Expected one lm-eval result, found {len(results)}")
    accuracy = json.loads(results[0].read_text())["results"]["gsm8k_local"]
    for key in ("exact_match,strict-match", "exact_match,flexible-extract"):
        metrics[f"{role}.gsm8k.{key.replace(',', '.')}"] = accuracy[key]
    points = []
    for concurrency in concurrencies:
        export = output / f"c{concurrency}" / "client_aiperf_results" / "profile_export_aiperf.json"
        profile = json.loads(export.read_text())
        if (
            not profile.get("is_complete")
            or profile.get("was_cancelled")
            or profile.get("error_summary")
        ):
            raise ValueError(f"Incomplete AIPerf export: {export}")
        phases = [p for p in profile["input_config"]["phases"] if p["kind"] == "profiling"]
        if len(phases) != 1 or phases[0]["concurrency"] != concurrency:
            raise ValueError(f"Unexpected AIPerf profiling phase: {export}")
        if profile["request_count"]["avg"] != max(8, concurrency * 4):
            raise ValueError(f"AIPerf did not complete the request budget: {export}")
        point = {"concurrency": concurrency}
        for name, unit in (
            ("output_token_throughput", "tokens/sec"),
            ("output_token_throughput_per_user", "tokens/sec/user"),
            ("time_to_first_token", "ms"),
            ("inter_token_latency", "ms"),
        ):
            if profile[name]["unit"] != unit:
                raise ValueError(f"Unexpected {name} unit: {export}")
            point[name] = profile[name]["avg"]
            metrics[f"{role}.c{concurrency}.{name}"] = point[name]
        points.append(point)
    with (output / "curve.csv").open("w") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(points[0]))
        writer.writeheader()
        writer.writerows(points)
    return metrics


def evaluate(args, client_python, workspace):
    """Run both arms sequentially in one single-GPU allocation."""
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    models = {"baseline": args.baseline.resolve(), "candidate": args.candidate.resolve()}
    cache = workspace / "hf-cache"
    tokenizers = {role: cache_tokenizer(path, role, cache) for role, path in models.items()}
    save(output / "tokenizers.json", tokenizers)
    client_env = dict(
        os.environ,
        HF_HUB_CACHE=str(cache),
        HF_HUB_OFFLINE="1",
        TRANSFORMERS_OFFLINE="1",
        HF_DATASETS_OFFLINE="1",
        TOKENIZERS_PARALLELISM="false",
    )
    check = subprocess.check_output(
        [str(client_python), "-c", TOKENIZER_CHECK, str(output / "tokenizers.json")],
        env=client_env,
        text=True,
    )
    preflight = json.loads(check)
    if preflight["versions"]["aiperf"] != "0.13.0" or preflight["versions"]["lm_eval"] != "0.4.12":
        raise ValueError("The prepared client environment does not match requirements.txt")
    save(output / "preflight.json", preflight)
    gpu = subprocess.check_output(
        [
            sys.executable,
            "-c",
            (
                "import torch,json,socket,vllm; assert torch.cuda.device_count()==1; "
                "print(json.dumps({'name':torch.cuda.get_device_name(0), 'count':1, "
                "'node':socket.gethostname(),'engine_version':vllm.__version__,"
                "'capability':torch.cuda.get_device_capability(0)}))"
            ),
        ],
        text=True,
    )
    save(output / "gpu.json", json.loads(gpu))
    dataset = args.dataset.resolve() / "main"
    files = {split: str(dataset / f"{split}-00000-of-00001.parquet") for split in ("train", "test")}
    for path in files.values():
        if not Path(path).is_file():
            raise ValueError(f"Missing pinned GSM8K parquet: {path}")
    task = workspace / "gsm8k_local.yaml"
    save(
        task,
        {
            "include": preflight["gsm8k_task"],
            "task": "gsm8k_local",
            "dataset_path": "parquet",
            "dataset_name": None,
            "dataset_kwargs": {"data_files": files},
        },
    )
    metrics = {}
    url = f"http://127.0.0.1:{args.port}"
    for role, checkpoint in models.items():
        arm = output / role
        arm.mkdir()
        model = "modelopt-comparison"
        server = [
            sys.executable,
            "-m",
            "vllm.entrypoints.openai.api_server",
            "--model",
            str(checkpoint),
            "--served-model-name",
            model,
            "--host",
            "127.0.0.1",
            "--port",
            str(args.port),
            "--tensor-parallel-size",
            "1",
            "--pipeline-parallel-size",
            "1",
            "--dtype",
            "bfloat16",
            "--kv-cache-dtype",
            "auto",
            "--max-model-len",
            "40960",
            "--max-num-seqs",
            "8",
            "--gpu-memory-utilization",
            "0.85",
            "--enable-prefix-caching",
            "--seed",
            "42",
        ]
        if role == "candidate" and args.candidate_linear_backend != "auto":
            server += ["--linear-backend", args.candidate_linear_backend]
        if args.compilation_mode == "none":
            server += ["-cc.mode=0"]
        save(
            arm / "recipe.json",
            {
                "model": getattr(args, role + "_ref"),
                "speculation": "disabled",
                "gpu_count_used": 1,
                "client": "aiperf==0.13.0",
                "accuracy_samples": args.samples,
                "temperature": 0,
                "candidate_linear_backend": args.candidate_linear_backend,
                "compilation_mode": args.compilation_mode,
            },
        )
        (arm / "server-command.txt").write_text(shlex.join(server) + "\n")
        with (arm / "server.log").open("w") as log:
            process = subprocess.Popen(
                server, stdout=log, stderr=subprocess.STDOUT, start_new_session=True
            )
            try:
                wait_for_server(process, url)
                command = accuracy_command(
                    client_python, model, task, arm / "accuracy", args.samples, url
                )
                save(arm / "accuracy-command.json", command)
                run(command, arm / "accuracy.log", env=client_env)
                for concurrency in args.concurrency:
                    cell = arm / f"c{concurrency}"
                    cell.mkdir()
                    command = performance_command(
                        client_python,
                        model,
                        tokenizers[role],
                        concurrency,
                        cell / "client_aiperf_results",
                        url,
                    )
                    (cell / "benchmark-command.txt").write_text(shlex.join(command) + "\n")
                    save(cell / "execution.json", {"started_at": datetime.now(UTC).isoformat()})
                    run(command, cell / "client.log", env=client_env)
                metrics.update(collect_metrics(arm, role, args.concurrency))
                save(output / "metrics.json", metrics)
            finally:
                stop_server(process)


def main():
    """Prepare isolated benchmark dependencies, execute, and retain tracked results."""
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("baseline", "candidate", "dataset", "output"):
        parser.add_argument("--" + name, required=True, type=Path)
    for name in ("baseline-ref", "candidate-ref"):
        parser.add_argument("--" + name, required=True)
    parser.add_argument("--client-python", type=Path)
    parser.add_argument("--candidate-linear-backend", choices=("auto", "marlin"), default="auto")
    parser.add_argument("--compilation-mode", choices=("default", "none"), default="default")
    parser.add_argument("--samples", type=int, default=64)
    parser.add_argument("--concurrency", type=int, nargs="+", default=[1, 4, 8])
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()
    if args.samples < 1 or not args.concurrency or any(c < 1 or c > 8 for c in args.concurrency):
        parser.error("Use positive samples and concurrency values in 1..8")
    if len(set(args.concurrency)) != len(args.concurrency):
        parser.error("Concurrency values must be unique")
    if args.output.exists() and (not args.output.is_dir() or any(args.output.iterdir())):
        parser.error("Use an empty results directory to avoid mixing measurements")
    status = "failed"
    with tempfile.TemporaryDirectory(prefix="modelopt-eval-") as temporary:
        workspace = Path(temporary)
        python = args.client_python
        if python is None:
            subprocess.run([sys.executable, "-m", "venv", str(workspace / "client")], check=True)
            python = workspace / "client/bin/python"
            subprocess.run(
                [
                    str(python),
                    "-m",
                    "pip",
                    "install",
                    "-r",
                    str(Path(__file__).with_name("requirements.txt")),
                ],
                check=True,
            )
        try:
            evaluate(args, python.absolute(), workspace)
            status = "completed"
        finally:
            if args.output.is_dir():
                save(args.output / "status.json", {"status": status})
                if os.environ.get("MLFLOW_RUN_ID"):
                    # Tracking is optional for public users and supplied by the v2 wrapper internally.
                    from mlflow.tracking import MlflowClient

                    client = MlflowClient()
                    run_id = os.environ["MLFLOW_RUN_ID"]
                    client.log_artifacts(run_id, str(args.output), artifact_path="evaluation")
                    manifest = os.environ.get("MODELOPT_INPUTS_MANIFEST")
                    inputs = json.loads(Path(manifest).read_text()) if manifest else []
                    model_ids = {item.get("role"): item.get("model_id") for item in inputs}
                    metrics_path = args.output / "metrics.json"
                    if metrics_path.is_file():
                        for name, value in json.loads(metrics_path.read_text()).items():
                            client.log_metric(
                                run_id, name, value, model_id=model_ids.get(name.split(".")[0])
                            )


if __name__ == "__main__":
    main()
