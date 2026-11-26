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
"""The on-policy loop on CPU: 2 torchrun nodes, each with 2 fake rollout servers."""

import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

import httpx
import pytest
import transformers
from _test_utils.examples.run_command import MODELOPT_ROOT
from _test_utils.torch.distributed.utils import get_free_port
from _test_utils.torch.transformers_models import create_tiny_qwen3_dir
from safetensors.torch import load_file

sys.path.insert(0, str(MODELOPT_ROOT / "examples/speculative_decoding/rejmem_onpolicy"))
import rollout

FAKES = Path(__file__).with_name("rejmem_onpolicy_fakes.py")
NODES, SERVERS, NS, HIDDEN, AUX_LAYERS = 2, 2, 3, 32, [0, 2]
ENV = {
    **os.environ,
    "CUDA_VISIBLE_DEVICES": "",  # two ranks on one host: gloo
    "PYTHONPATH": os.pathsep.join([str(MODELOPT_ROOT), os.environ.get("PYTHONPATH", "")]),
    "OMP_NUM_THREADS": "2",
}


@pytest.fixture(scope="module")
def experiment(tmp_path_factory):
    root = tmp_path_factory.mktemp("rejmem_onpolicy")
    model = create_tiny_qwen3_dir(
        root, with_tokenizer=True, hidden_size=HIDDEN, num_hidden_layers=4
    )
    corpus = root / "corpus"
    corpus.mkdir()
    for s in range(8):
        rows = [
            {
                "uuid": f"u{s}-{r}",
                "conversations": [
                    *([{"role": "system", "content": "sys"}] if r % 3 == 0 else []),
                    {"role": "user", "content": f"q{s}{r}"},
                    {"role": "assistant", "content": "why</think>ans" if r % 2 else "ans"},
                ],
            }
            for r in range(6)
        ]
        (corpus / f"train-{s:05d}.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
    config = {
        "metadata": {"description": "rejmem on-policy CPU test"},
        "model": {"model_name_or_path": str(model), "use_fake_base_for_offline": True},
        "training": {
            "max_steps": 4,
            "save_steps": 2,
            "logging_steps": 1,
            "training_seq_len": 64,
            "per_device_train_batch_size": 2,
            "learning_rate": 1e-3,
            "warmup_steps": 1,
            "lr_scheduler_type": "cosine",
            "seed": 7,
            "report_to": "none",
            "ddp_timeout": 120,
        },
        "dflash": {
            "dflash_block_size": NS + 1,
            "dflash_num_anchors": 8,
            "dflash_mask_token_id": 127,
            "dflash_self_logit_distillation": True,
            "dflash_architecture_config": {
                "num_hidden_layers": 1,
                "target_layer_ids": AUX_LAYERS,
                "head_dim": 8,
                "num_attention_heads": 4,
                "num_key_value_heads": 2,
                "intermediate_size": 64,
            },
        },
        "onpolicy": {
            "num_speculative_tokens": NS,
            "data_dir": str(corpus),
            "max_prompt_len": 40,
            "prompts_per_step": 4,
            "request_timeout": 60,
            "record_timeout": 30,
        },
    }
    (root / "config.yaml").write_text(json.dumps(config))  # JSON is YAML
    return root


def _run(root, name, resume=None, ignore_pushes_after=(-1, -1)):
    """One run on fresh servers; returns the exit codes and each node's server log."""
    ports = [[get_free_port() for _ in range(SERVERS)] for _ in range(NODES)]
    servers, trainers, master_port = [], [], get_free_port()
    for node in range(NODES):
        capture = root / f"{name}_capture{node}"
        capture.mkdir()
        for port in ports[node]:
            cmd = [
                *(sys.executable, FAKES, "serve"),
                f"--port={port}",
                f"--ns={NS}",
                f"--hidden={HIDDEN}",
                "--vocab=128",
                f"--aux_layers={json.dumps(AUX_LAYERS)}",
                f"--capture_dir={capture}",
                f"--log={root / f'{name}_{node}.jsonl'}",
                f"--ignore_pushes_after={ignore_pushes_after[node]}",
            ]
            servers.append(subprocess.Popen(cmd, env=ENV))
    for port in [p for node_ports in ports for p in node_ports]:
        for _ in range(300):
            try:
                httpx.get(f"http://127.0.0.1:{port}/")
                break
            except httpx.HTTPError:
                time.sleep(0.1)
    for node in range(NODES):
        urls = ",".join(f"http://127.0.0.1:{port}" for port in ports[node])
        cmd = [
            *(sys.executable, "-m", "torch.distributed.run"),
            f"--nnodes={NODES}",
            f"--node_rank={node}",
            "--nproc_per_node=1",
            "--master_addr=127.0.0.1",
            f"--master_port={master_port}",
            *(FAKES, "train", f"--config={root / 'config.yaml'}"),
            *([f"--resume_from={resume}"] if resume else []),
            f"training.output_dir={root / name}",
            f"onpolicy.server_urls=[{urls}]",
            f"onpolicy.capture_dir={root / f'{name}_capture{node}'}",
        ]
        log = open(root / f"{name}_trainer{node}.log", "w")
        trainers.append(subprocess.Popen(list(map(str, cmd)), env=ENV, stdout=log, stderr=log))
    codes = [p.wait(timeout=600) for p in trainers]
    for p in servers:
        p.kill()
    logs = [
        [json.loads(line) for line in (root / f"{name}_{node}.jsonl").read_text().splitlines()]
        for node in range(NODES)
    ]
    for log in logs:
        for r in log:
            r["step"], r["i"] = map(int, r["tag"].split("-s")[1].split("-"))
        log.sort(key=lambda r: (r["step"], r["i"]))
    return codes, logs


def _metrics(path):
    return [json.loads(line) for line in (path / "metrics.jsonl").read_text().splitlines()]


def test_onpolicy_loop_shards_resumes_exactly_and_rejects_stale_records(experiment):
    root = experiment
    codes, logs = _run(root, "a")
    assert codes == [0, 0]
    assert [m["step"] for m in _metrics(root / "a")] == [1, 2, 3, 4]
    # Strictly on-policy: the trainer pushed before the first rollout and after every step.
    assert all(r["version"] == r["step"] + 1 for log in logs for r in log)
    prompts = [{json.dumps(r["prompt"]) for r in log} for log in logs]
    assert len(prompts[0]) == len(prompts[1]) == 16
    assert not prompts[0] & prompts[1]  # each node has its own shard
    assert not list((root / "a_capture0").iterdir())  # records are consumed once

    # Resume from checkpoint-2 in a copy of the output, as after a crash before checkpoint-4, on
    # fresh servers: steps 3-4 replay exactly and replace their metrics.
    shutil.copytree(root / "a", root / "b", ignore=shutil.ignore_patterns("checkpoint-4"))
    codes, resumed_logs = _run(root, "b", resume=root / "b" / "checkpoint-2")
    assert codes == [0, 0]
    assert all(r["version"] == r["step"] - 1 for log in resumed_logs for r in log)
    for node in range(NODES):
        assert [r["prompt"] for r in resumed_logs[node]] == [
            r["prompt"] for r in logs[node] if r["step"] >= 2
        ]
    untimed = [
        [{k: v for k, v in m.items() if not k.startswith("t_")} for m in _metrics(root / run)]
        for run in ("a", "b")
    ]
    assert untimed[0] == untimed[1]
    final_a = load_file(root / "a/checkpoint-4/export/model.safetensors")
    final_b = load_file(root / "b/checkpoint-4/export/model.safetensors")
    assert final_a.keys() == final_b.keys()
    assert all(final_a[k].equal(final_b[k]) for k in final_a)

    # Node 1's servers miss the push after step 2: its step-3 records are stale and every rank aborts.
    codes, _ = _run(root, "c", ignore_pushes_after=(-1, 2))
    assert codes[0] != 0 and codes[1] != 0
    assert [m["step"] for m in _metrics(root / "c")] == [1, 2]
    assert "OffPolicyError" in (root / "c_trainer1.log").read_text()


@pytest.mark.parametrize("memory", [True, False])
def test_serve_command_takes_the_regime_from_the_config(experiment, memory):
    recipe, cfg = rollout.load_config(
        str(experiment / "config.yaml"),
        [f"onpolicy.memory={memory}", "onpolicy.draft_sample_method=greedy"],
    )
    argv, env = rollout.serve_command(recipe, cfg, 65001, "/capture", draft="/draft")
    spec = json.loads(argv[argv.index("--speculative-config") + 1])
    assert spec == {
        "method": "dflash",
        "model": "/draft",
        "num_speculative_tokens": NS,
        "draft_sample_method": "greedy",
    }
    assert argv[argv.index("--host") + 1] == "127.0.0.1"  # the weight endpoints unpickle bodies
    assert env["MODELOPT_REJMEM_CAPTURE_DIR"] == "/capture"
    assert ("MODELOPT_REJMEM_CONSUMER" in env) == memory


def test_config_refuses_stop_strings(experiment):
    # The engine does not stop such requests itself, so the servers write no record for them.
    with pytest.raises(ValueError, match="stop"):
        rollout.load_config(str(experiment / "config.yaml"), ["onpolicy.sampling.stop=[x]"])


def test_prompt_source_mixes_single_subset_shards(experiment, tmp_path):
    """Each corpus shard holds one subset (thinking or not), yet every step mixes them."""
    for s in range(8):
        reply = "why</think>ans" if s % 2 else "ans"
        rows = [
            {
                "conversations": [
                    {"role": "user", "content": f"q{s}-{r}"},
                    {"role": "assistant", "content": reply},
                ]
            }
            for r in range(6)
        ]
        (tmp_path / f"train-{s:05d}.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
    recipe, _ = rollout.load_config(str(experiment / "config.yaml"), [])
    tokenizer = transformers.AutoTokenizer.from_pretrained(recipe.model.model_name_or_path)
    source = rollout.PromptSource(tmp_path, tokenizer, 40, seed=7, rank=0, world_size=1)
    first = source.take(6)  # as many prompts as a shard has rows
    assert {p.enable_thinking for p in first} == {True, False}
