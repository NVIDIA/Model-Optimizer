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

"""End-to-end CI test for the streaming speculative-decoding workflow.

Drives ``launch_train.sh`` in ``data.mode=streaming`` against an in-process
mocked HTTP server that mimics ``vllm serve`` with
``ExampleHiddenStatesConnector``. Exercises ``main.py``'s mode dispatch,
``EagleVllmStreamingDataset``, the resume-callback wiring, and the full
Trainer step — without requiring real vLLM (which would need a separate GPU,
several minutes of warmup, and the connector plugin).
"""

import json
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path

import pytest
import safetensors.torch
import torch
from _test_utils.examples.run_command import MODELOPT_ROOT, run_example_command

EAGLE3_YAML = str(
    MODELOPT_ROOT / "modelopt_recipes" / "general" / "speculative_decoding" / "eagle3.yaml"
)

# tiny_llama_path fixture builds num_hidden_layers=2 + hidden_size=512;
# default_eagle_aux_layer_ids(2) = [0, 1] -> 2 aux + 1 final = 3 captured layers.
_HIDDEN_SIZE = 512
_N_CAPTURED_LAYERS = 3

# Tiny EAGLE architecture overrides (dotlist entries) — same as offline test.
_TINY_EAGLE_ARCH = [
    "eagle.eagle_architecture_config.max_position_embeddings=128",
    "eagle.eagle_architecture_config.num_hidden_layers=1",
    "eagle.eagle_architecture_config.intermediate_size=64",
    "eagle.eagle_architecture_config.num_attention_heads=2",
    "eagle.eagle_architecture_config.num_key_value_heads=2",
    "eagle.eagle_architecture_config.head_dim=64",
]


def _make_handler(scratch_dir: Path):
    """Stdlib HTTP handler that mimics vLLM's ExampleHiddenStatesConnector:
    per request, write a safetensors file and reply with its path."""
    counter = {"n": 0}

    class _Handler(BaseHTTPRequestHandler):
        def do_POST(self):
            length = int(self.headers.get("Content-Length", "0"))
            body = json.loads(self.rfile.read(length))
            prompt = body["prompt"]
            seq = len(prompt)
            counter["n"] += 1
            path = scratch_dir / f"req_{counter['n']}.safetensors"
            safetensors.torch.save_file(
                {
                    "token_ids": torch.tensor(prompt, dtype=torch.int64),
                    "hidden_states": torch.randn(seq, _N_CAPTURED_LAYERS, _HIDDEN_SIZE),
                },
                str(path),
            )
            payload = json.dumps({"kv_transfer_params": {"hidden_states_path": str(path)}}).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(payload)))
            self.end_headers()
            self.wfile.write(payload)

        def log_message(self, *args, **kwargs):
            pass  # keep test output quiet

    return _Handler


@pytest.fixture
def mock_vllm_server(tmp_path_factory):
    """Spin up a stdlib HTTPServer pretending to be a vLLM hidden-states endpoint.

    Yields the base URL; serves on a kernel-assigned loopback port so concurrent
    runs don't collide.
    """
    scratch = tmp_path_factory.mktemp("vllm_scratch")
    server = HTTPServer(("127.0.0.1", 0), _make_handler(scratch))
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield f"http://127.0.0.1:{server.server_port}"
    finally:
        server.shutdown()
        thread.join(timeout=5)


def test_streaming_eagle_training(
    tiny_llama_path, tiny_conversations_path, mock_vllm_server, tmp_path_factory
):
    """Train EAGLE3 in streaming mode end-to-end against the mocked server."""
    output_dir = tmp_path_factory.mktemp("eagle_streaming_ckpt")

    overrides = [
        f"model.model_name_or_path={tiny_llama_path}",
        f"data.data_path={tiny_conversations_path}",
        "data.mode=streaming",
        f"data.streaming_server_url={mock_vllm_server}",
        f"data.streaming_model_name={tiny_llama_path}",
        "data.streaming_max_seq_len=64",
        "data.streaming_prefetch=2",
        f"training.output_dir={output_dir}",
        "training.num_train_epochs=1",
        "training.learning_rate=1e-5",
        "training.training_seq_len=32",
        "training.save_steps=1",
        "training.dataloader_num_workers=0",  # enforced by StreamingDataset
        *_TINY_EAGLE_ARCH,
    ]

    run_example_command(
        ["./launch_train.sh", "--config", EAGLE3_YAML, *overrides],
        "speculative_decoding",
        setup_free_port=True,
    )

    # Training must produce at least one checkpoint (validates the full
    # mode-dispatch → dataset → collator → trainer step chain).
    ckpts = list(output_dir.rglob("checkpoint-*"))
    assert ckpts, f"no checkpoint produced in {output_dir}"
