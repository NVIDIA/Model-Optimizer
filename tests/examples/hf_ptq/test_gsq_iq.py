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

import json
import math
from types import SimpleNamespace

import pytest
import torch
import torch.nn.functional as F

from examples.hf_ptq.gsq_iq import evaluate_perplexity, source_identity, source_revision


class _ToyLanguageModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.register_buffer("table", torch.tensor([[0.0, 1.0], [2.0, 0.0]]))
        self.windows = []

    def forward(self, input_ids, labels, use_cache):
        assert not use_cache
        self.windows.append(input_ids.cpu().flatten().tolist())
        logits = self.table[input_ids]
        loss = F.cross_entropy(logits[:, :-1].flatten(0, 1), labels[:, 1:].flatten())
        return SimpleNamespace(loss=loss)


@pytest.mark.parametrize("token_count", [2048, 2049, 4095, 4100])
def test_perplexity_scores_every_target_once_including_final_tail(token_count):
    tokens = (torch.arange(token_count) % 3 == 0).long()
    model = _ToyLanguageModel()
    result = evaluate_perplexity(model, tokens, device="cpu")

    expected_nll = F.cross_entropy(model.table[tokens[:-1]], tokens[1:], reduction="sum").item()
    assert result["scored_tokens"] == token_count - 1
    assert result["negative_log_likelihood"] == pytest.approx(expected_nll, rel=1e-6)
    assert result["perplexity"] == pytest.approx(math.exp(expected_nll / (token_count - 1)))
    assert all(len(window) <= 2048 for window in model.windows)
    assert [target for window in model.windows for target in window[1:]] == tokens[1:].tolist()


def test_source_identity_verifies_revision_weight_bytes_and_tokenizer(tmp_path):
    revision = "1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0"
    cache = tmp_path / "models--Qwen--Qwen3.8-27B"
    snapshot = cache / "snapshots" / revision
    snapshot.mkdir(parents=True)
    blobs = cache / "blobs"
    blobs.mkdir()
    digest = "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"
    blob = blobs / digest
    blob.write_bytes(b"abc")
    (snapshot / "model-00001-of-00001.safetensors").symlink_to(blob)
    (snapshot / "model.safetensors.index.json").write_text(
        json.dumps({"weight_map": {"model.weight": "model-00001-of-00001.safetensors"}})
    )
    (snapshot / "config.json").write_text("{}")
    (snapshot / "tokenizer_config.json").write_text("{}")
    tokenizer = snapshot / "tokenizer.json"
    tokenizer.write_text("{}")

    identity = source_identity(snapshot, revision)
    assert identity["files"]["model-00001-of-00001.safetensors"]["sha256"] == digest
    with pytest.raises(ValueError):
        source_identity(snapshot, "0" * 40)
    blob.write_bytes(b"abd")
    with pytest.raises(ValueError):
        source_identity(snapshot, revision)
    blob.write_bytes(b"abc")
    tokenizer.write_text('{"changed":true}')
    assert source_identity(snapshot, revision) != identity


@pytest.mark.parametrize("layout", ["loose", "detached", "packed", "worktree"])
def test_source_revision_reads_git_metadata_without_git_executable(tmp_path, layout):
    revision = "4f08a12bb4246f8fb05fc3dc0fc5df5484ab801b"
    repository = tmp_path / "repository"
    repository.mkdir()
    metadata = repository / ".git"
    common = metadata
    if layout == "worktree":
        metadata = tmp_path / "metadata" / "worktree"
        common = tmp_path / "common"
        (repository / ".git").write_text("gitdir: ../metadata/worktree\n")
    metadata.mkdir(parents=True)
    common.mkdir(exist_ok=True)
    if layout == "worktree":
        (metadata / "commondir").write_text("../../common\n")
    if layout == "detached":
        target = metadata / "HEAD"
    else:
        (metadata / "HEAD").write_text("ref: refs/heads/main\n")
        if layout == "loose":
            target = common / "refs" / "heads" / "main"
            target.parent.mkdir(parents=True)
        else:
            target = common / "packed-refs"
    suffix = " refs/heads/main" if target.name == "packed-refs" else ""
    target.write_text(revision + suffix + "\n")
    assert source_revision(repository) == revision
    target.write_text("invalid" + suffix + "\n")
    with pytest.raises(ValueError):
        source_revision(repository)
