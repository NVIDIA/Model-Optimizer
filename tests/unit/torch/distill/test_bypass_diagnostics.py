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

"""Collective diagnostics reject local preparation failures before model collectives."""

from unittest import mock

import pytest
import torch


def _diagnostic_preparation_worker(rank, size, failure):
    import torch.distributed as dist
    from _test_utils.torch.distributed.utils import bounded_collectives
    from torch.distributed.device_mesh import init_device_mesh

    from examples.bypass import evaluate as diagnostics
    from modelopt.torch.distill.plugins.transformers import TransformersBypassAdapter

    mesh = init_device_mesh("cpu", (size,), mesh_dim_names=("dp",))
    seen = []

    class CollectiveModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.scores = torch.nn.Parameter(torch.tensor([0.0, 1.0]))

        def forward(self, input_ids):
            seen.append(tuple(input_ids.shape))
            dist.all_reduce(torch.ones(1), group=mesh.get_group())
            return self.scores.expand(*input_ids.shape, 2)

    model = CollectiveModel().train()
    TransformersBypassAdapter(mesh).parallelize(model)
    batch = {
        "input_ids": torch.ones(1, 4, dtype=torch.long),
        "_packed_seq_ids": torch.tensor([[1, 1, 2, 2] if rank == 0 else [1, 1, 1, 1]]),
    }

    class Loader:
        def __iter__(self):
            if rank == 1 and failure == "iterator":
                raise RuntimeError("Simulated diagnostic iterator failure")
            return iter([batch])

    original_empty = diagnostics.empty_document_batch
    original_documents = diagnostics.document_batches
    original_labels = torch.full_like

    def empty(value):
        if rank == 1 and failure == "padding":
            raise torch.OutOfMemoryError("Simulated diagnostic padding allocation failure")
        return original_empty(value)

    def documents(value):
        if rank == 1 and failure == "documents":
            raise torch.OutOfMemoryError("Simulated diagnostic splitting allocation failure")
        return original_documents(value)

    def labels(*args, **kwargs):
        if rank == 1 and failure == "labels":
            raise torch.OutOfMemoryError("Simulated diagnostic label allocation failure")
        return original_labels(*args, **kwargs)

    with (
        bounded_collectives(10),
        mock.patch.object(diagnostics, "empty_document_batch", empty),
        mock.patch.object(diagnostics, "document_batches", documents),
        mock.patch.object(torch, "full_like", labels),
        pytest.raises(ValueError, match="Simulated diagnostic") as refusal,
    ):
        diagnostics.evaluate_pair(model, model, Loader(), device_mesh=mesh)
    errors = [None] * size
    dist.all_gather_object(errors, str(refusal.value), group=mesh.get_group())
    assert len(set(errors)) == 1
    assert len(seen) == (2 if failure == "padding" else 0)
    assert model.training


@pytest.mark.parametrize("failure", ["iterator", "padding", "documents", "labels"])
def test_diagnostic_preparation_errors_agree_before_model_collectives(cpu_dist_workers, failure):
    pytest.importorskip("transformers", minversion="5.23")
    cpu_dist_workers.run(_diagnostic_preparation_worker, failure)
