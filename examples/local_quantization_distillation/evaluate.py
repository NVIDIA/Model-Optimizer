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

"""Paired held-out PPL and full-vocabulary KL(teacher || candidate)."""

import argparse
import inspect
import json
import math
import time
from pathlib import Path

import torch
import torch.distributed as dist

from modelopt.torch.utils import print_rank_0


@torch.no_grad()
def diagnostic_statistics(
    teacher_logits, candidate_logits, labels, valid_mask, *, token_chunk=128, vocabulary_chunk=8192
):
    """Return candidate NLL/count, KL/count and teacher NLL without a full FP32 copy.

    Labels are *already shifted*, within-document next-token targets. KL includes
    every valid input token, including prompts and document-final tokens. Vocabulary
    chunks use exact full-vocabulary normalizers; this is not a top-k approximation.
    """
    if teacher_logits.shape != candidate_logits.shape or teacher_logits.ndim < 2:
        raise ValueError("Paired logits must have identical token/vocabulary dimensions.")
    if labels.shape != teacher_logits.shape[:-1] or valid_mask.shape != labels.shape:
        raise ValueError("Diagnostic targets and mask must match the local token layout.")
    if min(token_chunk, vocabulary_chunk) < 1:
        raise ValueError("Diagnostic chunk sizes must be positive.")
    vocabulary = teacher_logits.shape[-1]
    teacher_logits = teacher_logits.reshape(-1, vocabulary)
    candidate_logits = candidate_logits.reshape(-1, vocabulary)
    labels, valid_mask = labels.flatten(), valid_mask.flatten().bool()
    totals = teacher_logits.new_zeros(5, dtype=torch.float64)
    for start in range(0, labels.numel(), token_chunk):
        stop = start + token_chunk
        target, valid = labels[start:stop], valid_mask[start:stop]
        predicted = valid & target.ne(-100)
        left, right = teacher_logits[start:stop], candidate_logits[start:stop]
        normalizers = []
        for logits in (left, right):
            norm = torch.full((logits.shape[0],), -torch.inf, device=logits.device)
            for column in range(0, vocabulary, vocabulary_chunk):
                block = logits[:, column : column + vocabulary_chunk].float()
                norm = torch.logaddexp(norm, block.logsumexp(-1))
            normalizers.append(norm)
        teacher_norm, candidate_norm = normalizers
        safe_target = target.masked_fill(~predicted, 0).unsqueeze(-1)
        totals[0] += (
            (candidate_norm - right.gather(-1, safe_target).squeeze(-1).float())[predicted]
            .double()
            .sum()
        )
        totals[1] += predicted.sum()
        totals[4] += (
            (teacher_norm - left.gather(-1, safe_target).squeeze(-1).float())[predicted]
            .double()
            .sum()
        )
        for column in range(0, vocabulary, vocabulary_chunk):
            teacher_logp = (
                left[:, column : column + vocabulary_chunk].float() - teacher_norm[:, None]
            )
            candidate_logp = (
                right[:, column : column + vocabulary_chunk].float() - candidate_norm[:, None]
            )
            kl = (teacher_logp.exp() * (teacher_logp - candidate_logp)).sum(-1)
            totals[2] += kl[valid].double().sum()
        totals[3] += valid.sum()
    return totals


def _logits(output):
    if isinstance(output, torch.Tensor):
        return output
    return output.logits if hasattr(output, "logits") else output[0]


def _perplexity(mean_cross_entropy):
    # PPL is secondary: an unrepresentable exponential must not hide finite KL.
    try:
        value = math.exp(mean_cross_entropy)
    except OverflowError:
        return None
    return value if math.isfinite(value) else None


@torch.no_grad()
def evaluate_pair(teacher, candidate, dataloader, *, mesh_context=None, padding_token_id=0):
    """Use identical held-out native AutoModel packed/CP inputs for both models.

    Every process owns the same vocab (or a DTensor that can materialize it).
    Token counts are reduced over DP×CP exactly once; TP replicas are not counted.
    This simple diagnostic entry point accepts full models, not PP stage modules.
    """
    if mesh_context is not None and mesh_context.pp_size > 1:
        raise ValueError("Run these full-model diagnostics without pipeline stage splitting.")
    teacher_flags = {
        module: module.training for model in (teacher, candidate) for module in model.modules()
    }
    device = next(candidate.parameters()).device
    if device.type == "cuda":
        torch.cuda.synchronize(device)
        torch.cuda.reset_peak_memory_stats(device)
    started = time.perf_counter()
    totals = torch.zeros(5, device=device, dtype=torch.float64)
    try:
        teacher.eval()
        candidate.eval()
        if mesh_context is not None:
            from modelopt.torch.distill.plugins.automodel import AutoModelLocalDistillationAdapter

            adapter = AutoModelLocalDistillationAdapter(mesh_context)
            adapter._configure_native_attention(teacher)
            adapter._configure_native_attention(candidate)
        for index, batch in enumerate(dataloader, 1):
            from modelopt.torch.distill.plugins.automodel.packing import prepare_local_batch

            context, local = prepare_local_batch(
                teacher, batch, mesh_context, padding_token_id=padding_token_id
            )
            labels, valid = local["labels"], local["valid_mask"]
            kwargs = {
                name: value
                for name, value in local.items()
                if name not in ("labels", "valid_mask", "sample_ids", "document_ids", "loss_mask")
            }
            if "use_cache" in inspect.signature(teacher.forward).parameters:
                kwargs["use_cache"] = False
            with context():
                target = _logits(teacher(**kwargs))
                prediction = _logits(candidate(**kwargs))
                from torch.distributed.tensor import DTensor

                if isinstance(target, DTensor):
                    target = target.full_tensor()
                if isinstance(prediction, DTensor):
                    prediction = prediction.full_tensor()
                totals += diagnostic_statistics(
                    target,
                    prediction,
                    labels.reshape(target.shape[:-1]),
                    valid.reshape(target.shape[:-1]),
                )
                del target, prediction
            if index == 1 or index % 4 == 0:
                print_rank_0(
                    json.dumps(
                        {
                            "diagnostic_batches_local": index,
                            "elapsed_seconds": time.perf_counter() - started,
                        }
                    )
                )
        if mesh_context is not None and dist.is_initialized():
            axis = "dp_cp" if "dp_cp" in mesh_context.device_mesh.mesh_dim_names else "dp"
            # Flattened DP/CP is a named submesh even when not a root dimension.
            try:
                mesh = mesh_context.device_mesh["dp_cp"]
            except (KeyError, RuntimeError):
                mesh = mesh_context.device_mesh[axis]
            if mesh.size() > 1:
                dist.all_reduce(totals, group=mesh.get_group())
        nll, target_count, kl, token_count, teacher_nll = totals.tolist()
        if min(target_count, token_count) <= 0:
            raise ValueError("Held-out diagnostics require valid tokens and next-token targets.")
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        return {
            "ppl": _perplexity(nll / target_count),
            "mean_cross_entropy": nll / target_count,
            "teacher_ppl": _perplexity(teacher_nll / target_count),
            "teacher_mean_cross_entropy": teacher_nll / target_count,
            "kld_teacher_candidate": kl / token_count,
            "next_token_targets": int(target_count),
            "kl_tokens": int(token_count),
            "elapsed_seconds_local": time.perf_counter() - started,
            "peak_allocated_bytes_local": torch.cuda.max_memory_allocated(device)
            if device.type == "cuda"
            else 0,
        }
    finally:
        for module, training in teacher_flags.items():
            module.training = training


def main():
    """Compare one native ModelOpt checkpoint with its original teacher."""
    from examples.local_quantization_distillation.prepare_data import PackedBatchSequence
    from examples.local_quantization_distillation.repair import load_models

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--data", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    student, teacher, adapter, settings = load_models(args.config, checkpoint=args.checkpoint)
    mesh = adapter.mesh_context
    data = PackedBatchSequence(
        args.data,
        "diagnostic",
        global_batch_size=mesh.dp_size or 1,
        dp_rank=mesh.device_mesh["dp"].get_local_rank(),
        dp_size=mesh.dp_size or 1,
        device=next(student.parameters()).device,
    )
    report = evaluate_pair(teacher, student, data, mesh_context=mesh)
    report["data_fingerprint"] = data.fingerprint
    report["checkpoint"] = str(Path(args.checkpoint).resolve())
    if not dist.is_initialized() or dist.get_rank() == 0:
        Path(args.output).write_text(json.dumps(report, indent=2) + "\n")
        print(json.dumps(report, indent=2))
    if dist.is_initialized():
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
