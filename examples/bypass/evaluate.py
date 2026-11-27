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


"""Held-out full-vocabulary KL and next-token cross entropy with bounded head storage."""

import inspect
import math
import time

import torch
import torch.distributed as dist
from torch.distributed.tensor import DTensor

from modelopt.torch.distill._bypass_capture import _valid_token_mask
from modelopt.torch.distill._bypass_parallel import _all_rank_values
from modelopt.torch.distill.plugins.transformers import TransformersBypassAdapter
from modelopt.torch.distill.plugins.transformers.output_head import (
    OutputHeadProjection,
    capture_head_inputs,
)
from modelopt.torch.distill.plugins.transformers.packing import (
    document_batches,
    empty_document_batch,
)


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


@torch.no_grad()
def _projected_statistics(
    left, right, teacher, candidate, labels, valid, token_chunk, vocabulary_chunk, slots, sliceable
):
    left, right = left.reshape(-1, left.shape[-1]), right.reshape(-1, right.shape[-1])
    labels, valid = labels.flatten(), valid.flatten().bool()
    totals = left.new_zeros(5, dtype=torch.float64)
    vocabulary = teacher.head.weight.shape[0]
    for start in range(0, slots, token_chunk):
        end = min(start + token_chunk, labels.numel())
        target, mask = labels[start:end], valid[start:end]
        if start >= labels.numel():
            # Every model/FSDP rank must still execute the same projection schedule.
            a, b = left.new_zeros((1, left.shape[-1])), right.new_zeros((1, right.shape[-1]))
            target = labels.new_full((1,), -100)
            mask = valid.new_zeros((1,))
        else:
            a, b = left[start:end], right[start:end]
        if not sliceable:
            t_logits = teacher.project(a)
            s_logits = candidate.project(b)
            totals += diagnostic_statistics(
                t_logits,
                s_logits,
                target,
                mask,
                token_chunk=token_chunk,
                vocabulary_chunk=vocabulary_chunk,
            )
            del t_logits, s_logits
            continue
        predicted = mask & target.ne(-100)
        t_max = torch.full((a.shape[0],), -torch.inf, device=a.device)
        s_max = torch.full_like(t_max, -torch.inf)
        t_mass = torch.zeros_like(t_max, dtype=torch.float64)
        s_mass, difference = torch.zeros_like(t_mass), torch.zeros_like(t_mass)
        t_target, s_target = torch.zeros_like(t_max), torch.zeros_like(t_max)
        for column in range(0, vocabulary, vocabulary_chunk):
            columns = slice(column, min(column + vocabulary_chunk, vocabulary))
            t_logits, s_logits = (
                teacher.project(a, columns).float(),
                candidate.project(b, columns).float(),
            )
            next_t = torch.maximum(t_max, t_logits.amax(-1))
            next_s = torch.maximum(s_max, s_logits.amax(-1))
            t_scale = (t_max.double() - next_t.double()).exp()
            s_scale = (s_max.double() - next_s.double()).exp()
            t_exp = (t_logits - next_t[:, None]).exp()
            s_exp = (s_logits - next_s[:, None]).exp()
            t_mass = t_mass * t_scale + t_exp.sum(-1, dtype=torch.float64)
            s_mass = s_mass * s_scale + s_exp.sum(-1, dtype=torch.float64)
            difference = difference * t_scale + (t_exp * (t_logits - s_logits)).sum(
                -1, dtype=torch.float64
            )
            t_max, s_max = next_t, next_s
            owned = predicted & (target >= column) & (target < columns.stop)
            index = (target - column).clamp(0, columns.stop - column - 1).unsqueeze(-1)
            t_target += torch.where(owned, t_logits.gather(-1, index).squeeze(-1), 0)
            s_target += torch.where(owned, s_logits.gather(-1, index).squeeze(-1), 0)
        t_norm = t_max.double() + t_mass.log()
        s_norm = s_max.double() + s_mass.log()
        totals[0] += (s_norm - s_target)[predicted].sum()
        totals[1] += predicted.sum()
        totals[4] += (t_norm - t_target)[predicted].sum()
        # KL = E_teacher[teacher_logits - candidate_logits] + log Z_candidate - log Z_teacher.
        totals[2] += (difference / t_mass + s_norm - t_norm)[mask].sum()
        totals[3] += mask.sum()
    return totals, "chunked_linear" if sliceable else "chunked_head"


def _perplexity(mean_cross_entropy):
    # PPL is secondary: an unrepresentable exponential must not hide finite KL.
    try:
        value = math.exp(mean_cross_entropy)
    except OverflowError:
        return None
    return value if math.isfinite(value) else None


def _synchronize(adapter, error):
    failures = [error for error in _all_rank_values(adapter.context, error) if error is not None]
    if failures:
        raise ValueError("Diagnostic input/placement error: " + "; ".join(failures))


def _model_output(model, batch, adapter, projection):
    inputs = adapter.prepare_batch(model, batch)
    target = getattr(model, "thinker", model)
    parameters = inspect.signature(target.forward).parameters
    inputs.update(
        {
            name: value
            for name, value in (
                ("use_cache", False),
                ("logits_to_keep", 0),
                ("num_logits_to_keep", 0),
            )
            if name in parameters
        }
    )
    if not any(p.kind == inspect.Parameter.VAR_KEYWORD for p in parameters.values()):
        inputs = {key: value for key, value in inputs.items() if key in parameters}
    if projection.head is not None:
        with capture_head_inputs(projection.head) as captured:
            adapter.forward_teacher(model, (), inputs)
        if len(captured) != 1:
            raise ValueError("Chunked diagnostics require exactly one pointwise output-head call.")
        return captured[0]
    output = adapter.forward_teacher(model, (), inputs)
    logits = (
        output
        if isinstance(output, torch.Tensor)
        else output.logits
        if hasattr(output, "logits")
        else output[0]
    )
    return logits.full_tensor() if isinstance(logits, DTensor) else logits


@torch.no_grad()
def evaluate_pair(
    teacher, candidate, batches, *, device_mesh=None, token_chunk=128, vocabulary_chunk=8192
):
    """Evaluate actual text-task/head math; labels are already shifted within documents.

    CP diagnostics use the faithful replicated full context. Only unique data
    lanes contribute to counts; every model owner executes the same head schedule.
    Caller-owned native placement is adopted, never inferred from display settings.
    """
    if min(token_chunk, vocabulary_chunk) < 1:
        raise ValueError("Diagnostic chunk sizes must be positive.")
    adapter = TransformersBypassAdapter(device_mesh)
    adapter.student = candidate
    for model in (teacher, candidate):
        adapter.adopt(model)
    context = adapter.context
    if (
        context is not None
        and context.pipeline_mesh is not None
        and context.pipeline_mesh.size() != 1
    ):
        raise ValueError("Whole-model diagnostics require PP=1.")
    projections, error = None, None
    try:
        projections = (OutputHeadProjection(teacher), OutputHeadProjection(candidate))
    except (ValueError, TypeError, AttributeError) as exc:
        error = f"{type(exc).__name__}: {exc}"
    _synchronize(adapter, error)
    assert projections is not None
    use_head = all(p.head is not None for p in projections)
    _synchronize(
        adapter,
        None
        if len(set(_all_rank_values(context, use_head))) == 1
        else "Output-head capabilities differ across owners",
    )
    modes = {
        module: module.training for model in (teacher, candidate) for module in model.modules()
    }
    device = next(candidate.parameters()).device
    totals = torch.zeros(5, device=device, dtype=torch.float64)
    projection_modes = set()
    if device.type == "cuda":
        torch.cuda.synchronize(device)
        torch.cuda.reset_peak_memory_stats(device)
    started = time.perf_counter()
    try:
        teacher.eval()
        candidate.eval()
        iterator, error = None, None
        try:
            iterator = iter(batches)
        except Exception as exc:
            error = f"{type(exc).__name__}: {exc}"
        _synchronize(adapter, error)
        assert iterator is not None
        exhausted = object()
        while True:
            batch, error = exhausted, None
            try:
                batch = next(iterator)
            except StopIteration:
                pass
            except Exception as exc:
                error = f"{type(exc).__name__}: {exc}"
            _synchronize(adapter, error)
            present = _all_rank_values(context, batch is not exhausted)
            _synchronize(
                adapter,
                None if len(set(present)) == 1 else "Data lanes have different diagnostic lengths",
            )
            if batch is exhausted:
                break
            documents, error = [], None
            try:
                documents = document_batches(batch)
            except Exception as exc:
                error = f"{type(exc).__name__}: {exc}"
            _synchronize(adapter, error)
            for index in range(max(_all_rank_values(context, len(documents)))):
                error = None
                try:
                    document = (
                        documents[index] if index < len(documents) else empty_document_batch(batch)
                    )
                    labels = document.get("labels", torch.full_like(document["input_ids"], -100))
                    valid = _valid_token_mask(document)
                    valid = torch.ones_like(labels, dtype=torch.bool) if valid is None else valid
                    if (
                        labels.shape != document["input_ids"].shape
                        or valid.shape != labels.shape
                        or labels.dtype != torch.long
                    ):
                        error = "Diagnostic labels/validity must match token IDs and use long integer labels"
                    if use_head:
                        assert projections[0].head is not None and projections[1].head is not None
                        vocabulary = projections[0].head.weight.shape[0]
                        if vocabulary != projections[1].head.weight.shape[0] or bool(
                            (
                                valid & labels.ne(-100) & ((labels < 0) | (labels >= vocabulary))
                            ).any()
                        ):
                            error = "Diagnostic target/output vocabulary mismatch"
                except Exception as exc:
                    error = f"{type(exc).__name__}: {exc}"
                _synchronize(adapter, error)
                left = _model_output(teacher, document, adapter, projections[0])
                right = _model_output(candidate, document, adapter, projections[1])
                _synchronize(
                    adapter,
                    None
                    if left.shape[:-1] == right.shape[:-1] == labels.shape
                    else "Outputs must cover every diagnostic token",
                )
                if use_head:
                    slots = max(_all_rank_values(context, labels.numel()))
                    with projections[0].weight_window(), projections[1].weight_window():
                        sliceable = all(
                            _all_rank_values(context, all(p.sliceable for p in projections))
                        )
                        values, mode = _projected_statistics(
                            left,
                            right,
                            *projections,
                            labels,
                            valid,
                            token_chunk,
                            vocabulary_chunk,
                            slots,
                            sliceable,
                        )
                    totals += values
                    projection_modes.add(mode)
                else:
                    error = (
                        None if left.shape == right.shape else "Paired vocabulary dimensions differ"
                    )
                    if bool(
                        (
                            valid & labels.ne(-100) & ((labels < 0) | (labels >= left.shape[-1]))
                        ).any()
                    ):
                        error = "Diagnostic targets are outside the output vocabulary"
                    _synchronize(adapter, error)
                    totals += diagnostic_statistics(
                        left,
                        right,
                        labels,
                        valid,
                        token_chunk=token_chunk,
                        vocabulary_chunk=vocabulary_chunk,
                    )
                    projection_modes.add("full_logits")
                del left, right
        data = None if context is None else context.data_mesh
        if data is not None and data.size() > 1:
            dist.all_reduce(totals, group=data.get_group())
        nll, targets, kl, tokens, teacher_nll = totals.tolist()
        if tokens <= 0 or not math.isfinite(kl):
            raise ValueError("Diagnostics require finite outputs and valid input tokens.")
        ce, teacher_ce = (nll / targets, teacher_nll / targets) if targets else (None, None)
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        return {
            "kld_teacher_candidate": kl / tokens,
            "mean_cross_entropy": ce,
            "teacher_mean_cross_entropy": teacher_ce,
            "ppl": None if ce is None else _perplexity(ce),
            "teacher_ppl": None if teacher_ce is None else _perplexity(teacher_ce),
            "kl_tokens": int(tokens),
            "next_token_targets": int(targets),
            "projection_modes": sorted(projection_modes),
            "context_parallel_mode": "replicated_full_context",
            "elapsed_seconds_local": time.perf_counter() - started,
            "peak_allocated_bytes_local": torch.cuda.max_memory_allocated(device)
            if device.type == "cuda"
            else 0,
        }
    finally:
        for module, training in modes.items():
            module.training = training
