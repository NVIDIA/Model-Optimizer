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

"""Temporary CI-only, single-unit IQ2_S before/after probe. Never dispatches jobs."""

import argparse
import ast
import gc
import hashlib
import importlib
import json
import statistics
import subprocess
import sys
import time
import types
from pathlib import Path

BASELINE = "a5b31b3630a3ff7f175fd14c0919301e98bbef94"
CANDIDATE = "8d2fa213c171b9ae6fa4884337ec11a5c0f57e0b"
PR1 = "c182c371d69c6753c217f8777c762a388bed10aa"
FORMATS = "modelopt/torch/quantization/repair_formats.py"
PROBE_PATH = ".github/scripts/bypass_scale_probe.py"
ALLOWLIST = {".github/workflows/gpu_tests.yml", PROBE_PATH}
SHAPE = (4096, 14336)
ELEMENTS = 4096 * 14336
GPU_BOUND = 4 * 1024**3
REPEATS = 3


def git(*args):
    """Read Git objects without changing refs or files."""
    return subprocess.check_output(["git", *args], text=True).strip()


def sha(value):
    """Fingerprint exact source bytes."""
    return hashlib.sha256(value).hexdigest()


def source_at(ref):
    """Read the complete original formats module."""
    return subprocess.check_output(["git", "show", f"{ref}:{FORMATS}"], text=True)


def source_gate():
    """Refuse production or test changes from the candidate."""
    baseline = source_at(BASELINE)
    candidate = source_at(CANDIDATE)
    assert (
        sha(baseline.encode()) == "4b04b2d0fd6e48f80dc99341f8c45aafce5eed4092c9bd27958046ca3463792b"
    )
    allowed_source = {FORMATS, "tests/unit/torch/quantization/test_repair_formats.py"}
    assert set(git("diff", "--name-only", BASELINE, CANDIDATE).splitlines()) == allowed_source
    assert set(git("diff", "--name-only", CANDIDATE, "HEAD").splitlines()) == ALLOWLIST
    parents = [
        line.split()[1]
        for line in git("cat-file", "-p", "HEAD").splitlines()
        if line.startswith("parent ")
    ]
    assert parents == [CANDIDATE], "Validation commit must be directly parented by candidate"
    for folder in ("modelopt", "tests"):
        assert git("rev-parse", f"{CANDIDATE}:{folder}") == git("rev-parse", f"HEAD:{folder}")
    subprocess.run(["git", "diff", "--exit-code", "HEAD", "--", "modelopt", "tests"], check=True)
    assert Path(FORMATS).read_bytes() == candidate.encode()
    classes = {}
    for label, text in (("baseline", baseline), ("candidate", candidate), ("pr1", source_at(PR1))):
        node = next(
            n
            for n in ast.parse(text).body
            if isinstance(n, ast.ClassDef) and n.name == "_ReplayScales"
        )
        body = ast.get_source_segment(text, node)
        assert body is not None
        backward = next(
            n for n in node.body if isinstance(n, ast.FunctionDef) and n.name == "backward"
        )
        classes[label] = {
            "sha256": sha(body.encode()),
            "saved_tensor_accesses": sum(
                isinstance(n, ast.Attribute) and n.attr == "saved_tensors"
                for n in ast.walk(backward)
            ),
        }
    assert classes["pr1"]["sha256"] == classes["baseline"]["sha256"]
    assert classes["baseline"]["saved_tensor_accesses"] == 2
    assert classes["candidate"]["saved_tensor_accesses"] == 1
    return baseline, {
        "baseline": BASELINE,
        "candidate": CANDIDATE,
        "validation": git("rev-parse", "HEAD"),
        "baseline_source_sha256": sha(baseline.encode()),
        "candidate_source_sha256": sha(candidate.encode()),
        "replay_classes": classes,
        "candidate_production_and_tests_identical": True,
        "CI_only_overlay_paths": sorted(ALLOWLIST),
    }


def main():
    """Run the bounded probe after verifying its immutable inputs."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--verify-only", action="store_true")
    args = parser.parse_args()
    baseline_source, provenance = source_gate()
    if args.verify_only:
        print("Bypass probe source gate: " + json.dumps(provenance, sort_keys=True), flush=True)
        return

    # Keep --verify-only stdlib-only before installing the container test dependencies.
    import torch
    from torch import nn

    import modelopt.torch.quantization as mtq
    from modelopt.torch.quantization.ggml import IQ_FORMAT_REGISTRY
    from modelopt.torch.quantization.ggml.common import pin_packed_weight, pinned_packed_weight

    candidate_module = importlib.import_module("modelopt.torch.quantization.repair_formats")
    assert candidate_module.__file__ is not None
    assert Path(candidate_module.__file__).resolve() == (Path.cwd() / FORMATS).resolve()
    baseline_module = types.ModuleType("modelopt.torch.quantization._bypass_probe_baseline")
    baseline_module.__package__ = "modelopt.torch.quantization"
    baseline_module.__file__ = f"<git:{BASELINE}:{FORMATS}>"
    sys.modules[baseline_module.__name__] = baseline_module
    exec(compile(baseline_source, baseline_module.__file__, "exec"), baseline_module.__dict__)
    assert torch.cuda.is_available(), "Probe requires hosted CUDA; no CPU substitution"
    torch.cuda.set_device(0)
    torch.manual_seed(107)
    dtype = torch.float16
    fmt = IQ_FORMAT_REGISTRY["iq2_s"]
    model = nn.Linear(SHAPE[1], SHAPE[0], bias=False, device="cuda", dtype=dtype)
    mtq.quantize(
        model,
        {
            "quant_cfg": [
                {"quantizer_name": "*", "enable": False},
                {
                    "quantizer_name": "weight_quantizer",
                    "cfg": {"num_bits": "iq2_s", "backend": "ggml", "block_sizes": {-1: 256}},
                },
            ],
            "algorithm": None,
        },
    )
    model.requires_grad_(False)
    weight = model._parameters["weight"]
    quantizer = model.weight_quantizer
    blocks = ELEMENTS // fmt.block_size
    packed = torch.randint(0, 256, (blocks, fmt.block_bytes), device="cuda", dtype=torch.uint8)
    assert fmt.replace_payload_scales is not None
    packed = fmt.replace_payload_scales(
        packed, torch.full((blocks,), 2**-8, device="cuda", dtype=dtype)
    )
    weight.copy_(fmt.dequantize(packed, torch.tensor(SHAPE), dtype=dtype))
    if quantizer.is_static_block_quant:
        quantizer._setup_for_blockquant(weight)
        processed = quantizer._process_for_blockquant(weight)
    else:
        processed = weight
    pin_packed_weight(quantizer, fmt.name, packed, source=processed)
    assert pinned_packed_weight(quantizer, fmt.name) is not None
    assert torch.equal(fmt.pack(processed, quantizer).reshape_as(packed), packed)

    def clean():
        gc.collect()
        torch.cuda.synchronize()

    def check_bound():
        peak = torch.cuda.max_memory_allocated()
        assert peak <= GPU_BOUND, (peak, GPU_BOUND)
        return peak

    def digest(tensor):
        value = tensor.detach().contiguous().reshape(-1)
        h = hashlib.sha256()
        for start in range(0, value.numel(), 262144):
            h.update(value[start : start + 262144].cpu().numpy().tobytes())
        return h.hexdigest()

    def sample(fn):
        clean()
        before = torch.cuda.memory_allocated()
        torch.cuda.reset_peak_memory_stats()
        start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        wall = time.perf_counter()
        start.record()
        value = fn()
        end.record()
        torch.cuda.synchronize()
        result = {
            "cuda_ms": start.elapsed_time(end),
            "wall_ms": (time.perf_counter() - wall) * 1000,
            "allocated_before_bytes": before,
            "allocated_delta_bytes": torch.cuda.memory_allocated() - before,
            "extra_peak_bytes": check_bound() - before,
        }
        return value, result

    def summarize(samples):
        return {
            key: {
                "min": min(s[key] for s in samples),
                "median": statistics.median(s[key] for s in samples),
                "max": max(s[key] for s in samples),
            }
            for key in samples[0]
        }

    results = []
    input_peak_bytes = check_bound()
    for label, module, expected_loads in (
        ("before", baseline_module, 2),
        ("after", candidate_module, 1),
    ):

        def factory(module=module):
            return module.IQScaleState(quantizer, processed, SHAPE, field_prefix="weight_quantizer")

        state = factory()
        check_bound()
        del state
        clean()
        construction = []
        for iteration in range(REPEATS):
            state, measured = sample(factory)
            construction.append(measured)
            if iteration < REPEATS - 1:
                del state
        assert state.local_base.min().item() > 0
        with torch.no_grad():
            state.global_delta.fill_(0.125)
            state.local_delta.fill_(0.125)
            state.global_base[0] = 0
            state.global_delta[0] = 0
        phases = {}
        for name, fn in (
            ("scales_grad_forward", state.scales),
            ("projected_steps_grad_forward", state.projected_steps),
            ("project_no_grad", state.project_),
        ):
            warm = fn()
            del warm
            samples = []
            for _ in range(REPEATS):
                value, measured = sample(fn)
                if name != "project_no_grad":
                    tensors = value if isinstance(value, tuple) else (value,)
                    assert any(t is not None and t.requires_grad for t in tensors)
                    del tensors
                samples.append(measured)
                del value
            phases[name] = summarize(samples)
        del fn
        parameters = tuple(state.parameters())
        backward_samples = []
        parity = None
        for iteration in range(REPEATS + 1):
            clean()
            counts = {
                "level_packs": 0,
                "level_loads": 0,
                "level_H2D_bytes": 0,
                "all_offloaded_bytes": 0,
            }
            offload = torch.autograd.graph.save_on_cpu(pin_memory=True)

            def pack_hook(tensor, offload=offload, counts=counts):
                is_levels = tensor.dtype == torch.int8 and tensor.numel() == ELEMENTS
                counts["level_packs"] += int(is_levels)
                counts["all_offloaded_bytes"] += tensor.numel() * tensor.element_size()
                return offload.pack_hook(tensor), is_levels

            def unpack_hook(saved, offload=offload, counts=counts):
                data, is_levels = saved
                if is_levels:
                    counts["level_loads"] += 1
                    counts["level_H2D_bytes"] += ELEMENTS
                return offload.unpack_hook(data)

            with torch.autograd.graph.saved_tensors_hooks(pack_hook, unpack_hook):
                steps = state.projected_steps()
                levels, unused = fmt.decode_int8(state.packed, torch.tensor(SHAPE))
                levels = levels.reshape(-1, steps.shape[-1], fmt.int8_group_size)
                gradient = levels.to(dtype)
                output = module._ReplayScales.apply(levels, steps, dtype)
            del levels, steps, unused
            assert counts["all_offloaded_bytes"] <= 512 * 1024**2
            actual, measured = sample(
                lambda output=output, parameters=parameters, gradient=gradient: torch.autograd.grad(
                    output, parameters, gradient
                )
            )
            assert counts["level_packs"] == 1 and counts["level_loads"] == expected_loads, counts
            measured.update(counts)
            if iteration:
                backward_samples.append(measured)
            if iteration == REPEATS:
                native = state.native_payload()
                decoded = fmt.dequantize(native, torch.tensor(SHAPE), dtype=dtype)
                assert torch.equal(output.reshape(SHAPE), decoded), "Native numeric parity failed"
                ref_steps = state.projected_steps()
                ref_levels, unused = fmt.decode_int8(state.packed, torch.tensor(SHAPE))
                expected = (
                    ref_levels.reshape(-1, ref_steps.shape[-1], fmt.int8_group_size).float()
                    * ref_steps.unsqueeze(-1)
                ).to(dtype)
                reference = torch.autograd.grad(expected, parameters, gradient)
                assert all(torch.equal(a, r) for a, r in zip(actual, reference)), (
                    "Native VJP parity failed"
                )
                assert actual[0][0].item() > 0, "Global zero recovery derivative was lost"
                parity = {
                    "native_exact": True,
                    "VJP_exact": True,
                    "global_zero_gradient_positive": True,
                    "native_sha256": digest(native),
                    "output_sha256": digest(output),
                    "master_gradient_sha256": [digest(t) for t in actual],
                }
                with torch.no_grad():
                    state.global_delta[0] = 0.125
                assert state.scales()[0][0].item() > 0, "Global zero scale cannot recover"
                parity["global_zero_recovers"] = True
                del native, decoded, ref_steps, ref_levels, unused, expected, reference
                verification_peak = check_bound()
            del output, gradient, actual, offload
        results.append(
            {
                "implementation": label,
                "construction": summarize(construction),
                **phases,
                "backward_real_CPU_offload": summarize(backward_samples),
                "parity": parity,
                "verification_peak_bytes": verification_peak,
            }
        )
        print("Bypass state probe: " + json.dumps(results[-1], sort_keys=True), flush=True)
        del parameters, state
        clean()
    assert results[0]["parity"] == results[1]["parity"], (
        "Before/after numeric or VJP fingerprints differ"
    )
    print(
        "Bypass state probe summary: "
        + json.dumps(
            {
                "provenance": provenance,
                "device": torch.cuda.get_device_name(0),
                "visible_GPUs": torch.cuda.device_count(),
                "active_probe_device": 0,
                "torch": torch.__version__,
                "shape": SHAPE,
                "format": "iq2_s",
                "dtype": str(dtype),
                "elements": ELEMENTS,
                "one_levels_tensor_bytes": ELEMENTS,
                "probe_absolute_GPU_bound_bytes": GPU_BOUND,
                "input_peak_bytes": input_peak_bytes,
                "logical_offloaded_tensor_bound_bytes": 512 * 1024**2,
                "repeats": REPEATS,
                "before_after_exact_parity": True,
                "scope": (
                    "One native unit. Forward scales retain autograd; project_ is no-grad. "
                    "Backward includes real save_on_cpu hooks and field-master VJPs; portable replay "
                    "is used explicitly, without replacing native kernels. Verification allocations "
                    "are outside timed phases."
                ),
            },
            sort_keys=True,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
