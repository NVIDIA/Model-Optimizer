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

import dataclasses
import json
from copy import deepcopy
from hashlib import sha256
from types import SimpleNamespace

import onnx
import pytest
from _test_utils.onnx.quantization.autotune.models import _create_simple_conv_onnx_model

from modelopt.onnx.quantization.autotune import workflows


class _FakeAutotuner:
    instances = []

    def __init__(self, model):
        self.model = model
        self.regions = []
        self.profiled_patterns = []
        self.baseline_latency_ms = None
        self.force_no_qdq = False
        self.final_decision = None
        self.proxy_decision = None
        self.proxy_record_count = 0
        self.current_profile_region = None
        self.save_count = 0
        self.config = None
        self.resume_fingerprint = None
        self.__class__.instances.append(self)

    def initialize(self, config, pattern_cache=None):
        self.config = config

    def set_resume_fingerprint(self, **fingerprint):
        self.resume_fingerprint = fingerprint

    def export_onnx(
        self,
        output_path=None,
        insert_qdq=True,
        best=False,
        model_transform=None,
    ):
        del best
        model = deepcopy(self.model)
        if insert_qdq and not self.force_no_qdq:
            metadata = model.metadata_props.add()
            metadata.key = "autotune_test_qdq"
            metadata.value = "true"
            model.graph.node.append(
                onnx.helper.make_node(
                    "QuantizeLinear",
                    inputs=["input", "conv_weight"],
                    outputs=["unused_quantized_output"],
                    name="autotune_test_q",
                )
            )
        if model_transform is not None:
            model = model_transform(model)
        if output_path is not None:
            onnx.save(model, output_path)
        return model.SerializeToString()

    def submit(self, latency_ms, success=True):
        assert success
        self.baseline_latency_ms = latency_ms

    def save_state(self, output_path):
        del output_path
        self.save_count += 1

    def set_force_no_qdq(self, enabled=True):
        self.force_no_qdq = enabled

    def record_final_decision(self, **decision):
        self.final_decision = decision

    def record_proxy_decision(self, **decision):
        self.proxy_decision = decision
        self.proxy_record_count += 1

    def import_insertion_points(self, quantized_tensors):
        del quantized_tensors

    def load_state(self, input_path):
        del input_path

    def set_profile_region(self, region, commit=True):
        del region, commit


class _FakePatternSchemes:
    def __init__(self):
        self.schemes = []
        self.selected_scheme = None

    def select_best(self, performance_threshold):
        del performance_threshold
        self.selected_scheme = min(self.schemes, key=lambda scheme: scheme.latency_ms)

    @property
    def profiled_override_count(self):
        return max(0, len(self.schemes) - 1)


class _FakeRegionAutotuner(_FakeAutotuner):
    def __init__(self, model):
        super().__init__(model)
        self.regions = [SimpleNamespace(id=0, level=0)]
        self.graph = SimpleNamespace(nodes=[])
        self.current_profile_pattern_schemes = None
        self.events = []
        self.generated_count = 0

    def set_profile_region(self, region, commit=True):
        del commit
        if region is not None:
            self.current_profile_region = region
            self.current_profile_pattern_schemes = _FakePatternSchemes()

    def begin_inherit_profile(self):
        self.events.append("begin_inherit")
        return 0

    def submit_inherit(self, latency_ms, success=True):
        self.events.append("submit_inherit")
        assert success
        self.current_profile_pattern_schemes.schemes.append(SimpleNamespace(latency_ms=latency_ms))

    def generate(self):
        self.events.append("generate")
        self.generated_count += 1
        return self.generated_count

    def submit(self, latency_ms, success=True):
        if self.baseline_latency_ms is None:
            self.events.append("submit_baseline")
            self.baseline_latency_ms = latency_ms
        else:
            self.events.append("submit_override")
            assert success
            self.current_profile_pattern_schemes.schemes.append(
                SimpleNamespace(latency_ms=latency_ms)
            )

    def save_state(self, output_path):
        self.events.append("save")
        super().save_state(output_path)


class _FakeResumedRegionAutotuner(_FakeRegionAutotuner):
    def load_state(self, input_path):
        del input_path
        self.baseline_latency_ms = 1.0
        self.current_profile_region = self.regions[0]
        self.current_profile_pattern_schemes = _FakePatternSchemes()
        self.current_profile_pattern_schemes.schemes.extend(
            [
                SimpleNamespace(latency_ms=0.9, is_profiled=True),
                SimpleNamespace(latency_ms=0.85, is_profiled=True),
            ]
        )

    def set_profile_region(self, region, commit=True):
        if region is self.current_profile_region:
            return
        super().set_profile_region(region, commit)

    def begin_inherit_profile(self):
        return -1


class _FakePersistedProxyAutotuner(_FakeAutotuner):
    def load_state(self, input_path):
        del input_path
        self.baseline_latency_ms = 1.0

        self.set_force_no_qdq(False)
        candidate = onnx.load_from_string(self.export_onnx(insert_qdq=True))
        candidate_sha256 = sha256(candidate.SerializeToString(deterministic=True)).hexdigest()

        self.set_force_no_qdq(True)
        selected = onnx.load_from_string(self.export_onnx(insert_qdq=False))
        selected_sha256 = sha256(selected.SerializeToString(deterministic=True)).hexdigest()
        self.proxy_decision = {
            "proxy_selection": "no_qdq",
            "baseline_latency_ms": self.baseline_latency_ms,
            "candidate_latency_ms": 0.99,
            "candidate_quantization_site_count": 1,
            "candidate_model_sha256": candidate_sha256,
            "selected_model_sha256": selected_sha256,
        }


@pytest.fixture(autouse=True)
def _patch_autotuner(monkeypatch):
    _FakeAutotuner.instances.clear()
    monkeypatch.setattr(workflows, "QDQAutotuner", _FakeAutotuner)


def _run_workflow(monkeypatch, tmp_path, latencies, model_transform=None):
    measurements = iter(latencies)
    monkeypatch.setattr(
        workflows,
        "benchmark_onnx_model",
        lambda *args, **kwargs: next(measurements),
    )
    autotuner = workflows.region_pattern_autotuning_workflow(
        _create_simple_conv_onnx_model(),
        output_dir=tmp_path,
        num_schemes_per_region=1,
        model_transform=model_transform,
    )
    return autotuner, onnx.load(tmp_path / "optimized_final.onnx")


def _has_qdq_marker(model):
    return any(prop.key == "autotune_test_qdq" for prop in model.metadata_props)


@pytest.mark.parametrize(
    ("reference", "candidate", "expected"),
    [
        (1.0, 1.0 / 1.019, False),
        (1.0, 1.0 / 1.02, True),
        (1.0, 1.0 / 1.021, True),
        (1.0, float("nan"), False),
        (1.0, float("inf"), False),
        (1.0, 0.0, False),
        (1.0, -1.0, False),
    ],
)
def test_performance_threshold(reference, candidate, expected):
    assert workflows._meets_performance_threshold(reference, candidate, 1.02) is expected


@pytest.mark.parametrize("num_schemes_per_region", [0, -1])
def test_override_budget_must_include_no_qdq(tmp_path, num_schemes_per_region):
    with pytest.raises(ValueError, match="must be at least 1"):
        workflows.region_pattern_autotuning_workflow(
            _create_simple_conv_onnx_model(),
            output_dir=tmp_path,
            num_schemes_per_region=num_schemes_per_region,
        )


@pytest.mark.parametrize("baseline_latency", [float("nan"), float("inf"), 0.0, -1.0])
def test_invalid_baseline_aborts(monkeypatch, tmp_path, baseline_latency):
    monkeypatch.setattr(
        workflows,
        "benchmark_onnx_model",
        lambda *args, **kwargs: baseline_latency,
    )

    with pytest.raises(RuntimeError, match="valid baseline latency"):
        workflows.region_pattern_autotuning_workflow(
            _create_simple_conv_onnx_model(), output_dir=tmp_path
        )


def test_subthreshold_final_reexports_no_qdq(monkeypatch, tmp_path):
    autotuner, final_model = _run_workflow(monkeypatch, tmp_path, [1.0, 0.99])

    assert autotuner.force_no_qdq
    assert not _has_qdq_marker(final_model)
    assert autotuner.save_count == 2
    assert autotuner.final_decision is None
    assert autotuner.proxy_decision["proxy_selection"] == "no_qdq"
    assert autotuner.proxy_decision["baseline_latency_ms"] == 1.0
    assert autotuner.proxy_decision["candidate_latency_ms"] == 0.99
    assert autotuner.proxy_decision["candidate_quantization_site_count"] == 1
    baseline_sha256 = sha256(
        onnx.load(tmp_path / "baseline.onnx").SerializeToString(deterministic=True)
    ).hexdigest()
    assert autotuner.proxy_decision["selected_model_sha256"] == baseline_sha256
    assert (
        autotuner.proxy_decision["candidate_model_sha256"]
        != autotuner.proxy_decision["selected_model_sha256"]
    )


def test_final_above_threshold_keeps_qdq(monkeypatch, tmp_path):
    autotuner, final_model = _run_workflow(monkeypatch, tmp_path, [1.0, 0.95])

    assert not autotuner.force_no_qdq
    assert _has_qdq_marker(final_model)
    assert autotuner.final_decision is None
    assert autotuner.proxy_decision["proxy_selection"] == "qdq"
    assert autotuner.proxy_decision["candidate_latency_ms"] == 0.95
    assert (
        autotuner.proxy_decision["candidate_model_sha256"]
        == autotuner.proxy_decision["selected_model_sha256"]
    )


@pytest.mark.parametrize("final_latency", [float("nan"), float("inf"), 0.0, -1.0])
def test_invalid_final_reexports_no_qdq(monkeypatch, tmp_path, final_latency):
    autotuner, final_model = _run_workflow(monkeypatch, tmp_path, [1.0, final_latency])

    assert autotuner.force_no_qdq
    assert not _has_qdq_marker(final_model)
    assert autotuner.final_decision is None
    assert autotuner.proxy_decision["proxy_selection"] == "no_qdq"
    assert autotuner.proxy_decision["candidate_latency_ms"] is None


def test_matching_proxy_decision_skips_benchmark_and_preserves_fallback(monkeypatch, tmp_path):
    monkeypatch.setattr(workflows, "QDQAutotuner", _FakePersistedProxyAutotuner)
    monkeypatch.setattr(
        workflows,
        "benchmark_onnx_model",
        lambda *args, **kwargs: pytest.fail("a validated proxy decision must not be remeasured"),
    )
    state_path = tmp_path / "state.yaml"
    state_path.touch()

    autotuner = workflows.region_pattern_autotuning_workflow(
        _create_simple_conv_onnx_model(),
        output_dir=tmp_path,
        state_file=str(state_path),
        num_schemes_per_region=1,
    )

    assert autotuner.force_no_qdq
    assert autotuner.proxy_decision["proxy_selection"] == "no_qdq"
    assert autotuner.proxy_record_count == 0
    assert not _has_qdq_marker(onnx.load(tmp_path / "optimized_final.onnx"))


def test_stale_proxy_candidate_hash_is_remeasured(monkeypatch, tmp_path):
    class StaleProxyAutotuner(_FakePersistedProxyAutotuner):
        def load_state(self, input_path):
            super().load_state(input_path)
            self.proxy_decision["candidate_model_sha256"] = "stale"

    monkeypatch.setattr(workflows, "QDQAutotuner", StaleProxyAutotuner)
    measurements = []

    def benchmark(*args, **kwargs):
        measurements.append(args[0])
        return 0.95

    monkeypatch.setattr(workflows, "benchmark_onnx_model", benchmark)
    state_path = tmp_path / "state.yaml"
    state_path.touch()

    autotuner = workflows.region_pattern_autotuning_workflow(
        _create_simple_conv_onnx_model(),
        output_dir=tmp_path,
        state_file=str(state_path),
        num_schemes_per_region=1,
    )

    assert len(measurements) == 1
    assert not autotuner.force_no_qdq
    assert autotuner.proxy_decision["proxy_selection"] == "qdq"
    assert autotuner.proxy_decision["candidate_model_sha256"] != "stale"
    assert autotuner.proxy_record_count == 1


def test_model_transform_applies_to_all_benchmark_exports(monkeypatch, tmp_path):
    transformed_models = []

    def transform(model):
        model.producer_name = "transformed"
        transformed_models.append(model)
        return model

    _, final_model = _run_workflow(monkeypatch, tmp_path, [1.0, 0.99], transform)

    assert len(transformed_models) == 3
    assert final_model.producer_name == "transformed"


def test_resume_fingerprint_includes_workflow_and_caller_options(monkeypatch, tmp_path):
    monkeypatch.setattr(workflows, "_benchmark_instance", None)
    monkeypatch.setattr(
        workflows,
        "benchmark_onnx_model",
        lambda *args, **kwargs: 1.0,
    )
    pattern_cache_path = tmp_path / "private-pattern-cache.yaml"
    workflows.PatternCache().save(str(pattern_cache_path))
    qdq_baseline_path = tmp_path / "private-qdq-baseline.onnx"
    onnx.save(_create_simple_conv_onnx_model(), qdq_baseline_path)

    autotuner = workflows.region_pattern_autotuning_workflow(
        _create_simple_conv_onnx_model(),
        output_dir=tmp_path,
        num_schemes_per_region=7,
        quant_type="fp8",
        default_dq_dtype="float16",
        node_filter_list=["private_node*"],
        pattern_cache_file=str(pattern_cache_path),
        qdq_baseline_model=str(qdq_baseline_path),
        resume_fingerprint={"private_runtime_option": "private_value"},
    )

    fingerprint = autotuner.resume_fingerprint
    assert fingerprint["config"] == dataclasses.asdict(autotuner.config)
    assert fingerprint["search"]["num_schemes_per_region"] == 7
    assert (
        fingerprint["search"]["pattern_cache"]["sha256"]
        == sha256(pattern_cache_path.read_bytes()).hexdigest()
    )
    assert (
        fingerprint["search"]["qdq_baseline"]["sha256"]
        == sha256(qdq_baseline_path.read_bytes()).hexdigest()
    )
    assert autotuner.resume_fingerprint["benchmark"] == {"backend": None}
    serialized = json.dumps(fingerprint)
    assert "private_node" not in serialized
    assert "private-pattern" not in serialized
    assert "private-qdq" not in serialized
    assert "private_runtime_option" not in serialized
    assert "private_value" not in serialized


def test_input_artifact_fingerprints_are_content_sensitive_and_redacted(tmp_path):
    artifact_path = tmp_path / "private-cache-name.yaml"
    artifact_path.write_text("first")
    first = workflows._get_file_identity(str(artifact_path))
    artifact_path.write_text("second")
    second = workflows._get_file_identity(str(artifact_path))

    assert first["sha256"] != second["sha256"]
    assert str(tmp_path) not in json.dumps(first)
    assert artifact_path.name not in json.dumps(first)


def test_timing_cache_fingerprint_uses_only_redacted_path_identity(tmp_path):
    timing_cache = tmp_path / "private-timing.cache"
    timing_cache.write_bytes(b"first")
    first = workflows._get_path_identity(str(timing_cache))
    timing_cache.write_bytes(b"changed-by-tensorrt")
    second = workflows._get_path_identity(str(timing_cache))

    assert first == second
    assert "sha256" not in first
    assert str(tmp_path) not in json.dumps(first)
    assert timing_cache.name not in json.dumps(first)


def test_trtexec_fingerprint_uses_resolved_binary_and_redacts_inputs(monkeypatch, tmp_path):
    class FakeTrtExecBenchmark:
        timing_cache_file = str(tmp_path / "private-timing.cache")
        warmup_runs = 5
        timing_runs = 10
        plugin_libraries = [str(tmp_path / "private-plugin.so")]
        trtexec_args = ["--private-option=secret"]

    trtexec_path = tmp_path / "trtexec"
    trtexec_path.write_bytes(b"binary-v1")
    plugin_path = tmp_path / "private-plugin.so"
    plugin_path.write_bytes(b"plugin-v1")
    monkeypatch.setattr(workflows, "TrtExecBenchmark", FakeTrtExecBenchmark)
    monkeypatch.setattr(workflows, "_benchmark_instance", FakeTrtExecBenchmark())
    monkeypatch.setattr(
        workflows.shutil,
        "which",
        lambda command: str(trtexec_path) if command == "trtexec" else None,
    )

    commands = []

    def run_trtexec(args=None, timeout=None):
        commands.append((args, timeout))
        return SimpleNamespace(
            stdout="TensorRT.trtexec [TensorRT v10.13.3]", stderr="", returncode=0
        )

    monkeypatch.setattr(
        workflows,
        "_run_trtexec",
        run_trtexec,
    )

    fingerprint = workflows._get_benchmark_fingerprint()

    assert fingerprint["trtexec"]["available"]
    assert fingerprint["trtexec"]["version"] == "10.13.3"
    assert commands == [(None, 10)]
    assert fingerprint["trtexec"]["binary"]["sha256"] == sha256(b"binary-v1").hexdigest()
    assert fingerprint["plugin_libraries"][0]["sha256"] == sha256(b"plugin-v1").hexdigest()
    serialized = json.dumps(fingerprint)
    assert str(tmp_path) not in serialized
    assert "private-plugin" not in serialized
    assert "private-option" not in serialized
    assert "secret" not in serialized


def test_inherit_control_is_outside_override_budget(monkeypatch, tmp_path):
    monkeypatch.setattr(workflows, "QDQAutotuner", _FakeRegionAutotuner)
    measurements = iter([1.0, 0.9, 0.88, 0.87, 0.8])
    monkeypatch.setattr(
        workflows,
        "benchmark_onnx_model",
        lambda *args, **kwargs: next(measurements),
    )

    autotuner = workflows.region_pattern_autotuning_workflow(
        _create_simple_conv_onnx_model(),
        output_dir=tmp_path,
        num_schemes_per_region=2,
    )

    assert autotuner.events.count("begin_inherit") == 1
    assert autotuner.events.count("submit_inherit") == 1
    assert autotuner.events.count("generate") == 2
    for index, event in enumerate(autotuner.events):
        if event.startswith("submit_"):
            assert autotuner.events[index + 1] == "save"


def test_resume_uses_remaining_override_budget(monkeypatch, tmp_path):
    monkeypatch.setattr(workflows, "QDQAutotuner", _FakeResumedRegionAutotuner)
    state_path = tmp_path / "state.yaml"
    state_path.touch()
    measurements = iter([0.84, 0.83, 0.8])
    monkeypatch.setattr(
        workflows,
        "benchmark_onnx_model",
        lambda *args, **kwargs: next(measurements),
    )

    autotuner = workflows.region_pattern_autotuning_workflow(
        _create_simple_conv_onnx_model(),
        output_dir=tmp_path,
        state_file=str(state_path),
        num_schemes_per_region=3,
    )

    assert autotuner.events.count("generate") == 2
    assert autotuner.events.count("submit_inherit") == 0


def test_resume_preserves_timing_cache_flush_sequence(monkeypatch, tmp_path):
    def run(autotuner_class, output_dir, state_file=None):
        flushes = []

        def benchmark(*args, **kwargs):
            if "scheme_" in str(args[1]):
                flushes.append(kwargs.get("flush_timing_cache", False))
            return 1.0

        monkeypatch.setattr(workflows, "QDQAutotuner", autotuner_class)
        monkeypatch.setattr(workflows, "benchmark_onnx_model", benchmark)
        workflows.region_pattern_autotuning_workflow(
            _create_simple_conv_onnx_model(),
            output_dir=output_dir,
            state_file=state_file,
            num_schemes_per_region=10,
        )
        return flushes

    uninterrupted_flushes = run(_FakeRegionAutotuner, tmp_path / "uninterrupted")

    class ResumedAutotuner(_FakeResumedRegionAutotuner):
        def load_state(self, input_path):
            super().load_state(input_path)
            self.current_profile_pattern_schemes.schemes.extend(
                [
                    SimpleNamespace(latency_ms=0.84, is_profiled=True),
                    SimpleNamespace(latency_ms=0.83, is_profiled=True),
                    SimpleNamespace(latency_ms=0.82, is_profiled=True),
                ]
            )

    resumed_dir = tmp_path / "resumed"
    resumed_dir.mkdir()
    state_path = resumed_dir / "state.yaml"
    state_path.touch()
    resumed_flushes = run(ResumedAutotuner, resumed_dir, str(state_path))

    assert uninterrupted_flushes == [False] * 8 + [True, False]
    assert resumed_flushes == uninterrupted_flushes[4:]
