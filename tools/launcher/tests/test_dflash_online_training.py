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

"""dflash_online_training.sh: export validation and regression checks."""

import json
import os
import shutil
import subprocess
from pathlib import Path

import pytest

_COMMON = Path(__file__).resolve().parents[1] / "common"
_EXPORT = "modules/Model-Optimizer/examples/speculative_decoding/scripts/export_hf_checkpoint.py"
_CONFIG = '{"model_type": "qwen3"}'
_MV = shutil.which("mv")


def _run(tmp_path, train_exit=0, loss=4.0, checkpoint_time=None, **env_overrides):
    """Run the script in a sandbox: stub pip/nvidia-smi/accelerate/export/torch, real gate.

    Returns (process, verdict or None, whether training was launched).
    """
    (tmp_path / "common" / "specdec").mkdir(parents=True, exist_ok=True)
    for name in ("service_utils.sh", "check_regression.py", "specdec/dflash_online_training.sh"):
        shutil.copy(_COMMON / name, tmp_path / "common" / name)
    (tmp_path / "modules/Model-Optimizer/modelopt").mkdir(
        parents=True, exist_ok=True
    )  # service_utils appends here
    export = tmp_path / _EXPORT
    export.parent.mkdir(parents=True, exist_ok=True)
    export.write_text(
        "import os, sys\n"
        "src = sys.argv[sys.argv.index('--model_path') + 1]\n"
        f"open({str(tmp_path / 'exports.log')!r}, 'a').write(src + '\\n')\n"
        "d =sys.argv[sys.argv.index('--export_path') + 1]\n"
        "os.makedirs(d, exist_ok=True)\n"
        "open(os.path.join(d, 'model.safetensors'), 'w').write('new')\n"
        "if os.environ.get('EXPORT_FAIL'):\n"
        "    sys.exit(1)\n"
        f"config = os.environ.get('EXPORT_CONFIG', {_CONFIG!r})\n"
        "open(os.path.join(d, 'config.json'), 'w').write(config)\n"
    )
    # The script counts GPUs with torch; without it that line trips the ERR trap and every
    # run exits 1, which would hide the exit code under test.
    pystub = tmp_path / "pystub"
    pystub.mkdir(exist_ok=True)
    (pystub / "torch.py").write_text(
        "class cuda:\n    @staticmethod\n    def device_count():\n        return 1\n"
    )
    launched = tmp_path / "launched"
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir(exist_ok=True)
    for tool, body in (
        ("pip", "exit 0"),
        ("nvidia-smi", "echo 1"),
        ("accelerate", f"touch {launched}; exit {train_exit}"),
        (
            "mv",
            (
                '[ -n "$PUBLISH_FAIL" ] && [[ "$2" == *.tmp ]] && exit 1\n'
                '[ -n "$RECOVER_FAIL" ] && [[ "$2" == *.old ]] && exit 1\n'
                f'exec {_MV} "$@"'
            ),
        ),
    ):
        (bin_dir / tool).write_text(f"#!/bin/bash\n{body}\n")
        (bin_dir / tool).chmod(0o755)

    # A checkpoint with the given loss, and a passing verdict from an earlier run.
    out = tmp_path / "out"
    (out / "checkpoint-2000").mkdir(parents=True, exist_ok=True)
    (out / "checkpoint-2000" / "trainer_state.json").write_text(
        json.dumps({"log_history": [{"loss": loss}]})
    )
    if checkpoint_time is not None:
        os.utime(
            out / "checkpoint-2000" / "trainer_state.json",
            (checkpoint_time, checkpoint_time),
        )
    (out / "regression_check.json").write_text('{"pass": true}')

    env = {
        **os.environ,
        "PATH": f"{bin_dir}:{os.environ['PATH']}",
        "PYTHONPATH": str(pystub),
        "MAX_FINAL_LOSS": "5.0",
        **env_overrides,
    }
    env.pop("SLURM_PROCID", None)
    proc = subprocess.run(
        ["bash", "common/specdec/dflash_online_training.sh", f"training.output_dir={out}"],
        cwd=tmp_path,
        env=env,
        capture_output=True,
        text=True,
    )
    verdict = out / "regression_check.json"
    return proc, json.loads(verdict.read_text()) if verdict.exists() else None, launched.exists()


def _seed_export(path, config):
    path.mkdir(parents=True)
    (path / "model.safetensors").write_text("old")
    if config is not None:
        (path / "config.json").write_text(config)


@pytest.mark.parametrize("final", [False, True])
@pytest.mark.parametrize("config", [None, "", '{"model_ty', "{}", "[]", _CONFIG])
def test_incomplete_export_is_repaired(tmp_path, final, config):
    """Repair invalid exports and leave complete exports untouched."""
    out = tmp_path / "out"
    if final:
        _seed_export(out / "exported-checkpoint-2000", _CONFIG)
        (out / "modelopt_state.pth").write_text("")
    export = out / f"exported-checkpoint-{'final' if final else '2000'}"
    _seed_export(export, config)
    proc, verdict, _ = _run(tmp_path, checkpoint_time=1)
    assert proc.returncode == 0, proc.stderr
    assert verdict["pass"] is True
    assert (export / "config.json").read_text() == _CONFIG
    assert (export / "model.safetensors").read_text() == ("old" if config == _CONFIG else "new")


@pytest.mark.parametrize("stale", [False, True])
def test_final_export_tracks_root_model(tmp_path, stale):
    """Refresh a valid final only when the root model is newer."""
    out = tmp_path / "out"
    _seed_export(out / "exported-checkpoint-2000", _CONFIG)
    export = out / "exported-checkpoint-final"
    _seed_export(export, _CONFIG)
    state = out / "modelopt_state.pth"
    state.write_text("")
    t = (export / "config.json").stat().st_mtime + (10 if stale else -10)
    os.utime(state, (t, t))
    proc, _, _ = _run(tmp_path, checkpoint_time=1)
    assert proc.returncode == 0, proc.stderr
    assert (export / "model.safetensors").read_text() == ("new" if stale else "old")


@pytest.mark.parametrize("final", [False, True])
@pytest.mark.parametrize("failure", [{"EXPORT_FAIL": "1"}, {"EXPORT_CONFIG": ""}])
def test_failed_export_writes_no_verdict(tmp_path, final, failure):
    """Exporter errors and invalid output stop the script before grading training."""
    out = tmp_path / "out"
    out.mkdir()
    (out / "modelopt_state.pth").write_text("")
    if final:
        _seed_export(out / "exported-checkpoint-2000", _CONFIG)
    proc, verdict, _ = _run(tmp_path, checkpoint_time=1, **failure)
    assert proc.returncode == 1, proc.stderr
    assert verdict is None
    if not final:
        assert not (out / "exported-checkpoint-final").exists()


def test_repeat_run_keeps_complete_exports(tmp_path):
    """A repeated run does not rewrite completed step or final exports."""
    out = tmp_path / "out"
    out.mkdir()
    (out / "modelopt_state.pth").write_text("")
    proc, _, _ = _run(tmp_path, checkpoint_time=1)
    assert proc.returncode == 0, proc.stderr
    weights = [
        out / f"exported-checkpoint-{step}" / "model.safetensors" for step in ("2000", "final")
    ]
    for path in weights:
        os.utime(path, (1, 1))
    proc, _, _ = _run(tmp_path, checkpoint_time=1)
    assert proc.returncode == 0, proc.stderr
    assert all(path.stat().st_mtime == 1 for path in weights)


@pytest.mark.parametrize("config", [None, ""])
def test_sharded_export_failure_preserves_leftover_final(tmp_path, config):
    """A leftover final cannot bypass an unexportable newest checkpoint."""
    out = tmp_path / "out"
    checkpoint = out / "checkpoint-2000"
    (checkpoint / "pytorch_model_fsdp_0").mkdir(parents=True)
    state = out / "modelopt_state.pth"
    state.write_text("")
    os.utime(state, (100, 100))
    final = out / "exported-checkpoint-final"
    _seed_export(final, _CONFIG)
    os.utime(final / "config.json", (110, 110))
    step_export = out / "exported-checkpoint-2000"
    if config is not None:
        _seed_export(step_export, config)

    # The stub exporter would succeed, so the script must reject the missing required export.
    proc, verdict, _ = _run(tmp_path, checkpoint_time=200)
    assert proc.returncode == 1, proc.stderr
    assert verdict is None
    assert (final / "model.safetensors").read_text() == "old"
    assert (final / "config.json").read_text() == _CONFIG
    assert not (tmp_path / "exports.log").exists()
    if config is None:
        assert not step_export.exists()
    else:
        assert (step_export / "model.safetensors").read_text() == "old"
        assert (step_export / "config.json").read_text() == config


@pytest.mark.parametrize(
    ("incomplete_step", "current_final"),
    [("1600", None), ("2000", "existing"), ("2000", "export")],
)
def test_incomplete_shards_allow_current_export(tmp_path, incomplete_step, current_final):
    """An obsolete shard export must not block a complete newest step or current final."""
    out = tmp_path / "out"
    for step in ("1600", "2000"):
        (out / f"checkpoint-{step}" / "pytorch_model_fsdp_0").mkdir(parents=True)
        _seed_export(
            out / f"exported-checkpoint-{step}", "" if step == incomplete_step else _CONFIG
        )
    final = out / "exported-checkpoint-final"
    if current_final:
        state = out / "modelopt_state.pth"
        state.write_text("")
        os.utime(state, (300, 300))
        if current_final == "existing":
            _seed_export(final, _CONFIG)
            os.utime(final / "config.json", (310, 310))

    proc, verdict, _ = _run(tmp_path, checkpoint_time=200)
    assert proc.returncode == 0, proc.stderr
    assert verdict["pass"] is True
    required = final if current_final else out / "exported-checkpoint-2000"
    assert json.loads((required / "config.json").read_text())
    log = tmp_path / "exports.log"
    assert (log.read_text().splitlines() if log.exists() else []) == (
        [str(out)] if current_final == "export" else []
    )
    assert _latest_export(tmp_path) == str(required)


@pytest.mark.parametrize(
    "failure",
    [
        {"EXPORT_FAIL": "1"},
        {"EXPORT_CONFIG": ""},
        {"PUBLISH_FAIL": "1"},
        {"PUBLISH_FAIL": "1", "RECOVER_FAIL": "1"},
    ],
)
def test_failed_replacement_preserves_final(tmp_path, failure):
    """Export and publication failures retain the previous usable final."""
    out = tmp_path / "out"
    _seed_export(out / "exported-checkpoint-2000", _CONFIG)
    final = out / "exported-checkpoint-final"
    _seed_export(final, _CONFIG)
    os.utime(final / "config.json", (250, 250))
    state = out / "modelopt_state.pth"
    state.write_text("")
    os.utime(state, (300, 300))
    proc, verdict, _ = _run(tmp_path, checkpoint_time=200, **failure)
    assert proc.returncode == 1, proc.stderr
    assert verdict is None
    backup = final.with_name(final.name + ".old")
    preserved = backup if failure.get("RECOVER_FAIL") else final
    assert (preserved / "model.safetensors").read_text() == "old"
    assert (preserved / "config.json").read_text() == _CONFIG
    assert backup.exists() == bool(failure.get("RECOVER_FAIL"))


@pytest.mark.parametrize(
    ("published", "recovery_fails"), [(False, False), (True, False), (False, True)]
)
def test_interrupted_publication_recovery(tmp_path, published, recovery_fails):
    """Recover a swap, stopping without deleting the backup if restoration fails."""
    out = tmp_path / "out"
    _seed_export(out / "exported-checkpoint-2000", _CONFIG)
    state = out / "modelopt_state.pth"
    state.write_text("")
    os.utime(state, (300, 300))
    final = out / "exported-checkpoint-final"
    backup = final.with_name(final.name + ".old")
    _seed_export(backup, _CONFIG)
    os.utime(backup / "config.json", (310, 310))
    if published:
        _seed_export(final, _CONFIG)
        (final / "model.safetensors").write_text("published")
        os.utime(final / "config.json", (310, 310))
    proc, verdict, _ = _run(
        tmp_path, checkpoint_time=200, **({"RECOVER_FAIL": "1"} if recovery_fails else {})
    )
    assert proc.returncode == int(recovery_fails), proc.stderr
    assert not (tmp_path / "exports.log").exists()
    if recovery_fails:
        assert verdict is None
        assert not final.exists()
        assert (backup / "model.safetensors").read_text() == "old"
        assert (backup / "config.json").read_text() == _CONFIG
    else:
        assert verdict["pass"] is True
        assert not backup.exists()
        assert (final / "model.safetensors").read_text() == ("published" if published else "old")


@pytest.mark.parametrize("existing_final", [False, True])
@pytest.mark.parametrize("root_time", [100, 200, 300])
def test_final_selection_tracks_latest_checkpoint(tmp_path, existing_final, root_time):
    """Retire finals only when the checkpoint is strictly newer than the root."""
    out = tmp_path / "out"
    step_export = out / "exported-checkpoint-2000"
    _seed_export(step_export, _CONFIG)
    state = out / "modelopt_state.pth"
    state.write_text("")
    current_root = root_time >= 200
    os.utime(state, (root_time, root_time))
    final = out / "exported-checkpoint-final"
    if existing_final:
        _seed_export(final, _CONFIG)
        os.utime(final / "config.json", (root_time + 10, root_time + 10))

    proc, verdict, _ = _run(tmp_path, checkpoint_time=200)
    assert proc.returncode == 0, proc.stderr
    assert verdict["pass"] is True
    assert verdict["trainer_state"].endswith("checkpoint-2000/trainer_state.json")
    assert final.exists() == current_root
    assert (step_export / "model.safetensors").read_text() == "old"
    if current_root:
        assert (final / "model.safetensors").read_text() == ("old" if existing_final else "new")

    assert _latest_export(tmp_path) == str(final if current_root else step_export)


def _latest_export(tmp_path):
    selected = subprocess.run(
        [
            "bash",
            "-c",
            (
                "source common/service_utils.sh >/dev/null; "
                'latest_checkpoint "$1" exported-checkpoint-'
            ),
            "bash",
            str(tmp_path / "out"),
        ],
        cwd=tmp_path,
        env={**os.environ, "PATH": f"{tmp_path / 'bin'}:{os.environ['PATH']}"},
        capture_output=True,
        text=True,
        check=True,
    )
    return selected.stdout.strip()


@pytest.mark.parametrize("train_exit", [0, 1])
def test_failed_training_writes_no_verdict(tmp_path, train_exit):
    """A crash must not let a surviving checkpoint, or an old verdict, pass the gate."""
    proc, verdict, _ = _run(tmp_path, train_exit)
    if train_exit:
        assert verdict is None, verdict
        assert proc.returncode == 1, proc.stderr
        assert "training failed; skipping export and regression check" in proc.stderr
    else:  # control: the same sandbox grades the checkpoint when training succeeds
        assert verdict is not None, proc.stderr
        assert verdict["pass"] is True
        assert verdict["trainer_state"].endswith("checkpoint-2000/trainer_state.json")
        assert proc.returncode == 0, proc.stderr
        assert "[ERROR]" not in proc.stderr


@pytest.mark.parametrize(("strict", "exit_code"), [("1", 1), ("0", 0)])
def test_strict_mode_fails_the_task_on_regression(tmp_path, strict, exit_code):
    """REGRESSION_STRICT=1 turns a failed check into a failed task; 0 only records it."""
    proc, verdict, _ = _run(tmp_path, loss=6.0, REGRESSION_STRICT=strict)
    assert verdict["failure_class"] == "TRAIN_REGRESSION", proc.stderr
    assert proc.returncode == exit_code, proc.stderr


def test_invalid_regression_strict_fails_before_training(tmp_path):
    """A typo fails at once, without training, and leaves no earlier run's verdict behind."""
    proc, verdict, launched = _run(tmp_path, REGRESSION_STRICT="true")
    assert proc.returncode == 1
    assert "REGRESSION_STRICT must be 0 or 1, got 'true'" in proc.stderr
    assert not launched
    assert verdict is None
