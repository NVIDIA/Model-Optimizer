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
"""Tests for ``ExampleRunner``: example steps run in a long-lived worker and leave no state behind."""

import contextlib
import json
import os
import re
import signal
import subprocess
import sys
import textwrap
import threading
import time
from pathlib import Path

import pytest
from _test_utils.examples import run_command
from _test_utils.examples.example_runner import (
    ExampleRunner,
    StepFailed,
    StepTimeout,
    keep_post_conversion_plugins,
)

pytestmark = pytest.mark.usefixtures("skip_on_windows")


class _Lane:
    """One example directory of a runner, with helpers to write and run its scripts."""

    def __init__(self, runner, directory):
        directory.mkdir(parents=True)
        self.runner, self.dir = runner, directory

    def write(self, name, body):
        (self.dir / name).write_text(textwrap.dedent(body))

    def run(self, script, *args, **kwargs):
        return self.runner(["python", script, *args], self.dir.name, **kwargs)

    def probe(self, script="probe.py"):
        return json.loads(self.run(script).strip().splitlines()[-1])


def _is_alive(pid):
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    try:  # a zombie is dead; nothing reaps it where pid 1 does not
        return Path(f"/proc/{pid}/stat").read_text().rsplit(")", 1)[1].split()[0] != "Z"
    except OSError:
        return True


def _wait_until_dead(pid):
    for _ in range(100):
        if not _is_alive(pid):
            return
        time.sleep(0.1)
    os.kill(pid, signal.SIGKILL)
    pytest.fail(f"process {pid} outlived its step")


@contextlib.contextmanager
def _in_process_lane(tmp_path, hooks=()):
    runner = ExampleRunner((), hooks, tmp_path, preload=(), in_process=True)
    try:
        yield _Lane(runner, tmp_path / "example")
    finally:
        runner.close()


def _lane_factory(runner, examples_dir, request):
    name = re.sub(r"\W+", "_", request.node.name)
    return lambda suffix="": _Lane(runner, examples_dir / f"{name}{suffix}")


@pytest.fixture(scope="module")
def examples_dir(tmp_path_factory):
    return tmp_path_factory.mktemp("examples")


@pytest.fixture(scope="module")
def worker_runner(examples_dir):
    # No preload: every failed step recycles the worker, and importing torch takes seconds.
    runner = ExampleRunner((), examples_dir=examples_dir, preload=(), in_process=False)
    yield runner
    runner.close()


@pytest.fixture(scope="module", params=["worker", "in_process"])
def backend(request):
    return request.param


@pytest.fixture(scope="module")
def any_runner(backend, examples_dir, worker_runner):
    if backend == "worker":
        yield worker_runner
        return
    runner = ExampleRunner((), examples_dir=examples_dir, preload=(), in_process=True)
    yield runner
    runner.close()


@pytest.fixture
def make_lane(worker_runner, examples_dir, request):
    """Lanes of the shared worker."""
    return _lane_factory(worker_runner, examples_dir, request)


@pytest.fixture
def make_lane_on_both_backends(any_runner, examples_dir, request):
    """Lanes of the shared worker, and of an in-process runner, as the test runs once per backend."""
    return _lane_factory(any_runner, examples_dir, request)


@pytest.fixture
def worker_lane(tmp_path):
    """A lane of a worker of its own, for tests that kill or stop it."""
    runner = ExampleRunner((), examples_dir=tmp_path, preload=(), in_process=False)
    yield _Lane(runner, tmp_path / "example")
    runner.close()


def test_step_runs_like_python_script_py(make_lane_on_both_backends):
    lane = make_lane_on_both_backends()
    lane.write(
        "s.py",
        """
        import argparse, os, sys

        def main():
            parser = argparse.ArgumentParser()
            parser.add_argument("--x", type=int)
            cwd, argv0 = os.path.basename(os.getcwd()), os.path.basename(sys.argv[0])
            print("x", parser.parse_args().x, "cwd", cwd, "argv0", argv0)

        if __name__ == "__main__":
            main()
        """,
    )
    assert f"x 3 cwd {lane.dir.name} argv0 s.py" in lane.run("s.py", "--x", "3")


def test_failed_step_raises_step_failed_and_the_runner_recovers(make_lane_on_both_backends):
    lane = make_lane_on_both_backends()
    lane.write("bad.py", "raise ValueError('bad')")
    lane.write("ok.py", "print('still fine')")
    with pytest.raises(StepFailed) as failure:
        lane.run("bad.py")
    assert isinstance(failure.value, subprocess.CalledProcessError)
    assert failure.value.returncode == 1
    assert "ValueError: bad" in failure.value.output
    assert failure.value.captured_output == failure.value.output
    assert "still fine" in lane.run("ok.py")


@pytest.mark.parametrize(
    ("body", "returncode", "text"),
    [
        ("import sys; sys.exit(3)", 3, ""),
        ("import sys; sys.exit('boom')", 1, "boom"),
        ("import argparse; argparse.ArgumentParser().parse_args(['--nope'])", 2, "unrecognized"),
    ],
)
def test_exit_statuses_map_to_the_return_code(make_lane, body, returncode, text):
    lane = make_lane()
    lane.write("bad.py", body)
    with pytest.raises(StepFailed) as failure:
        lane.run("bad.py")
    assert failure.value.returncode == returncode
    assert text in failure.value.output


def test_sys_exit_zero_is_success(make_lane):
    lane = make_lane()
    lane.write("z.py", "import sys; print('done'); sys.exit(0)")
    assert "done" in lane.run("z.py")


def test_process_state_changed_by_a_step_is_restored(make_lane_on_both_backends):
    lane = make_lane_on_both_backends()
    lane.write(
        "leak.py",
        """
        import logging, os, signal, sys, warnings

        os.environ["LEAK"] = "1"
        os.chdir("/")
        sys.path.insert(0, "/nonexistent-leak")
        signal.signal(signal.SIGTERM, lambda *args: None)
        logger = logging.getLogger("__main__")
        logger.addHandler(logging.StreamHandler(sys.stdout))
        logger.setLevel(10)
        logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))
        logging.disable(logging.CRITICAL)
        warnings.simplefilter("ignore")
        sys.argv.append("leak")
        """,
    )
    lane.write(
        "probe.py",
        """
        import json, logging, os, signal, sys, warnings

        print(json.dumps({
            "leak": "LEAK" in os.environ,
            "path": "/nonexistent-leak" in sys.path,
            "argv": len(sys.argv),
            "sigterm": str(signal.getsignal(signal.SIGTERM)),
            "main_handlers": len(logging.getLogger("__main__").handlers),
            "root_handlers": len(logging.getLogger().handlers),
            "root_level": logging.getLogger().level,
            "disabled": logging.root.manager.disable,
            "filters": len(warnings.filters),
        }))
        """,
    )
    cwd, before = os.getcwd(), lane.probe()
    lane.run("leak.py")
    after = lane.probe()
    assert after == before
    assert after["leak"] is False
    assert after["main_handlers"] == 0
    assert os.getcwd() == cwd


def test_random_state_start_method_and_backend_flags_are_restored(backend, examples_dir):
    # Preload torch up front, as the real lanes do.
    runner = ExampleRunner(
        (), examples_dir=examples_dir, preload=("torch",), in_process=backend == "in_process"
    )
    lane = _Lane(runner, examples_dir / f"state_{backend}")
    lane.write(
        "mutate.py",
        """
        import multiprocessing, random
        import numpy, torch

        multiprocessing.set_start_method("forkserver", force=True)
        random.seed(1)
        numpy.random.seed(1)
        torch.manual_seed(1)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = True
        """,
    )
    lane.write(
        "probe.py",
        """
        import json, multiprocessing, random
        import numpy, torch

        print(json.dumps({
            "start_method": multiprocessing.get_start_method(allow_none=True),
            "python": random.random(),
            "numpy": float(numpy.random.rand()),
            "torch": float(torch.rand(1)),
            "deterministic": torch.backends.cudnn.deterministic,
            "benchmark": torch.backends.cudnn.benchmark,
        }))
        """,
    )
    try:
        before = lane.probe()
        lane.run("mutate.py")
        assert lane.probe() == before
    finally:
        runner.close()


def test_handlers_added_by_each_step_do_not_accumulate(make_lane):
    # What diffusers' quantize.py does with its setup_logging().
    lane = make_lane()
    lane.write(
        "logs.py",
        """
        import logging, sys

        log = logging.getLogger(__name__)
        log.setLevel(logging.INFO)
        log.addHandler(logging.StreamHandler(sys.stdout))
        log.info("LINE")
        """,
    )
    for _ in range(3):
        assert lane.run("logs.py").count("LINE") == 1


def test_modules_of_the_example_dir_are_not_shared_but_imports_stay_cached(make_lane):
    first, second = make_lane("_a"), make_lane("_b")
    for lane, value in ((first, "A"), (second, "B")):
        lane.write("utils.py", f"VALUE = '{value}'")
        lane.write(
            "main.py", "import sys, utils\nprint('got', utils.VALUE, 'colorsys' in sys.modules)"
        )
    first.write("warm.py", "import colorsys")
    first.run("warm.py")
    seen = [lane.run("main.py") for lane in (first, second, first)]
    assert ["got A True" in seen[0], "got B True" in seen[1], "got A True" in seen[2]] == [True] * 3


def test_env_is_the_whole_environment_of_the_step_and_is_restored(make_lane):
    lane = make_lane()
    lane.write("keys.py", "import os; print('KEYS', sorted(os.environ))")
    assert "KEYS ['ONLY']" in lane.run("keys.py", env={"ONLY": "1"})
    assert "ONLY" not in lane.run("keys.py")


def test_output_of_fds_and_child_processes_is_captured(make_lane):
    lane = make_lane()
    lane.write(
        "o.py",
        """
        import os, subprocess, sys

        print("py")
        print("err", file=sys.stderr)
        os.write(1, b"fd1\\n")
        subprocess.run(["echo", "child"])
        """,
    )
    output = lane.run("o.py")
    assert all(text in output for text in ("py", "err", "fd1", "child"))


def test_step_can_start_child_processes(make_lane):
    # DataLoader workers and Pools: a daemonic worker would refuse.
    lane = make_lane()
    lane.write(
        "fork.py",
        """
        import multiprocessing

        def child():
            print("child ran")

        if __name__ == "__main__":
            process = multiprocessing.get_context("fork").Process(target=child)
            process.start()
            process.join()
            print("exit", process.exitcode)
        """,
    )
    output = lane.run("fork.py")
    assert "child ran" in output
    assert "exit 0" in output


def test_hooks_wrap_every_step(tmp_path):
    events = []

    @contextlib.contextmanager
    def hook():
        events.append("enter")
        try:
            yield
        finally:
            events.append("exit")

    with _in_process_lane(tmp_path, (hook,)) as lane:
        lane.write("s.py", "print('ran')")
        assert "ran" in lane.run("s.py")
    assert events == ["enter", "exit"]


def test_failing_hook_fails_the_step(tmp_path):
    def hook():
        raise RuntimeError("hook failed")

    with _in_process_lane(tmp_path, (hook,)) as lane:
        lane.write("s.py", "print('ran')")
        with pytest.raises(StepFailed) as failure:
            lane.run("s.py")
    assert "hook failed" in failure.value.output
    assert "ran" not in failure.value.output


def test_keep_post_conversion_plugins_undoes_the_registrations_of_a_step(tmp_path):
    from modelopt.torch.quantization.plugins.custom import CUSTOM_POST_CONVERSION_PLUGINS

    before = set(CUSTOM_POST_CONVERSION_PLUGINS)
    with _in_process_lane(tmp_path, (keep_post_conversion_plugins,)) as lane:
        lane.write(
            "s.py",
            """
            from modelopt.torch.quantization.plugins.custom import CUSTOM_POST_CONVERSION_PLUGINS

            CUSTOM_POST_CONVERSION_PLUGINS.add(lambda model: None)
            """,
        )
        lane.run("s.py")
    assert before == CUSTOM_POST_CONVERSION_PLUGINS


def test_worker_is_reused_until_a_step_fails(worker_lane):
    worker_lane.write("pid.py", "import os; print('pid', os.getpid())")
    worker_lane.write("bad.py", "raise SystemExit(1)")

    def pid():
        return worker_lane.run("pid.py").split()[-1]

    first = pid()
    assert pid() == first != str(os.getpid())
    with pytest.raises(StepFailed):
        worker_lane.run("bad.py")
    assert pid() != first


def test_timeout_kills_the_hung_step_with_its_children_and_the_next_step_gets_a_fresh_worker(
    worker_lane,
):
    worker_lane.write("pid.py", "import os; print('pid', os.getpid())")
    worker_lane.write(
        "hang.py",
        """
        import subprocess, time

        child = subprocess.Popen(["sleep", "60"])
        print("child", child.pid, flush=True)
        time.sleep(60)
        """,
    )
    first = worker_lane.run("pid.py")
    started = time.monotonic()
    with pytest.raises(StepTimeout) as timeout:
        worker_lane.run("hang.py", timeout=2)
    assert time.monotonic() - started < 15
    child = int(re.search(r"child (\d+)", timeout.value.captured_output).group(1))
    _wait_until_dead(child)
    assert worker_lane.run("pid.py") != first


@pytest.mark.parametrize(
    ("death", "returncode"),
    [("os._exit(7)", 7), ("os.kill(os.getpid(), signal.SIGKILL)", -signal.SIGKILL)],
)
def test_worker_death_fails_the_step_not_the_session(worker_lane, death, returncode):
    worker_lane.write("die.py", f"import os, signal\nprint('about to die', flush=True)\n{death}")
    worker_lane.write("ok.py", "print('alive')")
    with pytest.raises(StepFailed) as failure:
        worker_lane.run("die.py")
    assert failure.value.returncode == returncode
    assert "about to die" in failure.value.output
    assert "alive" in worker_lane.run("ok.py")


def test_interrupt_while_waiting_kills_the_step(worker_lane):
    # What pytest-timeout's signal method does: raise in the main thread while the runner waits.
    class Interrupted(BaseException):
        pass

    def interrupt(*_):
        raise Interrupted

    worker_lane.write(
        "hang.py",
        """
        import os, pathlib, time

        pathlib.Path(__file__).with_name("pid").write_text(str(os.getpid()))
        print("before hang", flush=True)
        time.sleep(60)
        """,
    )
    previous = signal.signal(signal.SIGUSR1, interrupt)
    main_thread = threading.main_thread().ident
    timer = threading.Timer(1.0, signal.pthread_kill, (main_thread, signal.SIGUSR1))
    timer.start()
    try:
        with pytest.raises(Interrupted) as interrupted:
            worker_lane.run("hang.py", timeout=30)
    finally:
        timer.cancel()
        signal.signal(signal.SIGUSR1, previous)
    assert "before hang" in interrupted.value.captured_output
    _wait_until_dead(int((worker_lane.dir / "pid").read_text()))


def test_close_lets_the_worker_exit_cleanly(worker_lane):
    # A clean exit is what lets coverage write the worker's data.
    worker_lane.write(
        "marker.py",
        """
        import atexit, pathlib

        marker = pathlib.Path(__file__).with_name("exited")
        atexit.register(marker.write_text, "clean")
        """,
    )
    worker_lane.run("marker.py")
    marker = worker_lane.dir / "exited"
    assert not marker.exists()
    worker_lane.runner.close()
    assert marker.read_text() == "clean"


def test_accepts_only_listed_scripts_run_with_python():
    runner = ExampleRunner({"a.py"}, preload=())
    try:
        assert runner.accepts(["python", "a.py", "--x", "1"])
        assert runner.accepts(["python3", "a.py"])
        assert runner.accepts([sys.executable, "a.py"])
        assert not runner.accepts(["python", "b.py"])
        assert not runner.accepts(["bash", "a.py"])
        assert not runner.accepts(["python", "-m", "a"])
        assert not runner.accepts(["python"])
    finally:
        runner.close()


@pytest.fixture
def installed_runner(tmp_path, monkeypatch):
    """A worker for ``in_worker.py`` installed in ``run_command``; yields ``examples/demo``."""
    monkeypatch.setattr(run_command, "MODELOPT_ROOT", tmp_path)
    demo = tmp_path / "examples" / "demo"
    demo.mkdir(parents=True)
    runner = ExampleRunner({"in_worker.py"}, examples_dir=tmp_path / "examples", preload=())
    run_command.set_in_process_runner(runner)
    yield demo
    run_command.set_in_process_runner(None)
    runner.close()


def _run_demo(script, **kwargs):
    return run_command.run_example_command([sys.executable, script], "demo", **kwargs)


def _demo_pid(script):
    return int(_run_demo(script).split()[-1])


def test_run_example_command_gives_listed_scripts_to_the_runner(installed_runner):
    for name in ("in_worker.py", "elsewhere.py"):
        (installed_runner / name).write_text("import os; print('pid', os.getpid())")
    worker = _demo_pid("in_worker.py")
    assert worker != os.getpid()
    assert _demo_pid("in_worker.py") == worker
    assert _demo_pid("elsewhere.py") not in (worker, os.getpid())


def test_kill_switch_runs_listed_scripts_as_subprocesses(installed_runner, monkeypatch):
    (installed_runner / "in_worker.py").write_text("import os; print('pid', os.getpid())")
    monkeypatch.setenv("MODELOPT_EXAMPLE_RUNNER", "subprocess")
    assert _demo_pid("in_worker.py") != _demo_pid("in_worker.py")


def test_run_example_command_passes_env_to_the_runner(installed_runner):
    (installed_runner / "in_worker.py").write_text("import os; print('KEYS', sorted(os.environ))")
    assert "KEYS ['ONLY']" in _run_demo("in_worker.py", env={"ONLY": "1"})


def test_failed_step_raises_called_process_error_through_run_example_command(installed_runner):
    (installed_runner / "in_worker.py").write_text("raise ValueError('nope')")
    with pytest.raises(subprocess.CalledProcessError) as failure:
        _run_demo("in_worker.py")
    assert "ValueError: nope" in failure.value.output


def test_transient_hub_errors_are_retried_through_the_runner(installed_runner):
    (installed_runner / "in_worker.py").write_text(
        textwrap.dedent(
            """
            import pathlib

            marker = pathlib.Path(__file__).with_name("tried")
            if not marker.exists():
                marker.write_text("1")
                raise RuntimeError("ConnectionError: Max retries exceeded")
            print("recovered")
            """
        )
    )
    with pytest.warns(UserWarning, match="transient HuggingFace access error"):
        assert "recovered" in _run_demo("in_worker.py", hf_retry_delay_s=0)


def test_plain_runners_keep_their_contract(tmp_path, monkeypatch):
    monkeypatch.setattr(run_command, "MODELOPT_ROOT", tmp_path)
    (tmp_path / "examples" / "demo").mkdir(parents=True)
    calls = []

    def plain_runner(cmd_parts, example_path):
        calls.append((cmd_parts, example_path))
        return "ran in process"

    run_command.set_in_process_runner(plain_runner)
    try:
        assert run_command.run_example_command(["x.py"], "demo") == "ran in process"
        with pytest.warns(UserWarning, match="env= given"):
            output = run_command.run_example_command(
                [sys.executable, "-c", "print('subprocess')"], "demo", env=os.environ.copy()
            )
    finally:
        run_command.set_in_process_runner(None)
    assert "subprocess" in output
    assert calls == [(["x.py"], "demo")]
