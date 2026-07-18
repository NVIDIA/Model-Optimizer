from pathlib import Path

from modelopt.torch.puzzletron.campaigns.launcher import allocation_for_topology
from modelopt.torch.puzzletron.campaigns.schema import ParallelTopology


def test_full_two_node_topology_is_exclusive() -> None:
    allocation = allocation_for_topology(
        ParallelTopology(tp=2, cp=2, pp=2, fsdp=2, ep=1)
    )

    assert allocation.nodes == 2
    assert allocation.gpus_per_node == 8
    assert allocation.nproc_per_node == 8
    assert allocation.exclusive is True


def test_four_rank_exception_is_nonexclusive() -> None:
    allocation = allocation_for_topology(
        ParallelTopology(tp=1, cp=2, pp=1, fsdp=2, ep=1)
    )

    assert allocation.nodes == 1
    assert allocation.gpus_per_node == 4
    assert allocation.nproc_per_node == 4
    assert allocation.exclusive is False


def test_topology_larger_than_two_eight_gpu_nodes_is_rejected() -> None:
    try:
        allocation_for_topology(ParallelTopology(tp=4, cp=2, pp=2, fsdp=2, ep=1))
    except ValueError as error:
        assert "16 GPUs" in str(error)
    else:
        raise AssertionError("campaign allocations must fit in two eight-GPU nodes")


def test_multinode_runner_uses_slurm_submit_directory_as_repository_root() -> None:
    runner = Path("examples/puzzletron/run_multinode_stage.sh").read_text()

    assert 'ROOT=${PUZZLETRON_ROOT:-${SLURM_SUBMIT_DIR:-"${SCRIPT_ROOT}"}}' in runner


def test_generic_launchers_default_to_the_campaign_venv() -> None:
    launchers = (
        Path("examples/puzzletron/run_multinode_stage.sh"),
        Path("examples/puzzletron/run_single_gpu_replace_scoring.sh"),
        Path("examples/puzzletron/run_axis_diagnostic_workers.sh"),
    )

    for launcher in launchers:
        contents = launcher.read_text()
        assert ".venv_new" in contents
        assert 'PUZZLETRON_VENV' in contents
        assert 'source "${PUZZLETRON_VENV}/bin/activate"' in contents


def test_axis_diagnostic_launcher_uses_one_configured_model_instance_per_worker() -> None:
    launcher = Path("examples/puzzletron/run_axis_diagnostic_workers.sh").read_text()
    task = Path("examples/puzzletron/run_axis_diagnostic_task.sh").read_text()

    assert "AXIS_DIAGNOSTIC_NPROC_PER_NODE" in launcher
    assert "AXIS_DIAGNOSTIC_GPUS_PER_WORKER" in launcher
    assert "--gpus-per-task=2" not in launcher
    assert "--ntasks-per-node=4" not in launcher
    assert '"--nproc-per-node=${AXIS_DIAGNOSTIC_NPROC_PER_NODE}"' in task
    assert "CUDA_VISIBLE_DEVICES=\"${first_gpu}" not in task
