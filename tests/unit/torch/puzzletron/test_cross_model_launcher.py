from modelopt.torch.puzzletron.campaigns.launcher import allocation_for_topology
from modelopt.torch.puzzletron.campaigns.schema import ParallelTopology
from examples.puzzletron.cross_model_campaign_inventory import filter_rows, inventory_rows
from pathlib import Path


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


def test_launcher_inventory_uses_stdlib_preflight_without_importing_modelopt(tmp_path) -> None:
    preflight = tmp_path / "preflight.json"
    preflight.write_text(
        '{"models": ['
        '{"model_id": "full", "topology": {"tp": 2, "cp": 2, "pp": 2, "fsdp": 2, "ep": 1}},'
        '{"model_id": "partial", "topology": {"tp": 1, "cp": 2, "pp": 1, "fsdp": 2, "ep": 1}}'
        ']}\n'
    )

    rows = inventory_rows(preflight)

    assert rows == (
        ("full", 2, 8, 8, True),
        ("partial", 1, 4, 4, False),
    )


def test_stage_runner_disables_xet_for_reliable_checkpoint_conversion() -> None:
    runner = Path("examples/puzzletron/run_cross_model_stage.sh").read_text()

    assert "HF_HUB_DISABLE_XET" in runner


def test_stage_runner_forwards_stage_overrides_to_main() -> None:
    runner = Path("examples/puzzletron/run_cross_model_stage.sh").read_text()

    assert 'EXTRA_ARGS=("${@:3}")' in runner
    assert '"${EXTRA_ARGS[@]}"' in runner


def test_stage_runner_prioritizes_active_venv_torch_libraries() -> None:
    runner = Path("examples/puzzletron/run_cross_model_stage.sh").read_text()

    assert 'VENV_TORCH_LIB="${VIRTUAL_ENV}/lib/python${PYTHON_VERSION}/site-packages/torch/lib"' in runner
    assert 'export LD_LIBRARY_PATH="${VENV_TORCH_LIB}:${LD_LIBRARY_PATH:-}"' in runner


def test_stage_runner_accepts_an_isolated_campaign_venv() -> None:
    runner = Path("examples/puzzletron/run_cross_model_stage.sh").read_text()

    assert 'PUZZLETRON_VENV=${PUZZLETRON_VENV:-"${ROOT}/.venv"}' in runner
    assert 'source "${PUZZLETRON_VENV}/bin/activate"' in runner


def test_stage_runner_honors_descriptor_owned_eager_stage_policy() -> None:
    runner = Path("examples/puzzletron/run_cross_model_stage.sh").read_text()

    assert 'torch_compile_disabled_stages' in runner
    assert 'export TORCH_COMPILE_DISABLE=1' in runner


def test_stage_runner_parallelizes_checkpoint_sorting() -> None:
    runner = Path("examples/puzzletron/run_cross_model_stage.sh").read_text()

    assert "activation|sort|sort_equivalence|activation_diagnostic" in runner


def test_stage_runner_parallelizes_bypass_overfit() -> None:
    runner = Path("examples/puzzletron/run_cross_model_stage.sh").read_text()

    assert "bypass|bypass_overfit" in runner


def test_multinode_runner_uses_slurm_submit_directory_as_repository_root() -> None:
    runner = Path("examples/puzzletron/run_multinode_stage.sh").read_text()

    assert 'ROOT=${PUZZLETRON_ROOT:-${SLURM_SUBMIT_DIR:-"${SCRIPT_ROOT}"}}' in runner


def test_main_accepts_multi_node_checkpoint_sorting() -> None:
    main = Path("examples/puzzletron/main.py").read_text()
    multi_node_stages = main.split("MULTI_NODE_STAGES = (", 1)[1].split(")", 1)[0]

    assert '"sort",' in multi_node_stages


def test_main_accepts_multi_node_bypass_overfit() -> None:
    main = Path("examples/puzzletron/main.py").read_text()
    multi_node_stages = main.split("MULTI_NODE_STAGES = (", 1)[1].split(")", 1)[0]

    assert '"bypass_overfit",' in multi_node_stages


def test_launcher_can_resume_from_first_failed_model() -> None:
    rows = (
        ("qwen", 2, 8, 8, True),
        ("llama", 1, 4, 4, False),
        ("nemotron", 2, 8, 8, True),
    )

    assert filter_rows(rows, start_model="llama") == rows[1:]


def test_activation_probe_launcher_is_isolated_and_respects_exclusive_policy() -> None:
    launcher = Path(
        "examples/puzzletron/launch_cross_model_activation_probes.sh"
    ).read_text()

    assert "probes/activation" in launcher
    assert "pruning.eval_samples=2" in launcher
    assert "pruning.micro_batch_size=2" in launcher
    assert "data.calibration.num_samples=2" in launcher
    assert "data.calibration.micro_batch_size=2" in launcher
    assert '[[ "${exclusive}" == 0 ]] || srun_args+=(--exclusive)' in launcher
