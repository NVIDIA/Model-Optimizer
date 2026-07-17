from examples.puzzletron.main import build_worker_command


def test_worker_command_propagates_gpu_count_to_composite_followups():
    command = build_worker_command(
        config_path="experiment.yaml",
        stage="build_library",
        overrides=(),
        gpus_per_node=1,
    )

    assert command[command.index("--gpus-per-node") + 1] == "1"
