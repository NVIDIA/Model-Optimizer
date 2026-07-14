from pathlib import Path


def test_profile_distillation_config_uses_streaming_data_and_all_objectives(tmp_path: Path):
    from examples.puzzletron.run_profile_distillation_worker import (
        build_profile_distillation_config,
    )

    config = {
        "experiment": {"dir": str(tmp_path)},
        "model": {"force_hf": False},
        "scoring": {"dataset_path": str(tmp_path / "data")},
    }
    actual = build_profile_distillation_config(
        config,
        teacher_dir=tmp_path / "teacher",
        student_dir=tmp_path / "student",
        output_dir=tmp_path / "output",
        descriptor="qwen3_5",
        sequence_length=2048,
        global_batch_size=32,
        local_batch_size=4,
        max_steps=256,
        learning_rate=5.0e-5,
        seed=445,
        checkpoint_every=64,
        tp=1,
        cp=4,
        pp=2,
        dp=2,
        ep=1,
    )

    kd = actual["distillation"]
    assert (kd["tp"], kd["cp"], kd["pp"], kd["dp"], kd["ep"]) == (1, 4, 2, 2, 1)
    assert kd["global_batch_size"] == 32
    assert kd["local_batch_size"] == 4
    assert kd["max_steps"] == 256
    assert {
        name: term["weight"] for name, term in kd["objective"].items()
    } == {"main_ce": 1.0, "main_kd": 1.0, "mtp_ce": 1.0, "mtp_kd": 1.0}
    assert kd["objective"]["main_kd"]["chunk_size"] == 128
    assert kd["objective"]["mtp_kd"]["chunk_size"] == 128
    dataset = kd["metadata"]["llm"]["dataset"]
    assert dataset["_target_"].endswith("make_puzzletron_llm_dataset")
    assert not dataset["_target_"].endswith("make_puzzletron_llm_overfit_dataset")
    assert dataset["num_samples"] == 32 * 256
    assert dataset["seq_length"] == 2048
    assert kd["metadata"]["recipe_overrides"]["step_scheduler"]["ckpt_every_steps"] == 64
