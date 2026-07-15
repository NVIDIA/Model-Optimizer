from types import SimpleNamespace

from modelopt.torch.puzzletron.distributed_eval.config import (
    build_campaign_manifest,
    parallelism_from_config,
)


def test_parallelism_converts_automodel_ep_overlaid_dp_to_logical_dp():
    config = {
        "scoring": {
            "automodel": {
                "recipe": {
                    "distributed": {
                        "dp_size": 8,
                        "tp_size": 1,
                        "cp_size": 1,
                        "ep_size": 4,
                        "pp_size": 2,
                    },
                    "distributed_config": {
                        "_target_": "nemo_automodel.components.distributed.config.FSDP2Config"
                    },
                }
            },
            "distributed_eval": {"gpus_per_task": 16},
        }
    }

    parallelism = parallelism_from_config(config, world_size=16)

    assert parallelism.dp_size == 2
    assert parallelism.ep_size == 4


def test_parallelism_preserves_automodel_dp_without_expert_parallelism():
    config = {
        "scoring": {
            "automodel": {
                "recipe": {
                    "distributed": {"dp_size": 8, "pp_size": 2},
                    "distributed_config": {
                        "_target_": "nemo_automodel.components.distributed.config.FSDP2Config"
                    },
                }
            },
            "distributed_eval": {"gpus_per_task": 16},
        }
    }

    parallelism = parallelism_from_config(config, world_size=16)

    assert parallelism.dp_size == 8


def test_parallelism_prefers_public_replacement_scoring_config():
    public_recipe = {
        "distributed": {"dp_size": 8, "ep_size": 4, "pp_size": 2},
        "distributed_config": {
            "_target_": "nemo_automodel.components.distributed.config.FSDP2Config"
        },
    }
    config = {
        "replacement_scoring": {
            "automodel": {"recipe": public_recipe},
            "distributed_eval": {"gpus_per_task": 16},
        },
        "scoring": {
            "automodel": {
                "recipe": {
                    "distributed": {"dp_size": 1},
                    "distributed_config": {},
                }
            }
        },
    }

    parallelism = parallelism_from_config(config, world_size=16)

    assert parallelism.dp_size == 2
    assert parallelism.ep_size == 4
    assert parallelism.pp_size == 2


def test_rpc_manifest_infers_descriptor_from_scoring_parent(monkeypatch, tmp_path):
    source = tmp_path / "scoring-parent"
    source.mkdir()
    (source / "config.json").write_text("{}\n")
    config = {
        "experiment": {"dir": str(tmp_path)},
        "model": {"force_hf": False, "trust_remote_code": True},
        "scoring": {
            "source_checkpoint_dir": str(source),
            "automodel": {"force_hf": False},
        },
        "data": {},
    }
    monkeypatch.setattr(
        "modelopt.torch.puzzletron.distributed_eval.config.load_plain_pipeline_config",
        lambda *args, **kwargs: config,
    )
    monkeypatch.setattr(
        "modelopt.torch.puzzletron.distributed_eval.config._load_recipe",
        lambda _config: {
            "model": {"torch_dtype": "bf16"},
            "distributed": {},
            "dist_env": {"backend": "nccl"},
        },
    )
    monkeypatch.setattr(
        "modelopt.torch.puzzletron.distributed_eval.config.resolve_descriptor_from_pretrained",
        lambda path, **kwargs: SimpleNamespace(name="llama"),
        raising=False,
    )

    manifest = build_campaign_manifest(tmp_path / "config.yaml", world_size=1)

    assert manifest.descriptor == "llama"
    assert manifest.model["checkpoint_dir"] == str(source.resolve())
