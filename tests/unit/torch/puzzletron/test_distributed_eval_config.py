import sys
from types import SimpleNamespace

import pytest

from modelopt.torch.puzzletron.distributed_eval.config import (
    build_campaign_manifest,
    distributed_stage_config,
    parallelism_from_config,
)


def test_parallelism_converts_automodel_ep_overlaid_dp_to_logical_dp():
    config = {
        "scoring": {
            "automodel": {
                "parallel": {
                    "tp": 1,
                    "cp": 1,
                    "pp": 2,
                    "ep": 4,
                    "dp_shard": 4,
                    "dp_replicate": 2,
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
                "parallel": {
                    "tp": 1,
                    "cp": 1,
                    "pp": 2,
                    "ep": 1,
                    "dp_shard": 8,
                    "dp_replicate": 1,
                }
            },
            "distributed_eval": {"gpus_per_task": 16},
        }
    }

    parallelism = parallelism_from_config(config, world_size=16)

    assert parallelism.dp_size == 8


def test_parallelism_prefers_public_replacement_scoring_config():
    public_parallel = {
        "tp": 1,
        "cp": 1,
        "pp": 2,
        "ep": 4,
        "dp_shard": 4,
        "dp_replicate": 2,
    }
    config = {
        "replacement_scoring": {
            "automodel": {"parallel": public_parallel},
            "distributed_eval": {"gpus_per_task": 16},
        },
        "scoring": {
            "automodel": {
                "parallel": {
                    "tp": 1,
                    "cp": 1,
                    "pp": 1,
                    "ep": 1,
                    "dp_shard": 1,
                    "dp_replicate": 1,
                }
            }
        },
    }

    parallelism = parallelism_from_config(config, world_size=16)

    assert parallelism.dp_size == 2
    assert parallelism.ep_size == 4
    assert parallelism.pp_size == 2


def test_depth_distributed_stage_uses_depth_source_data_and_recipe():
    config = {
        "puzzle_dir": "/puzzle",
        "scoring": {
            "source_checkpoint_dir": "/puzzle/ckpts/sorted_teacher",
            "eval_samples": 16,
            "automodel": {
                "force_hf": False,
                "lm_head_backend": "flash_kld",
                "parallel": {"tp": 1, "cp": 1, "pp": 1, "ep": 1},
            },
        },
        "depth": {
            "source_checkpoint_dir": "/puzzle/ckpts/teacher",
            "output_dir": "/puzzle/depth/iterative",
            "eval_samples": 128,
            "micro_batch_size": 1,
            "block_size": 8192,
            "automodel": {
                "parallel": {
                    "tp": 1,
                    "cp": 1,
                    "pp": 2,
                    "ep": 4,
                    "dp_shard": 4,
                    "dp_replicate": 1,
                }
            },
        },
    }

    effective = distributed_stage_config(config, stage="depth")

    assert "replacement_scoring" not in effective
    assert effective["scoring"]["source_checkpoint_dir"] == "/puzzle/ckpts/teacher"
    assert effective["scoring"]["target_teacher_dir"] == "/puzzle/ckpts/teacher"
    assert effective["scoring"]["eval_samples"] == 128
    assert effective["scoring"]["block_size"] == 8192
    assert effective["scoring"]["automodel"] == {
        "force_hf": False,
        "lm_head_backend": "flash_kld",
        "parallel": {
            "tp": 1,
            "cp": 1,
            "pp": 2,
            "ep": 4,
            "dp_shard": 4,
            "dp_replicate": 1,
        },
    }


def test_rpc_manifest_infers_descriptor_from_scoring_parent(monkeypatch, tmp_path):
    source = tmp_path / "scoring-parent"
    source.mkdir()
    (source / "config.json").write_text("{}\n")
    config = {
        "experiment": {"dir": str(tmp_path)},
        "model": {"force_hf": False, "trust_remote_code": True},
        "scoring": {
            "source_checkpoint_dir": str(source),
            "packed_token_cache_path": str(tmp_path / "validation.tokens"),
            "automodel": {"force_hf": False},
        },
        "data": {},
    }
    monkeypatch.setattr(
        "modelopt.torch.puzzletron.distributed_eval.config.load_plain_pipeline_config",
        lambda *args, **kwargs: config,
    )
    monkeypatch.setattr(
        "modelopt.torch.puzzletron.distributed_eval.config._stage_recipe",
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
    assert manifest.data["scoring"]["packed_token_cache_path"] == str(
        tmp_path / "validation.tokens"
    )


def test_worker_failure_is_reported_before_distributed_cleanup(capsys):
    from modelopt.torch.puzzletron.distributed_eval.cli import _run_worker_with_cleanup

    class BrokenWorker:
        def run(self):
            raise RuntimeError("worker failed before cleanup")

    def cleanup():
        print("cleanup called", file=sys.stderr)

    with pytest.raises(RuntimeError, match="worker failed before cleanup"):
        _run_worker_with_cleanup(BrokenWorker().run, cleanup)

    stderr = capsys.readouterr().err
    assert "RuntimeError: worker failed before cleanup" in stderr
    assert stderr.index("RuntimeError: worker failed before cleanup") < stderr.index("cleanup called")
