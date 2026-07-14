from types import SimpleNamespace

from modelopt.torch.puzzletron.distributed_eval.config import build_campaign_manifest


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
