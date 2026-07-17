"""Generate the immutable per-model configs for the stage-by-stage campaign."""

from __future__ import annotations

import argparse
from pathlib import Path

from transformers import AutoConfig

from modelopt.torch.puzzletron.campaigns.config_generation import generate_campaign_configs
from modelopt.torch.puzzletron.campaigns.preflight import load_preflight
from modelopt.torch.puzzletron.campaigns.schema import load_campaign


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--campaign",
        type=Path,
        required=True,
        help="Cross-model campaign YAML to compile.",
    )
    parser.add_argument(
        "--preflight",
        type=Path,
        default=Path(
            "puzzle_runs/clean/acceptance/2026-07-06-cross-model-stage-matrix/"
            "campaign/preflight.json"
        ),
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("puzzle_runs/clean/acceptance/2026-07-06-cross-model-stage-matrix"),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    # AutoModel installs native config aliases eagerly.
    import nemo_automodel._transformers.registry  # noqa: F401

    campaign = load_campaign(args.campaign)
    preflight = load_preflight(args.preflight)

    def load_source_config(model, record):
        return AutoConfig.from_pretrained(
            model.hf_id,
            revision=record.immutable_revision,
            trust_remote_code=True,
        )

    outputs = generate_campaign_configs(
        campaign,
        preflight,
        output_root=args.output_root,
        config_loader=load_source_config,
    )
    for output in outputs:
        print(output)


if __name__ == "__main__":
    main()
