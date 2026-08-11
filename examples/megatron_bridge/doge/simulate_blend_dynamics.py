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
"""Simulate data-blend dynamics for DoGE-style distillation.

The simulator is intentionally simple: every source has an expected KD-loss drop vector over
target domains. Adaptive methods score those source vectors against the current target objective
and update source weights. This is not an LLM training simulator; it is a controlled sanity check
for score dynamics such as cosine alignment versus dot-product alignment.
"""

from __future__ import annotations

import argparse
import csv
import html
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, cast

Scorer = Literal["cosine", "raw_dot", "dot", "kd_gap_dot"]
Preset = Literal["two_domain", "twenty_four_domain"]


@dataclass(frozen=True)
class Source:
    """One train-data source and its expected effect on target KD losses."""

    name: str
    group: str
    effects: tuple[float, ...]


@dataclass(frozen=True)
class Scenario:
    """A complete simulated distillation setup."""

    name: str
    target_names: tuple[str, ...]
    target_weights: tuple[float, ...]
    initial_losses: tuple[float, ...]
    sources: tuple[Source, ...]
    initial_source_weights: tuple[float, ...]


@dataclass(frozen=True)
class SimulationConfig:
    """Generic simulation hyperparameters."""

    blocks: int = 200
    train_lr: float = 0.015
    meta_lr: float = 2.0
    raw_dot_scale: float = 100.0
    min_source_weight: float = 0.0


@dataclass(frozen=True)
class SimulationRecord:
    """One run/block summary."""

    run: str
    scorer: str
    block: int
    target_loss: float
    ptv2_math_loss: float
    ptv2_chat_loss: float
    ptv2_math_score: float
    ptv2_chat_score: float
    ptv2_math_weight: float
    ptv2_chat_weight: float
    math_group_weight: float
    chat_group_weight: float
    weak_broad_group_weight: float
    noise_group_weight: float


def _dot(left: tuple[float, ...], right: tuple[float, ...]) -> float:
    return sum(left_value * right_value for left_value, right_value in zip(left, right))


def _norm(values: tuple[float, ...]) -> float:
    return math.sqrt(sum(value * value for value in values))


def _normalize(values: tuple[float, ...]) -> tuple[float, ...]:
    total = sum(values)
    if total <= 0.0:
        return tuple(1.0 / len(values) for _ in values)
    return tuple(value / total for value in values)


def _normalize_positive(values: tuple[float, ...]) -> tuple[float, ...]:
    return _normalize(tuple(max(0.0, value) for value in values))


def _with_values(size: int, values: dict[int, float]) -> tuple[float, ...]:
    return tuple(values.get(index, 0.0) for index in range(size))


def _two_domain_scenario() -> Scenario:
    target_names = ("ptv2_math", "ptv2_chat")
    sources = (
        Source("ptv2_math", "math", (1.3, 0.25)),
        Source("ptv2_chat", "chat", (0.02, 1.0)),
    )
    return Scenario(
        name="two_domain",
        target_names=target_names,
        target_weights=(0.5, 0.5),
        initial_losses=(1.0, 1.0),
        sources=sources,
        initial_source_weights=(0.5, 0.5),
    )


def _twenty_four_domain_scenario() -> Scenario:
    target_names = (
        "ptv2_math",
        "ptv2_chat",
        "ptv2_code",
        "ptv2_stem",
        "ptv2_multilingual_it",
        "ptv2_science",
        "ptv2_reasoning_on",
        "ptv2_reasoning_off",
        "ptv2_general",
        "wikitext",
    )
    target_count = len(target_names)

    def src(name: str, group: str, values: dict[int, float]) -> Source:
        return Source(name, group, _with_values(target_count, values))

    sources = [
        src("ptv2_math", "math", {0: 2.4, 2: 0.45, 3: 0.70, 5: 0.55, 6: 0.35, 7: 0.30}),
        src("ptv2_chat", "chat", {1: 1.45, 4: 0.25, 6: 0.22, 7: 0.22, 8: 0.38}),
        src("ptv2_code", "code", {2: 1.8, 0: 0.28, 6: 0.25}),
        src("ptv2_stem", "stem", {3: 1.6, 5: 0.80, 0: 0.35}),
        src("ptv2_multilingual_it", "chat", {4: 1.2, 1: 0.22, 8: 0.20}),
        src("ptv2_science", "stem", {5: 1.5, 3: 0.55, 0: 0.28}),
        src("ptv2_reasoning_on", "reasoning", {6: 1.4, 0: 0.32, 2: 0.22}),
        src("ptv2_reasoning_off", "reasoning", {7: 1.3, 1: 0.20, 8: 0.22}),
        src("ptv2_general", "general", {8: 1.1, 1: 0.35, 9: 0.30}),
        src("wikitext", "general", {9: 1.2, 8: 0.35, 1: 0.10}),
        src("nemotron_math_v1", "math", {0: 1.7, 3: 0.35, 5: 0.28}),
        src("nemotron_code", "code", {2: 1.35, 6: 0.22}),
        src("nemotron_general", "general", {8: 0.85, 9: 0.50, 1: 0.28}),
        src("stem_v1", "stem", {3: 0.95, 5: 0.60, 0: 0.15}),
    ]
    # Weak broad sources are deliberately highly aligned with the average target vector but low
    # magnitude. Cosine can over-credit them; dot product should down-weight them.
    weak_broad = tuple(0.11 for _ in range(target_count))
    sources.extend(
        Source(f"weak_broad_{index + 1}", "weak_broad", weak_broad) for index in range(8)
    )
    sources.extend(
        Source(f"synth_noise_{index + 1}", "noise", tuple(0.01 for _ in range(target_count)))
        for index in range(2)
    )
    if len(sources) != 24:
        raise AssertionError(f"Expected 24 sources, got {len(sources)}")
    return Scenario(
        name="twenty_four_domain",
        target_names=target_names,
        target_weights=tuple(1.0 / target_count for _ in target_names),
        initial_losses=(0.82, 1.08, 0.74, 0.78, 0.70, 0.76, 0.72, 0.70, 0.64, 0.58),
        sources=tuple(sources),
        initial_source_weights=tuple(1.0 / len(sources) for _ in sources),
    )


def build_scenario(preset: Preset) -> Scenario:
    """Build one of the supported simulator presets."""

    if preset == "two_domain":
        return _two_domain_scenario()
    if preset == "twenty_four_domain":
        return _twenty_four_domain_scenario()
    raise ValueError(f"Unknown preset: {preset}")


def _source_scores(
    scenario: Scenario, scorer: Scorer, losses: tuple[float, ...], config: SimulationConfig
) -> tuple[float, ...]:
    """Return raw source scores for the requested scoring rule."""

    target_priority = tuple(
        target_weight * loss for target_weight, loss in zip(scenario.target_weights, losses)
    )
    target_norm = _norm(target_priority)
    raw_scores: list[float] = []
    for source in scenario.sources:
        source_dot = _dot(source.effects, target_priority)
        if scorer == "raw_dot":
            raw_scores.append(config.raw_dot_scale * source_dot)
        elif scorer == "dot":
            raw_scores.append(source_dot)
        elif scorer == "kd_gap_dot":
            raw_scores.append(source_dot * _dot(source.effects, losses))
        elif scorer == "cosine":
            source_norm = _norm(source.effects)
            if source_norm <= 0.0 or target_norm <= 0.0:
                raw_scores.append(0.0)
            else:
                raw_scores.append(source_dot / (source_norm * target_norm))
        else:
            raise ValueError(f"Unknown scorer: {scorer}")
    return tuple(raw_scores)


def _apply_floor(weights: tuple[float, ...], min_weight: float) -> tuple[float, ...]:
    if min_weight <= 0.0:
        return weights
    if min_weight * len(weights) >= 1.0:
        raise ValueError("--min-source-weight is too large for the number of sources.")
    return _normalize(tuple(max(min_weight, weight) for weight in weights))


def _update_weights(
    weights: tuple[float, ...], scores: tuple[float, ...], config: SimulationConfig
) -> tuple[float, ...]:
    updated = tuple(
        weight * math.exp(config.meta_lr * score) for weight, score in zip(weights, scores)
    )
    return _apply_floor(_normalize(updated), config.min_source_weight)


def _update_losses(
    scenario: Scenario,
    weights: tuple[float, ...],
    losses: tuple[float, ...],
    config: SimulationConfig,
) -> tuple[float, ...]:
    source_count = len(scenario.sources)
    drops = []
    for target_index, loss in enumerate(losses):
        effective_rate = sum(
            weights[source_index] * scenario.sources[source_index].effects[target_index]
            for source_index in range(source_count)
        )
        drops.append(config.train_lr * effective_rate * loss)
    return tuple(max(0.0, loss - drop) for loss, drop in zip(losses, drops))


def _target_loss(scenario: Scenario, losses: tuple[float, ...]) -> float:
    return _dot(scenario.target_weights, losses)


def _target_index(scenario: Scenario, name: str) -> int:
    try:
        return scenario.target_names.index(name)
    except ValueError:
        return 0


def _weight_sum(scenario: Scenario, weights: tuple[float, ...], group: str) -> float:
    return sum(weight for weight, source in zip(weights, scenario.sources) if source.group == group)


def _record(
    run: str,
    scorer: str,
    block: int,
    scenario: Scenario,
    losses: tuple[float, ...],
    weights: tuple[float, ...],
    scores: tuple[float, ...],
) -> SimulationRecord:
    ptv2_math_index = _target_index(scenario, "ptv2_math")
    ptv2_chat_index = _target_index(scenario, "ptv2_chat")
    ptv2_math_source_index = next(
        index for index, source in enumerate(scenario.sources) if source.name == "ptv2_math"
    )
    ptv2_chat_source_index = next(
        index for index, source in enumerate(scenario.sources) if source.name == "ptv2_chat"
    )
    score_masses = _normalize_positive(scores)
    return SimulationRecord(
        run=run,
        scorer=scorer,
        block=block,
        target_loss=_target_loss(scenario, losses),
        ptv2_math_loss=losses[ptv2_math_index],
        ptv2_chat_loss=losses[ptv2_chat_index],
        ptv2_math_score=score_masses[ptv2_math_source_index],
        ptv2_chat_score=score_masses[ptv2_chat_source_index],
        ptv2_math_weight=weights[ptv2_math_source_index],
        ptv2_chat_weight=weights[ptv2_chat_source_index],
        math_group_weight=_weight_sum(scenario, weights, "math"),
        chat_group_weight=_weight_sum(scenario, weights, "chat"),
        weak_broad_group_weight=_weight_sum(scenario, weights, "weak_broad"),
        noise_group_weight=_weight_sum(scenario, weights, "noise"),
    )


def simulate(
    scenario: Scenario, config: SimulationConfig, scorers: tuple[Scorer, ...]
) -> list[SimulationRecord]:
    """Run fixed uniform plus all requested adaptive scorers."""

    records: list[SimulationRecord] = []
    runs: list[tuple[str, Scorer | None]] = [("uniform", None)]
    runs.extend((f"adaptive_{scorer}", scorer) for scorer in scorers)
    for run, scorer in runs:
        losses = scenario.initial_losses
        weights = scenario.initial_source_weights
        for block in range(config.blocks + 1):
            if scorer is None:
                scores = tuple(1.0 / len(scenario.sources) for _ in scenario.sources)
            else:
                scores = _source_scores(scenario, scorer, losses, config)
            records.append(
                _record(
                    run=run,
                    scorer="fixed" if scorer is None else scorer,
                    block=block,
                    scenario=scenario,
                    losses=losses,
                    weights=weights,
                    scores=scores,
                )
            )
            if block < config.blocks:
                if scorer is not None:
                    weights = _update_weights(weights, scores, config)
                losses = _update_losses(scenario, weights, losses, config)
    return records


def _write_csv(path: Path, records: list[SimulationRecord]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(SimulationRecord.__dataclass_fields__))
        writer.writeheader()
        for record in records:
            writer.writerow(record.__dict__)


def _path_data(
    points: list[tuple[float, float]],
    *,
    x_min: float,
    x_max: float,
    y_min: float,
    y_max: float,
    left: float,
    top: float,
    width: float,
    height: float,
) -> str:
    def sx(value: float) -> float:
        return left + width * (value - x_min) / (x_max - x_min)

    def sy(value: float) -> float:
        return top + height * (1.0 - (value - y_min) / (y_max - y_min))

    return " ".join(
        f"{'M' if index == 0 else 'L'} {sx(x_value):.2f} {sy(y_value):.2f}"
        for index, (x_value, y_value) in enumerate(points)
    )


def _write_svg(
    path: Path,
    *,
    title: str,
    y_label: str,
    series: list[tuple[str, list[tuple[float, float]], str]],
    y_min: float,
    y_max: float,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    x_values = [x_value for _, points, _ in series for x_value, _ in points]
    x_min = min(x_values)
    x_max = max(x_values)
    left, top, width, height = 72.0, 52.0, 700.0, 350.0
    svg_width, svg_height = 980, 500
    elements = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{svg_width}" height="{svg_height}" '
        f'viewBox="0 0 {svg_width} {svg_height}">',
        "<style>text{font-family:Arial,Helvetica,sans-serif;font-size:13px}"
        ".title{font-size:18px;font-weight:700}.label{font-size:14px}</style>",
        f'<text class="title" x="{svg_width / 2:.0f}" y="26" text-anchor="middle">'
        f"{html.escape(title)}</text>",
        f'<line x1="{left}" y1="{top + height}" x2="{left + width}" '
        f'y2="{top + height}" stroke="#444" />',
        f'<line x1="{left}" y1="{top}" x2="{left}" y2="{top + height}" stroke="#444" />',
    ]
    for tick in range(6):
        fraction = tick / 5
        y_pos = top + height * (1.0 - fraction)
        y_value = y_min + fraction * (y_max - y_min)
        x_pos = left + width * fraction
        x_value = x_min + fraction * (x_max - x_min)
        elements.extend(
            [
                f'<line x1="{left}" y1="{y_pos:.2f}" x2="{left + width}" '
                f'y2="{y_pos:.2f}" stroke="#ddd" stroke-width="0.8" />',
                f'<text x="{left - 10}" y="{y_pos + 4:.2f}" text-anchor="end">{y_value:.2f}</text>',
                f'<text x="{x_pos:.2f}" y="{top + height + 24}" text-anchor="middle">'
                f"{x_value:.0f}</text>",
            ]
        )
    elements.extend(
        [
            f'<text class="label" x="{left + width / 2:.0f}" y="{svg_height - 20}" '
            'text-anchor="middle">block</text>',
            f'<text class="label" x="18" y="{top + height / 2:.0f}" text-anchor="middle" '
            f'transform="rotate(-90 18 {top + height / 2:.0f})">{html.escape(y_label)}</text>',
        ]
    )
    legend_x = left + width + 22
    for index, (name, points, color) in enumerate(series):
        path_data = _path_data(
            points,
            x_min=x_min,
            x_max=x_max,
            y_min=y_min,
            y_max=y_max,
            left=left,
            top=top,
            width=width,
            height=height,
        )
        legend_y = top + 14 + index * 24
        elements.extend(
            [
                f'<path d="{path_data}" fill="none" stroke="{color}" stroke-width="2.5" />',
                f'<line x1="{legend_x}" y1="{legend_y}" '
                f'x2="{legend_x + 20}" y2="{legend_y}" '
                f'stroke="{color}" stroke-width="2.5" />',
                f'<text x="{legend_x + 26}" y="{legend_y + 4}">{html.escape(name)}</text>',
            ]
        )
    elements.append("</svg>")
    path.write_text("\n".join(elements) + "\n")


def _series(
    records: list[SimulationRecord], field: str, runs: tuple[str, ...]
) -> list[tuple[str, list[tuple[float, float]], str]]:
    colors = {
        "uniform": "#777777",
        "adaptive_cosine": "#E45756",
        "adaptive_raw_dot": "#B279A2",
        "adaptive_dot": "#54A24B",
        "adaptive_kd_gap_dot": "#4C78A8",
    }
    labels = {
        "uniform": "Uniform",
        "adaptive_cosine": "Cosine DoGE",
        "adaptive_raw_dot": "Raw-dot DoGE",
        "adaptive_dot": "Scaled-dot DoGE",
        "adaptive_kd_gap_dot": "KD-gap dot",
    }
    return [
        (
            labels[run],
            [
                (record.block, float(getattr(record, field)))
                for record in records
                if record.run == run
            ],
            colors[run],
        )
        for run in runs
    ]


def write_plots(output_dir: Path, preset: Preset, records: list[SimulationRecord]) -> None:
    """Write SVG plots for the simulated trajectories."""

    runs = tuple(dict.fromkeys(record.run for record in records))
    max_loss = max(record.target_loss for record in records)
    dot_records = [record for record in records if record.run == "adaptive_dot"]
    cosine_records = [record for record in records if record.run == "adaptive_cosine"]
    _write_svg(
        output_dir / f"blend_sim_{preset}_target_loss.svg",
        title=f"{preset}: target KD loss",
        y_label="target KD loss",
        series=_series(records, "target_loss", runs),
        y_min=0.0,
        y_max=max_loss,
    )
    _write_svg(
        output_dir / f"blend_sim_{preset}_ptv2_losses.svg",
        title=f"{preset}: PT-v2 Math/Chat KD loss",
        y_label="KD loss",
        series=[
            (
                "Dot PT-v2 Math",
                [(record.block, record.ptv2_math_loss) for record in dot_records],
                "#54A24B",
            ),
            (
                "Dot PT-v2 Chat",
                [(record.block, record.ptv2_chat_loss) for record in dot_records],
                "#F58518",
            ),
            (
                "Cos PT-v2 Math",
                [(record.block, record.ptv2_math_loss) for record in cosine_records],
                "#72B7B2",
            ),
            (
                "Cos PT-v2 Chat",
                [(record.block, record.ptv2_chat_loss) for record in cosine_records],
                "#E45756",
            ),
        ],
        y_min=0.0,
        y_max=max(record.ptv2_chat_loss for record in records),
    )
    _write_svg(
        output_dir / f"blend_sim_{preset}_weights.svg",
        title=f"{preset}: selected adaptive weights",
        y_label="source/group weight",
        series=[
            (
                "Dot Math group",
                [(record.block, record.math_group_weight) for record in dot_records],
                "#54A24B",
            ),
            (
                "Dot Chat group",
                [(record.block, record.chat_group_weight) for record in dot_records],
                "#F58518",
            ),
            (
                "Dot weak broad",
                [(record.block, record.weak_broad_group_weight) for record in dot_records],
                "#B279A2",
            ),
            (
                "Cos Math group",
                [(record.block, record.math_group_weight) for record in cosine_records],
                "#72B7B2",
            ),
            (
                "Cos Chat group",
                [(record.block, record.chat_group_weight) for record in cosine_records],
                "#E45756",
            ),
            (
                "Cos weak broad",
                [(record.block, record.weak_broad_group_weight) for record in cosine_records],
                "#FF9DA6",
            ),
        ],
        y_min=0.0,
        y_max=1.0,
    )
    _write_svg(
        output_dir / f"blend_sim_{preset}_ptv2_scores.svg",
        title=f"{preset}: PT-v2 normalized score mass",
        y_label="normalized score mass",
        series=[
            (
                "Dot PT-v2 Math score",
                [(record.block, record.ptv2_math_score) for record in dot_records],
                "#54A24B",
            ),
            (
                "Dot PT-v2 Chat score",
                [(record.block, record.ptv2_chat_score) for record in dot_records],
                "#F58518",
            ),
            (
                "Cos PT-v2 Math score",
                [(record.block, record.ptv2_math_score) for record in cosine_records],
                "#72B7B2",
            ),
            (
                "Cos PT-v2 Chat score",
                [(record.block, record.ptv2_chat_score) for record in cosine_records],
                "#E45756",
            ),
        ],
        y_min=0.0,
        y_max=0.25,
    )


def _print_report(records: list[SimulationRecord], report_every: int) -> None:
    print(
        "run                  block  target_loss  ptv2_math  ptv2_chat  "
        "math_score  chat_score  math_group_w  chat_group_w  weak_broad_w"
    )
    for record in records:
        if record.block % report_every == 0:
            print(
                f"{record.run:20s} {record.block:5d} {record.target_loss:12.4f} "
                f"{record.ptv2_math_loss:10.4f} {record.ptv2_chat_loss:10.4f} "
                f"{record.ptv2_math_score:10.4f} {record.ptv2_chat_score:10.4f} "
                f"{record.math_group_weight:12.4f} {record.chat_group_weight:12.4f} "
                f"{record.weak_broad_group_weight:12.4f}"
            )
    print("\nFinal summary:")
    for run in dict.fromkeys(record.run for record in records):
        final = max(
            (record for record in records if record.run == run), key=lambda record: record.block
        )
        print(
            f"{run:20s} target={final.target_loss:.4f} math={final.ptv2_math_loss:.4f} "
            f"chat={final.ptv2_chat_loss:.4f} math_w={final.math_group_weight:.4f} "
            f"chat_w={final.chat_group_weight:.4f} weak_w={final.weak_broad_group_weight:.4f}"
        )


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--preset", choices=["two_domain", "twenty_four_domain"], default="twenty_four_domain"
    )
    parser.add_argument("--blocks", type=int, default=200)
    parser.add_argument("--train-lr", type=float, default=0.015)
    parser.add_argument("--meta-lr", type=float, default=2.0)
    parser.add_argument("--min-source-weight", type=float, default=0.0)
    parser.add_argument(
        "--scorers",
        choices=["cosine", "raw_dot", "dot", "kd_gap_dot"],
        nargs="+",
        default=["cosine", "raw_dot", "dot", "kd_gap_dot"],
    )
    parser.add_argument(
        "--raw-dot-scale",
        type=float,
        default=100.0,
        help="Multiplier for raw-dot scores to mimic unnormalized real gradient-dot scale.",
    )
    parser.add_argument("--output-dir", type=Path, default=Path("/tmp/plots"))
    parser.add_argument("--report-every", type=int, default=20)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    scenario = build_scenario(args.preset)
    config = SimulationConfig(
        blocks=args.blocks,
        train_lr=args.train_lr,
        meta_lr=args.meta_lr,
        raw_dot_scale=args.raw_dot_scale,
        min_source_weight=args.min_source_weight,
    )
    records = simulate(scenario, config, cast("tuple[Scorer, ...]", tuple(args.scorers)))
    args.output_dir.mkdir(parents=True, exist_ok=True)
    _write_csv(args.output_dir / f"blend_sim_{args.preset}.csv", records)
    write_plots(args.output_dir, args.preset, records)
    _print_report(records, args.report_every)
    print(f"\nWrote CSV and SVG plots to: {args.output_dir}")


if __name__ == "__main__":
    main()
