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
"""Simulate simple data-blend dynamics for DoGE-style distillation.

The default two-domain scenario is intentionally small:

* Math data improves Math and also transfers to Chat.
* Chat data improves Chat directly more than Math data transfers to Chat.
* Math-to-Chat transfer fades as Math KD loss saturates.
* Per-domain effective learning rates model task difficulty.

This creates the sanity-check case where Math KD can approach zero while Chat KD is
still high; at that point a useful adaptive scorer should flip toward Chat.
"""

from __future__ import annotations

import argparse
import csv
import html
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Literal

Scorer = Literal["true", "cosine", "dot"]


@dataclass(frozen=True)
class SimulationConfig:
    """Configuration for the two-domain toy dynamics."""

    blocks: int = 100
    target_math_weight: float = 0.5
    target_chat_weight: float = 0.5
    initial_math_loss: float = 1.0
    initial_chat_loss: float = 1.0
    initial_math_source_weight: float = 0.5
    meta_lr: float = 0.2
    train_lr: float = 0.08
    math_effective_lr: float = 1.0
    chat_effective_lr: float = 1.0
    math_learning_scale: float = 1.5
    chat_learning_scale: float = 1.0
    math_to_chat_transfer: float = 0.6
    min_source_weight: float = 0.0


@dataclass(frozen=True)
class SimulationRecord:
    """One simulated block."""

    block: int
    run: str
    scorer: str
    math_loss: float
    chat_loss: float
    target_loss: float
    math_source_weight: float
    chat_source_weight: float
    math_score: float
    chat_score: float


def _normalize(values: tuple[float, float]) -> tuple[float, float]:
    values = (max(0.0, values[0]), max(0.0, values[1]))
    total = values[0] + values[1]
    if total <= 0.0:
        return 0.5, 0.5
    return values[0] / total, values[1] / total


def _source_effects(
    config: SimulationConfig, math_loss: float, chat_loss: float
) -> tuple[tuple[float, float], tuple[float, float]]:
    """Return expected per-source KD-loss drops on target domains (Math, Chat)."""

    math_source = (
        config.math_learning_scale * math_loss,
        config.math_to_chat_transfer * math_loss,
    )
    chat_source = (0.0, config.chat_learning_scale * chat_loss)
    return math_source, chat_source


def _raw_scores(
    scorer: Scorer, config: SimulationConfig, math_loss: float, chat_loss: float
) -> tuple[float, float]:
    math_effect, chat_effect = _source_effects(config, math_loss, chat_loss)
    target_weights = (config.target_math_weight, config.target_chat_weight)

    if scorer == "true":
        return (
            target_weights[0] * math_effect[0] + target_weights[1] * math_effect[1],
            target_weights[0] * chat_effect[0] + target_weights[1] * chat_effect[1],
        )

    target_priority = (
        target_weights[0] * math_loss,
        target_weights[1] * chat_loss,
    )
    math_dot = math_effect[0] * target_priority[0] + math_effect[1] * target_priority[1]
    chat_dot = chat_effect[0] * target_priority[0] + chat_effect[1] * target_priority[1]
    if scorer == "dot":
        return math_dot, chat_dot

    if scorer == "cosine":
        target_norm = math.hypot(*target_priority)
        math_norm = math.hypot(*math_effect)
        chat_norm = math.hypot(*chat_effect)
        if target_norm <= 0.0:
            return 0.0, 0.0
        return (
            0.0 if math_norm <= 0.0 else math_dot / (math_norm * target_norm),
            0.0 if chat_norm <= 0.0 else chat_dot / (chat_norm * target_norm),
        )

    raise ValueError(f"Unknown scorer: {scorer}")


def _apply_floor(weights: tuple[float, float], floor: float) -> tuple[float, float]:
    if floor <= 0.0:
        return weights
    if floor >= 0.5:
        raise ValueError("--min_source_weight must be below 0.5.")
    floored = (max(floor, weights[0]), max(floor, weights[1]))
    return _normalize(floored)


def _update_weights(
    weights: tuple[float, float], scores: tuple[float, float], config: SimulationConfig
) -> tuple[float, float]:
    """Apply the exponentiated update w_i <- normalize(w_i * exp(meta_lr * score_i))."""

    updated = (
        weights[0] * math.exp(config.meta_lr * scores[0]),
        weights[1] * math.exp(config.meta_lr * scores[1]),
    )
    return _apply_floor(_normalize(updated), config.min_source_weight)


def _update_losses(
    weights: tuple[float, float], config: SimulationConfig, math_loss: float, chat_loss: float
) -> tuple[float, float]:
    math_effect, chat_effect = _source_effects(config, math_loss, chat_loss)
    math_drop = (
        config.train_lr
        * config.math_effective_lr
        * (weights[0] * math_effect[0] + weights[1] * chat_effect[0])
    )
    chat_drop = (
        config.train_lr
        * config.chat_effective_lr
        * (weights[0] * math_effect[1] + weights[1] * chat_effect[1])
    )
    return max(0.0, math_loss - math_drop), max(0.0, chat_loss - chat_drop)


def _target_loss(config: SimulationConfig, math_loss: float, chat_loss: float) -> float:
    return config.target_math_weight * math_loss + config.target_chat_weight * chat_loss


def simulate(config: SimulationConfig, scorer: Scorer) -> list[SimulationRecord]:
    """Return fixed-uniform and adaptive trajectories."""

    records: list[SimulationRecord] = []
    for run, adaptive in (("uniform", False), ("adaptive", True)):
        weights = (config.initial_math_source_weight, 1.0 - config.initial_math_source_weight)
        math_loss = config.initial_math_loss
        chat_loss = config.initial_chat_loss
        for block in range(config.blocks + 1):
            scores = _normalize(_raw_scores(scorer, config, math_loss, chat_loss))
            records.append(
                SimulationRecord(
                    block=block,
                    run=run,
                    scorer="fixed" if not adaptive else scorer,
                    math_loss=math_loss,
                    chat_loss=chat_loss,
                    target_loss=_target_loss(config, math_loss, chat_loss),
                    math_source_weight=weights[0],
                    chat_source_weight=weights[1],
                    math_score=0.5 if not adaptive else scores[0],
                    chat_score=0.5 if not adaptive else scores[1],
                )
            )
            if block < config.blocks:
                if adaptive:
                    weights = _update_weights(weights, scores, config)
                math_loss, chat_loss = _update_losses(weights, config, math_loss, chat_loss)
    return records


def _write_csv(path: Path, records: list[SimulationRecord]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(SimulationRecord.__dataclass_fields__))
        writer.writeheader()
        for record in records:
            writer.writerow(asdict(record))


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
    def sx(x_value: float) -> float:
        return left + width * (x_value - x_min) / (x_max - x_min)

    def sy(y_value: float) -> float:
        return top + height * (1.0 - (y_value - y_min) / (y_max - y_min))

    commands = []
    for index, (x_value, y_value) in enumerate(points):
        commands.append(f"{'M' if index == 0 else 'L'} {sx(x_value):.2f} {sy(y_value):.2f}")
    return " ".join(commands)


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
    left, top, width, height = 72.0, 52.0, 680.0, 350.0
    svg_width, svg_height = 880, 500

    elements = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{svg_width}" height="{svg_height}" '
        f'viewBox="0 0 {svg_width} {svg_height}">',
        "<style>text{font-family:Arial,Helvetica,sans-serif;font-size:13px}"
        ".title{font-size:18px;font-weight:700}.label{font-size:14px}</style>",
        f'<text class="title" x="{svg_width / 2:.0f}" y="26" text-anchor="middle">'
        f"{html.escape(title)}</text>",
        f'<line x1="{left}" y1="{top + height}" x2="{left + width}" y2="{top + height}" '
        'stroke="#444" />',
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
                f'<line x1="{left}" y1="{y_pos:.2f}" x2="{left + width}" y2="{y_pos:.2f}" '
                'stroke="#ddd" stroke-width="0.8" />',
                f'<text x="{left - 10}" y="{y_pos + 4:.2f}" text-anchor="end">{y_value:.2f}</text>',
                f'<text x="{x_pos:.2f}" y="{top + height + 24}" text-anchor="middle">{x_value:.0f}</text>',
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
        d = _path_data(
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
                f'<path d="{d}" fill="none" stroke="{color}" stroke-width="2.5" />',
                f'<line x1="{legend_x}" y1="{legend_y}" x2="{legend_x + 20}" '
                f'y2="{legend_y}" stroke="{color}" stroke-width="2.5" />',
                f'<text x="{legend_x + 26}" y="{legend_y + 4}">{html.escape(name)}</text>',
            ]
        )
    elements.append("</svg>")
    path.write_text("\n".join(elements) + "\n")


def write_plots(output_dir: Path, scorer: Scorer, records: list[SimulationRecord]) -> None:
    """Write the requested source-weight and target-loss plots."""

    adaptive = [record for record in records if record.run == "adaptive"]
    uniform = [record for record in records if record.run == "uniform"]

    _write_svg(
        output_dir / f"blend_dynamics_{scorer}_weights.svg",
        title=f"Adaptive source weights ({scorer} scorer)",
        y_label="source weight",
        series=[
            (
                "Math source",
                [(record.block, record.math_source_weight) for record in adaptive],
                "#4C78A8",
            ),
            (
                "Chat source",
                [(record.block, record.chat_source_weight) for record in adaptive],
                "#F58518",
            ),
        ],
        y_min=0.0,
        y_max=1.0,
    )
    _write_svg(
        output_dir / f"blend_dynamics_{scorer}_target_loss.svg",
        title=f"Target KD loss: uniform vs adaptive ({scorer})",
        y_label="target KD loss",
        series=[
            (
                "Uniform 50/50",
                [(record.block, record.target_loss) for record in uniform],
                "#777777",
            ),
            (
                f"DoGE adaptive ({scorer})",
                [(record.block, record.target_loss) for record in adaptive],
                "#54A24B",
            ),
        ],
        y_min=0.0,
        y_max=max(record.target_loss for record in records),
    )


def _print_report(records: list[SimulationRecord], report_every: int) -> None:
    print(
        "run       scorer   block  math_loss  chat_loss  target_loss  "
        "math_weight  chat_weight  math_score  chat_score"
    )
    for record in records:
        if record.block % report_every == 0:
            print(
                f"{record.run:9s} {record.scorer:7s} {record.block:5d} "
                f"{record.math_loss:10.4f} {record.chat_loss:10.4f} {record.target_loss:12.4f} "
                f"{record.math_source_weight:11.4f} {record.chat_source_weight:11.4f} "
                f"{record.math_score:10.4f} {record.chat_score:10.4f}"
            )


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--blocks", type=int, default=100)
    parser.add_argument("--scorer", choices=["true", "cosine", "dot"], default="cosine")
    parser.add_argument("--output-dir", type=Path, default=Path("/tmp/plots"))
    parser.add_argument("--report-every", type=int, default=10)
    parser.add_argument("--meta-lr", type=float, default=0.2)
    parser.add_argument("--train-lr", type=float, default=0.08)
    parser.add_argument(
        "--math-effective-lr",
        type=float,
        default=1.0,
        help="Multiplier for Math KD-loss reduction; lower values make Math harder.",
    )
    parser.add_argument(
        "--chat-effective-lr",
        type=float,
        default=1.0,
        help="Multiplier for Chat KD-loss reduction; lower values make Chat harder.",
    )
    parser.add_argument("--math-learning-scale", type=float, default=1.5)
    parser.add_argument("--chat-learning-scale", type=float, default=1.0)
    parser.add_argument("--math-to-chat-transfer", type=float, default=0.6)
    parser.add_argument("--min-source-weight", type=float, default=0.0)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    config = SimulationConfig(
        blocks=args.blocks,
        meta_lr=args.meta_lr,
        train_lr=args.train_lr,
        math_effective_lr=args.math_effective_lr,
        chat_effective_lr=args.chat_effective_lr,
        math_learning_scale=args.math_learning_scale,
        chat_learning_scale=args.chat_learning_scale,
        math_to_chat_transfer=args.math_to_chat_transfer,
        min_source_weight=args.min_source_weight,
    )
    records = simulate(config, args.scorer)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    _write_csv(args.output_dir / f"blend_dynamics_{args.scorer}.csv", records)
    write_plots(args.output_dir, args.scorer, records)
    _print_report(records, args.report_every)
    print(f"\nWrote CSV and plots to: {args.output_dir}")


if __name__ == "__main__":
    main()
