"""Build canonical distributed requests from Puzzletron one-block solutions."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable

from .campaign import Campaign
from .identity import canonicalize, content_id
from .schema import EvaluationRequest


def build_replace_block_requests(
    campaign: Campaign,
    *,
    solutions_path: str | Path,
    solution_ids: Iterable[int] | None = None,
    sort_solutions_by: str | None = None,
    bigger_is_better: bool = False,
) -> list[EvaluationRequest]:
    from ..plugins.automodel.solution_launch import _extract_single_sequence_replacement
    from ..tools.validate_puzzle_with_multi_replacements import load_puzzle_solutions

    solutions = load_puzzle_solutions(
        Path(solutions_path),
        sort_solutions_by,
        bigger_is_better,
    )
    ids = list(solution_ids) if solution_ids is not None else list(range(len(solutions)))
    requests: list[EvaluationRequest] = []
    for solution_id in ids:
        solution = solutions[solution_id]
        replacement = _extract_single_sequence_replacement(solution)
        solution_payload = solution.get("puzzle_solution", solution)
        hidden_width = solution_payload.get("hidden_width")
        hidden_width = None if hidden_width is None else int(hidden_width)
        replacement_dict = (
            replacement.to_dict() if hasattr(replacement, "to_dict") else replacement
        )
        candidate_id = content_id(
            "candidate",
            {
                "layer_replacement": replacement_dict,
                "hidden_width": hidden_width,
            },
        )
        requests.append(
            EvaluationRequest(
                campaign_id=campaign.campaign_id,
                handler="replace_block",
                payload={
                    "candidate_id": candidate_id,
                    "solution_id": solution_id,
                    "hidden_width": hidden_width,
                    "layer_replacement": replacement_dict,
                    "puzzle_solution": canonicalize(solution),
                },
                model=campaign.manifest.model,
                data=campaign.manifest.data,
                metrics=campaign.manifest.metrics,
                precision=campaign.manifest.precision,
                evaluator_revision=campaign.manifest.evaluator_revision,
                metadata={"solutions_path": str(Path(solutions_path).resolve())},
            )
        )
    return requests
