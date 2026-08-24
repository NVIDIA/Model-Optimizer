# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

from collections import defaultdict
from statistics import mean

from .specbench import SpecBench

REQUEST_AL_PER_SAMPLE_PER_TURN = "Request_AL_Per_Turn"
CATEGORY_AL_PER_SAMPLE_PER_TURN = "Category_AL_Per_Sample_Per_Turn"
AVERAGE_AL_PER_SAMPLE_PER_TURN = "Average_AL_Per_Sample_Per_Turn"
AVERAGE_AL_PER_TURN = "Average_AL_Per_Turn"
SAMPLE_COUNT_PER_TURN = "Sample_Count_Per_Turn"


class Agentic(SpecBench):
    def _aggregate_request_metrics(self, request_metrics, failed_request_ids):
        super()._aggregate_request_metrics(request_metrics, failed_request_ids)
        self.out[REQUEST_AL_PER_SAMPLE_PER_TURN] = {}

        per_sample_turn_als = defaultdict(list)
        per_turn_als = defaultdict(list)
        sample_category = {}

        for request_metric in request_metrics:
            request = request_metric.request
            request_al = request_metric.request_al
            sample_id = request.question_id
            per_sample_turn_als[sample_id].append(request_al)
            sample_category[sample_id] = request.category
            if request.step is not None:
                per_turn_als[request.step].append(request_al)

        for sample_id, turn_als in per_sample_turn_als.items():
            self.out[REQUEST_AL_PER_SAMPLE_PER_TURN][sample_id] = mean(turn_als)

        if per_turn_als:
            self.out[AVERAGE_AL_PER_TURN] = {
                turn: mean(als) for turn, als in sorted(per_turn_als.items())
            }
            self.out[SAMPLE_COUNT_PER_TURN] = {
                turn: len(als) for turn, als in sorted(per_turn_als.items())
            }

        per_category_turn_als = defaultdict(list)
        for sample_id, al in self.out[REQUEST_AL_PER_SAMPLE_PER_TURN].items():
            per_category_turn_als[sample_category[sample_id]].append(al)
        self.out[CATEGORY_AL_PER_SAMPLE_PER_TURN] = {}
        for category_name, category_al in per_category_turn_als.items():
            if len(category_al) > 0:
                category_al = mean(category_al)
                self.out[CATEGORY_AL_PER_SAMPLE_PER_TURN][category_name] = category_al
        self.out[CATEGORY_AL_PER_SAMPLE_PER_TURN] = dict(
            sorted(self.out[CATEGORY_AL_PER_SAMPLE_PER_TURN].items())
        )
        average_per_turn_ar = mean(self.out[REQUEST_AL_PER_SAMPLE_PER_TURN].values())
        self.out[AVERAGE_AL_PER_SAMPLE_PER_TURN] = average_per_turn_ar

    def _pretty_print_results(self):
        from rich.console import Console
        from rich.table import Table

        console = Console()

        if AVERAGE_AL_PER_TURN in self.out:
            turn_table = Table(
                title="AL per Trajectory Turn",
                show_header=True,
                header_style="bold magenta",
            )
            turn_table.add_column("Turn", style="cyan", no_wrap=True)
            turn_table.add_column(AVERAGE_AL_PER_TURN, justify="right", style="green")
            turn_table.add_column("# Samples", justify="right", style="dim")

            sorted_turns = sorted(self.out[AVERAGE_AL_PER_TURN].items())
            if len(sorted_turns) > 10:
                display_turns = [*sorted_turns[:5], None, *sorted_turns[-5:]]
            else:
                display_turns = sorted_turns
            for entry in display_turns:
                if entry is None:
                    turn_table.add_row("...", "...", "...")
                else:
                    turn, al = entry
                    turn_table.add_row(
                        str(turn),
                        f"{al:.4f}",
                        str(self.out[SAMPLE_COUNT_PER_TURN][turn]),
                    )

            console.print(turn_table)

        table = Table(
            title="Agentic Acceptance Rate Results",
            show_header=True,
            header_style="bold magenta",
        )
        table.add_column("Category", style="cyan", no_wrap=True)
        table.add_column("Average_AL", justify="right", style="green")
        table.add_column(AVERAGE_AL_PER_SAMPLE_PER_TURN, justify="right", style="green")

        for category_name, category_ar in sorted(self.out["Category_AL"].items()):
            table.add_row(
                category_name,
                f"{category_ar:.4f}",
                f"{self.out[CATEGORY_AL_PER_SAMPLE_PER_TURN][category_name]:.4f}",
            )

        table.add_section()
        table.add_row(
            "[bold]Overall Average[/bold]",
            f"[bold]{self.out['Average_AL']:.4f}[/bold]",
            f"[bold]{self.out[AVERAGE_AL_PER_SAMPLE_PER_TURN]:.4f}[/bold]",
        )

        console.print(table)
        console.print(
            "Definitions: Request_AL is accepted tokens / drafted tokens per sample. "
            "Average_AL is the mean of Request_AL over samples. "
            f"{REQUEST_AL_PER_SAMPLE_PER_TURN} is the mean of per-turn accepted/drafted "
            "for each sample. "
            f"{AVERAGE_AL_PER_SAMPLE_PER_TURN} is the mean of "
            f"{REQUEST_AL_PER_SAMPLE_PER_TURN} over samples. "
            f"{AVERAGE_AL_PER_TURN} is the mean accepted/drafted rate for each turn.",
            style="dim",
        )

    def _build_visualization_dataframe(self, text_outputs):
        import pandas as pd

        sample_category = {}
        sample_response_lengths = defaultdict(list)
        for request, messages in zip(self.requests, text_outputs):
            sample_category[request.question_id] = request.category
            response_lengths = [len(c["content"]) for c in messages if c["role"] == "assistant"]
            if response_lengths:
                sample_response_lengths[request.question_id].append(mean(response_lengths))

        sample_ids = list(self.out["Request_AL"].keys())
        return pd.DataFrame.from_dict(
            {
                "question_id": sample_ids,
                "acceptance_rate": [self.out["Request_AL"][sample_id] for sample_id in sample_ids],
                "category": [sample_category[sample_id] for sample_id in sample_ids],
                "response_length": [
                    mean(sample_response_lengths[sample_id])
                    if sample_response_lengths[sample_id]
                    else 0
                    for sample_id in sample_ids
                ],
            }
        )
