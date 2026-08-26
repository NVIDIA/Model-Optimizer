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

"""Navigable prompt sessions for setup v2."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from puzzletron_setup import SetupError

from .prompts import BACK, InteractiveBackend, NonInteractiveBackend, PromptBackend, PromptChoice
from .state import PromptFrame, WizardState

if TYPE_CHECKING:
    from collections.abc import Callable, Sequence

__all__ = ["BACK", "WizardSession"]


class WizardSession:
    """Bind prompt interactions to atomic answer and navigation state."""

    def __init__(
        self,
        state: WizardState,
        backend: PromptBackend | None = None,
        *,
        guided: bool = False,
    ) -> None:
        """Bind wizard state to an interactive or scripted prompt backend."""
        self.state = state
        self.backend = backend or InteractiveBackend()
        self.guided = bool(guided)
        self._section = "campaign"
        self._collection: str | None = None
        self._item_id: str | None = None
        self._cursor: int | None = None
        self._replay: list[tuple[PromptFrame, Any, int]] = []
        self._replay_cursor = 0
        self._back_target: PromptFrame | None = None

    @property
    def current_frame(self) -> PromptFrame | None:
        """Return the most recent prompt frame, if one exists."""
        frames = self.state.frames
        return frames[-1] if frames else None

    def begin(self, section: str) -> None:
        """Begin or resume prompt replay for a wizard section."""
        self._section = section
        self._collection = None
        self._item_id = None
        self._cursor = None
        self._replay = [
            record for record in self.state.answered_frames if record[0].section == section
        ]
        self._replay_cursor = 0

    def enter_collection(
        self,
        collection: str,
        *,
        item_id: str | None,
        cursor: int | None,
    ) -> None:
        """Enter one item in a repeatable prompt collection."""
        self._collection = collection
        self._item_id = item_id
        self._cursor = cursor

    def leave_collection(self) -> None:
        """Leave the active repeatable prompt collection."""
        self._collection = None
        self._item_id = None
        self._cursor = None

    def collection_cursor(self, collection: str) -> int | None:
        """Return the saved cursor for an active collection."""
        frame = self.current_frame
        if frame is not None and frame.collection == collection:
            return frame.cursor
        return None

    def back(self) -> PromptFrame | None:
        """Remove and return the current prompt frame."""
        return self.state.pop_frame()

    def consume_back_target(self) -> PromptFrame | None:
        """Return and clear the prompt that should be asked again after Back."""
        target = self._back_target
        self._back_target = None
        return target

    def _frame(self, prompt_id: str) -> PromptFrame:
        return PromptFrame(
            section=self._section,
            prompt_id=prompt_id,
            collection=self._collection,
            item_id=self._item_id,
            cursor=self._cursor,
        )

    def _ask(self, prompt_id: str, invoke: Callable[[], Any]) -> Any:
        frame = self._frame(prompt_id)
        if self._replay_cursor < len(self._replay):
            replay_frame, replay_value, frame_index = self._replay[self._replay_cursor]
            if replay_frame == frame:
                self._replay_cursor += 1
                return replay_value
            self.state.truncate_frames(frame_index)
            self._replay = self._replay[: self._replay_cursor]
        self.state.push_frame(frame)
        value = invoke()
        if value is BACK:
            self.state.pop_frame()
            target = self.current_frame
            if target is not None:
                self.state.pop_frame()
            self._back_target = target
            return BACK
        self.state.answer_frame(frame, value)
        return value

    @staticmethod
    def describe_default(value: Any, source: str) -> None:
        """Print a resolved default and its provenance."""
        print(f"  Default: {value!r} ({source})")

    @staticmethod
    def _choices(choices: Sequence[Any]) -> list[PromptChoice]:
        rendered = []
        for item in choices:
            if isinstance(item, PromptChoice):
                rendered.append(item)
            elif isinstance(item, tuple):
                rendered.append(PromptChoice(str(item[0]), item[1]))
            else:
                rendered.append(PromptChoice(str(item), item))
        return rendered

    def text(
        self,
        prompt_id: str,
        message: str,
        *,
        default: str = "",
        validate: Callable[[Any], bool | str] | None = None,
    ) -> Any:
        """Request, validate, and persist a text answer."""
        while True:
            value = self._ask(prompt_id, lambda: self.backend.text(message, default))
            if value is BACK:
                return BACK
            rendered = str(value)
            verdict = True if validate is None else validate(rendered)
            if verdict is True:
                return rendered
            self.state.pop_frame()
            if isinstance(self.backend, NonInteractiveBackend):
                raise SetupError(str(verdict))
            print(f"  {verdict}")

    def integer(
        self,
        prompt_id: str,
        message: str,
        *,
        default: int,
        minimum: int = 0,
        maximum: int | None = None,
    ) -> Any:
        """Request a bounded integer answer."""

        def validate(value: str) -> bool | str:
            try:
                parsed = int(value)
            except ValueError:
                return "Enter an integer."
            if parsed < minimum:
                return f"Enter a value of at least {minimum}."
            if maximum is not None and parsed > maximum:
                return f"Enter a value of at most {maximum}."
            return True

        value = self.text(
            prompt_id,
            message,
            default=str(default),
            validate=validate,
        )
        return value if value is BACK else int(value)

    def select(
        self,
        prompt_id: str,
        message: str,
        choices: Sequence[Any],
        *,
        default: Any = None,
    ) -> Any:
        """Request one answer from a set of choices."""
        rendered = self._choices(choices)
        if len(rendered) == 1:
            print(f"  {message} {rendered[0].title} (only option)")
            return rendered[0].value
        return self._ask(
            prompt_id,
            lambda: self.backend.select(message, rendered, default),
        )

    def confirm(
        self,
        prompt_id: str,
        message: str,
        *,
        default: bool,
    ) -> Any:
        """Request a yes-or-no answer."""
        return self.select(
            prompt_id,
            message,
            [("Yes", True), ("No", False)],
            default=default,
        )

    def checkbox(
        self,
        prompt_id: str,
        message: str,
        choices: Sequence[Any],
        *,
        defaults: Sequence[Any] = (),
        validate: Callable[[Any], bool | str] | None = None,
    ) -> Any:
        """Request and validate multiple selected choices."""
        rendered_choices = self._choices(choices)

        def disabled_verdict(values: Sequence[Any]) -> bool | str:
            for value in values:
                choice = next(
                    (item for item in rendered_choices if item.value == value),
                    None,
                )
                if choice is not None and choice.disabled:
                    return f"{choice.title} is unavailable: {choice.disabled}."
            return True

        if len(rendered_choices) == 1:
            selected = [rendered_choices[0].value]
            verdict = disabled_verdict(selected)
            if verdict is True and validate is not None:
                verdict = validate(selected)
            if verdict is True:
                print(
                    f"  {message} {rendered_choices[0].title} (only option, selected automatically)"
                )
                return selected
        while True:
            value = self._ask(
                prompt_id,
                lambda: self.backend.checkbox(
                    message,
                    rendered_choices,
                    defaults,
                ),
            )
            if value is BACK:
                return BACK
            rendered = list(value)
            verdict = disabled_verdict(rendered)
            if verdict is True and validate is not None:
                verdict = validate(rendered)
            if verdict is True:
                return rendered
            self.state.pop_frame()
            if isinstance(self.backend, NonInteractiveBackend):
                raise SetupError(str(verdict))
            print(f"  {verdict}")
