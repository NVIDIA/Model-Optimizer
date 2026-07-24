# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Navigable prompt sessions for setup v2."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import Any, Optional, Union

from .prompts import BACK, InteractiveBackend, PromptBackend, PromptChoice
from .state import PromptFrame, WizardState

__all__ = ["BACK", "WizardSession"]

Validator = Callable[[Any], Union[bool, str]]


class WizardSession:
    """Bind prompt interactions to atomic answer and navigation state."""

    def __init__(
        self,
        state: WizardState,
        backend: Optional[PromptBackend] = None,
    ) -> None:
        self.state = state
        self.backend = backend or InteractiveBackend()
        self._section = "campaign"
        self._collection: Optional[str] = None
        self._item_id: Optional[str] = None
        self._cursor: Optional[int] = None

    @property
    def current_frame(self) -> Optional[PromptFrame]:
        frames = self.state.frames
        return frames[-1] if frames else None

    def begin(self, section: str) -> None:
        self._section = section
        self._collection = None
        self._item_id = None
        self._cursor = None

    def enter_collection(
        self,
        collection: str,
        *,
        item_id: Optional[str],
        cursor: Optional[int],
    ) -> None:
        self._collection = collection
        self._item_id = item_id
        self._cursor = cursor

    def leave_collection(self) -> None:
        self._collection = None
        self._item_id = None
        self._cursor = None

    def collection_cursor(self, collection: str) -> Optional[int]:
        frame = self.current_frame
        if frame is not None and frame.collection == collection:
            return frame.cursor
        return None

    def back(self) -> Optional[PromptFrame]:
        return self.state.pop_frame()

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
        self.state.push_frame(frame)
        value = invoke()
        if value is BACK:
            self.state.pop_frame()
            return BACK
        return value

    @staticmethod
    def describe_default(value: Any, source: str) -> None:
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
        validate: Optional[Validator] = None,
    ) -> Any:
        while True:
            value = self._ask(prompt_id, lambda: self.backend.text(message, default))
            if value is BACK:
                return BACK
            rendered = str(value)
            verdict = True if validate is None else validate(rendered)
            if verdict is True:
                return rendered
            print(f"  {verdict}")

    def integer(
        self,
        prompt_id: str,
        message: str,
        *,
        default: int,
        minimum: int = 0,
        maximum: Optional[int] = None,
    ) -> Any:
        def validate(value: str) -> Union[bool, str]:
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
        return self._ask(
            prompt_id,
            lambda: self.backend.select(message, self._choices(choices), default),
        )

    def confirm(
        self,
        prompt_id: str,
        message: str,
        *,
        default: bool,
    ) -> Any:
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
        validate: Optional[Validator] = None,
    ) -> Any:
        while True:
            value = self._ask(
                prompt_id,
                lambda: self.backend.checkbox(
                    message,
                    self._choices(choices),
                    defaults,
                ),
            )
            if value is BACK:
                return BACK
            rendered = list(value)
            verdict = True if validate is None else validate(rendered)
            if verdict is True:
                return rendered
            print(f"  {verdict}")
