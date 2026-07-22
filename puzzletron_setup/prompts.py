# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Small questionary adapter used by the Puzzletron question flow."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import TYPE_CHECKING, Any, Union

from . import SetupError

if TYPE_CHECKING:
    from .state import AnswerState

__all__ = ["PromptSession"]

# This alias is evaluated at import time; keep it usable by lightweight Python
# 3.9 setup environments even though the ModelOpt package itself starts at 3.10.
Validator = Callable[[Any], Union[bool, str]]  # noqa: UP007


def _questionary():
    try:
        import questionary
    except ImportError as error:
        raise SetupError(
            "questionary is required. Install examples/puzzletron/requirements-setup.txt."
        ) from error
    return questionary


def _answer(question) -> Any:
    value = question.ask()
    if value is None:
        raise KeyboardInterrupt
    return value


class PromptSession:
    """Render consistent prompts while keeping questionary out of wizard logic."""

    def __init__(self) -> None:
        """Create an unbound session; `begin` attaches persisted section state."""
        self._state: AnswerState | None = None
        self._section: str | None = None
        self._transcript: list[dict[str, Any]] = []
        self._cursor = 0

    def begin(self, state: AnswerState, section: str) -> None:
        """Bind prompts to one resumable answer section."""
        self._state = state
        self._section = section
        self._transcript = state.partial(section)
        self._cursor = 0

    def reset(self) -> None:
        """Discard the current section transcript after an invalid retry."""
        if self._state is not None and self._section is not None:
            self._state.truncate_partial(self._section)
        self._transcript = []
        self._cursor = 0

    def checkpoint(self) -> int:
        """Return a transcript position that can be retried safely."""
        return self._cursor

    def rewind(self, checkpoint: int) -> None:
        """Forget answers at and after a checkpoint so invalid input can be corrected."""
        if checkpoint < 0 or checkpoint > self._cursor:
            raise ValueError(f"Invalid prompt checkpoint: {checkpoint}")
        if self._state is not None and self._section is not None:
            self._state.truncate_partial(self._section, checkpoint)
        self._transcript = self._transcript[:checkpoint]
        self._cursor = checkpoint

    def _replay(self, message: str) -> tuple[bool, Any]:
        if self._cursor >= len(self._transcript):
            return False, None
        record = self._transcript[self._cursor]
        if record.get("prompt") != message:
            if self._state is not None and self._section is not None:
                self._state.truncate_partial(self._section, self._cursor)
            self._transcript = self._transcript[: self._cursor]
            return False, None
        self._cursor += 1
        return True, record.get("value")

    def _record(self, message: str, value: Any) -> Any:
        if self._state is not None and self._section is not None:
            self._state.record_partial(self._section, message, value)
            self._transcript.append({"prompt": message, "value": value})
            self._cursor += 1
        return value

    def _describe(self, description: str | None) -> None:
        if description:
            print(f"  {description}")

    def text(
        self,
        message: str,
        *,
        default: str | None = None,
        description: str | None = None,
        validate: Validator | None = None,
    ) -> str:
        """Ask for a non-cancelled string."""
        replayed, value = self._replay(message)
        if replayed:
            rendered = str(value)
            if validate is None or validate(rendered) is True:
                return rendered
            self.rewind(self._cursor - 1)
        self._describe(description)
        questionary = _questionary()
        value = str(
            _answer(
                questionary.text(
                    message,
                    default=default or "",
                    validate=validate,
                )
            )
        )
        return str(self._record(message, value))

    def integer(
        self,
        message: str,
        *,
        default: int,
        minimum: int = 1,
        maximum: int | None = None,
        description: str | None = None,
    ) -> int:
        """Ask for a bounded integer."""

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

        return int(
            self.text(
                message,
                default=str(default),
                description=description,
                validate=validate,
            )
        )

    def confirm(
        self,
        message: str,
        *,
        default: bool,
        description: str | None = None,
    ) -> bool:
        """Ask a yes/no question."""
        replayed, value = self._replay(message)
        if replayed:
            return bool(value)
        self._describe(description)
        value = bool(_answer(_questionary().confirm(message, default=default)))
        return bool(self._record(message, value))

    def select(
        self,
        message: str,
        choices: Sequence[Any],
        *,
        default: Any = None,
        description: str | None = None,
    ) -> Any:
        """Choose one value from labels or `(label, value)` pairs."""
        replayed, value = self._replay(message)
        if replayed:
            return value
        self._describe(description)
        questionary = _questionary()
        rendered = [
            questionary.Choice(title=item[0], value=item[1]) if isinstance(item, tuple) else item
            for item in choices
        ]
        value = _answer(questionary.select(message, choices=rendered, default=default))
        return self._record(message, value)

    def checkbox(
        self,
        message: str,
        choices: Sequence[Any],
        *,
        defaults: Sequence[Any] = (),
        description: str | None = None,
        validate: Validator | None = None,
    ) -> list[Any]:
        """Choose multiple values with explicit checked defaults."""
        replayed, value = self._replay(message)
        if replayed:
            selected = list(value)
            if validate is None or validate(selected) is True:
                return selected
            self.rewind(self._cursor - 1)
        self._describe(description)
        questionary = _questionary()
        checked = set(defaults)
        rendered = []
        for item in choices:
            if isinstance(item, tuple):
                title, value = item
            else:
                title = value = item
            rendered.append(
                questionary.Choice(title=str(title), value=value, checked=value in checked)
            )
        value = list(_answer(questionary.checkbox(message, choices=rendered, validate=validate)))
        return list(self._record(message, value))
