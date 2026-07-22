# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Small questionary adapter used by the Puzzletron question flow."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import Any

from . import SetupError

__all__ = ["PromptSession"]

Validator = Callable[[Any], bool | str]


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

        self._describe(description)
        questionary = _questionary()
        return str(
            _answer(
                questionary.text(
                    message,
                    default=default or "",
                    validate=validate,
                )
            )
        )

    def integer(
        self,
        message: str,
        *,
        default: int,
        minimum: int = 1,
        description: str | None = None,
    ) -> int:
        """Ask for a bounded integer."""

        def validate(value: str) -> bool | str:
            try:
                parsed = int(value)
            except ValueError:
                return "Enter an integer."
            return True if parsed >= minimum else f"Enter a value of at least {minimum}."

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

        self._describe(description)
        return bool(_answer(_questionary().confirm(message, default=default)))

    def select(
        self,
        message: str,
        choices: Sequence[Any],
        *,
        default: Any = None,
        description: str | None = None,
    ) -> Any:
        """Choose one value from labels or `(label, value)` pairs."""

        self._describe(description)
        questionary = _questionary()
        rendered = [
            questionary.Choice(title=item[0], value=item[1])
            if isinstance(item, tuple)
            else item
            for item in choices
        ]
        return _answer(questionary.select(message, choices=rendered, default=default))

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
        return list(
            _answer(questionary.checkbox(message, choices=rendered, validate=validate))
        )
