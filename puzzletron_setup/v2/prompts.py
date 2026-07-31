# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Prompt backends with a universal Back action."""

from __future__ import annotations

from collections import deque
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any, Protocol

from puzzletron_setup import SetupError

__all__ = [
    "BACK",
    "InteractiveBackend",
    "PromptBackend",
    "PromptChoice",
    "ScriptedBackend",
]


class _Back:
    def __repr__(self) -> str:
        return "BACK"


BACK = _Back()


@dataclass(frozen=True)
class PromptChoice:
    """One backend-neutral prompt choice."""

    title: str
    value: Any
    disabled: str | None = None


class PromptBackend(Protocol):
    """Minimal backend used by the navigable wizard session."""

    def text(self, message: str, default: str) -> Any:
        raise NotImplementedError

    def select(
        self,
        message: str,
        choices: Sequence[PromptChoice],
        default: Any,
    ) -> Any:
        raise NotImplementedError

    def checkbox(
        self,
        message: str,
        choices: Sequence[PromptChoice],
        defaults: Sequence[Any],
    ) -> Any:
        raise NotImplementedError


def _questionary():
    try:
        import questionary
    except ImportError as error:
        raise SetupError(
            "questionary is required. Install examples/puzzletron/requirements-setup.txt."
        ) from error
    return questionary


def _answer(question: Any) -> Any:
    value = question.ask()
    if value is None:
        raise KeyboardInterrupt
    return value


def _choice_style(questionary: Any) -> Any:
    return questionary.Style(
        [
            ("highlighted", "noreverse bg:default"),
            ("selected", "noreverse bg:default"),
        ]
    )


def _bind_escape_back(question: Any) -> Any:
    """Make Escape return the same sentinel for every interactive widget."""
    key_bindings = question.application.key_bindings
    if not hasattr(key_bindings, "add"):
        from prompt_toolkit.key_binding import KeyBindings, merge_key_bindings

        escape_bindings = KeyBindings()
        question.application.key_bindings = merge_key_bindings(
            [key_bindings, escape_bindings]
        )
        key_bindings = escape_bindings

    @key_bindings.add("escape", eager=True)
    def go_back(event):
        event.app.exit(result=BACK)

    return question


class InteractiveBackend:
    """Questionary-backed prompts with visible Back controls."""

    _BACK_TITLE = "← Back"

    def text(self, message: str, default: str) -> Any:
        print("  Press Esc to go back (or type :back).")
        question = _bind_escape_back(_questionary().text(message, default=default))
        value = _answer(question)
        if value is BACK:
            return BACK
        value = str(value)
        return BACK if value.strip().lower() == ":back" else value

    def select(
        self,
        message: str,
        choices: Sequence[PromptChoice],
        default: Any,
    ) -> Any:
        questionary = _questionary()
        rendered = [
            questionary.Choice(
                title=choice.title,
                value=choice.value,
                disabled=choice.disabled,
            )
            for choice in choices
        ]
        rendered.append(questionary.Choice(title=self._BACK_TITLE, value=BACK))
        return _answer(
            _bind_escape_back(
                questionary.select(
                    message,
                    choices=rendered,
                    default=default,
                    style=_choice_style(questionary),
                )
            )
        )

    def checkbox(
        self,
        message: str,
        choices: Sequence[PromptChoice],
        defaults: Sequence[Any],
    ) -> Any:
        questionary = _questionary()
        selected = set(defaults)
        rendered = [
            questionary.Choice(
                title=choice.title,
                value=choice.value,
                checked=choice.value in selected,
                disabled=choice.disabled,
            )
            for choice in choices
        ]
        rendered.append(questionary.Separator(f"  {self._BACK_TITLE} (press Esc)"))
        question = _bind_escape_back(
            questionary.checkbox(
                message,
                choices=rendered,
                instruction=(
                    "(Use arrow keys to move, <space> to select, <a> to toggle, "
                    "<i> to invert, <esc> to go back)"
                ),
                style=_choice_style(questionary),
            )
        )

        values = _answer(question)
        if values is BACK:
            return BACK
        return list(values)


class ScriptedBackend:
    """Deterministic non-interactive backend for embedding and automation."""

    def __init__(self, answers: Sequence[Any]) -> None:
        self._answers = deque(answers)

    @property
    def remaining(self) -> int:
        return len(self._answers)

    def _next(self) -> Any:
        if not self._answers:
            raise SetupError("Scripted setup input was exhausted.")
        value = self._answers.popleft()
        return BACK if value == ":back" else value

    def text(self, message: str, default: str) -> Any:
        del message, default
        return self._next()

    def select(
        self,
        message: str,
        choices: Sequence[PromptChoice],
        default: Any,
    ) -> Any:
        del message, choices, default
        return self._next()

    def checkbox(
        self,
        message: str,
        choices: Sequence[PromptChoice],
        defaults: Sequence[Any],
    ) -> Any:
        del message, choices, defaults
        return self._next()
