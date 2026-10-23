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

"""Build the prompt for, and post the verdict of, the ``/claude review`` workflow.

``prepare`` turns the PR data that .github/workflows/claude_review.yml fetched into one prompt (harness
notes, shared review prompt, ModelOpt guidelines, untrusted PR context) and one patch file per changed
file. ``post`` applies the PR size budget and the test-plan requirement to the model's structured
verdict and writes the review payloads the workflow posts with ``gh api``.

Neither step touches the network or a token, because the PR text they handle is attacker-controlled.
Standard library only: it runs on the runner's stock Python.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import secrets
import sys
from collections import defaultdict
from pathlib import Path, PurePosixPath
from typing import Any

LOGIC_LINE_BUDGET = 500
DECISIONS = ("approve", "lgtm", "nudge", "comment")
CLAUDE_LOGINS = frozenset({"github-actions[bot]", "claude[bot]"})
SPLIT_MARKER = ":scissors:"

# Byte budgets. The action passes the prompt (twice) through environment variables, and Linux
# caps one variable at 128 KB, so the prompt must stay well under that.
MAX_PROMPT_BYTES = 100_000
_BODY_BYTES = 8_000
_COMMENT_BYTES = 2_000
_PREVIOUS_COMMENTS_BYTES = 10_000
_CHANGES_BYTES = 36_000  # the changed-file list and the patches inlined in it
_INLINE_PATCH_BYTES = 26_000
_MAX_REVIEW_BODY = 60_000  # GitHub caps a review body at 65,536 characters
_REVIEW_FIRST = {
    "modelopt": 0,
    "examples": 1,
}  # these directories get the inline patch budget first

VERDICT_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "decision": {"type": "string", "enum": list(DECISIONS)},
        "summary": {"type": "string"},
        "comments": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "file": {"type": "string"},
                    "line": {"type": "integer"},
                    "body": {"type": "string"},
                },
                "required": ["file", "line", "body"],
            },
            "description": "Inline comments, each on a line of the PR head file inside a diff hunk.",
        },
        "nudge_reason": {"type": "string"},
        "size_exception": {"type": "string"},
    },
    "required": ["decision", "summary"],
}

_LOGIC_SUFFIXES = frozenset(
    {".py", ".pyi", ".cu", ".cuh", ".c", ".cc", ".cpp", ".cxx", ".h", ".hh", ".hpp", ".hxx"}
    | {".sh", ".bash", ".js", ".jsx", ".ts", ".tsx"}
)
# Tests and docs never count toward the budget, whatever their suffix.
_NOT_LOGIC = re.compile(
    r"(^|/)(tests?|__tests__|docs?|documentation)/|(^|/)(test_[^/]+\.py|[^/]+_test\.py|conftest\.py)$",
    re.IGNORECASE,
)
_DOMINANCE = 0.7  # a grouping where one slice holds more than this share is not a split
_SPLIT_EXCEPTION = re.compile(
    r"renam|mechanical move|moved verbatim|generated|codegen|self.contained|new example|new model"
    r"|new backend|no working intermediate",
    re.IGNORECASE,
)
_FLAGS = re.IGNORECASE | re.MULTILINE
_SPLIT_ASK = re.compile(
    r"^\s*[-*]\s*\[[ xX]\].*\b(splits?|splitting|break(?:ing)?\s+(?:up|out|it|this|the)|carve)\b",
    _FLAGS,
)
_NEEDS_ACTION = re.compile(r"^\s*needs action:\s*$", _FLAGS)
_NO_ACTION = re.compile(r"^\s*no action needed\.?\s*$", _FLAGS)
# Every real summary has one of these; one with none is the model narrating, not reviewing.
_REVIEW_SHAPE = re.compile(r"^\s*(?:Needs action:|No action needed|- \[[ xX]\])", re.MULTILINE)

_NO_CODE_SUFFIXES = (".md", ".rst", ".txt", ".yaml", ".yml")
_TEST_HEADING = re.compile(r"^\s{0,3}#+\s*(test|testing|tests|how.*tested|verif)", _FLAGS)
_TEST_LABEL = re.compile(r"^\s*\*{0,2}\s*(test\s*plan|testing|tests)\s*\*{0,2}\s*:", _FLAGS)
_UNCHECKED = re.compile(r"^\s*[-*]\s*\[\s\]", re.MULTILINE)
_CHECKED = re.compile(r"^\s*[-*]\s*\[[xX]\]", re.MULTILINE)

_CLOSE_TAG = re.compile(r"</(untrusted_[a-z0-9_]+)>", re.IGNORECASE)
_SPECIAL_TOKEN = re.compile(r"<\|([^|>\n]{1,60})\|>")
_UNSAFE_LABEL = re.compile(r"[^\w./+@=-]")
_PREVIOUS_COMMENTS_PREFACE = (
    "This is a re-review. Classify each comment below as critical or minor, then check whether the "
    "current diff addresses it, as the re-review guidelines of the review prompt describe.\n"
)


def _wrap(label: str, text: str | None, limit: int) -> str:
    """Wrap attacker-controlled text so it cannot close the block or fake a chat-template token."""
    data = (text or "").encode()
    body = data[:limit].decode(errors="ignore") + ("\n[truncated]" if len(data) > limit else "")
    body = _SPECIAL_TOKEN.sub(r"<\\|\1\\|>", _CLOSE_TAG.sub(r"<\\/\1>", body))
    return f"<untrusted_{label}>\n{body}\n</untrusted_{label}>"


def _login(item: dict[str, Any]) -> str:
    return (item.get("user") or {}).get("login", "")


def _is_bot(item: dict[str, Any]) -> bool:
    return (item.get("user") or {}).get("type") == "Bot"


def _flatten(pages: list[Any]) -> list[Any]:
    """Join the pages that ``gh api --paginate --slurp`` returns."""
    return [item for page in pages for item in page]


def _group(entries: list[tuple[str, int]], depth: int) -> dict[str, list[int]]:
    groups: dict[str, list[int]] = defaultdict(lambda: [0, 0])
    for path, lines in entries:
        parts = PurePosixPath(path).parts[:-1]
        group = groups["/".join(parts[:depth]) if parts else "repo-root"]
        group[0] += lines
        group[1] += 1
    return groups


def assess_size(files: list[dict[str, Any]]) -> dict[str, Any]:
    """Count the PR's added lines of core logic, grouped by the shallowest directory that splits it."""
    entries = [
        (f["filename"], f["additions"])
        for f in files
        if f["additions"]
        and PurePosixPath(f["filename"]).suffix.lower() in _LOGIC_SUFFIXES
        and not _NOT_LOGIC.search(f["filename"])
    ]
    total = sum(lines for _, lines in entries)
    for depth in range(1, 9):  # the shallowest depth that divides the diff, else the deepest
        groups = _group(entries, depth)
        if len(groups) > 1 and max(g[0] for g in groups.values()) <= total * _DOMINANCE:
            break
    slices = sorted(
        ((_UNSAFE_LABEL.sub("?", label)[:120], g[0], g[1]) for label, g in groups.items()),
        key=lambda s: (-s[1], s[0]),
    )
    return {"logic_lines": total, "is_oversized": total > LOGIC_LINE_BUDGET, "slices": slices}


def size_notice(size: dict[str, Any]) -> str:
    """Render the ``## PR size`` block that tells the model the PR is over the budget."""
    by_dir = "\n".join(
        f"- `{label}`: {n} lines in {f} file(s)" for label, n, f in size["slices"][:6]
    )
    return f"""## PR size: split required
This PR adds {size["logic_lines"]} lines of core logic, over the {LOGIC_LINE_BUDGET}-line budget. \
Long PRs get skimmed, not reviewed, so the author has to be asked to split it, however clean the \
code is.

Core logic by directory (the shallowest depth that divides it):
{by_dir}

Your summary must open its `Needs action:` block with a `- [ ] {SPLIT_MARKER} Split ...` item that \
names what goes in each stacked `[x/N]` PR, cutting by module, then directory, then file, and by \
feature only where none of those divides the change. The marker is required and the item applies to \
every decision. Do not choose `approve` or `lgtm`: use `nudge`, or `comment` if you also found real \
problems. See focus area 3 of the review prompt.

Only a mechanical rename, generated files or a self-contained drop (a new example, or a new \
model/backend with no working intermediate state) is exempt: set `size_exception` in your verdict, \
naming it and the evidence in this diff. Cohesive, readable and urgent are not exceptions.
"""


def _ensure_split_item(summary: str, size: dict[str, Any]) -> str:
    """Make sure the summary carries the split ask, leading its ``Needs action:`` block."""
    if _SPLIT_ASK.search(summary):
        return summary
    areas = ", ".join(f"`{label}` ({n})" for label, n, _ in size["slices"][:3])
    item = (
        f"- [ ] {SPLIT_MARKER} Split this PR into stacked `[x/N]` PRs: {size['logic_lines']} added "
        f"lines of core logic is over the {LOGIC_LINE_BUDGET}-line budget. Largest areas: {areas}."
    )
    summary = _NO_ACTION.sub("", summary)
    summary, found = _NEEDS_ACTION.subn(lambda m: f"{m.group(0)}\n{item}", summary, count=1)
    return summary if found else f"{summary}\n\nNeeds action:\n{item}"


def check_test_plan(body: str) -> str:
    """Return why the PR description lacks a finished test plan, or an empty string if it has one."""
    if not (_TEST_HEADING.search(body) or _TEST_LABEL.search(body)):
        return "No test plan or testing section found in the PR description."
    unchecked = len(_UNCHECKED.findall(body))
    if unchecked:
        total = unchecked + len(_CHECKED.findall(body))
        return f"Test plan has {unchecked} unchecked item(s) out of {total}. Finish or remove them."
    return ""


def _is_no_code_change(files: list[dict[str, Any]]) -> bool:
    def prose(f: dict[str, Any]) -> bool:
        renamed = f["status"] == "renamed" and not f["additions"] and not f["deletions"]
        return renamed or f["filename"].lower().endswith(_NO_CODE_SUFFIXES)

    return bool(files) and all(prose(f) for f in files)


def apply_gates(
    verdict: dict[str, Any], files: list[dict[str, Any]], pr_body: str
) -> tuple[str, str]:
    """Apply the size budget and the test-plan requirement; return the final decision and summary."""
    decision, summary = verdict["decision"], verdict["summary"]
    size = assess_size(files)
    if size["is_oversized"] and not _SPLIT_EXCEPTION.search(verdict["size_exception"]):
        decision = "nudge" if decision in ("approve", "lgtm") else decision
        summary = _ensure_split_item(summary, size)
    if decision in ("approve", "lgtm") and not _is_no_code_change(files):
        missing = check_test_plan(pr_body)
        if missing:
            decision, summary = "nudge", f"{summary}\n\nTesting: {missing}"
    return decision, summary


def _right_lines(patch: str) -> set[int]:
    """New-file line numbers inside a unified-diff patch: the lines GitHub lets us comment on."""
    lines: set[int] = set()
    new_line: int | None = None
    for raw in patch.split("\n"):  # not splitlines(): it also splits on form feeds
        if raw.startswith("@@"):
            match = re.search(r"\+(\d+)", raw)
            new_line = int(match.group(1)) if match else None
        elif new_line is not None and raw[:1] in ("+", " "):
            lines.add(new_line)
            new_line += 1
    return lines


def _fold(summary: str, comments: list[dict[str, Any]], heading: str) -> str:
    bullets = "\n".join(f"- `{c['file']}:{c['line']}`: {c['body']}" for c in comments)
    return f"{summary}\n\n---\n\n**{heading}**\n\n{bullets}" if comments else summary


def build_review(
    decision: str,
    summary: str,
    comments: list[dict[str, Any]],
    files: list[dict[str, Any]],
    head_sha: str,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Build the review payload and a summary-only fallback for ``POST /pulls/{n}/reviews``.

    A comment off the diff would make GitHub reject the whole review, so it is folded into the
    summary. The fallback folds every comment, for when GitHub rejects the review anyway.
    """
    valid = {f["filename"]: _right_lines(f.get("patch") or "") for f in files}
    inline, outside = [], []
    for comment in comments:
        (inline if comment["line"] in valid.get(comment["file"], ()) else outside).append(comment)
    payload: dict[str, Any] = {
        "commit_id": head_sha,
        "event": "APPROVE" if decision == "approve" else "COMMENT",
        "body": _fold(summary, outside, "Additional comments (outside the PR diff):")[
            :_MAX_REVIEW_BODY
        ],
    }
    if inline:
        payload["comments"] = [
            {"path": c["file"], "line": c["line"], "side": "RIGHT", "body": c["body"]}
            for c in inline
        ]
    fallback = {
        "commit_id": head_sha,
        "event": "COMMENT",
        "body": _fold(summary, comments, "Inline comments:")[:_MAX_REVIEW_BODY],
    }
    return payload, fallback


def _previous_comments(discussion: dict[str, list[dict[str, Any]]]) -> str:
    """Render earlier discussion as a ``## Previous Review Comments`` block, empty on a first review.

    A first review is one with no review or inline comment by Claude. Other bots are left out.
    """
    reviews, inline = discussion["reviews"], discussion["review_comments"]
    if not any(_login(item) in CLAUDE_LOGINS for item in [*reviews, *inline]):
        return ""

    def claude_or_human(item: dict[str, Any]) -> bool:
        return not _is_bot(item) or _login(item) in CLAUDE_LOGINS

    issue = discussion["issue_comments"]
    entries = [
        (r.get("submitted_at") or "", r, "") for r in reviews if r["body"] and claude_or_human(r)
    ]
    entries += [(c["created_at"], c, c["path"]) for c in inline if claude_or_human(c)]
    entries += [
        (c["created_at"], c, "") for c in issue if not _is_bot(c) and not c["body"].startswith("/")
    ]
    lines: list[str] = []
    used = 0
    for _, item, path in sorted(entries, key=lambda e: e[0], reverse=True):
        where = f" ({_wrap('comment_file', path, 300)})" if path else ""
        author = _wrap("comment_author", _login(item), 100)
        line = f"- **{author}**{where}: {_wrap('comment_body', item['body'], _COMMENT_BYTES)}"
        used += len(line.encode())
        if used > _PREVIOUS_COMMENTS_BYTES:
            break
        lines.append(line)
    return "\n".join(["## Previous Review Comments", _PREVIOUS_COMMENTS_PREFACE, *reversed(lines)])


def _strip_title(text: str) -> str:
    return re.sub(r"\A# .*\n+", "", text)


def build_prompt(
    pr: dict[str, Any],
    files: list[dict[str, Any]],
    discussion: dict[str, list[dict[str, Any]]],
    reviewer_comment: str,
    harness: str,
    review: str,
    guidelines: str,
    patch_dir: Path,
) -> str:
    """Compose the review prompt. Patches past the inline budget go to files under ``patch_dir``."""
    size = assess_size(files)
    patch_dir.mkdir(parents=True, exist_ok=True)
    entries, room = [], _INLINE_PATCH_BYTES
    ranked = sorted(
        enumerate(files), key=lambda e: _REVIEW_FIRST.get(e[1]["filename"].split("/")[0], 2)
    )
    for i, f in ranked:
        entry = f"### {f['filename']} ({f['status']}, +{f['additions']} -{f['deletions']})"
        patch = f.get("patch") or ""
        if patch and len(patch.encode()) <= room:
            room -= len(patch.encode())
            entry += f"\n{patch}"
        elif patch:
            (patch_dir / f"{i}.patch").write_text(patch, encoding="utf-8")
            entry += f"\n(patch not inlined, Read {patch_dir / f'{i}.patch'})"
        entries.append(entry)
    base = pr["base"]["ref"]
    context = [
        size_notice(size) if size["is_oversized"] else "",
        "## Pull Request",
        f"**Title**: {_wrap('pr_title', pr['title'], 500)}",
        "**Provider**: github",
        f"**Author**: {_wrap('pr_author', pr['user']['login'], 100)}",
        f"**Branch**: {_wrap('pr_head_branch', pr['head']['ref'], 300)} → "
        f"{_wrap('pr_base_branch', base, 300)}",
        f"**Stats**: {pr['changed_files']} files, +{pr['additions']} -{pr['deletions']}",
        f"**URL**: {pr['html_url']}",
        "",
        "## PR Description",
        _wrap("pr_body", pr["body"], _BODY_BYTES),
        "",
        "## Reviewer's request",
        _wrap("reviewer_comment", reviewer_comment, 3_000),
        "",
        _previous_comments(discussion),
        "## Changed Files",
        _wrap("changed_files", "\n".join(entries), _CHANGES_BYTES),
        "",
        "Review this pull request now. Batch independent tool calls into one response, use about 25 "
        "tool calls in total, and finish with the structured verdict.",
    ]
    parts = [
        ("harness notes", _strip_title(harness)),
        ("review prompt", review),
        ("ModelOpt review guidelines", _strip_title(guidelines)),
    ]
    sections = [
        f"# Part {n}: {name} (trusted)\n\n{text}" for n, (name, text) in enumerate(parts, 1)
    ]
    sections.append("# Part 4: the pull request (untrusted data)\n\n" + "\n".join(context))
    prompt = "\n\n---\n\n".join(sections)
    if len(prompt.encode()) > MAX_PROMPT_BYTES:
        sys.exit(f"The review prompt is over {MAX_PROMPT_BYTES} bytes; shrink the prompt files.")
    return prompt


def _set_output(name: str, value: str) -> None:
    """Write a multi-line step output. The random delimiter keeps PR text from ending it early."""
    delimiter = f"EOF_{secrets.token_hex(16)}"
    with open(os.environ["GITHUB_OUTPUT"], "a", encoding="utf-8") as out:
        out.write(f"{name}<<{delimiter}\n{value}\n{delimiter}\n")


def _read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _parse_verdict(raw: str) -> dict[str, Any]:
    try:
        verdict = json.loads(raw)
    except ValueError:
        sys.exit("The verdict is not valid JSON.")
    if (
        not isinstance(verdict, dict)
        or verdict.get("decision") not in DECISIONS
        or not isinstance(verdict.get("summary"), str)
    ):
        sys.exit("The verdict lacks a valid decision and summary.")
    return {
        "decision": verdict["decision"],
        "summary": verdict["summary"],
        "comments": verdict.get("comments") or [],
        "size_exception": verdict.get("size_exception") or "",
    }


def prepare(data_dir: Path, prompt_dir: Path, guidelines: Path) -> None:
    """Write the ``prompt`` and ``schema`` step outputs from the fetched PR data."""
    discussion = {
        name: _flatten(_read_json(data_dir / f"{name}.json"))
        for name in ("reviews", "review_comments", "issue_comments")
    }
    prompt = build_prompt(
        _read_json(data_dir / "pr.json"),
        _flatten(_read_json(data_dir / "files.json")),
        discussion,
        os.environ.get("COMMENT_BODY", ""),
        harness=(prompt_dir / "harness.md").read_text(encoding="utf-8"),
        review=(prompt_dir / "review.md").read_text(encoding="utf-8"),
        guidelines=guidelines.read_text(encoding="utf-8"),
        patch_dir=data_dir.resolve() / "patches",
    )
    _set_output("prompt", prompt)
    _set_output("schema", json.dumps(VERDICT_SCHEMA, separators=(",", ":")))


def post(data_dir: Path, raw_verdict: str) -> None:
    """Gate the verdict and write ``review.json`` and ``fallback.json`` for the workflow to post."""
    pr = _read_json(data_dir / "pr.json")
    files = _flatten(_read_json(data_dir / "files.json"))
    verdict = _parse_verdict(raw_verdict)
    decision, summary = apply_gates(verdict, files, pr["body"] or "")
    if not _REVIEW_SHAPE.search(summary):
        sys.exit("The verdict summary has no Needs action / No action needed structure.")
    payload, fallback = build_review(
        decision, summary, verdict["comments"], files, pr["head"]["sha"]
    )
    (data_dir / "review.json").write_text(json.dumps(payload), encoding="utf-8")
    (data_dir / "fallback.json").write_text(json.dumps(fallback), encoding="utf-8")
    print(f"decision={decision} inline_comments={len(payload.get('comments', []))}")


def main(argv: list[str] | None = None) -> None:
    """Run the ``prepare`` or ``post`` step."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=["prepare", "post"])
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--prompt-dir", type=Path, help="prepare: harness.md and review.md")
    parser.add_argument("--guidelines", type=Path, help="prepare: the ModelOpt guidelines file")
    args = parser.parse_args(argv)
    if args.command == "prepare":
        prepare(args.data_dir, args.prompt_dir, args.guidelines)
    else:
        post(args.data_dir, os.environ.get("VERDICT", ""))


if __name__ == "__main__":
    main()
