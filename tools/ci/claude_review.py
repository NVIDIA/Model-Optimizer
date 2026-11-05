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

``prepare`` turns the PR data that .github/workflows/claude_review.yml fetched into one prompt: the
harness notes, the review baseline and the ModelOpt rubric (read at the PR's base commit) and the
untrusted PR context, with the patches that do not fit in the prompt written to files. ``post`` turns
the model's structured verdict into the review payloads the workflow posts with ``gh api``; only a
complete ``approve`` against an existing baseline becomes an approval.

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

CORE_LINE_GUIDELINE = 500
DECISIONS = ("approve", "lgtm", "nudge", "comment")
CLAUDE_LOGINS = frozenset({"github-actions[bot]", "claude[bot]"})
BASELINE = "code-review.md"
RUBRIC = "modelopt.md"

# Byte budgets. The action passes the prompt (twice) through environment variables, and Linux
# caps one variable at 128 KB, so the prompt must stay well under that.
MAX_PROMPT_BYTES = 100_000
_BODY_BYTES = 8_000
_COMMENT_BYTES = 2_000
_PREVIOUS_COMMENTS_BYTES = 10_000
_CHANGES_BYTES = 36_000  # the changed-file list and the patches inlined in it
_INLINE_PATCH_BYTES = 26_000
_MAX_REVIEW_BODY = 60_000  # GitHub caps a review body at 65,536 characters
_REVIEW_FIRST = {"modelopt": 0, "examples": 1}  # these get the inline patch budget first

VERDICT_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "decision": {"type": "string", "enum": list(DECISIONS)},
        "summary": {"type": "string"},
        "complete": {"type": "boolean"},
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
    },
    "required": ["decision", "summary", "complete"],
}

# AGENTS.md's rule for the size guideline: tests, docs, config and blank lines do not count.
_NOT_CORE = re.compile(
    r"(^|/)(tests|docs)/|(^|/)test_[^/]*\.py$|_test\.go$|\.spec\.[^/]+$"
    r"|\.(ya?ml|json|toml|ini|xml|lock|md|rst|txt|ipynb)$"
)
_DOMINANCE = 0.7  # a grouping where one slice holds more than this share is not a split
# Every real summary has one of these; one with none is the model narrating, not reviewing.
_REVIEW_SHAPE = re.compile(r"^\s*(?:Needs action:|No action needed|- \[[ xX]\])", re.MULTILINE)
_CLOSE_TAG = re.compile(r"</(untrusted_[a-z0-9_]+)>", re.IGNORECASE)
_SPECIAL_TOKEN = re.compile(r"<\|([^|>\n]{1,60})\|>")
_UNSAFE_LABEL = re.compile(r"[^\w./+@=-]")
_NO_BASELINE = (
    "No review baseline exists at the PR's base commit. Review with your own judgement and the "
    "rubric below, if there is one, say in the summary that no baseline was available, and use "
    "`nudge` or `comment`: a review without a baseline is never published as an approval."
)
_PREVIOUS_COMMENTS_PREFACE = (
    "This is a re-review. Classify each comment below as critical or minor, then check whether the "
    "current diff addresses it, as the re-review guidelines of the baseline describe.\n"
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


def _core_lines(f: dict[str, Any]) -> int:
    """Added non-blank lines of one file, 0 if it is a test, doc or config file."""
    if _NOT_CORE.search(f["filename"]):
        return 0
    patch = f.get("patch")
    if patch is None:  # GitHub omits very large patches; the plain count is an upper bound
        return int(f["additions"])
    return sum(1 for line in patch.split("\n") if line.startswith("+") and line[1:].strip())


def _group(entries: list[tuple[str, int]], depth: int) -> dict[str, list[int]]:
    groups: dict[str, list[int]] = defaultdict(lambda: [0, 0])
    for path, lines in entries:
        parts = PurePosixPath(path).parts[:-1]
        group = groups["/".join(parts[:depth]) if parts else "repo-root"]
        group[0] += lines
        group[1] += 1
    return groups


def assess_size(files: list[dict[str, Any]]) -> dict[str, Any]:
    """Count the PR's added core-logic lines, grouped by the shallowest directory that splits it."""
    entries = [(f["filename"], n) for f in files if (n := _core_lines(f))]
    total = sum(lines for _, lines in entries)
    for depth in range(1, 9):  # the shallowest depth that divides the diff, else the deepest
        groups = _group(entries, depth)
        if len(groups) > 1 and max(g[0] for g in groups.values()) <= total * _DOMINANCE:
            break
    slices = sorted(
        ((_UNSAFE_LABEL.sub("?", label)[:120], g[0], g[1]) for label, g in groups.items()),
        key=lambda s: (-s[1], s[0]),
    )
    return {"core_lines": total, "is_over": total > CORE_LINE_GUIDELINE, "slices": slices}


def size_notice(size: dict[str, Any]) -> str:
    """Render the ``## PR size`` block: a trusted upper bound, shown only above the guideline."""
    by_dir = "\n".join(
        f"- `{label}`: {n} lines in {f} file(s)" for label, n, f in size["slices"][:6]
    )
    return f"""## PR size: above the ~{CORE_LINE_GUIDELINE}-line guideline
This PR adds {size["core_lines"]} lines of core logic by the AGENTS.md count: tests, docs, config and \
blank lines are excluded, comments and docstrings are not, so the count is an upper bound. A PR a \
little over the guideline needs no split ask, so subtract comments and docstrings by eye when the \
count is close, then apply the baseline's size rule.

Core logic by directory (the shallowest depth that divides it):
{by_dir}
"""


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
    approve: bool,
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
        "event": "APPROVE" if approve else "COMMENT",
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
    texts: dict[str, str | None],
    patch_dir: Path,
) -> str:
    """Compose the review prompt. Patches past the inline budget go to files under ``patch_dir``.

    ``texts`` holds the trusted ``harness`` and, when they exist at the base commit, the ``baseline``
    and the ``rubric``.
    """
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
    size = assess_size(files)
    context = [
        size_notice(size) if size["is_over"] else "",
        "## Pull Request",
        f"**Title**: {_wrap('pr_title', pr['title'], 500)}",
        "**Provider**: github",
        f"**Author**: {_wrap('pr_author', pr['user']['login'], 100)}",
        (
            f"**Branch**: {_wrap('pr_head_branch', pr['head']['ref'], 300)} → "
            f"{_wrap('pr_base_branch', pr['base']['ref'], 300)}"
        ),
        f"**Head SHA**: {os.environ['HEAD_SHA']}",
        f"**Base SHA**: {os.environ['BASE_SHA']}",
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
        (
            "Review this pull request now. Batch independent tool calls into one response, use about "
            "25 tool calls in total, and finish with the structured verdict."
        ),
    ]
    parts = [
        ("Harness notes", _strip_title(texts["harness"] or "")),
        ("Review baseline", _strip_title(texts["baseline"]) if texts["baseline"] else _NO_BASELINE),
    ]
    if texts["rubric"]:
        parts.append(("ModelOpt review rubric", _strip_title(texts["rubric"])))
    sections = [f"# {name} (trusted)\n\n{text}" for name, text in parts]
    sections.append("# The pull request (untrusted data)\n\n" + "\n".join(context))
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


def _read_text(path: Path) -> str | None:
    return path.read_text(encoding="utf-8") if path.exists() else None


def _parse_verdict(raw: str) -> dict[str, Any]:
    try:
        verdict = json.loads(raw)
    except ValueError:
        sys.exit("The verdict is not valid JSON.")
    if (
        not isinstance(verdict, dict)
        or verdict.get("decision") not in DECISIONS
        or not isinstance(verdict.get("summary"), str)
        or not _REVIEW_SHAPE.search(verdict["summary"])
    ):
        sys.exit("The verdict lacks a decision and a summary with Needs action / No action needed.")
    return {
        "decision": verdict["decision"],
        "summary": verdict["summary"],
        "complete": verdict.get("complete") is True,
        "comments": verdict.get("comments") or [],
    }


def prepare(data_dir: Path, harness: Path, guidance_dir: Path) -> None:
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
        {
            "harness": harness.read_text(encoding="utf-8"),
            "baseline": _read_text(guidance_dir / BASELINE),
            "rubric": _read_text(guidance_dir / RUBRIC),
        },
        patch_dir=data_dir.resolve() / "patches",
    )
    _set_output("prompt", prompt)
    _set_output("schema", json.dumps(VERDICT_SCHEMA, separators=(",", ":")))


def post(data_dir: Path, raw_verdict: str, guidance_dir: Path) -> None:
    """Write ``review.json`` and ``fallback.json`` for the workflow to post.

    Only a complete ``approve`` against an existing baseline is published as an approval; ``lgtm``,
    ``nudge`` and ``comment`` are comment reviews. The footer names the reviewed head and baseline.
    """
    files = _flatten(_read_json(data_dir / "files.json"))
    verdict = _parse_verdict(raw_verdict)
    head, base = os.environ["HEAD_SHA"], os.environ["BASE_SHA"]
    baseline = (guidance_dir / BASELINE).exists()
    footer = (
        f"Baseline: .agents/review-guidelines/{BASELINE} at {base[:12]}."
        if baseline
        else f"No review baseline at {base[:12]}, so this is a comment review."
    )
    approve = verdict["decision"] == "approve" and verdict["complete"] and baseline
    payload, fallback = build_review(
        approve,
        f"{verdict['summary']}\n\n<sub>Reviewed {head[:12]}. {footer}</sub>",
        verdict["comments"],
        files,
        head,
    )
    (data_dir / "review.json").write_text(json.dumps(payload), encoding="utf-8")
    (data_dir / "fallback.json").write_text(json.dumps(fallback), encoding="utf-8")
    print(f"event={payload['event']} inline_comments={len(payload.get('comments', []))}")


def main(argv: list[str] | None = None) -> None:
    """Run one step. ``HEAD_SHA`` and ``BASE_SHA`` come from the environment."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=["prepare", "post"])
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument(
        "--guidance-dir", type=Path, required=True, help="baseline and rubric files"
    )
    parser.add_argument("--harness", type=Path, help="prepare: harness.md")
    args = parser.parse_args(argv)
    if args.command == "prepare":
        prepare(args.data_dir, args.harness, args.guidance_dir)
    else:
        post(args.data_dir, os.environ.get("VERDICT", ""), args.guidance_dir)


if __name__ == "__main__":
    main()
