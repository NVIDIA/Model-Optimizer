#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Build the public Model Optimizer announcement site."""

from __future__ import annotations

import argparse
import html
import json
import re
import shutil
from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parent
SITE_SRC = ROOT / "site_src"
POSTS_DIR = SITE_SRC / "posts"
STATIC_DIR = SITE_SRC / "static"
TOOLS_DIR = SITE_SRC / "tools"
SOURCE_ASSETS = ROOT / "source" / "assets"


def parse_frontmatter(text: str) -> tuple[dict, str]:
    if not text.startswith("---"):
        return {}, text
    end = text.find("\n---", 3)
    if end < 0:
        return {}, text
    metadata = yaml.safe_load(text[3:end]) or {}
    return metadata, text[end + 4 :].strip()


def slug_from_path(path: Path) -> str:
    slug = path.stem
    match = re.match(r"\d{4}-\d{2}-\d{2}-(.*)", slug)
    return match.group(1) if match else slug


def inline_markup(text: str) -> str:
    escaped = html.escape(text)
    escaped = re.sub(r"`([^`]+)`", r"<code>\1</code>", escaped)
    escaped = re.sub(r"\*\*([^*]+)\*\*", r"<strong>\1</strong>", escaped)
    escaped = re.sub(r"\[([^\]]+)\]\(([^)]+)\)", r'<a href="\2">\1</a>', escaped)
    return escaped


def page_path(path: str, prefix: str) -> str:
    if not path or "://" in path or path.startswith("#") or path.startswith("mailto:"):
        return path
    if path.startswith("/"):
        return prefix + path.lstrip("/")
    return path


def render_markdown(markdown: str, prefix: str) -> str:
    lines = markdown.splitlines()
    out: list[str] = []
    paragraph: list[str] = []
    list_open = False
    code_open = False
    code_lines: list[str] = []

    def flush_paragraph() -> None:
        nonlocal paragraph
        if paragraph:
            out.append(f"<p>{inline_markup(' '.join(paragraph))}</p>")
            paragraph = []

    def close_list() -> None:
        nonlocal list_open
        if list_open:
            out.append("</ul>")
            list_open = False

    for raw in lines:
        line = raw.rstrip()
        stripped = line.strip()
        if stripped.startswith("```"):
            flush_paragraph()
            close_list()
            if code_open:
                out.append(f"<pre><code>{html.escape(chr(10).join(code_lines))}</code></pre>")
                code_lines = []
                code_open = False
            else:
                code_open = True
            continue
        if code_open:
            code_lines.append(raw)
            continue
        if not stripped:
            flush_paragraph()
            close_list()
            continue
        if stripped.startswith("<") and stripped.endswith(">"):
            flush_paragraph()
            close_list()
            out.append(re.sub(r'(src|href)="(/[^"]*)"', lambda m: f'{m.group(1)}="{page_path(m.group(2), prefix)}"', stripped))
            continue
        if stripped.startswith("### "):
            flush_paragraph()
            close_list()
            out.append(f"<h3>{inline_markup(stripped[4:])}</h3>")
            continue
        if stripped.startswith("## "):
            flush_paragraph()
            close_list()
            out.append(f"<h2>{inline_markup(stripped[3:])}</h2>")
            continue
        image = re.match(r"!\[([^\]]*)\]\(([^)]+)\)", stripped)
        if image:
            flush_paragraph()
            close_list()
            alt, src = image.groups()
            src = page_path(src, prefix)
            out.append(
                '<figure class="post-image">'
                f'<img src="{html.escape(src, quote=True)}" alt="{html.escape(alt, quote=True)}">'
                f"<figcaption>{inline_markup(alt)}</figcaption>"
                "</figure>"
            )
            continue
        if stripped.startswith("- "):
            flush_paragraph()
            if not list_open:
                out.append("<ul>")
                list_open = True
            out.append(f"<li>{inline_markup(stripped[2:])}</li>")
            continue
        paragraph.append(stripped)

    flush_paragraph()
    close_list()
    if code_open:
        out.append(f"<pre><code>{html.escape(chr(10).join(code_lines))}</code></pre>")
    return "\n".join(out)


def load_posts() -> list[dict]:
    posts = []
    for path in sorted(POSTS_DIR.glob("*.md")):
        metadata, body = parse_frontmatter(path.read_text(encoding="utf-8"))
        if not metadata.get("title"):
            continue
        slug = slug_from_path(path)
        posts.append(
            {
                "title": str(metadata.get("title", "")),
                "author": str(metadata.get("author", "")),
                "date": str(metadata.get("date", ""))[:10],
                "summary": str(metadata.get("summary", "")),
                "tags": [str(tag) for tag in metadata.get("tags", [])],
                "image": str(metadata.get("image", "")),
                "slug": slug,
                "url": f"announcements/{slug}/",
                "body": body,
            }
        )
    posts.sort(key=lambda post: post["date"], reverse=True)
    return posts


def topnav(prefix: str, active: str) -> str:
    home = prefix or "./"
    links = [
        ("Announcements", home, "announcements"),
        ("API Docs", f"{prefix}api/", "api"),
        ("GitHub", "https://github.com/NVIDIA/Model-Optimizer", "github"),
    ]
    rendered = []
    for label, href, key in links:
        cls = ' class="active"' if active == key else ""
        rendered.append(f'<a href="{href}"{cls}>{label}</a>')
    return (
        '<nav class="topnav">'
        f'<a href="{home}" class="logo">Model Optimizer</a>'
        f'<div class="nav-links">{"".join(rendered)}'
        '<button class="theme-toggle" type="button" aria-label="Toggle color theme" aria-pressed="false">'
        '<span class="theme-toggle-track"><span class="theme-toggle-thumb"></span></span>'
        '<span class="theme-toggle-label">Light</span>'
        "</button>"
        "</div>"
        "</nav>"
    )


def render_index(posts: list[dict]) -> str:
    tags = sorted({tag for post in posts for tag in post["tags"]})
    cards = []
    for post in posts:
        tag_html = "".join(f"<span>#{html.escape(tag)}</span>" for tag in post["tags"])
        cards.append(
            '<a class="post-card" '
            f'href="{post["url"]}" '
            f'data-title="{html.escape(post["title"].lower(), quote=True)}" '
            f'data-summary="{html.escape(post["summary"].lower(), quote=True)}" '
            f'data-tags="{html.escape(" ".join(post["tags"]).lower(), quote=True)}">'
            f'<time>{html.escape(post["date"])}</time>'
            f'<h2>{html.escape(post["title"])}</h2>'
            f'<p>{html.escape(post["summary"])}</p>'
            f'<div class="card-meta"><span>{html.escape(post["author"])}</span><div class="card-tags">{tag_html}</div></div>'
            "</a>"
        )
    tag_buttons = "".join(f'<button type="button" data-tag="{html.escape(tag)}">#{html.escape(tag)}</button>' for tag in tags)
    posts_json = json.dumps(
        [
            {
                "title": p["title"],
                "summary": p["summary"],
                "tags": p["tags"],
                "url": p["url"],
                "date": p["date"],
            }
            for p in posts
        ]
    )
    return f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>Model Optimizer Announcements</title>
  <script>
    (function () {{
      var theme = localStorage.getItem("modelopt-theme") || "dark";
      document.documentElement.dataset.theme = theme;
    }})();
  </script>
  <link rel="stylesheet" href="static/css/blog.css">
</head>
<body>
  {topnav("", "announcements")}
  <main>
    <section class="blog-hero">
      <div>
        <p class="eyebrow">NVIDIA Model Optimizer</p>
        <h1>Announcements</h1>
        <p class="subtitle">Release notes, technical updates, examples, and deployment stories from the Model Optimizer team.</p>
      </div>
    </section>

    <section class="toolbar" aria-label="Announcement filters">
      <label class="search-box">
        <span>Search announcements</span>
        <input id="post-search" type="search" placeholder="Search title, summary, or tag">
      </label>
      <div class="tag-filter" id="tag-filter">
        <button type="button" class="active" data-tag="">All</button>
        {tag_buttons}
      </div>
    </section>

    <section class="blog-grid" id="post-grid">
      {"".join(cards)}
    </section>

    <p class="empty-state" id="empty-state" hidden>No announcements match the current filters.</p>

    <section class="authoring">
      <h2>Add an announcement by PR</h2>
      <p>Create a Markdown file in <code>docs/site_src/posts/</code> with YAML frontmatter. Images can live under <code>docs/site_src/static/images/</code> and be referenced from the post body.</p>
    </section>
  </main>
  <script id="post-data" type="application/json">{html.escape(posts_json)}</script>
  <script src="static/js/blog.js"></script>
</body>
</html>
"""


def render_post(post: dict) -> str:
    tags = "".join(f"<span>#{html.escape(tag)}</span>" for tag in post["tags"])
    prefix = "../../"
    image_src = page_path(post["image"], prefix)
    image = f'<img class="post-hero-image" src="{html.escape(image_src, quote=True)}" alt="">' if image_src else ""
    content_html = render_markdown(post["body"], prefix)
    return f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>{html.escape(post["title"])} | Model Optimizer Announcements</title>
  <script>
    (function () {{
      var theme = localStorage.getItem("modelopt-theme") || "dark";
      document.documentElement.dataset.theme = theme;
    }})();
  </script>
  <link rel="stylesheet" href="../../static/css/blog.css">
</head>
<body>
  {topnav(prefix, "announcements")}
  <main class="post-content">
    <a class="back-link" href="../../">Back to announcements</a>
    <header class="post-header">
      <time>{html.escape(post["date"])}</time>
      <h1>{html.escape(post["title"])}</h1>
      <p>{html.escape(post["summary"])}</p>
      <div class="post-meta"><span>{html.escape(post["author"])}</span><div class="card-tags">{tags}</div></div>
      {image}
    </header>
    <article>{content_html}</article>
  </main>
</body>
</html>
"""


def build(output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    posts = load_posts()
    (output_dir / "index.html").write_text(render_index(posts), encoding="utf-8")
    for post in posts:
        post_dir = output_dir / "announcements" / post["slug"]
        post_dir.mkdir(parents=True, exist_ok=True)
        (post_dir / "index.html").write_text(render_post(post), encoding="utf-8")
    if STATIC_DIR.exists():
        shutil.copytree(STATIC_DIR, output_dir / "static", dirs_exist_ok=True)
    if TOOLS_DIR.exists():
        shutil.copytree(TOOLS_DIR, output_dir / "tools", dirs_exist_ok=True)
    if SOURCE_ASSETS.exists():
        image_dir = output_dir / "static" / "images"
        image_dir.mkdir(parents=True, exist_ok=True)
        for asset in SOURCE_ASSETS.iterdir():
            if asset.is_file():
                shutil.copy2(asset, image_dir / asset.name)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=ROOT / "build" / "html")
    args = parser.parse_args()
    build(args.output)


if __name__ == "__main__":
    main()
