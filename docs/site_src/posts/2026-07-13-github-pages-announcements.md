---
title: "Model Optimizer announcements are moving to GitHub Pages"
author: Model Optimizer Team
date: 2026-07-13
tags: [release, docs, github-pages]
summary: "The public Model Optimizer site is gaining a PR-updated announcements landing page while keeping API documentation one click away."
image: /static/images/model-optimizer-banner.png
---

The Model Optimizer GitHub Pages site is expanding from API documentation into a lightweight announcement hub. The goal is to make releases, technical notes, examples, and deployment writeups easier to discover without introducing a separate publishing system.

## What changes

- Announcements live as Markdown files reviewed through pull requests.
- The landing page defaults to announcements, with API documentation available from the top navigation.
- Posts support tags, search, filtering, and embedded images.

![Model Optimizer banner](/static/images/model-optimizer-banner.png)

## Authoring flow

Create a Markdown file under `docs/site_src/posts/` with YAML frontmatter:

```
---
title: "Your announcement"
author: Your Name
date: 2026-07-13
tags: [quantization, release]
summary: "One sentence summary."
---
```

The GitHub Pages workflow rebuilds the static site from the committed source, so every announcement follows the same review path as code and docs.
