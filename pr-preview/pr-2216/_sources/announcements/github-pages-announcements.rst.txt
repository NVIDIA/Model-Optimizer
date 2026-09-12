:orphan:

Model Optimizer Announcements Are Moving to GitHub Pages
#########################################################

:Author: Model Optimizer Team
:Date: August 13, 2026
:Tags: release, docs, github-pages

The Model Optimizer GitHub Pages site is expanding from API documentation into a lightweight announcement hub. The goal is to make releases, technical notes, examples, and deployment writeups easier to discover without introducing a separate publishing system.

What Changes
************

* Announcements live in the documentation source and are reviewed through pull requests.
* The landing page defaults to announcements.
* Existing API documentation remains available from the Sphinx left navigation.
* Announcement pages support tags, search, filtering, and embedded images.

Authoring Flow
**************

Add a Sphinx page under ``docs/source/announcements/``, add its ``.announcement-card`` metadata and link to ``docs/source/index.rst``, and link it from the announcements toctree when applicable. The GitHub Pages workflow rebuilds the static site from committed source, so every announcement follows the same review path as code and docs.
