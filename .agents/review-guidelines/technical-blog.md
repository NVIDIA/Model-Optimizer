# Technical blog review guideline

Use this rubric when a pull-request comment requests a **technical blog** or
**technical announcement** review. It applies to public-facing documentation
such as `docs/source/announcements/` and complements the repository's normal
code-review guidance.

## Review scope

Review the changed announcement and its landing-page card together. Do not
review unrelated source files unless they provide evidence for a claim in the
post.

## Checks

1. **Factual support** — Every technical claim, performance number, and
   comparison must be supported by a cited public source, a clearly identified
   reproducible measurement, or a qualified statement. Flag claims that
   overstate what the cited source establishes.
2. **Citation integrity** — Check that cited papers, repositories, checkpoints,
   and issue or PR links exist and match the surrounding claim. Publication
   dates must not precede the cited source's availability.
3. **Technical precision** — Preserve meaningful distinctions: measured versus
   inferred results, training versus serving behavior, throughput versus
   latency, architecture versus implementation detail, and public facts versus
   internal context.
4. **Figure provenance** — Images need an accurate alt text and a source or
   provenance that makes their public use appropriate. Captions and nearby
   text must not imply a result the figure does not show.
5. **Public-release suitability** — Do not expose private infrastructure,
   unreleased products, confidential benchmark data, credentials, internal
   URLs, or claims that cannot be independently supported by public material.
6. **Reader clarity** — Verify the title, date, author, summary, tags, and
   announcement-card metadata agree. Prefer precise terminology over marketing
   shorthand when the two could be confused.

## Findings

Raise only material findings. Each finding should identify the exact claim,
explain the public-facing risk, and propose a concrete correction. Do not
duplicate routine style, spelling, or formatting feedback already handled by
CodeRabbit.
