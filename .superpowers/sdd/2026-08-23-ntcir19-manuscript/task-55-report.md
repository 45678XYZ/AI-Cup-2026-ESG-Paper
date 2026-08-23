# Task 5.5 report — four title-block layout placeholders

Date verified: 2026-08-23

## Amendment implemented

- `metadata.tex` now renders exactly four visible ACM title-block names:
  `Student Author 1`, `Student Author 2`, `Student Author 3`, and
  `Student Author 4`.
- No affiliation, institution, team name, email, or real identity was added.
- Draft mode emits `WARNING: layout-only author placeholders remain in this
  draft` and otherwise passes.
- Final mode emits `ERROR: layout-only author placeholders must be replaced
  before final submission`. The checker treats an explicit placeholder author
  (or the explicit `TODO(author-metadata)` marker) as a blocker even when a
  syntactically email-like `\\email{...@...}` is present. It does not use a
  broad `Student` heuristic.
- The approved spec, plan global/final-check descriptions, paper plan, and
  README descriptions now record this layout-only amendment. Real metadata is
  still deferred to the user.

## TDD evidence

- RED: the new controlled fixture failed before implementation because the old
  checker emitted neither a placeholder warning nor a final error when the four
  placeholder authors and `\\email{student@example.org}` were present.
- RED: the clean compiled-PDF integration test failed before implementation
  because `Student Author 1`--`Student Author 4` were absent from extracted
  page-one text.
- RED: the separate explicit `TODO(author-metadata)` fixture failed until the
  checker recognized that exact marker; no generic `TODO` or `Student` match is
  used.
- GREEN: focused placeholder fixtures and the clean compiled-PDF integration
  test passed after the minimal checker and metadata changes.

## Render and checks

- Normal `make -C manuscript check`: exit 0; only the intended layout-only
  placeholder warning.
- `make -C manuscript check-final`: nonzero; its only error is the required
  layout-placeholder replacement error.
- The normal and isolated-font clean builds passed. The latter is exercised by
  `test_clean_compiled_manuscript_renders_without_system_fonts`.
- PDF extraction finds every one of the four literal names on page 1. Visual
  inspection of the rendered page shows the four centered title-block slots,
  with no invented details.
- PDF: 6 pages (within the eight-page limit). The draft check also passed page,
  embedded-font, build-log, undefined-citation/reference, and overfull-box
  checks; the rendered PDF was inspected for title-block layout and float/order
  continuity.
- `pytest -q tests/test_manuscript.py`: 25 passed.
- `pytest -q`: 386 passed, 7 skipped.
- `python -m paper.validate --all`: 72 artifacts checked clean.
- `git diff --check`: clean. Diff against `112b81c` confirms no changes to
  frozen probabilities, predictions, results, splits, tables, figures, or
  `run_manifest.json`.

## Remaining submission action

Replace all four placeholders with real user-supplied author names and valid
email metadata, plus any required affiliation details, before running the final
submission gate.
