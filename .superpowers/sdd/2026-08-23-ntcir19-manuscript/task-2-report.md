# Task 2 report — official NTCIR LaTeX scaffold

## RED

R1 disallows source-grep tests for LaTeX configuration and prose, so the
scaffold was validated through its executable contract. Before implementation,
`make -C manuscript build` exited 2 with `No rule to make target 'build'`.
The requested build interface and compiled deliverable were therefore absent.

## GREEN

Added the official `acmart` sigconf/article shell, disabled ACM reference-block
printing and copyright footnote, gobbled page numbering, and used plain page
style. The source provides the approved title and abstract, seven sectional
stubs, intentionally blank author metadata, an ACM-format bibliography
placeholder, and frozen generated inputs for Tables 1--4 and Figure 1.

Added a repeatable `Makefile`. `build` writes `build/main.pdf`; `check` invokes
the existing checker with `--root .` and `--repo-root ..` while exposing the
repository package through `PYTHONPATH=..`; `check-final` adds the final author
metadata gate. `/manuscript/build/` is ignored.

During GREEN verification, the first integration build revealed two real
boundary issues: invoking the checker from `manuscript/` could not import the
repository package, and generated Tables 3--4 exceeded their provisional
layout widths. The Makefile now supplies the repository import path, and the
section-local wrappers first reduce the measurement font then resize each table
to its intended single- or double-column width. This preserves the generated
numeric bodies and avoids broad suppression of TeX overflow diagnostics.

## Verification evidence

- `make -C manuscript build`: passes and produces `manuscript/build/main.pdf`.
- `make -C manuscript check`: exits 0; it reports only the intended warning
  that author metadata is absent from the draft.
- `make -C manuscript check-final`: exits 2 as intended, with only
  `author metadata is required for final submission` as the policy error.
- PDF inspection: 4 pages; `font_errors(...) == []` (all required fonts
  embedded).
- Build-log inspection: no unresolved references or undefined citations; the
  only remaining overfull hbox is 1.66953pt, below the checker's 2pt threshold.
- `pytest tests/test_manuscript.py -q`: 21 passed.
- `pytest -q`: 382 passed, 7 skipped.
- `git diff --check`: passes.

The only deferred submission requirement is real author, affiliation, and
email metadata, which remains intentionally absent until supplied by the user.

## Fix round 1 — rendered Table 3 regression

### RED

The source included the frozen Table 3 fragment, but the one-column float was
stranded behind later floats and did not appear in a clean compiled PDF. Added
`test_clean_compiled_manuscript_renders_all_generated_tables`, which performs
`make clean` and a real Tectonic build, extracts the resulting PDF text, and
requires the four visible generated-table captions. It also rejects any blank
PDF page. At base/head `5368830`, the test failed because the PDF text lacked
`Comparison of the two evaluation estimands.`; the source include alone was
therefore insufficient.

### GREEN

Changed only the placement wrapper for Table 3: it is now a non-floating,
captioned table within the Results section. Its frozen generated tabular body
remains unchanged and is locally resized to the single-column width. This
prevents it from being deferred behind Table 4's two-column float.

The regression test now passes after a clean real build. Extracted PDF text
contains the captions for Tables 1--4, and every rendered page has text. Visual
inspection confirms that Table 3 is readable on page 1 and that the complete
Table 4 remains readable on page 3. The rebuilt PDF has 3 pages, no blank
trailing page, and embedded fonts. Draft check passes; final check fails only
for intentionally absent author metadata. The log has no unresolved references
or citations and only a 1.66953pt overfull hbox, below the checker's 2pt limit.
