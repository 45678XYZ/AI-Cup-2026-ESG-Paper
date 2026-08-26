# Paper Figures Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the focused manuscript's technical-looking Figure 1 and multilingual table float with two reproducible, publication-quality vector figures.

**Architecture:** Keep Figure 1's existing label-derived macro pipeline and restyle only its TikZ drawing. Add a Figure 2 pipeline that parses the canonical generated multilingual table, writes TeX macros, and builds a standalone TikZ PDF; integrate both figures into the manuscript and provenance checks.

**Tech Stack:** Python 3.10, pytest, TikZ/pdfLaTeX via latexmk, Tectonic, pypdf, GNU Make

**Spec:** `docs/superpowers/specs/2026-08-26-paper-figures-design.md`

## Global Constraints

- Work in the existing `paper/ntcir19-manuscript` workspace; do not create a worktree.
- Do not change methods, experiment arms, numerical results, or inferential claims.
- Keep M0--M6 only; M7 remains excluded.
- Generate every printed number from existing canonical data; do not hand-copy results into a drawing.
- Keep all figure output vector, deterministic, font-embedded, colour-blind-friendly, and legible in greyscale.
- Keep `manuscript/build/main.pdf` tracked and the final manuscript at eight pages or fewer.

---

### Task 1: Restyle Figure 1 with an observable vector-colour contract

**Files:**
- Modify: `tests/test_analysis_figure1.py`
- Modify: `figures/figure1_hierarchy.tex`
- Regenerate: `figures/figure1_hierarchy.pdf`

**Interfaces:**
- Consumes: `analysis.figure1.build(out_path)` and `figure1_defs.tex`
- Produces: a deterministic vector PDF containing multiple non-grey fill colours while preserving all generated count macros

- [ ] **Step 1: Write the failing rendered-PDF test**

Add a test that builds Figure 1, reads the page content stream, extracts RGB fill operators, and requires at least three distinct non-grey fills. The current mostly white figure must fail this test.

- [ ] **Step 2: Run the focused test and verify RED**

Run: `pytest -q tests/test_analysis_figure1.py -k non_grey_fill`

Expected: FAIL because the current drawing has fewer than three non-grey fills.

- [ ] **Step 3: Implement the approved Figure 1 treatment**

Edit the TikZ source to add labelled panel bands, filled hierarchy nodes, filled route cards, legal/invalid status chips, and stronger alignment. Continue to reference `\NumStates`, `\NumCombinations`, `\NumInvalid`, and the other generated macros rather than typing their values.

- [ ] **Step 4: Rebuild and verify GREEN**

Run: `pytest -q tests/test_analysis_figure1.py`

Expected: all Figure 1 tests pass, including vector, font, reproducibility, and generated-count checks.

- [ ] **Step 5: Commit the Figure 1 unit**

Run:

```bash
git add tests/test_analysis_figure1.py figures/figure1_hierarchy.tex figures/figure1_hierarchy.pdf
git commit -m "docs(figure): clarify task and decision architecture"
```

### Task 2: Add the data-derived multilingual Figure 2 generator

**Files:**
- Create: `tests/test_analysis_figure2.py`
- Create: `analysis/figure2.py`
- Create: `figures/figure2_multilingual.tex`
- Create: `figures/figure2_defs.tex`
- Create: `figures/figure2_multilingual.pdf`
- Modify: `analysis/__main__.py`
- Modify: `analysis/preview.py`
- Modify: `tests/test_analysis_preview.py`
- Modify: `tests/test_run_manifest.py`

**Interfaces:**
- Consumes: `tables/table6_multilingual.tex`
- Produces: `parse_table(path) -> list[dict]`, `write_defs(rows, path) -> Path`, and `build(table_path, out_path) -> Path`

- [ ] **Step 1: Write parser and rendering contract tests**

Use a five-row literal TeX fixture. Assert that parsing returns the exact invalid rate, repair nets, and score deltas; macro output preserves the values and derived signs; the committed table produces five rows; and a rendered PDF is non-empty, vector-only, font-embedded, and deterministic.

- [ ] **Step 2: Run the focused test and verify RED**

Run: `pytest -q tests/test_analysis_figure2.py`

Expected: collection fails because `analysis.figure2` does not exist.

- [ ] **Step 3: Implement the minimal parser, macro writer, and builder**

Parse only the canonical seven-column table row shape, validate exactly five expected languages, escape generated labels, compute plot coordinates from the parsed deltas, write `figure2_defs.tex`, and compile the standalone TikZ source with the same deterministic environment as Figure 1.

- [ ] **Step 4: Draw the approved Figure 2**

Create five aligned rows with invalid-rate badges, signed N/A and substantive ledger cards, and a common delta axis using distinct marker shapes for weighted F1 and tuple accuracy. Print exact values through generated macros and add a text takeaway stating the derived 5/5 versus 2/5 sign count.

- [ ] **Step 5: Integrate one-command generation and preview**

Call the Figure 2 builder after table generation in `analysis/__main__.py`, include both figures in the analysis preview, and extend the run-manifest output test to require `figure2_multilingual.pdf`.

- [ ] **Step 6: Run the focused tests and verify GREEN**

Run: `pytest -q tests/test_analysis_figure2.py tests/test_analysis_preview.py tests/test_run_manifest.py`

Expected: all selected tests pass.

- [ ] **Step 7: Commit the Figure 2 unit**

Run:

```bash
git add analysis/figure2.py analysis/__main__.py analysis/preview.py tests/test_analysis_figure2.py tests/test_analysis_preview.py tests/test_run_manifest.py figures/figure2_multilingual.tex figures/figure2_defs.tex figures/figure2_multilingual.pdf
git commit -m "feat(analysis): visualize multilingual repair effects"
```

### Task 3: Integrate Figure 2 into the focused manuscript

**Files:**
- Modify: `tests/test_manuscript.py`
- Modify: `manuscript/check.py`
- Modify: `manuscript/sections/06_results.tex`
- Modify: `manuscript/sections/03_task_and_data.tex`
- Modify: `run_manifest.json`
- Regenerate: `manuscript/build/main.pdf`

**Interfaces:**
- Consumes: both committed figure PDFs and the existing multilingual result prose
- Produces: a manuscript requiring canonical Figure 2 inclusion while retaining Table 6 as a generated provenance artifact

- [ ] **Step 1: Write failing manuscript-policy tests**

Change the policy fixture to include the canonical Figure 2. Add tests that reject a missing Figure 2 inclusion and an external decoy, and update the canonical asset fixture to require both PDFs. The current policy must fail these tests.

- [ ] **Step 2: Run the focused tests and verify RED**

Run: `pytest -q tests/test_manuscript.py -k 'multilingual or canonical_repository_assets'`

Expected: FAIL because the policy and manuscript still require the multilingual table float rather than Figure 2.

- [ ] **Step 3: Update manuscript policy and figure references**

Require `figures/figure2_multilingual.pdf` as a tracked frozen asset and canonical manuscript inclusion. Replace the Table 6 float with a Figure 2 float and revise the surrounding paragraph and captions without altering any numerical claim. Tighten Figure 1's caption to match the restyled diagram.

- [ ] **Step 4: Regenerate the run manifest and manuscript PDF**

Run:

```bash
python -m paper.run_manifest
make -C manuscript build
```

Expected: `run_manifest.json` indexes both figure PDFs and Tectonic exits zero.

- [ ] **Step 5: Verify policy and PDF structure**

Run: `pytest -q tests/test_manuscript.py` and `make -C manuscript check-final`.

Expected: tests pass; the PDF is readable, uses embedded fonts, has no unresolved references or overfull boxes above policy, and has at most eight pages.

- [ ] **Step 6: Visually inspect every changed figure and manuscript page**

Render both standalone figures and all manuscript pages to PNG. Check text size, arrow direction, marker legend, colour/greyscale redundancy, float placement, title wrapping, collisions, clipping, and excess whitespace. If a visual defect is found, add the narrowest automated guard that can represent it before changing the source.

- [ ] **Step 7: Run full verification and commit the manuscript unit**

Run:

```bash
pytest -q
make -C manuscript check-final
git status --short
```

After checking the complete output and diff, stage the source, generated manifests, both figure PDFs, and `manuscript/build/main.pdf`, then commit with:

```bash
git commit -m "docs(paper): add visual hierarchy and multilingual evidence"
```
