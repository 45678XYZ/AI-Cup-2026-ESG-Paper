# Focused Multilingual Manuscript Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Merge the frozen multilingual replication into the paper branch and publish a focused, at-most-eight-page NTCIR manuscript that centers the confirmatory Chinese study and the repeated multilingual M1--M0 contrast pattern.

**Architecture:** Keep the existing modular LaTeX manuscript and frozen Chinese analysis pipeline. Add one small analysis module that selects the five fixed document-disjoint, lambda-zero arms from committed prediction-derived artifacts and renders a provenance-tracked LaTeX table. Extend the manuscript checker with focused-version claim and asset gates, then rewrite the paper section by section without changing frozen predictions, probabilities, splits, or result JSON.

**Tech Stack:** Git, Python 3, NumPy, pytest, JSON, LaTeX/acmart, BibTeX, Tectonic 0.15.0, pypdf, pdftotext/pdfinfo.

**Spec:** `docs/superpowers/specs/2026-08-25-focused-multilingual-manuscript-design.md`

## Global Constraints

- Work on `paper/ntcir19-manuscript`; merge `multilingual-replication` with a merge commit and do not rebase either history.
- Preserve checkpoint `54cb418` and approved design commit `8ca29cb`.
- Preserve the `/manuscript/build/main.pdf` exception in `.gitignore`; the final PDF must remain tracked.
- Do not modify frozen probabilities, predictions, result JSON, split manifests, run metadata, or redistributed datasets except by the already-approved branch merge.
- Do not add, restore, describe, or cite M7. Active manuscript source and repository changes for this revision must contain no M7 design.
- Chinese official weighted macro-F1 is the only confirmatory family. Chinese tuple exact-match is a separate pre-specified secondary family. C-wF1 and hF remain post-hoc descriptive checks.
- External comparisons are within-language M1--M0 contrasts. Do not rank raw scores across languages or claim a pure language effect.
- On ML-Promise, call the weighted score “AI CUP weights applied to ML-Promise (not official).”
- Use fixed `pdf_group`, `lambda=0` arms in the five-language table: Chinese frozen large backbone, English RoBERTa-large, and French/Japanese/Korean XLM-R-large.
- Report the external direction pattern as tuple positive in `32/32` arms and weighted macro-F1 positive in `14/32`, with the per-language counts available in prose.
- Correct the Chinese audit to 49 companies and remove the false claim that one report contains two companies.
- Remove the selected-best same-document Table 3 from the manuscript. Keep at most a short same-document sensitivity sentence.
- Structural-loss and backbone evidence is exploratory, compact, and subordinate to the main Chinese and multilingual results.
- The compiled PDF, including references, must contain at most eight pages.

---

### Task 1: Merge the frozen replication branch without disturbing the manuscript

**Files:**
- Modify during conflict resolution: `.gitignore`
- Possibly modify during conflict resolution: `README.md`
- Import unchanged from `multilingual-replication`: `analysis/`, `architecture_screen/`, `dataset/`, `docs/`, `paper/`, `runs_en/`, `runs_fr/`, `runs_ja/`, `runs_ko/`, `scripts/`, `splits_*`, `structural_arm/`, and their committed artifacts
- Preserve unchanged: `manuscript/**`

- [x] **Step 1: Confirm the pre-merge identities and clean state**

Run:

```bash
git status --short --branch
git rev-parse HEAD
git rev-parse multilingual-replication
git merge-base HEAD multilingual-replication
```

Expected: current branch is `paper/ntcir19-manuscript`, `HEAD` is `8ca29cb...`, the replication tip is `6f4914b...`, and the worktree is clean.

- [x] **Step 2: Merge with explicit history preservation**

Run:

```bash
git merge --no-ff multilingual-replication -m "merge: integrate multilingual replication evidence"
```

Expected: `.gitignore` requires manual resolution; no file under `manuscript/` is deleted or replaced.

- [x] **Step 3: Resolve only real content conflicts**

For `.gitignore`, retain all replication artifact exclusions and finish with:

```gitignore
/manuscript/build/*
!/manuscript/build/main.pdf
```

For `README.md`, retain both the manuscript build entry and the ML-Promise replication/data notices. Use `apply_patch`; do not take either whole side.

- [x] **Step 4: Verify merge protection gates**

Run:

```bash
git diff --check
git diff 8ca29cb -- manuscript
git grep -n -E 'M7|batch decoder' -- ':!docs/superpowers/plans/**' ':!docs/superpowers/specs/**'
git status --short
```

Expected: no manuscript diff immediately attributable to the merge, no M7 hit, and only resolved merge changes are staged.

- [x] **Step 5: Finish the merge commit if Git paused for conflicts**

Run:

```bash
git add .gitignore README.md
git commit -m "merge: integrate multilingual replication evidence"
```

Expected: a two-parent merge commit exists and `git status --short` is empty.

---

### Task 2: Make frozen summary reproducibility portable across environments

**Files:**
- Modify: `tests/test_multilingual_replication.py`
- Modify: `tests/test_multilingual_config.py`
- Modify: `scripts/queue_after_english.py`
- Modify early as a post-merge baseline prerequisite: `manuscript/check.py`, `tests/test_manuscript.py`

**Interfaces:**
- Add a test-only recursive comparator that keeps strings, keys, integers, and booleans exact while comparing floating leaves at `abs=1e-12`, `rel=1e-12`.
- Keep the queue-root unit test about non-collision; do not require a second developer’s `/tmp` worktree to exist.

- [x] **Step 1: Reproduce the two portability failures**

Run:

```bash
pytest tests/test_multilingual_replication.py tests/test_multilingual_config.py -q
```

Expected: exact JSON equality fails only at last-bit floating leaves for French/Japanese/Korean, and the queue test fails because a hard-coded external worktree is absent.

- [x] **Step 2: Add a failing unit test for the recursive comparator**

In `tests/test_multilingual_replication.py`, add a comparator fixture test that accepts `0.3` versus `0.30000000000000004` but rejects a changed key, string, list length, or a float difference of `1e-6`.

- [x] **Step 3: Implement the smallest test-only comparison helper**

Implement `_assert_json_close(actual, expected)` recursively. Use `pytest.approx` only at floating leaves; compare all non-floating values exactly. Replace `assert report == checked_in` with this helper.

- [x] **Step 4: Separate queue configuration from machine-local deployment**

Replace `test_queue_points_at_two_distinct_existing_worktrees` with a unit test that asserts `MULTI_ROOT != ENGLISH_ROOT` and verifies generated commands are rooted in their configured locations. Do not assert that another user’s `/tmp` directory exists.

- [x] **Step 5: Run focused and full regression tests**

Run:

```bash
pytest tests/test_multilingual_replication.py tests/test_multilingual_config.py -q
pytest -q
```

Expected: focused tests pass; the complete suite has no portability failures.

- [x] **Step 6: Commit the test portability fix**

```bash
git add tests/test_multilingual_replication.py tests/test_multilingual_config.py
git commit -m "test(multilingual): tolerate portable float rebuilds"
```

---

### Task 3: Generate the fixed-arm five-language table from committed evidence

**Files:**
- Create: `analysis/multilingual_table.py`
- Create: `tests/test_multilingual_table.py`
- Modify: `analysis/tables.py`
- Modify: `analysis/__main__.py`
- Generate: `tables/table6_multilingual.tex`
- Generate: `tables/table6_multilingual_caption.txt`
- Modify: `tables/manifest.json`

**Interfaces:**
- `fixed_arm_rows(chinese_summary: dict, audit: dict, repo_root: Path = REPO_ROOT) -> list[dict]`
- `render_multilingual_table(rows: list[dict]) -> str`
- `multilingual_table_inputs(repo_root: Path = REPO_ROOT) -> list[Path]`
- Extend `analysis.tables.TABLE_FILES`, `SOURCE_SCRIPTS`, `table_inputs`, `build_captions`, and `write_tables` for `table6_multilingual.tex`.

- [ ] **Step 1: Write selection and rendering tests**

Create tests that load the real committed summaries and pin the five selected rows after rounding:

```python
expected = [
    ("Chinese", 2000, 49, 0.1255, -0.0011, 0.0350),
    ("English", 400, 9, 0.2317, 0.0032, 0.0725),
    ("French", 400, 9, 0.2108, 0.0036, 0.0625),
    ("Japanese", 400, 19, 0.3125, -0.0131, 0.0708),
    ("Korean", 500, 32, 0.2273, -0.0192, 0.0293),
]
```

Also assert that English selects `roberta_large`, the other external languages select `xlm_roberta_large`, every row selects `lambda_0.0/pdf_group`, and the renderer contains no M2--M6 score column.

- [ ] **Step 2: Run the new tests and confirm failure**

Run:

```bash
pytest tests/test_multilingual_table.py -q
```

Expected: collection fails because `analysis.multilingual_table` does not exist.

- [ ] **Step 3: Implement fixed-arm extraction**

Use the already-computed Chinese `summaries["pdf_group"]["methods"]` supplied by `analysis.__main__`. Read external summaries from:

```text
runs_en/summary.json  -> summaries/roberta_large/lambda_0.0/pdf_group
runs_fr/summary.json  -> summaries/xlm_roberta_large/lambda_0.0/pdf_group
runs_ja/summary.json  -> summaries/xlm_roberta_large/lambda_0.0/pdf_group
runs_ko/summary.json  -> summaries/xlm_roberta_large/lambda_0.0/pdf_group
```

For every row, compute M0 invalidity and the M1--M0 deltas from the selected M0/M1 means. Never store the intended rounded values as the source of truth.

- [ ] **Step 4: Implement the compact tabular and caption**

Render six columns: language/corpus, rows, reports, M0 invalid percentage, `Delta wF1`, and `Delta tuple`. The caption must state fixed backbone choices, `lambda=0`, document-disjoint evaluation, three-seed means, and that ML-Promise weighted scores use AI CUP weights and are not official. Add a Korean footnote sentence about PDF-page inputs and zero gold `Misleading` examples.

- [ ] **Step 5: Integrate provenance into the one-command rebuild**

Add `table6_multilingual.tex` to the existing table registry. Its direct inputs are the six Chinese M0/M1 prediction files plus the four committed `runs_*/summary.json` files. Ensure the manifest hashes exactly these direct inputs and names `analysis/multilingual_table.py` as its source script.

- [ ] **Step 6: Run table tests and rebuild**

Run:

```bash
pytest tests/test_multilingual_table.py tests/test_analysis_tables.py -q
python -m analysis
git diff --check
```

Expected: tests pass; the table and caption are regenerated; `tables/manifest.json` contains a complete `table6_multilingual.tex` entry.

- [ ] **Step 7: Commit the generated table pipeline**

```bash
git add analysis/multilingual_table.py analysis/tables.py analysis/__main__.py \
  tests/test_multilingual_table.py tables/table6_multilingual.tex \
  tables/table6_multilingual_caption.txt tables/manifest.json
git commit -m "feat(analysis): generate five-language contrast table"
```

---

### Task 4: Extend manuscript policy gates for the focused version

**Files:**
- Modify: `manuscript/check.py`
- Modify: `tests/test_manuscript.py`
- Modify mechanically: `manuscript/sections/06_results.tex`

**Interfaces:**
- Treat `tables/table6_multilingual.tex` as a required canonical generated asset.
- Require the manuscript to include canonical Table 4 and canonical multilingual Table 6.
- Reject active-source M7 mentions and selected-best `table3_regimes.tex` inclusion.
- Check canonical generated assets for committed-tree drift against `HEAD`, not against the obsolete local `main` branch.
- Add focused-paper regression checks without banning legitimate repository result documents.

**Execution note:** The `git diff HEAD` asset-drift portion and its regression
test were completed during Task 2 because the corrected Table 1 otherwise made
the first post-merge full-suite verification fail. The remaining Table 6, M7,
and Table 3 policy work stays in this task.

- [x] **Step 1: Add failing policy tests**

Add tests that demonstrate all of the following:

- a decoy or missing Table 6 inclusion is rejected;
- an active `M7` mention is rejected;
- an active inclusion of `table3_regimes.tex` is rejected;
- a committed corrected Table 1 is accepted even though it differs from the old local `main` branch;
- an uncommitted edit to a canonical generated table is rejected relative to `HEAD`.

- [x] **Step 2: Run focused tests and confirm the new failures**

Run:

```bash
pytest tests/test_manuscript.py -q
```

Expected: the new policy tests fail against the old asset list, main-relative drift rule, and old Table 3 inclusion.

- [x] **Step 3: Implement minimal policy changes**

Add Table 6 to `REQUIRED_ASSETS` and `CANONICAL_FROZEN_ASSETS`, extend canonical-inclusion validation using the same containment rules as Table 4, add active-source M7 rejection, and reject an active Table 3 inclusion. Replace the `git diff main` asset check with `git diff --quiet HEAD -- <asset>` while retaining tracked-file and manifest-input hash checks. Update fixture manifest/table-name lists in the tests.

- [x] **Step 4: Mechanically switch the included result table**

In `06_results.tex`, remove the old `table3_regimes.tex` table environment and insert the canonical `table6_multilingual.tex` table environment with its generated caption. Replace only the immediately adjacent selected-best wording with a neutral fixed-arm introduction; the full Results rewrite remains Task 6.

- [x] **Step 5: Run policy tests and commit the foundation**

Run:

```bash
pytest tests/test_manuscript.py -q
```

Expected: every checker test and the real-manuscript source-policy test passes.

```bash
git add manuscript/check.py manuscript/sections/06_results.tex tests/test_manuscript.py
git commit -m "test(paper): enforce focused multilingual claims"
```

---

### Task 5: Update data, methods, evaluation hierarchy, and references

**Files:**
- Modify: `manuscript/references.bib`
- Modify: `manuscript/sections/02_related_work.tex`
- Modify: `manuscript/sections/03_task_and_data.tex`
- Modify: `manuscript/sections/04_methods.tex`
- Modify: `manuscript/sections/05_experiments.tex`
- Modify as generated by Task 3: `tables/table1_dataset.tex`, `tables/audit.json`, `tables/manifest.json`

- [x] **Step 1: Add the ML-Promise primary citation**

Add Seki et al., “ML-Promise: A Multilingual Dataset for Corporate Promise Verification,” EMNLP 2025, ACL Anthology `2025.emnlp-main.1028`. Cite it in Related Work and Task/Data; retain the PromiseEval citation and distinguish the multilingual dataset from the shared-task overview.

- [x] **Step 2: Correct the Chinese audit prose**

State 2,000 development rows, 49 reports, and 49 companies. Remove the statement that one report covers two companies. Retain the rare-class warning (`Misleading`, n=2) and document-level split rationale.

- [x] **Step 3: Add the external corpus protocol**

Describe English/French/Japanese as paragraph inputs and Korean as first-384-token PDF-page inputs. Record rows/reports, fixed primary backbones, four-backbone robustness matrix, `lambda in {0, 0.3}`, three seeds, five rotations, and document-disjoint `pdf_group` evaluation.

- [x] **Step 4: Declare the statistical hierarchy before Results**

State explicitly:

1. confirmatory Chinese official wF1, five contrasts, Holm;
2. pre-specified secondary Chinese tuple accuracy, separate five-contrast Holm family;
3. pre-registered external within-arm replication directions and English cluster bootstrap;
4. exploratory C-wF1, hF, structural loss, and alternate-backbone screens.

Do not elevate C-wF1/hF p-values and do not pool external descriptive directions into a new post-hoc significance test.

- [x] **Step 5: Rebuild and run section-level checks**

Run:

```bash
python -m analysis
pytest tests/test_analysis_audit.py tests/test_analysis_tables.py tests/test_manuscript.py -q
```

Expected: generated audit/table data say 49; generic policy tests pass; only Results/abstract focused-content tests may remain red.

- [x] **Step 6: Commit the protocol and citation revision**

```bash
git add manuscript/references.bib manuscript/sections/02_related_work.tex \
  manuscript/sections/03_task_and_data.tex manuscript/sections/04_methods.tex \
  manuscript/sections/05_experiments.tex tables/table1_dataset.tex \
  tables/audit.json tables/manifest.json
git commit -m "docs(paper): add multilingual replication protocol"
```

---

### Task 6: Rewrite the abstract, introduction, and Results around the focused evidence

**Files:**
- Modify: `manuscript/main.tex`
- Modify: `manuscript/sections/01_introduction.tex`
- Modify: `manuscript/sections/06_results.tex`

- [ ] **Step 1: Add failing focused-content regression tests**

In `tests/test_manuscript.py`, assert that the real manuscript source states 49 Chinese companies; contains `32/32`, `14/32`, and “AI CUP weights applied to ML-Promise”; includes the pre-specified tuple result in the abstract without headlining `p_Holm=.025`; and contains no raw cross-language winner claim. Run the new tests and confirm they fail for missing focused prose, not for checker infrastructure.

- [ ] **Step 2: Rewrite the title and abstract**

Lead with the mismatch between hierarchical output quality and field-wise scoring. Preserve the Chinese confirmatory null (`M1--M0` official wF1 approximately `-0.001`, `p_Holm=1.000`), report the pre-specified tuple gain (`+0.035`, `p_Holm=.001`), and summarize external repetition (`32/32` tuple-positive versus `14/32` weighted-positive arms). Do not headline post-hoc `p_Holm=.025`.

- [ ] **Step 3: Rewrite the introduction and contributions**

Frame three contributions only: controlled Chinese decision comparison, pre-specified whole-tuple evidence, and multilingual within-arm replication. Mention structural/backbone experiments only as robustness checks, not as additional main research questions.

- [ ] **Step 4: Finish the generated multilingual table integration**

Review the mechanically inserted `\input{../tables/table6_multilingual.tex}` environment in its final Results context. Remove all remaining “best calibrated projection/best decoder” selection language. Keep one short same-document sensitivity sentence only if it fits the page budget.

- [ ] **Step 5: Order Results by inferential priority**

Present:

1. Chinese official wF1 family, all five null;
2. Chinese tuple family, including M1--M0 `+0.035` and adverse M4--M1 `-0.006`;
3. fixed-arm five-language table;
4. all external arms: tuple `32/32`, weighted `14/32` (EN 7/8, FR 5/8, JA 0/8, KO 2/8);
5. one compact exploratory robustness paragraph: Chinese invalidity `12.55% -> 5.18%`, external structural-loss invalidity reduction `16/16`, mixed downstream effects, and non-conclusive backbone screens.

- [ ] **Step 6: Run focused manuscript tests**

Run:

```bash
pytest tests/test_manuscript.py -q
git grep -n -E 'M7|batch decoder' -- manuscript analysis/multilingual_table.py
```

Expected: claim tests for the abstract, table inclusion, 49 companies, and repeated direction counts pass; the M7 grep has no output.

- [ ] **Step 7: Commit the focused core rewrite**

```bash
git add manuscript/main.tex manuscript/sections/01_introduction.tex \
  manuscript/sections/06_results.tex tests/test_manuscript.py
git commit -m "docs(paper): center multilingual hierarchy replication"
```

---

### Task 7: Rewrite Discussion and Conclusion with calibrated scope

**Files:**
- Modify: `manuscript/sections/07_discussion.tex`

- [ ] **Step 1: Interpret the repeated pattern, not raw language scores**

Explain that projection always enforces validity and repeatedly improves exact whole-tuple correctness in the executed external arms, while field-wise weighted macro-F1 has inconsistent direction. State that this supports metric--structure mismatch, not language ranking or universal model superiority.

- [ ] **Step 2: Preserve the decoder caution**

Retain the adverse Chinese M4--M1 tuple result and explain that searching all legal states can improve one field while disrupting a previously correct whole tuple. Do not generalize this single confirmatory-dataset result to every decoder.

- [ ] **Step 3: State limitations compactly**

Include corpus/backbone confounding, small report-cluster counts, Korean page-level input and zero `Misleading` support, AI CUP weights being unofficial on ML-Promise, descriptive external directions, and exploratory structural/backbone evidence. State that the Korean committed predictions can be rescored from a fresh clone although the local extracted page-text training input is not redistributed.

- [ ] **Step 4: End with the licensed conclusion**

Conclude that field-wise weighted macro-F1 did not consistently reflect the validity and exact-match gains observed after hierarchical projection; recommend reporting both field-wise and structured whole-output measures.

- [ ] **Step 5: Run prose and policy checks**

Run:

```bash
pytest tests/test_manuscript.py -q
PYTHONPATH=. python manuscript/check.py --root manuscript --repo-root .
git diff --check
```

Expected: all manuscript tests and draft source checks pass.

- [ ] **Step 6: Commit the calibrated interpretation**

```bash
git add manuscript/sections/07_discussion.tex
git commit -m "docs(paper): calibrate multilingual discussion"
```

---

### Task 8: Rebuild all derived manuscript assets and prove provenance

**Files:**
- Regenerate: `tables/audit.json`
- Regenerate: `tables/table1_dataset.tex`
- Regenerate: `tables/table1_dataset_caption.txt`
- Regenerate: `tables/table2_main.tex`
- Regenerate: `tables/table2_main_caption.txt`
- Regenerate: `tables/table3_regimes.tex` (repository artifact only; not included in manuscript)
- Regenerate: `tables/table3_regimes_caption.txt`
- Regenerate: `tables/table4_contrasts.tex`
- Regenerate: `tables/table4_contrasts_caption.txt`
- Regenerate: `tables/table5_metrics.tex`
- Regenerate: `tables/table5_metrics_caption.txt`
- Regenerate: `tables/table6_multilingual.tex`
- Regenerate: `tables/table6_multilingual_caption.txt`
- Regenerate: `tables/manifest.json`

- [ ] **Step 1: Run the canonical rebuild**

Run:

```bash
python -m analysis
```

Expected: all six table fragments and captions are written from committed inputs; no raw frozen artifact is modified.

- [ ] **Step 2: Audit the generated diff**

Run:

```bash
git diff --check
git status --short
git diff -- predictions probs results splits runs_en runs_fr runs_ja runs_ko \
  probs_architecture probs_lambda_sweep probs_structural structural_arm
```

Expected: no frozen prediction/probability/result/split/summary diff. Intended generated changes are limited to `tables/`.

- [ ] **Step 3: Validate manifest coverage**

Run:

```bash
pytest tests/test_analysis_tables.py tests/test_artifacts.py tests/test_manuscript.py -q
PYTHONPATH=. python manuscript/check.py --root manuscript --repo-root .
```

Expected: every required table has a complete, repository-relative, checksum-verified manifest entry.

- [ ] **Step 4: Commit regenerated evidence**

```bash
git add tables
git commit -m "chore(paper): refresh focused manuscript tables"
```

---

### Task 9: Perform final statistical, build, and visual verification

**Files:**
- Regenerate and commit: `manuscript/build/main.pdf`
- Do not commit: `manuscript/build/main.log` and other build products

- [ ] **Step 1: Run the complete Python suite**

Run:

```bash
pytest -q
```

Expected: all tests pass, with only explicitly registered skips.

- [ ] **Step 2: Validate multilingual artifacts**

Run:

```bash
python -m paper.validate --all --corpus mlpromise_fr
python -m paper.validate --all --corpus mlpromise_ja
pytest tests/test_multilingual_replication.py -q
```

Expected: French and Japanese each report 360 clean artifacts. The summary-rebuild test also proves Korean prediction rescoring. Document that the full Korean validator additionally needs the intentionally undistributed `local_data/mlpromise_korean_pages.json`; do not fabricate or fetch it.

- [ ] **Step 3: Build and run final manuscript checks**

Run:

```bash
make -C manuscript check-final
pdfinfo manuscript/build/main.pdf | grep '^Pages:'
pdftotext manuscript/build/main.pdf - | rg '32/32|14/32|49|AI CUP weights applied'
```

Expected: final check exits zero, the PDF is at most eight pages, and the central focused claims survive compilation.

- [ ] **Step 4: Inspect every rendered page visually**

Render the PDF to a new `mktemp -d` directory with `pdftoppm -png -r 144`, then inspect every page with the image viewer. Check table legibility, clipping, font substitution, whitespace, float placement, bibliography breaks, and that no table or reference line crosses a page boundary awkwardly.

- [ ] **Step 5: Verify tracking and exclusion gates**

Run:

```bash
git check-ignore manuscript/build/main.pdf
git ls-files --error-unmatch manuscript/build/main.pdf
git grep -n -E 'M7|batch decoder' -- manuscript analysis/multilingual_table.py
git diff --check
git status --short
```

Expected: `git check-ignore` returns nonzero, `git ls-files` succeeds, M7 grep has no output, and only the intended final PDF/source adjustments remain.

- [ ] **Step 6: Commit the latest manuscript**

```bash
git add manuscript manuscript/build/main.pdf
git commit -m "docs(paper): publish focused multilingual revision"
```

- [ ] **Step 7: Verify the committed endpoint**

Run:

```bash
git status --short --branch
git log --oneline --decorate -12
make -C manuscript check-final
```

Expected: clean `paper/ntcir19-manuscript`, final checks still pass from the committed tree, and the history visibly preserves the checkpoint, approved design, merge commit, focused implementation commits, and final PDF.

---

## Final Acceptance Checklist

- [ ] `multilingual-replication` is merged with both histories preserved.
- [ ] `manuscript/build/main.pdf` is tracked and not ignored.
- [ ] The manuscript is focused on Chinese confirmatory evidence plus multilingual replication.
- [ ] Table 6 is generated from committed evidence and covered by checksum provenance.
- [ ] Chinese Companies is 49; no report is said to cover two companies.
- [ ] The abstract reports the pre-specified tuple result and does not headline the post-hoc C-wF1 p-value.
- [ ] External results say tuple `32/32` and weighted `14/32`, without raw cross-language ranking.
- [ ] ML-Promise weighted scores are explicitly labelled unofficial AI CUP-weighted scores.
- [ ] Korean input/support/reproducibility limitations are disclosed.
- [ ] The selected-best same-document Table 3 is absent from the manuscript.
- [ ] Structural loss and backbone screens remain compact and exploratory.
- [ ] No M7 design or result appears in the manuscript revision.
- [ ] Full tests and final manuscript checks pass.
- [ ] The visually inspected PDF is at most eight pages including references.
- [ ] The final branch is clean and the latest PDF is committed.
