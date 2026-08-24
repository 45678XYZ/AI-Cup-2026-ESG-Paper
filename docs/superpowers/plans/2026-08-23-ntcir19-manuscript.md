# NTCIR-19 AI CUP ESG Manuscript Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build and validate an English, submission-ready, at-most-eight-page NTCIR-19 AI CUP ESG manuscript from the repository's committed RTX 3090 study artifacts.

**Architecture:** The manuscript is a modular ACM/NTCIR LaTeX document under `manuscript/`. It inputs the existing generated table fragments and committed hierarchy figure rather than copying their numeric bodies, while a Python checker enforces claim, provenance, metadata, and page-budget policies. Scientific prose is split by section so evidence-heavy Results can be reviewed independently from framing and related work.

**Tech Stack:** LaTeX with the official NTCIR-19 `acmart` sample settings, BibTeX with `ACM-Reference-Format`, Tectonic 0.15.0, Python 3, pypdf 6.13.1, pytest.

**Spec:** `docs/superpowers/specs/2026-08-23-ntcir19-manuscript-design.md`

## Global Constraints

- Use only committed, frozen RTX 3090 artifacts indexed by `run_manifest.json`.
- Manuscript sources must contain neither of the two prohibited GPU model strings defined in `manuscript/check.py::PROHIBITED_HARDWARE`.
- Printed numerical claims come from `tables/*.tex` or `tables/*_caption.txt`; `tables/findings.md` controls statistical verdicts.
- Use `no detectable difference` for unresolved contrasts; do not claim equivalence.
- Path-constrained weighted macro-F1 and hierarchical F1 must be disclosed as post hoc.
- Significance decisions use Holm-adjusted p-values; printed 95% intervals are uncorrected.
- Report all five contrasts in each of the four Table 4 metric families.
- State that the path-constrained M1-M0 finding does not survive Holm correction under same-document evaluation.
- Make no improvement or significance claim for `Misleading` (n=2).
- Do not modify frozen probabilities, predictions, results, splits, manifests, or generated table bodies.
- The four team-supplied student author records are active in `metadata.tex` as
  of 2026-08-24. Draft checks still warn about any explicit placeholder (or
  author-metadata TODO marker), and final checks reject one even if an
  email-like field is present.
- The compiled PDF, including references, must contain at most eight pages.

---

## File map

### New manuscript files

- `manuscript/main.tex` — official NTCIR/ACM shell, abstract, section includes, bibliography.
- `manuscript/metadata.tex` — four team-supplied author, affiliation, location,
  and email records; no team name is invented.
- `manuscript/references.bib` — verified primary-source bibliography.
- `manuscript/Makefile` — `build`, `check`, `check-final`, and `clean` targets.
- `manuscript/README.md` — build commands, source authority, and final metadata rule.
- `manuscript/check.py` — source-policy, asset, page-count, citation-log, and font checks.
- `manuscript/sections/01_introduction.tex` — problem, gap, evidence preview, contributions.
- `manuscript/sections/02_related_work.tex` — hierarchical classification and path-constrained metrics.
- `manuscript/sections/03_task_and_data.tex` — ESG task, hierarchy, legal states, data audit.
- `manuscript/sections/04_methods.tex` — shared model, calibration, projection, valid-state decoder.
- `manuscript/sections/05_experiments.tex` — cross-fitting protocols, metrics, bootstrap, Holm families.
- `manuscript/sections/06_results.tex` — Tables 2--4, central contrast, adverse result, protocol gap.
- `manuscript/sections/07_discussion.tex` — interpretation, limitations, conclusion.
- `tests/test_manuscript.py` — automated manuscript-policy and PDF checks.

### Existing files read without modification

- `tables/table1_dataset.tex`, `tables/table2_main.tex`, `tables/table3_regimes.tex`, `tables/table4_contrasts.tex`.
- `tables/*_caption.txt`, `tables/findings.md`, `tables/manifest.json`.
- `figures/figure1_hierarchy.pdf`.
- `docs/study_report.md`, `docs/paper_plan.md`, `docs/related_work_citations.md`.
- `run_manifest.json`.

### Existing file modified

- `.gitignore` — ignore `/manuscript/build/` only.

---

### Task 1: Manuscript policy checker

**Files:**
- Create: `manuscript/check.py`
- Create: `tests/test_manuscript.py`

**Interfaces:**
- Consumes: manuscript root `Path`, optional compiled PDF `Path`, and `final: bool`.
- Produces: `source_errors(root: Path, final: bool = False) -> list[str]`, `source_warnings(root: Path) -> list[str]`, `asset_errors(repo_root: Path) -> list[str]`, `pdf_errors(pdf: Path, max_pages: int = 8) -> list[str]`, `font_errors(pdf: Path) -> list[str]`, `log_errors(log: Path) -> list[str]`, and a CLI exit status.

- [ ] **Step 1: Write policy tests**

Create fixtures with small `.tex` files and add these tests:

```python
import re
from pathlib import Path

from pypdf import PdfWriter

from paper.data import REPO_ROOT
from manuscript.check import (
    asset_errors,
    font_errors,
    log_errors,
    pdf_errors,
    source_errors,
    source_text,
    source_warnings,
)


def write_minimal(root: Path, body: str, metadata: str = "") -> None:
    root.mkdir()
    (root / "main.tex").write_text(body, encoding="utf-8")
    (root / "metadata.tex").write_text(metadata, encoding="utf-8")


def test_draft_rejects_prohibited_hardware_and_unqualified_null_claim(tmp_path):
    write_minimal(
        tmp_path / "m",
        "\\input{../tables/table4_contrasts.tex}\n"
        "Path-constrained wF1 and hierarchical F1 are post hoc. "
        "There was no difference on an L" + "40S run.",
    )
    errors = source_errors(tmp_path / "m")
    assert any("prohibited hardware" in error for error in errors)
    assert any("no detectable difference" in error for error in errors)


def test_draft_rejects_an_equivalence_claim(tmp_path):
    write_minimal(
        tmp_path / "m",
        "\\input{../tables/table4_contrasts.tex}\n"
        "Path-constrained wF1 and hierarchical F1 are post hoc. "
        "The two decision rules are equivalent.",
    )
    assert any("equivalence" in error for error in source_errors(tmp_path / "m"))


def test_draft_requires_full_table4_and_post_hoc_disclosure(tmp_path):
    write_minimal(tmp_path / "m", "Path-constrained wF1 and hierarchical F1.")
    errors = source_errors(tmp_path / "m")
    assert any("table4_contrasts.tex" in error for error in errors)
    assert any("post hoc" in error for error in errors)


def test_author_metadata_is_draft_warning_but_final_error(tmp_path):
    write_minimal(
        tmp_path / "m",
        "\\input{../tables/table4_contrasts.tex}\n"
        "Path-constrained wF1 and hierarchical F1 were adopted post hoc.",
    )
    assert not any("author metadata" in error for error in source_errors(tmp_path / "m"))
    assert any("author metadata" in warning for warning in source_warnings(tmp_path / "m"))
    assert any("author metadata" in error
               for error in source_errors(tmp_path / "m", final=True))


def test_pdf_page_limit(tmp_path):
    path = tmp_path / "nine-pages.pdf"
    writer = PdfWriter()
    for _ in range(9):
        writer.add_blank_page(width=612, height=792)
    with path.open("wb") as stream:
        writer.write(stream)
    assert any("9 pages" in error for error in pdf_errors(path, max_pages=8))


def test_log_rejects_unresolved_references(tmp_path):
    log = tmp_path / "main.log"
    log.write_text("LaTeX Warning: There were undefined references.", encoding="utf-8")
    assert log_errors(log)


def test_asset_check_names_missing_generated_files(tmp_path):
    errors = asset_errors(tmp_path)
    assert any("table4_contrasts.tex" in error for error in errors)
```

- [ ] **Step 2: Run the focused tests and confirm failure**

Run: `pytest tests/test_manuscript.py -q`

Expected: collection fails because `manuscript.check` does not exist.

- [ ] **Step 3: Implement the checker**

Create `manuscript/check.py` with:

```python
from __future__ import annotations

import argparse
import re
from pathlib import Path

from pypdf import PdfReader

PROHIBITED_HARDWARE = ("L" + "40", "L" + "40S")
SOURCE_GLOBS = ("*.tex", "sections/*.tex", "*.bib")
REQUIRED_ASSETS = (
    "tables/table1_dataset.tex",
    "tables/table2_main.tex",
    "tables/table3_regimes.tex",
    "tables/table4_contrasts.tex",
    "figures/figure1_hierarchy.pdf",
)


def source_text(root: Path) -> str:
    files = sorted({path for pattern in SOURCE_GLOBS for path in root.glob(pattern)})
    return "\n".join(path.read_text(encoding="utf-8") for path in files)


def source_errors(root: Path, final: bool = False) -> list[str]:
    text = source_text(root)
    lowered = text.lower()
    errors: list[str] = []
    if any(marker.lower() in lowered for marker in PROHIBITED_HARDWARE):
        errors.append("manuscript contains a prohibited hardware reference")
    if re.search(r"\bno difference\b", lowered):
        errors.append("replace 'no difference' with 'no detectable difference'")
    if re.search(r"\b(?:is|are|were) equivalent\b", lowered):
        errors.append("an equivalence claim requires an equivalence design")
    if "../tables/table4_contrasts.tex" not in text:
        errors.append("the full generated table4_contrasts.tex is not included")
    metrics_present = "path-constrained" in lowered or "hierarchical f1" in lowered
    if metrics_present and "post hoc" not in lowered:
        errors.append("path-constrained wF1 and hierarchical F1 require a post hoc disclosure")
    metadata = (root / "metadata.tex").read_text(encoding="utf-8")
    has_author = bool(re.search(r"\\author\{[^}]+\}", metadata))
    has_email = bool(re.search(r"\\email\{[^}]+@[^}]+\}", metadata))
    if final and not (has_author and has_email):
        errors.append("author metadata is required for final submission")
    return errors


def source_warnings(root: Path) -> list[str]:
    metadata = (root / "metadata.tex").read_text(encoding="utf-8")
    if re.search(r"\\author\{[^}]+\}", metadata):
        return []
    return ["author metadata is intentionally absent from this draft"]


def asset_errors(repo_root: Path) -> list[str]:
    return [f"required generated asset is missing: {path}"
            for path in REQUIRED_ASSETS if not (repo_root / path).exists()]


def pdf_errors(pdf: Path, max_pages: int = 8) -> list[str]:
    if not pdf.exists():
        return [f"compiled PDF is missing: {pdf}"]
    pages = len(PdfReader(pdf).pages)
    return [] if pages <= max_pages else [f"compiled PDF has {pages} pages; limit is {max_pages}"]


def font_errors(pdf: Path) -> list[str]:
    missing: set[str] = set()
    for page in PdfReader(pdf).pages:
        resources = page.get("/Resources", {}).get_object()
        fonts = resources.get("/Font", {}).get_object()
        for font_ref in fonts.values():
            font = font_ref.get_object()
            if font.get("/Subtype") == "/Type3":
                continue
            descendants = font.get("/DescendantFonts", [])
            concrete = [item.get_object() for item in descendants] or [font]
            for item in concrete:
                descriptor_ref = item.get("/FontDescriptor")
                descriptor = descriptor_ref.get_object() if descriptor_ref else None
                embedded = descriptor and any(
                    key in descriptor for key in ("/FontFile", "/FontFile2", "/FontFile3")
                )
                if not embedded:
                    missing.add(str(item.get("/BaseFont", "unnamed font")))
    return [f"font is not embedded: {name}" for name in sorted(missing)]


def log_errors(log: Path) -> list[str]:
    text = log.read_text(encoding="utf-8", errors="replace").lower()
    errors = []
    for pattern in (
        r"there were undefined references",
        r"(?:latex|package natbib) warning: citation .* undefined",
    ):
        if re.search(pattern, text):
            errors.append(f"build log matches unresolved-reference pattern: {pattern}")
    for width in re.findall(r"overfull \\hbox \(([\d.]+)pt too wide\)", text):
        if float(width) > 2.0:
            errors.append(f"build log has an overfull box {width}pt too wide")
    return errors


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path(__file__).parent)
    parser.add_argument("--pdf", type=Path)
    parser.add_argument("--log", type=Path)
    parser.add_argument("--repo-root", type=Path, default=Path(__file__).parents[1])
    parser.add_argument("--final", action="store_true")
    args = parser.parse_args()
    errors = source_errors(args.root, final=args.final)
    errors.extend(asset_errors(args.repo_root))
    if args.pdf:
        errors.extend(pdf_errors(args.pdf))
        errors.extend(font_errors(args.pdf))
    if args.log and args.log.exists():
        errors.extend(log_errors(args.log))
    for error in errors:
        print(f"ERROR: {error}")
    for warning in source_warnings(args.root):
        print(f"WARNING: {warning}")
    return bool(errors)


if __name__ == "__main__":
    raise SystemExit(main())
```

- [ ] **Step 4: Run focused tests**

Run: `pytest tests/test_manuscript.py -q`

Expected: all Task 1 tests pass.

- [ ] **Step 5: Commit the checker**

```bash
git add manuscript/check.py tests/test_manuscript.py
git commit -m "test(paper): enforce manuscript claim policies"
```

---

### Task 2: Official NTCIR LaTeX scaffold and repeatable build

**Files:**
- Create: `manuscript/main.tex`
- Create: `manuscript/metadata.tex`
- Create: `manuscript/Makefile`
- Create: `manuscript/README.md`
- Create: `manuscript/references.bib`
- Create: all seven `manuscript/sections/*.tex` files
- Modify: `.gitignore`
- Modify: `tests/test_manuscript.py`

**Interfaces:**
- Consumes: official NTCIR sample settings and existing generated assets.
- Produces: `make -C manuscript build` at `manuscript/build/main.pdf` and `make -C manuscript check`.

- [ ] **Step 1: Add scaffold-structure tests**

Add assertions that `main.tex` contains the official class and top-matter settings, includes all seven section files, uses ACM bibliography formatting, and references all four generated tables plus Figure 1.

```python
def test_main_uses_official_ntcir_shell():
    text = (REPO_ROOT / "manuscript" / "main.tex").read_text(encoding="utf-8")
    assert r"\documentclass[sigconf,article]{acmart}" in text
    assert r"\settopmatter{printacmref=false}" in text
    assert r"\bibliographystyle{ACM-Reference-Format}" in text
    for index in range(1, 8):
        assert f"sections/0{index}_" in text


def test_manuscript_inputs_generated_assets():
    text = source_text(REPO_ROOT / "manuscript")
    for name in ("table1_dataset.tex", "table2_main.tex",
                 "table3_regimes.tex", "table4_contrasts.tex"):
        assert f"../tables/{name}" in text
    assert "../figures/figure1_hierarchy.pdf" in text


def test_compiled_shell_embeds_fonts():
    pdf = REPO_ROOT / "manuscript" / "build" / "main.pdf"
    assert not font_errors(pdf)
```

- [ ] **Step 2: Run the focused tests and confirm failure**

Run: `pytest tests/test_manuscript.py -q`

Expected: the new tests fail because `main.tex` and section files are absent.

- [ ] **Step 3: Create the official shell**

Use the NTCIR-19 sample's exact class and publication settings:

```tex
\documentclass[sigconf,article]{acmart}
\settopmatter{printacmref=false}
\renewcommand\footnotetextcopyrightpermission[1]{}
\pagenumbering{gobble}
\usepackage{booktabs}
\usepackage{amsmath}
\usepackage{graphicx}
\usepackage{microtype}

\begin{document}
\title{When Field-Wise Metrics Miss Hierarchical Consistency:
  A Controlled Study on the AI CUP ESG Promise Verification Dataset}
\input{metadata}
\begin{abstract}
ESG promise verification requires jointly predicting whether a paragraph
contains a promise, its verification timeline, whether it supplies evidence,
and the quality of that evidence. These outputs obey hard dependencies, yet
the task metric scores them field by field. We conduct a controlled study of
seven decision rules using identical cross-fitted probabilities. Independent
argmax produces invalid tuples on 12.6\% of document-disjoint predictions;
deterministic projection removes them. The official weighted macro-F1 shows no
detectable projection effect, whereas a path-constrained variant that changes
only the treatment of ancestor-unsupported predictions yields a positive
Holm-corrected contrast. Pre-specified tuple accuracy corroborates this result,
while valid-state decoding is worse than projection on that metric. These
findings show that, for this dataset and backbone, field-wise scoring can
understate the utility of validity-preserving decisions.
\end{abstract}
\keywords{ESG, promise verification, hierarchical classification,
  constrained decoding, evaluation metrics}
\maketitle
\pagestyle{plain}
\input{sections/01_introduction}
\input{sections/02_related_work}
\input{sections/03_task_and_data}
\input{sections/04_methods}
\input{sections/05_experiments}
\input{sections/06_results}
\input{sections/07_discussion}
\bibliographystyle{ACM-Reference-Format}
\bibliography{references}
\end{document}
```

`metadata.tex` contains the four author records supplied by the team on
2026-08-24. Each section file contains its final `\section` heading and one
factual orientation sentence so the build contains no fake content.

- [ ] **Step 4: Add build commands and ignore only scratch output**

`manuscript/Makefile`:

```make
TECTONIC ?= tectonic
PYTHON ?= python
BUILD_DIR := build
PDF := $(BUILD_DIR)/main.pdf
LOG := $(BUILD_DIR)/main.log

.PHONY: build check check-final clean

build:
	mkdir -p $(BUILD_DIR)
	$(TECTONIC) main.tex --outdir $(BUILD_DIR) --keep-logs

check: build
	$(PYTHON) check.py --root . --pdf $(PDF) --log $(LOG)

check-final: build
	$(PYTHON) check.py --root . --pdf $(PDF) --log $(LOG) --final

clean:
	rm -rf $(BUILD_DIR)
```

Add `/manuscript/build/` to `.gitignore`. The Makefile's narrowly resolved build directory is the only recursive deletion target.

- [ ] **Step 5: Build the shell**

Run: `make -C manuscript build`

Expected: Tectonic produces `manuscript/build/main.pdf` using the official `acmart` bundle. Missing author data may warn but must not stop the draft build.

- [ ] **Step 6: Run scaffold and existing tests**

Run: `pytest tests/test_manuscript.py -q`

Run: `pytest -q`

Expected: manuscript tests and the existing suite pass.

- [ ] **Step 7: Commit the scaffold**

```bash
git add .gitignore manuscript tests/test_manuscript.py
git commit -m "docs(paper): scaffold NTCIR-19 LaTeX manuscript"
```

---

### Task 3: Task, model, methods, and experimental design

**Files:**
- Modify: `manuscript/sections/03_task_and_data.tex`
- Modify: `manuscript/sections/04_methods.tex`
- Modify: `manuscript/sections/05_experiments.tex`
- Modify: `tests/test_manuscript.py`

**Interfaces:**
- Consumes: `docs/paper_plan.md` Sections 2--4, `docs/study_report.md` Parts II and IV, Figure 1, Table 1.
- Produces: the reproducible technical core referred to by Results.

- [ ] **Step 1: Add technical-content guard tests**

```python
def test_technical_core_records_frozen_design_choices():
    text = source_text(REPO_ROOT / "manuscript")
    for required in (
        "17 legal", "120", "hfl/chinese-roberta-wwm-ext-large",
        "BERT architecture", "alpha_t=1", "10,000",
        "PDF-cluster", "49 source", "three seeds",
    ):
        assert required.lower() in text.lower()
    assert "post hoc" in text.lower()
```

- [ ] **Step 2: Run the focused test and confirm failure**

Run: `pytest tests/test_manuscript.py::test_technical_core_records_frozen_design_choices -q`

Expected: failure lists the design facts not yet written.

- [ ] **Step 3: Write Task and Data**

Write compact academic prose that:

- defines PS, VT, ES, and EQ and the implications `PS=No` and `ES=No`;
- derives `1 + 4(1+3)=17` legal tuples from 120 combinations;
- states that the study uses the AI CUP VeriPromiseESG development data;
- inputs Table 1 and explains 2,000 labeled paragraphs, 49 reports, 50 companies, 15 observed legal states, and `Misleading` n=2;
- states that all 49 development reports also occur in the unlabeled competition test data and that document-disjoint is company-disjoint here;
- includes Figure 1 at native width in a two-column `figure*`.

Do not infer labels for the unlabeled competition test split.

- [ ] **Step 4: Write Methods**

Specify:

- `hfl/chinese-roberta-wwm-ext-large` as a 24-layer, hidden-size-1024 BERT architecture pretrained with a RoBERTa-style recipe;
- one shared encoder and four independent linear heads;
- class-bias decision calibration `z_{t,c}(x)=\log p_{t,c}(x)+b_{t,c}`;
- deterministic coordinate ascent over [-3,3] in steps of 0.05, at most 20 passes;
- global versus hierarchy-conditioned bias estimation, including the three structurally unidentifiable child `N/A` biases fixed at zero;
- complete bidirectional projection;
- joint search over 17 states with `\alpha_t=1`;
- M0--M6 as alternatives over identical base probabilities, not sequential stages.

- [ ] **Step 5: Write Experimental Setup**

Specify:

- five-way rotating train/calibration/test cross-fitting;
- document-disjoint as primary and same-document as a second estimand;
- seeds 42, 123, and 456;
- the official weighted macro-F1 weights 0.20/0.15/0.30/0.35;
- tuple accuracy as pre-specified and C-wF1/hF as post hoc;
- paired PDF-cluster bootstrap with 49 source-report clusters, 10,000 resamples, and seed 20260814;
- five pre-specified contrasts corrected as a separate Holm family for each metric;
- bracketed intervals as uncorrected percentile intervals and `p_{\mathrm{Holm}}<.05` as the decision rule.

- [ ] **Step 6: Run checks and build**

Run: `pytest tests/test_manuscript.py -q`

Run: `make -C manuscript build`

Expected: both succeed; inspect the log for overfull technical equations.

- [ ] **Step 7: Commit the technical core**

```bash
git add manuscript/sections tests/test_manuscript.py
git commit -m "docs(paper): write task methods and experiment design"
```

---

### Task 4: Verified bibliography and related work

**Files:**
- Modify: `manuscript/references.bib`
- Modify: `manuscript/sections/02_related_work.tex`
- Modify: `tests/test_manuscript.py`

**Interfaces:**
- Consumes: `docs/related_work_citations.md` and primary publisher/proceedings records.
- Produces: cited context for ESG promise verification, climate-target extraction, BERT, Chinese whole-word masking, hierarchical metrics, and multiplicity correction.

- [ ] **Step 1: Add bibliography integrity tests**

```python
def test_required_primary_references_are_present_and_cited():
    root = REPO_ROOT / "manuscript"
    bib = (root / "references.bib").read_text(encoding="utf-8")
    tex = source_text(root)
    for key in (
        "devlin2019bert", "cui2021wwm", "yu2022constrained",
        "ji2023hierarchical", "plaud2024revisiting", "holm1979",
        "chen2025promiseeval", "schimanski2023climatebert",
        "aicup2026guidelines",
    ):
        assert f"{{{key}," in bib
        assert key in tex
```

- [ ] **Step 2: Run the focused test and confirm failure**

Run: `pytest tests/test_manuscript.py::test_required_primary_references_are_present_and_cited -q`

Expected: failure lists missing keys.

- [ ] **Step 3: Verify and add primary-source BibTeX**

Verify entries against these primary records before adding them:

- Devlin et al., BERT, ACL Anthology `N19-1423`.
- Cui et al., *Pre-Training with Whole Word Masking for Chinese BERT*, DOI `10.1109/TASLP.2021.3076303`.
- Yu, Shen, and Mao, SIGIR 2022, DOI `10.1145/3477495.3531765`.
- Ji et al., ACL 2023, DOI `10.18653/v1/2023.acl-long.164`.
- Plaud et al., CoNLL 2024, ACL Anthology `2024.conll-1.18`.
- Holm, 1979, DOI `10.2307/4615733`.
- Chen et al., the SemEval-2025 PromiseEval task overview, ACL Anthology `2025.semeval-1.321`.
- Schimanski et al., *ClimateBERT-NetZero*, ACL Anthology `2023.emnlp-main.975`.
- The versioned AI CUP 2026 sample-submission guide as a `@misc` entry; do not invent an unreleased task-overview paper.

- [ ] **Step 4: Write Related Work**

Write four compact paragraphs:

1. Connect the AI CUP data to PromiseEval's four-part ESG promise-verification formulation and distinguish verification from climate-target extraction in ClimateBERT-NetZero.
2. Place the output structure within hierarchical text classification without claiming that this paper discovered hierarchy constraints.
3. Attribute the path-constrained principle to Yu et al.; explain that Ji et al. call the family C-metrics; state that this paper adapts the principle to the official field-weighted macro-F1 rather than claiming to use the original C-MacroF1.
4. Address Plaud et al. directly: their point-estimate ranking observation and this paper's paired contrast inference answer different questions. State that hF also changes micro versus macro aggregation, so only C-wF1 isolates consistency treatment.

- [ ] **Step 5: Build and check references**

Run: `make -C manuscript build`

Run: `pytest tests/test_manuscript.py -q`

Expected: no undefined citation or reference warnings.

- [ ] **Step 6: Commit related work**

```bash
git add manuscript/references.bib manuscript/sections/02_related_work.tex tests/test_manuscript.py
git commit -m "docs(paper): add verified related work"
```

---

### Task 5: Results, adverse evidence, and limitations

**Files:**
- Modify: `manuscript/sections/06_results.tex`
- Modify: `manuscript/sections/07_discussion.tex`
- Modify: `tests/test_manuscript.py`

**Interfaces:**
- Consumes: generated Tables 2--4, their captions, `tables/findings.md`, and the frozen protocol prose.
- Produces: the paper's evidence chain and scoped conclusion.

- [ ] **Step 1: Add result-claim tests against generated artifacts**

The tests must derive values from existing table text rather than restating constants:

```python
def test_results_include_each_authoritative_headline_row():
    results = (REPO_ROOT / "manuscript" / "sections" / "06_results.tex").read_text(
        encoding="utf-8")
    table4 = (REPO_ROOT / "tables" / "table4_contrasts.tex").read_text(
        encoding="utf-8")
    current_metric = ""
    rows = {}
    for line in table4.splitlines():
        if not line.rstrip().endswith(r"\\"):
            continue
        cells = [cell.strip() for cell in line[:-2].split("&")]
        if len(cells) != 5 or cells[0] == "Metric":
            continue
        if cells[0]:
            current_metric = cells[0]
        rows[(current_metric, cells[1])] = line
    for contrast, metric in (
        ("M1-M0", "Weighted macro-F1"),
        ("M1-M0", "Path-constrained wF1"),
        ("M1-M0", "Tuple accuracy"),
        ("M4-M1", "Tuple accuracy"),
    ):
        row = next(line for (family, name), line in rows.items()
                   if name == contrast and family.startswith(metric))
        for number in re.findall(r"[-+]?\d+\.\d+", row):
            assert number in results or "../tables/table4_contrasts.tex" in results


def test_results_disclose_non_replication_and_adverse_result():
    text = source_text(REPO_ROOT / "manuscript").lower()
    assert "same-document" in text and "0.739" in text
    assert "m4-m1" in text and "0.028" in text
    assert "no detectable difference" in text
```

- [ ] **Step 2: Run the focused tests and confirm failure**

Run: `pytest tests/test_manuscript.py -q`

Expected: result-disclosure tests fail.

- [ ] **Step 3: Write Results with generated floats**

Structure Results as:

1. **Controlled decision comparison:** input Table 2; lead with 12.6% invalid M0 outputs and zero for M1--M6. State that `\pm` is seed spread over the full pipeline, not a confidence interval.
2. **Metric-constraint mismatch:** input the complete Table 4. State official M1-M0 -0.001 [-0.006, 0.003], `p_{\mathrm{Holm}}=1.000`; C-wF1 +0.004 [0.001, 0.007], `p_{\mathrm{Holm}}=.025`; hF +0.003, `p_{\mathrm{Holm}}=.017` as corroboration with a micro/macro confound; tuple accuracy +0.035, `p_{\mathrm{Holm}}=.001` as pre-specified corroboration.
3. **Adverse result:** state M4-M1 tuple accuracy -0.006 [-0.010, -0.002], `p_{\mathrm{Holm}}=.028`. Do not call valid-state decoding uniformly superior.
4. **Evaluation estimands:** input Table 3 and describe 0.012--0.015 gaps as differences between targets, not bias.
5. **Protocol robustness:** state that C-wF1 M1-M0 is +0.002 [-0.001, 0.005], `p_{\mathrm{Holm}}=.739` under same-document evaluation and therefore does not replicate there.

Every significance sentence names `p_{\mathrm{Holm}}`, and Table 4's caption states that intervals are uncorrected.

- [ ] **Step 4: Write Discussion, limitations, and conclusion**

Discussion will:

- attribute metric disagreement to ancestor-unsupported partial credit, without claiming a universal property;
- explain qualitatively that projection exchanges gains on structurally forced `N/A` labels against losses on substantive labels; omit exact mechanism counts because they are not printed in a generated TeX artifact;
- state that official wF1 ranks M5 first while hF ranks M6 first, so “best” requires a metric;
- explain why a more complex 17-state search need not improve whole-row correctness.

Limitations will compactly cover all design-mandated points: 49 effective clusters; one dataset, language, domain, and backbone; seed conflates split and training randomness; n=2 `Misleading`; no training-time structural loss, classifier chain, LLM, or alternate backbone; no temporal/industry split; primary C-wF1 result only under document-disjoint; two post-hoc metrics; one frozen hyperparameter configuration; corrected training and statistical-decision defects with all reported artifacts regenerated.

Conclusion will make one narrow claim: for this ESG dataset and backbone, field-wise scoring can fail to reflect validity-preserving decisions.

- [ ] **Step 5: Run policy checks and build**

Run: `pytest tests/test_manuscript.py -q`

Run: `make -C manuscript check`

Expected: checks pass except no final-author check is requested.

- [ ] **Step 6: Commit results and discussion**

```bash
git add manuscript/sections/06_results.tex manuscript/sections/07_discussion.tex tests/test_manuscript.py
git commit -m "docs(paper): report controlled results and limitations"
```

---

### Task 6: Introduction, abstract, integration, and eight-page fit

**Files:**
- Modify: `manuscript/main.tex`
- Modify: `manuscript/sections/01_introduction.tex`
- Modify: all section files only as needed for page fit and cross-references
- Modify: `manuscript/check.py`
- Modify: `tests/test_manuscript.py`

**Interfaces:**
- Consumes: completed technical, related-work, and results sections.
- Produces: coherent compiled draft at `manuscript/build/main.pdf`.

- [ ] **Step 1: Add integration tests**

```python
def test_introduction_states_question_and_four_contributions():
    text = (REPO_ROOT / "manuscript" / "sections" / "01_introduction.tex").read_text(
        encoding="utf-8")
    assert "field-wise" in text.lower()
    assert "17 legal" in text.lower()
    assert text.count(r"\item") == 4


def test_abstract_uses_scoped_language():
    text = (REPO_ROOT / "manuscript" / "main.tex").read_text(encoding="utf-8").lower()
    abstract = text.split(r"\begin{abstract}", 1)[1].split(r"\end{abstract}", 1)[0]
    assert "for this dataset and backbone" in abstract
    assert "state-of-the-art" not in abstract
    assert "competition score improvement" not in abstract
```

- [ ] **Step 2: Run the focused tests and confirm failure where prose is incomplete**

Run: `pytest tests/test_manuscript.py -q`

Expected: introduction contribution test fails until the final framing is written.

- [ ] **Step 3: Write Introduction**

Use four paragraphs:

1. ESG promise verification needs internally usable tuples, not four unrelated labels.
2. The official field-wise score can credit unsupported children; formulate the research gap around metric alignment.
3. Preview the controlled evidence without presenting a SOTA claim.
4. Provide exactly four contribution items matching Spec Section 3.3.

- [ ] **Step 4: Tighten the abstract**

Keep 150--190 words. Include task, controlled setup, 12.6% invalid rate, official unresolved contrast, positive C-wF1 contrast, pre-specified tuple corroboration, adverse decoder result, and the one-dataset/one-backbone scope. Do not add competition rank.

- [ ] **Step 5: Compile and fit within eight pages**

Run: `make -C manuscript build`

Run:

```bash
python - <<'PY'
from pypdf import PdfReader
pdf = "manuscript/build/main.pdf"
print(len(PdfReader(pdf).pages))
PY
```

If the draft exceeds eight pages, reduce in this order:

1. remove repeated definitions already visible in Figure 1 or tables;
2. shorten captions without dropping statistical provenance;
3. compress Related Work and limitations prose;
4. use `\small` for Tables 2 and 4;
5. adjust float placement.

Do not reduce body text below the ACM template default, hide any Table 4 row, remove adverse evidence, or delete post-hoc disclosures.

- [ ] **Step 6: Resolve LaTeX diagnostics**

Run: `make -C manuscript check`

Inspect `manuscript/build/main.log`. Fix every undefined citation/reference and material overfull box. A small overfull caused only by a long DOI in the bibliography may be addressed with `\Urlmuskip=0mu plus 1mu`; do not solve layout issues with global scaling.

- [ ] **Step 7: Run the full suite**

Run: `pytest -q`

Expected: all existing and manuscript tests pass.

- [ ] **Step 8: Commit the integrated draft**

```bash
git add manuscript tests/test_manuscript.py
git commit -m "docs(paper): complete integrated NTCIR-19 draft"
```

---

### Task 7: Independent review and final verification

**Files:**
- Modify: manuscript sources only for issues found by review.

**Interfaces:**
- Consumes: complete compiled draft and approved design.
- Produces: reviewed branch with evidence-backed handoff and active,
  non-placeholder author metadata.

- [ ] **Step 1: Request scientific-claim review**

Ask a reviewer to compare every claim in `manuscript/` against `tables/findings.md`, `tables/*.tex`, and the design spec. The reviewer must report unsupported generalizations, omitted adverse evidence, incorrect pre/post-specification language, and any hardware result outside the allowed artifact set.

- [ ] **Step 2: Request format and readability review**

Ask a separate reviewer to inspect the compiled PDF for eight-page compliance, float order, table readability, reference completeness, and whether the manuscript reads as an academic evaluation study rather than a competition report.

- [ ] **Step 3: Apply only evidence-backed corrections**

For each review item, locate the supporting artifact before editing. Do not add a new experiment, new numerical analysis, or new result family during this task.

- [ ] **Step 4: Run final draft verification**

Run:

```bash
make -C manuscript clean
make -C manuscript check
pytest -q
git diff --check
git status --short --branch
```

Expected:

- manuscript build succeeds;
- PDF has at most eight pages;
- all tests pass;
- no whitespace errors;
- branch contains only intentional committed changes;
- `make -C manuscript check-final` succeeds with the four team-supplied author
  records.

- [ ] **Step 5: Commit review corrections**

```bash
git add manuscript tests/test_manuscript.py
git commit -m "docs(paper): address manuscript review"
```

- [ ] **Step 6: Record handoff**

Report the branch, commit hashes, compiled PDF path, page count, test results,
and the remaining external submission action: confirm whether the AI CUP
Special Session expects the general NTCIR `Team Name`/`Subtasks` blocks.
