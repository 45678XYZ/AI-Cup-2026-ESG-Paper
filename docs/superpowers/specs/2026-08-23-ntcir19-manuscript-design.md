# NTCIR-19 AI CUP ESG Manuscript Design

**Date:** 2026-08-23

**Metadata updated:** 2026-08-24

**Branch:** `paper/ntcir19-manuscript`

**Submission:** AI CUP Special Session at NTCIR-19

**Working title:** *When Field-Wise Metrics Miss Hierarchical Consistency: A Controlled Study on the AI CUP ESG Promise Verification Dataset*

## 1. Objective

Produce a submission-ready English LaTeX manuscript of at most eight pages,
including references, that converts the repository's controlled RTX 3090 study
into an academic paper rather than a competition system report.

The manuscript will study whether a field-wise evaluation metric reflects the
utility of validity-preserving decisions when the four ESG promise-verification
outputs obey a hard hierarchy. It will not claim a competition-score
improvement or state-of-the-art performance.

## 2. Scope and authority

The paper may use only committed, frozen RTX 3090 artifacts indexed by
`run_manifest.json` and the generated outputs under `tables/`, `figures/`,
`results/`, `predictions/`, and `probs/`.

The paper must not use, mention, compare against, or derive claims from L40 or
L40S runs, temporary artifacts, or timing experiments. A manuscript check will
reject either string in manuscript sources.

Numerical authority is:

1. `tables/*.tex` and `tables/*_caption.txt` for printed paper numbers;
2. `tables/findings.md` for statistical verdicts and prohibited claims;
3. `tables/manifest.json` and `run_manifest.json` for provenance;
4. `docs/study_report.md` for interpretation, subject to the sources above;
5. `docs/paper_plan.md` as the immutable record of pre-specified analyses.

If sources disagree, the higher source in this list wins. Mechanism numbers
currently available only in hand-written prose may enter the manuscript only
after they are promoted to a generated, provenance-tracked artifact.

## 3. Scientific framing

### 3.1 Central question

The four labels form a hierarchy with 17 legal tuples out of 120 Cartesian
combinations. Independent heads can emit illegal tuples, while the official
weighted macro-F1 scores fields separately. The study asks whether this scoring
choice reflects the value of enforcing validity.

### 3.2 Causal argument

The manuscript will present this chain:

1. The task has hard parent-child constraints.
2. Independent argmax violates those constraints on 12.6% of predictions in
   the primary table, while projection and valid-state decoding are legal by
   construction.
3. The official field-wise metric can award credit to child predictions whose
   predicted ancestors do not support them.
4. Projection versus independent argmax has no detectable difference on the
   official metric: M1-M0 is -0.001, with uncorrected 95% CI
   [-0.006, 0.003] and Holm-adjusted p=1.000.
5. Changing only the treatment of ancestor-invalid predictions yields a
   positive path-constrained contrast: +0.004 [0.001, 0.007], with
   Holm-adjusted p=.025.
6. Pre-specified tuple accuracy corroborates the direction:
   +0.035 [0.028, 0.043], with Holm-adjusted p=.001.
7. Therefore, on this task and backbone, the field-wise metric can
   substantially understate the utility of structure-preserving decoding.

### 3.3 Contributions

The paper will claim four contributions:

1. It formalizes AI CUP ESG promise verification as constrained multi-task
   classification over 17 legal tuples and documents its severe imbalance.
2. It performs a controlled comparison of seven decision rules on identical
   base probabilities and test rows.
3. It quantifies a task-metric mismatch through a single-change comparison
   between official and path-constrained weighted macro-F1.
4. It reports both adverse and robustness findings: the 17-state decoder is
   worse than projection on tuple accuracy, and the same-document versus
   document-disjoint estimands differ by 0.012--0.015.

### 3.4 Claim boundaries

The manuscript must:

- say `no detectable difference`, never `no difference` or `equivalent`, when
  a contrast does not survive Holm correction;
- disclose path-constrained weighted macro-F1 and hierarchical F1 as post hoc;
- use Holm-adjusted p-values, not uncorrected intervals, for significance;
- state that the primary path-constrained finding does not survive correction
  in the same-document protocol;
- describe same-document and document-disjoint results as different estimands,
  not as biased and unbiased estimates;
- report the adverse M4-M1 tuple-accuracy result;
- avoid improvement or significance claims for `Misleading` (n=2);
- limit generalization to one dataset, one Chinese backbone, and the tested
  decision rules.

## 4. Manuscript organization and page budget

The manuscript will use the official NTCIR-19 ACM Master Article sample with
`\documentclass[sigconf,article]{acmart}` and ACM reference formatting.

| Page | Content | Main assets |
|---|---|---|
| 1 | Title, abstract, introduction | research gap, headline evidence, contributions |
| 2 | related work, task and dataset | Table 1, 17-state definition |
| 3 | methods | Figure 1, M0--M6, projection and decoder |
| 4 | experimental setup, main results | Table 2, protocols and statistics |
| 5 | metric-constraint mismatch | complete Table 4: four metric families by five contrasts |
| 6 | mechanism and evaluation targets | Table 3, protocol comparison and robustness |
| 7 | discussion, limitations, conclusion | adverse result and scope boundaries |
| 8 | references | cited primary literature |

Figure 1, Table 2, and Table 4 will span both columns. Tables 1 and 3 will use
one column unless legibility requires both. Table 5 will not appear as a float;
its metric-dependent ranking observation will be summarized in prose.

All twenty Table 4 contrasts will remain visible. The paper will not select
only favorable comparisons.

## 5. Repository layout

The existing `paper/` directory is a Python package and will not contain the
manuscript. New sources will live under:

```text
manuscript/
├── main.tex
├── metadata.tex
├── references.bib
├── Makefile
├── README.md
└── sections/
    ├── 01_introduction.tex
    ├── 02_related_work.tex
    ├── 03_task_and_data.tex
    ├── 04_methods.tex
    ├── 05_experiments.tex
    ├── 06_results.tex
    └── 07_discussion.tex
```

`main.tex` will include generated table fragments from `../tables/` and the
committed Figure 1 PDF from `../figures/`. Numerical table bodies will not be
duplicated in manuscript sources.

`metadata.tex` contains the four student author records supplied by the team on
2026-08-24, including their affiliation, location, and individual email
addresses. No author data or team name is invented. The draft build warns if an
explicit placeholder reappears, and the final readiness check rejects any
placeholder even if an email-like field has been added.

The AI CUP Special Session invitation governs submission-specific content. Its
instructions do not require the standard participant-paper `Team Name` and
`Subtasks` blocks, and AI CUP is not one of the task names enumerated on the
general NTCIR-19 participant page. The initial manuscript will therefore omit
those blocks. The final submission checklist will require confirmation from the
AI CUP organizers and will add the fields only if they request them and the user
provides the content.

## 6. Build and verification

The official sample settings will be preserved:

- `\settopmatter{printacmref=false}`;
- no copyright footnote;
- no page numbers or running headers;
- `ACM-Reference-Format` bibliography.

The repository has `tectonic` available. `manuscript/Makefile` will expose a
repeatable build and checks. The draft verification path will:

1. compile `main.tex` with Tectonic;
2. fail on unresolved citations or references;
3. fail if the PDF exceeds eight pages;
4. fail if manuscript sources contain `L40` or `L40S`;
5. confirm that all referenced generated tables and figures exist;
6. check that C-wF1 and hF are described as post hoc;
7. check that Table 4 is included in full;
8. inspect overfull boxes and embedded PDF fonts;
9. require four active, non-placeholder author and email records for final
   submission, while retaining the draft warning for any placeholder.

The manuscript is complete only when the English source builds, stays within
eight pages, includes all core results, contains no unresolved citations, and
passes every check with the team-supplied author metadata.

## 7. Bibliography policy

The bibliography will prioritize primary sources: the AI CUP task or dataset
documentation, the Chinese pretrained model paper, original hierarchical
classification and constrained-decoding work, Yu et al. for path-constrained
metrics, and the Ji and Plaud papers needed to explain terminology and the
ranking discussion.

Every entry will be verified against an official proceedings page, DOI record,
or the primary paper. The manuscript will not invent an overview-paper citation
that has not yet been released.

## 8. Git workflow

Development takes place directly in the clean checkout on
`paper/ntcir19-manuscript`, as approved by the user. Work will be split into
reviewable commits:

1. approved manuscript design;
2. official-template scaffold and build checks;
3. English scientific content and bibliography;
4. layout, page-budget, and final consistency corrections.

No experiment will be retrained and no frozen result artifact will be modified
as part of manuscript drafting.

## 9. Acceptance criteria

The manuscript design succeeds when it produces an academic, reproducible,
eight-page-or-shorter paper whose central contribution is supported by the
controlled evidence, whose unfavorable and post hoc results are disclosed, and
whose numerical claims can be traced to committed RTX 3090 artifacts.
