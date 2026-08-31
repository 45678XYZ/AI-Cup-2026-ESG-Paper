# Focused Multilingual NTCIR-19 Manuscript Revision

**Date:** 2026-08-25

**Status:** Approved direction; implementation pending

**Target:** NTCIR-19 AI CUP participant paper, at most eight pages including references

## 1. Objective

Revise the current six-page Chinese-only manuscript into a focused paper whose
primary scientific result remains the pre-specified Chinese controlled study
and whose main robustness result is an external replication on the English,
French, Japanese, and Korean ML-Promise corpora.

The revision must preserve the distinction between confirmatory, pre-specified
secondary, and exploratory evidence. It must not become a catalogue of every
completed experiment. Training-time structural loss and alternate-backbone
screens are retained as compact robustness evidence rather than promoted to
coequal research questions.

The unexecuted M7 batch decoder is outside this manuscript and its uncommitted
design has been removed.

## 2. Version-control sequence

The existing manuscript is preserved before any integration work:

1. Commit the current source state, `.gitignore` exception, and compiled
   `manuscript/build/main.pdf` as a checkpoint.
2. Commit this approved design separately.
3. Merge `multilingual-replication` into `paper/ntcir19-manuscript` without
   rebasing or rewriting either history.
4. Implement the focused manuscript and generated multilingual table.
5. Commit the completed revision only after tests, artifact validation,
   manuscript checks, and visual PDF inspection.

The checkpoint commit is `54cb418` (`docs(paper): track current manuscript
PDF`).

## 3. Central narrative

### 3.1 Primary Chinese evidence

The Chinese document-disjoint experiment remains the only confirmatory main
study. Seven decision rules consume identical cross-fitted probabilities and
test rows. The paper reports the complete pre-specified family of five
contrasts on the official AI CUP weighted macro-F1, with Holm correction.
None is detectable. For M1--M0, the effect is approximately `-0.001` with
`p_Holm = 1.000`.

Tuple exact-match was pre-specified as a secondary outcome and forms its own
family over the same five contrasts. M1--M0 is `+0.035`, with the existing
report-cluster interval and `p_Holm = .001`. M4--M1 is adverse at `-0.006`,
with `p_Holm = .028`. These results establish that projection can improve
whole-output correctness without earning a detectable official-score gain and
that a larger legal-state search is not automatically better.

### 3.2 External multilingual replication

The external replication uses four ML-Promise corpora and language-appropriate
fixed backbones. It does not compare raw scores between languages. It compares
M1 and M0 within the same language, backbone, structural-loss arm, probabilities,
and test rows.

Across English, French, Japanese, and Korean, there are 32 document-disjoint
backbone-by-lambda arms. M1--M0 tuple accuracy is positive in all `32/32` arms.
M1--M0 weighted macro-F1 is positive in only `14/32` arms:

- English: tuple positive `8/8`; weighted macro-F1 positive `7/8`.
- French: tuple positive `8/8`; weighted macro-F1 positive `5/8`.
- Japanese: tuple positive `8/8`; weighted macro-F1 positive `0/8`.
- Korean: tuple positive `8/8`; weighted macro-F1 positive `2/8`.

The cross-language claim is therefore about a repeated contrast pattern, not
about one language being easier or one raw score being higher. The supported
claim is that projection consistently improves complete-tuple correctness in
the executed external arms, while field-wise weighted macro-F1 does not have a
consistent direction.

### 3.3 Fixed-arm five-language table

The paper includes one compact generated table for the fixed primary arm in
each language. It contains:

- language and corpus;
- rows and report clusters;
- M0 invalid-tuple rate;
- M1--M0 weighted macro-F1 delta;
- M1--M0 tuple-accuracy delta.

The intended rounded values are:

| Language | Rows | Reports | M0 invalid | wF1 delta | Tuple delta |
|---|---:|---:|---:|---:|---:|
| Chinese | 2,000 | 49 | 12.55% | -0.0011 | +0.0350 |
| English | 400 | 9 | 23.17% | +0.0032 | +0.0725 |
| French | 400 | 9 | 21.08% | +0.0036 | +0.0625 |
| Japanese | 400 | 19 | 31.25% | -0.0131 | +0.0708 |
| Korean | 500 | 32 | 22.73% | -0.0192 | +0.0293 |

The table must be generated from committed prediction-derived summaries, not
hand-transcribed into LaTeX. English uses RoBERTa-large; French, Japanese, and
Korean use XLM-R-large; Chinese uses the frozen Chinese RoBERTa WWM Ext Large.
All five rows use `lambda = 0` and document-disjoint evaluation.

The English and other ML-Promise weighted scores must be called "AI CUP
weights applied to ML-Promise", not an ML-Promise official metric. Korean uses
PDF-page inputs, has no gold `Misleading` examples, and uses a different input
unit from the paragraph-based corpora; these facts belong in the table note or
limitations.

## 4. Supporting robustness evidence

Training-time structural loss and alternate-backbone screens receive one
compact paragraph or small descriptive block, subject to the page budget.
The paragraph reports only the pattern needed to delimit the main claim:

- On the Chinese anchor, structural loss reduces M0 invalidity from `12.55%`
  to `5.18%`, but the pre-registered M1 weighted macro-F1 effect is `+0.00250`
  with an interval crossing zero and `p = .427`.
- Across the 16 external-language backbone arms that have both `lambda = 0`
  and `lambda = 0.3`, structural loss reduces M0 invalidity in `16/16` arms,
  while its downstream M1 weighted macro-F1 and tuple effects are mixed.
- DeBERTa and ELECTRA screens likewise show large invalidity reductions but no
  conclusive official-score gain. Chinese RoBERTa-base falsifies the simple
  mechanism that a smaller model necessarily emits more invalid tuples.

The full per-backbone tables remain in repository result documents and are not
duplicated in the paper. These screens are exploratory and do not enter a Holm
family.

## 5. Statistical hierarchy

The revised manuscript explicitly declares the following hierarchy:

1. **Confirmatory:** official Chinese weighted macro-F1, five pre-specified
   decision contrasts, one Holm family.
2. **Pre-specified secondary:** Chinese tuple exact-match, the same five
   contrasts, a separate Holm family.
3. **External replication:** pre-registered within-arm multilingual contrast
   directions and the English report-cluster tuple test; language-specific
   field scores are descriptive where the effective report count is small.
4. **Exploratory:** C-wF1, hF, structural training, and alternate-backbone
   screens.

C-wF1 and hF no longer carry the central claim and are not presented as
separate Holm-corrected confirmatory families. They may be retained in one
short supplementary sentence showing directional agreement with tuple
accuracy. The abstract must not headline the post-hoc C-wF1 `p = .025`.

The paper never claims equivalence from a non-significant difference and never
calls an uncorrected interval a multiple-testing decision rule.

## 6. Manuscript structure

### Abstract

Lead with the task hierarchy, the controlled Chinese comparison, the
pre-specified tuple result, and the external `32/32` versus `14/32` replication
pattern. End with a bounded claim: field-wise scoring need not reflect gains in
complete hierarchical correctness. Do not enumerate every p-value.

### Introduction

Frame the paper as an evaluation-and-decision study rather than a competition
performance claim. Contributions become:

1. formalization and data audit;
2. controlled seven-rule Chinese study;
3. pre-specified whole-tuple evidence of metric--structure mismatch;
4. external replication over four languages and multiple backbones;
5. transparent adverse and non-replicating results.

### Related Work

Retain PromiseEval/ML-Promise, hierarchical inference, and hierarchical metric
work. Add the external-dataset citation and enough constrained-inference
context to position projection and valid-state decoding as controlled decision
rules rather than novel general algorithms.

### Task and Data

Correct the Chinese company count from 50 to 49 and remove the false statement
that one report covers two companies. Add a concise ML-Promise paragraph and a
small data summary, preferably integrated into the new five-language table
rather than creating another large table.

### Methods and Experimental Design

Keep the existing M0--M6 definitions. Add only the information needed to make
the replication comparable: common 17-state mapping, fixed per-language
backbones, `lambda` arms, report-disjoint cross-fitting, and within-language
paired interpretation. Structural loss receives a brief definition or a
direct reference to the repository protocol if space is insufficient.

### Results

Present results in this order:

1. Chinese controlled comparison and inferential hierarchy;
2. Chinese tuple result and adverse M4--M1 result;
3. five-language fixed-arm table;
4. `32/32` tuple versus `14/32` weighted-score direction pattern;
5. compact structural/backbone robustness paragraph;
6. one sentence on same-document sensitivity.

Remove the current three-row Table 3 that selects "best calibrated projection"
and "best valid-state decoder" after observing document-disjoint means. Do not
replace it with another selected-best table. If all seven regime rows do not
fit cleanly, the regime comparison remains prose plus repository artifact.

### Discussion and Conclusion

Explain the mechanism without universalizing: legality removes impossible
outputs, but field-wise macro-F1 trades repairs to structural `N/A` classes
against damage to substantive child classes. External data show that the sign
of that trade varies, while whole-tuple correctness improves consistently in
the executed arms.

Limitations include corpus/language confounding, small report counts (especially
English and French), different input units for Korean, absent Korean
`Misleading`, language-specific backbones, post-hoc structural metrics, and the
fact that the replication matrix is not a factorial causal estimate of a pure
language effect.

The conclusion contains two or three sentences, not the current one-line
ending.

## 7. Title and claim scope

The title should no longer say that the study is only "on the AI CUP ...
Dataset". It should retain the field-wise-metric versus hierarchical-consistency
theme while signaling controlled study plus multilingual replication. The
short title remains suitable for ACM headers.

No title or section may claim a universal failure of field-wise metrics. The
paper's strongest licensed wording is that field-wise weighted macro-F1 did not
consistently reflect the gains in tuple validity and exact-match observed in
these controlled and externally replicated arms.

## 8. Generated artifacts and code boundaries

The merge brings in `runs_en/summary.json`, `runs_fr/summary.json`,
`runs_ja/summary.json`, and `runs_ko/summary.json`. A focused renderer is added
to the analysis layer to generate the new LaTeX table from those summaries and
the frozen Chinese summary. Tests must pin its rows, selected arms, signs,
column count, and the warning that ML-Promise has no official weighted metric.

Existing prediction files remain the metric source of truth. The renderer does
not recompute training, select the best method, or mutate the frozen M0--M6
artifacts. No existing result number is manually edited to make it match the
paper.

## 9. Page and presentation budget

The target is seven to eight pages including references. Priority order under
space pressure is:

1. Chinese controlled table and inferential contrasts;
2. five-language replication table;
3. method and protocol reproducibility;
4. limitations and references;
5. compact structural/backbone paragraph;
6. post-hoc C-wF1/hF details.

The old selected-best regime table is removed first. Complete structural and
backbone result matrices remain outside the paper. Tables must remain legible
without reducing text below the current acceptable ACM sizing.

## 10. Verification and acceptance criteria

The revision is complete only when all of the following hold:

1. The current checkpoint remains reachable before the data-branch merge.
2. The merge retains manuscript sources and the tracked PDF.
3. The new five-language table is generated from committed summaries and its
   tests pass.
4. French and Japanese validators pass on all 360 artifacts each.
5. Korean scoring summaries rebuild from committed predictions; full Korean
   training-input validation is reported as unavailable in a fresh clone
   because locally reconstructed page text is intentionally not distributed.
6. All repository tests pass except environment-specific tests whose exact
   causes are documented; floating-point reproducibility assertions must use a
   justified tolerance rather than exact cross-environment JSON equality.
7. `make check-final` passes.
8. The PDF is at most eight pages, includes references, has no missing
   citations, and contains no illegible table or figure.
9. The manuscript states Companies = 49 and does not claim that one Chinese
   report covers two companies.
10. The abstract does not use post-hoc C-wF1 significance as confirmatory
    evidence and does report the pre-specified tuple result.
11. M7 does not appear in source, tables, abstract, or discussion.

## 11. Non-goals

- No new model training or hyperparameter search.
- No new M7 implementation or result.
- No attempt to rank languages by raw score.
- No claim that language alone causes the observed heterogeneity.
- No expansion of the confirmatory family after seeing results.
- No full reproduction of every exploratory architecture table in the paper.
- No push or pull-request creation unless separately requested.
