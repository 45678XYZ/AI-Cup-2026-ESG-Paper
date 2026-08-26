# Hierarchy-Constrained Decision Calibration and Decoding for Multi-Task ESG Promise Verification

Controlled study for the NTCIR-19 AI CUP special session paper.

## What the study compares

Each paragraph carries four dependent labels — promise status (PS), verification
timeline (VT), evidence status (ES), evidence quality (EQ) — whose hierarchy
admits only **17 of the 120** label tuples. Holding one fixed set of base model
probabilities constant, we compare seven decision rules:

| ID | Calibration | Output rule | Valid by construction |
|---|---|---|---|
| M0 | none | independent argmax | no |
| M1 | none | deterministic projection | yes |
| M2 | global bias | deterministic projection | yes |
| M3 | conditional bias | deterministic projection | yes |
| M4 | none | 17-state decoding | yes |
| M5 | global bias | 17-state decoding | yes |
| M6 | conditional bias | 17-state decoding | yes |

Because every method consumes identical probabilities on identical test rows,
differences are attributable to the decision stage rather than to the backbone
or the training data.

## Third-party data

`dataset/mlpromise_english.json` is a redistributed copy of the English portion
of **ML-Promise** (Seki et al., EMNLP 2025), used for the external replication
arm and licensed **CC BY-NC-SA 4.0**. It is byte-identical to the release; all
normalisation happens at load time. Attribution, source and licence terms are
in [`dataset/mlpromise_english.NOTICE`](dataset/mlpromise_english.NOTICE).

The multilingual replication additionally vendors the byte-identical French,
Japanese and Korean training releases. Their source file IDs, hashes, licence
and load-time normalisation are recorded in the corresponding
`dataset/mlpromise_{french,japanese,korean}.{NOTICE,json}` provenance files.
The Korean release contains no text field; report-page text is reconstructed
locally by `scripts/prepare_korean_pages.py` and is deliberately not committed.

Everything else in this repository is the study's own work. The AI CUP data
under `dataset/vpesg4k_*.json` is covered by the competition's own terms.

## Layout

```
paper/          study code: label space, data, training, contract artifacts
  labels.py       frozen enumerations + canonical 17-state space
  data.py         loading, canonical row order, data checksum
  score.py        official weighted macro-F1 (single source of truth)
  train_config.py frozen training recipe
  model.py        shared encoder, four heads
  dataset.py      tokenisation and collation
  splits.py       rotating three-way split generation (stdlib only)
  projection.py   hierarchy-constrained projection onto the 17 states
  decoder.py      joint decoding over the 17 legal states
  methods.py      the M0-M6 table, and scoring in log space
  calibration.py  class biases estimated on the Calibration partition only
  accumulation.py gradient-accumulation windows (torch-free, so CI checks them)
  structure_loss.py training-time legality objective (off in the frozen study)
  select_lambda.py  the structural arm's pre-registered selection criterion
  train_fold.py   trains one rotation, emits raw probabilities only
  run_training.py driver: split manifest in, contract bundle out
  run_decisions.py driver: probability bundles in, contract-3 files out
  artifacts.py    the only writer of contract files
  evaluate.py     per-row predictions -> the contract-3 results file
  validate.py     inbound conformance checks on received artifacts
  provenance.py   the git stamp every generated artifact carries
  run_manifest.py one index for the whole study, with cross-file verdicts
  corpus.py       which corpus a run is about, and where its artifacts live
  labels_en.py    the ML-Promise English vocabulary, mapped onto the frozen space
  data_en.py      the English release, loaded into the shape the study uses
  labels_ml.py    the French/Japanese/Korean label mapping, corrections counted
  data_ml.py      the non-English releases, loaded without modifying their files
  multilingual_config.py  the frozen corpus/model matrix for the replication
analysis/       audit, statistics, tables and figure (consumes the contracts)
  __main__.py     one command that regenerates every number C is responsible for
  audit.py        dataset and split audit; the numbers behind Table 1
  metrics.py      subset-aware weighted macro-F1, pinned to paper/score.py
  load.py         prediction sets aligned onto one canonical row order
  bootstrap.py    paired PDF-cluster bootstrap and Holm correction
  aggregate.py    cross-seed aggregation and the pre-specified contrasts
  cases.py        why removing every invalid tuple barely moves weighted macro-F1
  findings.py     what the intervals license D to write, computed not agreed
  tables.py       contract-4 tables, captions and provenance manifest
  legality_cost.py  what enforcing legality costs, measured in every arm (Table 3)
  multilingual_mechanism.py  the same mechanism across five corpora (Table 7)
  structural_arm.py       the structural arm against the frozen lambda=0 arm
  architecture_screen.py  one exploratory backbone's two lambda arms
  rbt_base.py             the pre-registered backbone generality check
  english_replication.py      the pre-registered English replication summary
  multilingual_replication.py the French/Japanese/Korean replication summaries
  figure1.py      Figure 1's counts, and the latexmk build of its source
  preview.py      every delivered tabular rendered into one PDF
scripts/        GPU campaign glue; no paper number depends on these
  prepare_korean_pages.py  derives the Korean model input, output not committed
  run_multilingual_experiments.py  one GPU's half of the frozen matrix
  (two queue/recovery helpers alongside them are machine-specific and kept
   only as a record of how the campaign was sequenced)
contracts/      interface schemas and example files (fabricated fixtures)
docs/           governance/ (plan, contract, inference families), preregistration/,
                results/ (one record per arm), writing/ (the report for D)
figures/        Figure 1 as standalone TikZ, its generated defs, and the PDF
tests/          pytest suite
dataset/        AI CUP data, plus the four vendored ML-Promise releases
```

### Artifact directories, by corpus

All version controlled; `.gitignore` records what each one keeps and why.

```
                       splits      probability bundles   predictions + results
AI CUP Chinese         splits/     probs/            30  predictions/, results/
  exploratory arms                 runs/*/probs/    135  runs/*/
ML-Promise English     splits_en/  runs_en/*/probs/ 150  runs_en/*/
ML-Promise French      splits_fr/  runs_fr/*/probs/ 150  runs_fr/*/
ML-Promise Japanese    splits_ja/  runs_ja/*/probs/ 150  runs_ja/*/
ML-Promise Korean      splits_ko/  runs_ko/*/probs/ 150  runs_ko/*/
```

`runs/` holds the five Chinese exploratory arms — `structural/`, `lambda_sweep/`,
`deberta_v2_320m/`, `electra_180g_large/`, `rbt_base/`. Each corpus with arms
nests them as `{corpus}/{backbone}/lambda_{x}/`; the frozen Chinese anchor has
no arm level and keeps the top-level names contract section 4 fixed.

```
tables/            the seven delivered tables, their captions, the dataset audit,
                   the mechanism and legality-cost JSON, and the manifest tying
                   each printed number to the inputs it came from
run_manifest.json  the generated study index, committed at results freeze
```

## Setup

Everything except training runs on CPU with five packages:

```bash
pip install numpy scikit-learn pandas pypdf pytest
pytest -q                      # no GPU, no model download
```

Figure 1 is standalone TikZ, so redrawing it needs a TeX installation carrying
`latexmk`. Without one `python -m analysis` still writes every table and simply
reports the figure as skipped — its counts come from `paper/labels.py` rather
than from the run, so the committed PDF cannot fall out of step with the
tables. The two tests that compile it skip on the same condition.

Any TeX Live or MacTeX installation will do. TeX Live also installs under
`$HOME` without root, which is how the committed figure was built:

```bash
./install-tl -scheme-full -texdir ~/texlive/2026    # then put its bin/ on PATH
```

Training additionally needs the pinned CUDA environment, which resolves only on
a CUDA machine (`environment.yml` pins `pytorch-cuda`):

```bash
conda env create -f environment.yml
conda activate aicup-esg-paper
```

## Protocol in one paragraph

Five-way rotating three-way cross-fitting. For each seed in {42, 123, 456} and
each rotation k: Test = fold k, Calibration = fold (k+1) mod 5, Train = the
remaining three folds. The model sees Train only; class biases are estimated on
Calibration only; scores are computed on Test only. After five rotations every
row has been in Test exactly once, and the five test partitions are concatenated
before a single score is computed — per-fold F1 is never averaged, because folds
differ in which rare classes are present. Two split protocols answer two
different questions: `pdf_group` (document-disjoint) and `row_strat`
(same-document). Full detail in [docs/governance/paper_plan.md](docs/governance/paper_plan.md).

## Boundaries between contributors

Work is handed off through file formats, not conversations. Schemas, invariants,
and example files live in [docs/governance/interface_contract.md](docs/governance/interface_contract.md).
Two boundary rules matter most:

- **Rows are identified by their `id` string, never by position.** Positional
  alignment fails silently and still produces plausible scores.
- **Training code emits raw probabilities and nothing else.** It applies no hard
  rules, no biases, no argmax. Any decision rule leaking into training would
  quietly stop M0 from being an unstructured baseline.

## Generating artifacts

```bash
python -m paper.splits                  # -> splits/{protocol}_seed{seed}.json  (6 files)
python -m contracts.export_states       # -> contracts/states.json
python -m contracts.make_fixtures       # -> synthetic probability fixtures   (5 bundles)
python -m contracts.make_examples       # -> synthetic results + placeholder tables (42 + 42)
pytest -q                               # asserts the invariants on what was generated
```

The portable multilingual driver runs the frozen French, Japanese and Korean
matrix with two disjoint workers. Run each command on its own GPU (the number
passed to `--gpu` is the physical CUDA device index):

```bash
python -m scripts.run_multilingual_experiments --worker large --gpu 0
python -m scripts.run_multilingual_experiments --worker base  --gpu 1
```

Both commands are resume-safe and generate decisions after training. Limit a
run with `--languages mlpromise_fr` (or `mlpromise_ja` / `mlpromise_ko`), and
use `--train-only` or `--decisions-only` when only one stage is needed. Validate
the completed language artifacts from any ordinary clone with:

```bash
python -m paper.validate --all --corpus mlpromise_fr
python -m paper.validate --all --corpus mlpromise_ja
python -m paper.validate --all --corpus mlpromise_ko
```

`scripts/queue_after_english.py` and `scripts/finalize_multilingual_recovery.py`
record the machine-specific sequencing used for the original campaign; they
are not required by this portable procedure.

## Tables and the figure

One command rebuilds every number the paper reports, from the per-row
predictions and nothing else:

```bash
python -m analysis --predictions-root contracts/examples   # against the synthetic set
python -m analysis                                         # against real results/
```

It writes all seven `tables/table*.tex`, their captions, `tables/audit.json`,
`tables/manifest.json` (recording the sha256 of every input, so any printed
number traces back to the artifacts behind it) and
`figures/figure1_hierarchy.pdf`. Rebuilding is a no-op on a clean tree: every
`.tex`, every caption, the figure and the preview come out byte-identical, and
only the `generated_at` and `git_sha` fields of the JSON move. The run takes
about six minutes, most of it the 10,000-resample paired PDF-cluster bootstrap
across the seven arms.

Nothing here transcribes a score: `analysis/metrics.py` is a vectorised
restatement of `paper/score.py`, fast enough for the bootstrap, and
`tests/test_analysis_metrics.py` asserts the two agree exactly — on the full
2,000 rows and on resampled subsets — so the intervals are computed with the
same metric the paper reports.

Run against `contracts/examples` the command prints a warning, because those
scores are fabricated: only the shapes are meaningful until B's probabilities
and A's decision stage exist.

Conformance is enforced where artifacts are created rather than checked
afterwards: the generator refuses to emit a split that breaks same-document
coverage, the bundle writer refuses arrays whose shape disagrees with the
manifest or that are not probability distributions, and the training driver
takes its rows from the manifest so alignment cannot drift. The test suite
asserts these properties on real generated files.

What a single write cannot see — whether a bundle was produced against the
split it names, whether five rotations belong to one partition, whether the
array on disk is still the one that was checksummed — is checked on receipt:

```bash
python -m paper.validate probs/pdf_group_seed42_r*
python -m paper.validate predictions/*.csv.gz
python -m paper.validate --all
```

Run it on every bundle as it arrives. These failures do not raise; they produce
a plausible results table, which is why they get a validator rather than a
convention.

One level up, a single command indexes the whole study — environment, commit,
the sha256 of every split, bundle, predictions and results file, and a verdict
on whether they are mutually consistent:

```bash
python -m paper.run_manifest                              # -> run_manifest.json
python -m paper.run_manifest --root contracts/examples --out /tmp/manifest.json
```

It records checksums and verdicts, never scores: the per-row files are the
authority on every number, and a copy here could disagree with them. It also
performs the one check nothing else does — that each results file's
`predictions_sha256` still matches the predictions file on disk. That link is
written once when the results are generated and never verified again, so a
predictions file regenerated afterwards would leave every table attributable to
a file that no longer exists in that form.

An incomplete study is indexed rather than refused (before B delivers, `probs`
is simply 0 of 30), and a manifest built from the synthetic fixtures says so in
its `warnings`, because otherwise it is indistinguishable from one built from a
real run. Generate it at results freeze and commit it alongside the artifacts.

Two further files are written beside them, neither a contract-4 deliverable:
`tables/case_analysis.json` counts which hierarchy rule the unconstrained
baseline breaks and what the repair costs, and `tables/findings.md` states
which claims the Holm-corrected intervals license — the distinction between
"no detectable difference" and "no difference" is a computed output rather
than an editorial decision.

Four tables past the three the contract names are written as well.

`tables/table3_legality_cost.tex` is the paper's central table: two rules that
both emit only legal tuples, scored under both pre-specified metrics, across
all seven (backbone, λ) arms. `tables/table4_contrasts.tex` puts the five
pre-specified contrasts against their Holm families side by side with the
corrected p-values; it is where the study's statistical result actually lives,
and its first block — the metric the competition ranks by — carries no bold at
all. `tables/table5_headroom.tex` decomposes the official score's shortfall by
field and prices what closing each part is worth; it is descriptive, so it is
the one to drop first against a page budget. `tables/table6_regimes.tex`
compares the two evaluation splits.

`tables/table7_multilingual_mechanism.tex` carries the external replication:
the same projection-versus-argmax contrast measured on all five corpora, with
the per-class ledger that explains why the official metric moves so little.

The last two are written by their own analysis modules rather than by
`analysis/tables.py`, because their inputs span arms and corpora rather than
the cross-seed summaries that file consumes. They are registered in
`analysis.tables.EXTERNAL_TABLES`, which the rebuild iterates — registering a
table and rebuilding it are the same act, so a delivered table cannot quietly
stop being regenerated.

The `.tex` files hold a bare `tabular` and do not compile alone. To see them
rendered, together with their captions and figure 1:

```bash
python -m analysis.preview                                # -> tables/preview.pdf
```

Building it needs a local `latexmk`, but reading it does not: the PDF is
committed, so the rendered tables are one click away for anyone without a TeX
installation. It is a convenience rather than a deliverable — no number is
computed here and the paper cites none of it — and rebuilding is a no-op,
because `SOURCE_DATE_EPOCH` and `\pdftrailerid{}` pin the two things pdfTeX
would otherwise vary between runs.

## Training runs

Requires a GPU and the conda environment.

```bash
python -m paper.run_training --protocol pdf_group --seed 42
```

One invocation trains all five rotations for that (protocol, seed) and writes
one contract-shaped bundle each; the full study is 2 protocols x 3 seeds = 6
invocations, 30 fits. `--skip-existing` makes a run resumable after a crash.

The driver reads its training rows straight out of the split manifest and
predicts in the manifest's own id order, so conformance is structural rather
than something the operator has to remember. It refuses to start unless
`MODEL_REVISION` is pinned; `--allow-unpinned-revision` overrides that for a
throwaway smoke test.

The backbone is fetched once per invocation and reused for all five rotations,
downloading it if the Hugging Face cache is cold. Because `from_pretrained`
ignores `revision` once it is handed a local directory, the driver checks that
the resolved snapshot really is the pinned commit and refuses to train
otherwise — that check, not the argument, is what enforces the pin.

## Decision runs

CPU only. This is the step between B's probabilities and the analysis: it
applies the seven decision rules and writes the contract-3 files.

```bash
python -m paper.run_decisions --protocol pdf_group --seed 42
python -m paper.run_decisions --protocol pdf_group --seed 42 \
    --methods M0 M1 M4 --probs-dir contracts/examples/probs --out-dir /tmp/smoke
```

Both arguments are required and each invocation covers one (protocol, seed), so
the full study is the same 6 invocations as the training stage — 42 predictions
files and 42 results files, a few seconds in total.

The completed training-time structural arm is kept separate from the frozen
study. Its selected-lambda bundles live in `runs/structural/probs/`; its 42 decision
files and the pre-registered cross-arm comparison live in `runs/structural/`.
Rebuild the comparison, including the 10,000-resample H2 bootstrap, with:

```bash
python -m analysis.structural_arm \
    --structural-root runs/structural \
    --probs-dir runs/structural/probs \
    --out runs/structural/comparison.json
```

The execution record and interpretation are in
[`docs/results/structural_training_results.md`](docs/results/structural_training_results.md).

One invocation loads the five rotations once and runs every requested method
over that one loaded set, which is what makes "identical probabilities on
identical rows" structural rather than an operating convention. All five
bundles are validated before anything is decided; the five test partitions are
then concatenated into one 2,000-row predictions file and scored once, because
per-fold F1 is not on a common scale across folds and must never be averaged.

The second form above is the fixture smoke test, and runs today against the
synthetic bundles in `contracts/examples/probs/`. All seven methods run; one
invocation of the full set takes about two seconds on the fixtures.

Biases are estimated per rotation on the Calibration partition and nowhere
else, against each field's own macro-F1 evaluated *before* the output rule is
applied. One estimate therefore serves both output rules — M2 and M5 are handed
the same global biases, M3 and M6 the same conditional ones — which is what
lets the results table be read as a factorial rather than as seven systems.

## Status

**All experiments are complete.** Five corpora, 765 probability bundles and
1,050 per-row prediction files:

| Corpus | Bundles | What it is |
|---|---:|---|
| AI CUP Chinese (frozen anchor) | 30 | the pre-specified study |
| AI CUP Chinese (exploratory arms) | 135 | structural-λ, DeBERTa, ELECTRA, RoBERTa-base |
| ML-Promise English | 150 | pre-registered external replication |
| ML-Promise French | 150 | multilingual replication |
| ML-Promise Japanese | 150 | multilingual replication |
| ML-Promise Korean | 150 | multilingual replication |

Every score in the paper is recomputed from those per-row predictions by
`python -m analysis`; nothing is transcribed. A rebuild on a clean tree
reproduces every table, caption, figure and the preview byte-for-byte.

`536 passed, 3 skipped` with TeX on `PATH` (the three are the torch-only
training-path tests). `paper.validate --all` reports clean for `aicup_zh` (72
artifacts) and for the English, French and Japanese corpora (360 each). Korean
bundle validation additionally needs the locally reconstructed page text, which
is deliberately not committed; its summary still reproduces from the committed
predictions, which is what the paper's numbers come from.

Files under `contracts/examples/` remain fabricated fixtures and must never be
used as paper results.

### What the study found

The pre-specified question — does enforcing the hierarchy raise the official
score — is answered **no**: none of the five pre-specified contrasts survives
Holm correction on the competition metric, and the whole spread across the
seven decision rules is 0.0082, which is itself the largest single contrast
(M5 is the highest-scoring rule and M6 the lowest) and still fails correction
at `p_Holm` = 0.171.

What replaced it is an account of *why*. Projection's only lever on a child
field is writing `N/A`, so its repairs land on one class and its damage on the
others, and a macro average over classes cancels them. That account is
arithmetic rather than linguistic, and `table7` shows it holding in all five
corpora: `N/A` net positive and substantive net negative five times out of
five, across an M0 illegal rate spanning 12.55% to 31.25%, while whole-row
accuracy rises in every corpus and the official metric rises in two and falls
in three.

One consequence is worth stating separately, because it is a property of the
metric rather than of any method: macro-F1 averages over the classes *present
in gold*, so a corpus missing one divides that field's weight by a smaller
number. Korean has no `Misleading` row, which makes a point of one
`evidence_quality` class worth 0.1167 there against 0.0875 everywhere else.
The metric's exchange rate is a property of each corpus, not of the schema —
which is also why cross-corpus scores are not compared here.

One design point in that code is easy to misread: under conditional
(hierarchy-constrained) estimation, the three child-field `N/A` biases are
*unidentifiable*, not merely unsupported — those classes never occur inside
their conditioning subset. They are pinned at 0.0 by
definition (`paper/labels.py::CONDITIONAL_PINNED_CLASSES`), which leaves M3
unaffected but gives M6 three pinned terms where M5 fits four. See
[docs/governance/paper_plan.md](docs/governance/paper_plan.md) §3.2.

The official GPU setup is frozen in `paper/train_config.py`: the exact Hugging
Face revision is pinned, the fixed budget is 12 epochs, and the last three epoch
states are averaged. All 30 cross-fitted probability bundles have been produced
and validated.

The 30 bundles were re-run on 2026-08-21 after the short-batch loss scaling was
corrected (`paper/accumulation.py::loss_scale`). All record the clean source
commit `35dea657eede733ea6c8945f3976a1561cfab80d`, the same training-configuration
hash and the RTX 3090 environment. `docs/results/gpu_training_progress.md` records the
original defect, the epoch evidence, the smoke test and the completed campaign.

The full official run has been materialised: `predictions/` and `results/`
contain all 42 protocol/seed/method outputs, and all 42 prediction files pass
the contract validator. Reproduce them from the committed bundles with the six
invocations under [Decision runs](#decision-runs) — one per (protocol, seed),
about a second and a half each — then `python -m paper.validate --all`.

The six decision invocations were regenerated from that clean source commit.
`python -m paper.run_manifest` indexes all 30 probability bundles, 42
predictions, 42 results and eight table artifacts; all six cross-file checks
pass with no warnings or notes.
