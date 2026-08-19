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
  train_fold.py   trains one rotation, emits raw probabilities only
  run_training.py driver: split manifest in, contract bundle out
  run_decisions.py driver: probability bundles in, contract-3 files out
  artifacts.py    the only writer of contract files
  evaluate.py     per-row predictions -> the contract-3 results file
  validate.py     inbound conformance checks on received artifacts
  provenance.py   the git stamp every generated artifact carries
  run_manifest.py one index for the whole study, with cross-file verdicts
analysis/       audit, statistics, tables and figure (consumes the contracts)
  audit.py        dataset and split audit; the numbers behind Table 1
  metrics.py      subset-aware weighted macro-F1, pinned to paper/score.py
  load.py         prediction sets aligned onto one canonical row order
  bootstrap.py    paired PDF-cluster bootstrap and Holm correction
  aggregate.py    cross-seed aggregation and the pre-specified contrasts
  tables.py       contract-4 tables, captions and provenance manifest
  figure1.py      Figure 1's counts, and the latexmk build of its source
contracts/      interface schemas and example files
docs/           paper plan and interface contract
figures/        Figure 1 as standalone TikZ, its generated defs, and the PDF
splits/         generated split manifests (version controlled)
tests/          pytest suite
dataset/        AI CUP development and test data
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
(same-document). Full detail in [docs/paper_plan.md](docs/paper_plan.md).

## Boundaries between contributors

Work is handed off through file formats, not conversations. Schemas, invariants,
and example files live in [docs/interface_contract.md](docs/interface_contract.md).
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

## Tables and the figure

One command rebuilds every number the paper reports, from the per-row
predictions and nothing else:

```bash
python -m analysis --predictions-root contracts/examples   # against the synthetic set
python -m analysis                                         # against real results/
```

It writes `tables/table{1,2,3}*.tex`, their captions, `tables/manifest.json`
(recording the sha256 of every input, so any printed number traces back to the
artifacts behind it) and `figures/figure1_hierarchy.pdf`. The 10,000-resample
paired PDF-cluster bootstrap takes about 90 seconds.

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

## Decision runs

CPU only. This is the step between B's probabilities and the analysis: it
applies the seven decision rules and writes the contract-3 files.

```bash
python -m paper.run_decisions --protocol pdf_group --seed 42
python -m paper.run_decisions --protocol pdf_group --seed 42 \
    --methods M0 M1 M4 --probs-dir contracts/examples/probs --out-dir /tmp/smoke
```

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

Done: frozen label space and 17-state definitions, data layer with checksums,
split generation for both protocols, the hierarchy-constrained projection, the
joint 17-state decoder, the M0-M6 method table, calibration-partition class
biases, the training and decision drivers, artifact validation, and synthetic
example files for every handoff.

The decision stage is complete: all seven methods run end to end on the
fixtures. What is missing is real input — no probabilities have been produced
yet, so every file under `contracts/examples/` is still fabricated and no
number anywhere in this repository is a result.

One design point in that code is easy to misread: under conditional
(hierarchy-constrained) estimation, the three child-field `N/A` biases are
*unidentifiable*, not merely unsupported — those classes never occur inside
their conditioning subset. They are pinned at 0.0 by
definition (`paper/labels.py::CONDITIONAL_PINNED_CLASSES`), which leaves M3
unaffected but gives M6 three pinned terms where M5 fits four. See
[docs/paper_plan.md](docs/paper_plan.md) §3.2.

Two values in `paper/train_config.py` are marked `TODO(B)` and must be settled
on the GPU machine before the first official run: `MODEL_REVISION` (pin to the
exact Hugging Face revision that gets downloaded) and `EPOCHS` (set from the
competition training logs — there is no early stopping, so the budget is
whatever this constant says).
