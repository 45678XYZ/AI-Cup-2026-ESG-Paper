# Reproduction Capsule Input Layout Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Reduce `ntcir19-esg-validity-layer` from sixteen to five first-level directories while preserving all 311 curated input files, all hashes, and the existing reproduction CLI.

**Architecture:** Move all immutable reproduction inputs below `inputs/<corpus>/`, with each corpus owning `splits/` and `runs/`. Keep `paper.corpus` as the corpus-to-path registry, make `reproduce.py` inventory the complete `inputs/` tree, and retain `figures/` and `reference/` as separate expected-output assets.

**Tech Stack:** Python 3.10+, pathlib, pytest, JSON/SHA-256 manifest, gzip CSV/JSON artifacts

**Spec:** `docs/superpowers/specs/2026-08-31-reproduction-input-layout-design.md`

## Global Constraints

- Do not delete or alter the bytes of any prediction, split, corpus-index, figure, or reference file.
- Keep `python reproduce.py`, `python reproduce.py --verify-inputs`, and `python reproduce.py --output-dir PATH` unchanged as public commands.
- The final non-generated first-level directories must be exactly `analysis`, `paper`, `inputs`, `figures`, and `reference`.
- Keep exactly 311 manifest artifacts, including exactly 300 prediction files, pinned to source commit `babd85e61368037d235607ac5ebc798378a7ef75`.
- Do not retain compatibility symlinks or duplicate copies of the old first-level input directories.
- Preserve unrelated staged and unstaged changes in the parent worktree.
- `ntcir19-esg-validity-layer/` is intentionally ignored by the parent repository and has no independent `.git`; do not force-add it or initialize a repository during this migration.
- Prefix every shell command with `rtk` as required by `/home/tom1030507/.codex/RTK.md`.

---

### Task 1: Establish the compact-layout public contract

**Files:**
- Modify: `tests/test_reproduction_release.py`
- Test: `tests/test_reproduction_release.py`

**Interfaces:**
- Consumes: the standalone capsule at `REPO_ROOT / "ntcir19-esg-validity-layer"`
- Produces: `EXPECTED_FIRST_LEVEL_DIRECTORIES: set[str]` and a failing layout test that defines the migration boundary

- [ ] **Step 1: Point the contract at the current capsule name**

Change the constant to:

```python
RELEASE_ROOT = REPO_ROOT / "ntcir19-esg-validity-layer"
```

- [ ] **Step 2: Add the first-level layout test**

Add this constant and test after `RELEASE_ROOT`:

```python
EXPECTED_FIRST_LEVEL_DIRECTORIES = {
    "analysis",
    "figures",
    "inputs",
    "paper",
    "reference",
}


def test_release_groups_all_curated_inputs_by_corpus():
    directories = {
        path.name for path in RELEASE_ROOT.iterdir()
        if path.is_dir() and path.name != "outputs"
    }
    assert directories == EXPECTED_FIRST_LEVEL_DIRECTORIES

    manifest = json.loads(
        (RELEASE_ROOT / "release_manifest.json").read_text(encoding="utf-8")
    )
    assert len(manifest["artifacts"]) == 311
    assert all(path.startswith("inputs/") for path in manifest["artifacts"])
```

- [ ] **Step 3: Move the corruption fixture contract to the new paths**

Replace its fixture paths and manifest keys with:

```python
prediction = (
    root / "inputs" / "aicup_zh" / "runs" / "main"
    / "predictions" / "paper.csv.gz"
)
corpus_index = root / "inputs" / "aicup_zh" / "corpus_index.json.gz"

# Inside the fixture manifest:
"artifacts": {
    "inputs/aicup_zh/runs/main/predictions/paper.csv.gz": "sha256:" + "0" * 64,
    "inputs/aicup_zh/corpus_index.json.gz": "sha256:" + "0" * 64,
},
```

- [ ] **Step 4: Run the new test and verify the expected failure**

Run:

```bash
rtk pytest -q tests/test_reproduction_release.py::test_release_groups_all_curated_inputs_by_corpus
```

Expected: FAIL because the current directory set contains the old `runs_*`,
`splits_*`, `predictions`, and `artifacts` roots and has no `inputs` root.

- [ ] **Step 5: Record a non-committing checkpoint**

Run:

```bash
rtk git diff -- tests/test_reproduction_release.py
rtk git diff --cached -- tests/test_reproduction_release.py
```

Expected: only the capsule-name fix, compact-layout test, and fixture path
changes are new; do not commit because this file already contains user-staged
work and the ignored capsule cannot participate in the same atomic commit.

---

### Task 2: Move the immutable inputs and rewrite the manifest paths

**Files:**
- Move: `ntcir19-esg-validity-layer/artifacts/aicup_corpus_index.json.gz`
- Move: `ntcir19-esg-validity-layer/{splits,predictions,runs}/`
- Move: `ntcir19-esg-validity-layer/{splits_en,runs_en}/`
- Move: `ntcir19-esg-validity-layer/{splits_fr,runs_fr}/`
- Move: `ntcir19-esg-validity-layer/{splits_ja,runs_ja}/`
- Move: `ntcir19-esg-validity-layer/{splits_ko,runs_ko}/`
- Modify mechanically: `ntcir19-esg-validity-layer/release_manifest.json`
- Test: `tests/test_reproduction_release.py`

**Interfaces:**
- Consumes: the twelve old first-level input directories and existing path-keyed SHA-256 manifest
- Produces: `inputs/<corpus>/{splits,runs}` plus `inputs/aicup_zh/corpus_index.json.gz`, with unchanged file bytes and rewritten manifest keys

- [ ] **Step 1: Confirm the exact move sources before mutation**

Run:

```bash
rtk ls -ld ntcir19-esg-validity-layer/artifacts \
  ntcir19-esg-validity-layer/predictions \
  ntcir19-esg-validity-layer/runs \
  ntcir19-esg-validity-layer/runs_en \
  ntcir19-esg-validity-layer/runs_fr \
  ntcir19-esg-validity-layer/runs_ja \
  ntcir19-esg-validity-layer/runs_ko \
  ntcir19-esg-validity-layer/splits \
  ntcir19-esg-validity-layer/splits_en \
  ntcir19-esg-validity-layer/splits_fr \
  ntcir19-esg-validity-layer/splits_ja \
  ntcir19-esg-validity-layer/splits_ko
```

Expected: all twelve directories exist exactly once beneath the capsule.

- [ ] **Step 2: Create corpus parents and move directories without copying bytes**

Run explicit moves from the parent repository root:

```bash
rtk mkdir -p ntcir19-esg-validity-layer/inputs/aicup_zh \
  ntcir19-esg-validity-layer/inputs/mlpromise_en \
  ntcir19-esg-validity-layer/inputs/mlpromise_fr \
  ntcir19-esg-validity-layer/inputs/mlpromise_ja \
  ntcir19-esg-validity-layer/inputs/mlpromise_ko
rtk mv ntcir19-esg-validity-layer/runs \
  ntcir19-esg-validity-layer/inputs/aicup_zh/runs
rtk mkdir -p ntcir19-esg-validity-layer/inputs/aicup_zh/runs/main
rtk mv ntcir19-esg-validity-layer/predictions \
  ntcir19-esg-validity-layer/inputs/aicup_zh/runs/main/predictions
rtk mv ntcir19-esg-validity-layer/splits \
  ntcir19-esg-validity-layer/inputs/aicup_zh/splits
rtk mv ntcir19-esg-validity-layer/artifacts/aicup_corpus_index.json.gz \
  ntcir19-esg-validity-layer/inputs/aicup_zh/corpus_index.json.gz
rtk rmdir ntcir19-esg-validity-layer/artifacts
rtk mv ntcir19-esg-validity-layer/runs_en \
  ntcir19-esg-validity-layer/inputs/mlpromise_en/runs
rtk mv ntcir19-esg-validity-layer/splits_en \
  ntcir19-esg-validity-layer/inputs/mlpromise_en/splits
rtk mv ntcir19-esg-validity-layer/runs_fr \
  ntcir19-esg-validity-layer/inputs/mlpromise_fr/runs
rtk mv ntcir19-esg-validity-layer/splits_fr \
  ntcir19-esg-validity-layer/inputs/mlpromise_fr/splits
rtk mv ntcir19-esg-validity-layer/runs_ja \
  ntcir19-esg-validity-layer/inputs/mlpromise_ja/runs
rtk mv ntcir19-esg-validity-layer/splits_ja \
  ntcir19-esg-validity-layer/inputs/mlpromise_ja/splits
rtk mv ntcir19-esg-validity-layer/runs_ko \
  ntcir19-esg-validity-layer/inputs/mlpromise_ko/runs
rtk mv ntcir19-esg-validity-layer/splits_ko \
  ntcir19-esg-validity-layer/inputs/mlpromise_ko/splits
```

`rmdir` is intentionally limited to the now-empty `artifacts` directory; it
must fail rather than remove anything if an unexpected file remains.

- [ ] **Step 3: Rewrite only manifest artifact path keys**

Perform these ordered replacements in `release_manifest.json` using a bulk
mechanical edit. Language-specific prefixes precede generic prefixes:

```text
artifacts/aicup_corpus_index.json.gz -> inputs/aicup_zh/corpus_index.json.gz
predictions/                         -> inputs/aicup_zh/runs/main/predictions/
runs_en/                             -> inputs/mlpromise_en/runs/
runs_fr/                             -> inputs/mlpromise_fr/runs/
runs_ja/                             -> inputs/mlpromise_ja/runs/
runs_ko/                             -> inputs/mlpromise_ko/runs/
runs/                                -> inputs/aicup_zh/runs/
splits_en/                           -> inputs/mlpromise_en/splits/
splits_fr/                           -> inputs/mlpromise_fr/splits/
splits_ja/                           -> inputs/mlpromise_ja/splits/
splits_ko/                           -> inputs/mlpromise_ko/splits/
splits/                              -> inputs/aicup_zh/splits/
```

Use a deterministic Perl edit, which changes paths but not JSON formatting:

```bash
rtk perl -0pi -e 's#artifacts/aicup_corpus_index\.json\.gz#inputs/aicup_zh/corpus_index.json.gz#g; s#"predictions/#"inputs/aicup_zh/runs/main/predictions/#g; s#"runs_en/#"inputs/mlpromise_en/runs/#g; s#"runs_fr/#"inputs/mlpromise_fr/runs/#g; s#"runs_ja/#"inputs/mlpromise_ja/runs/#g; s#"runs_ko/#"inputs/mlpromise_ko/runs/#g; s#"runs/#"inputs/aicup_zh/runs/#g; s#"splits_en/#"inputs/mlpromise_en/splits/#g; s#"splits_fr/#"inputs/mlpromise_fr/splits/#g; s#"splits_ja/#"inputs/mlpromise_ja/splits/#g; s#"splits_ko/#"inputs/mlpromise_ko/splits/#g; s#"splits/#"inputs/aicup_zh/splits/#g' ntcir19-esg-validity-layer/release_manifest.json
```

- [ ] **Step 4: Verify counts and that all manifest artifact keys moved**

Run:

```bash
rtk rg -c '"inputs/' ntcir19-esg-validity-layer/release_manifest.json
rtk rg -n '"(artifacts|predictions|runs(_en|_fr|_ja|_ko)?|splits(_en|_fr|_ja|_ko)?)/' ntcir19-esg-validity-layer/release_manifest.json
rtk proxy find ntcir19-esg-validity-layer/inputs -type f | rtk wc -l
```

Expected: 311 `inputs/` artifact keys, no old artifact keys, and 311 files
under `inputs/`.

---

### Task 3: Route all runtime consumers through the corpus-first layout

**Files:**
- Modify: `ntcir19-esg-validity-layer/reproduce.py`
- Modify: `ntcir19-esg-validity-layer/paper/data.py`
- Modify: `ntcir19-esg-validity-layer/paper/corpus.py`
- Modify: `ntcir19-esg-validity-layer/paper/artifacts.py`
- Modify: `ntcir19-esg-validity-layer/paper/provenance.py`
- Modify: `ntcir19-esg-validity-layer/analysis/load.py`
- Modify: `ntcir19-esg-validity-layer/analysis/audit.py`
- Modify: `ntcir19-esg-validity-layer/analysis/__main__.py`
- Modify: `ntcir19-esg-validity-layer/analysis/legality_cost.py`
- Modify: `ntcir19-esg-validity-layer/analysis/multilingual_mechanism.py`
- Modify: `ntcir19-esg-validity-layer/analysis/external_arms.py`
- Test: `tests/test_reproduction_release.py`

**Interfaces:**
- Consumes: `inputs/<corpus>/` paths created in Task 2
- Produces: unchanged `verify_inputs(root: Path = ROOT) -> dict` and reproduction CLI behavior using the new roots

- [ ] **Step 1: Define the new AI CUP paths in `paper.data`**

Use these constants and the split path in `data_checksum`:

```python
REPO_ROOT = Path(__file__).resolve().parent.parent
INPUTS_ROOT = REPO_ROOT / "inputs"
AICUP_ROOT = INPUTS_ROOT / "aicup_zh"
CORPUS_INDEX_PATH = AICUP_ROOT / "corpus_index.json.gz"
AICUP_SPLITS_DIR = AICUP_ROOT / "splits"
AICUP_PRIMARY_RUN_ROOT = AICUP_ROOT / "runs" / "main"

# In data_checksum:
split = AICUP_SPLITS_DIR / "pdf_group_seed42.json"
```

- [ ] **Step 2: Make `paper.corpus` the corpus path registry**

Set the path fields exactly as follows while preserving descriptions and lazy
loaders:

```python
"aicup_zh": {
    "splits_dir": "inputs/aicup_zh/splits",
    "decisions_root": "inputs/aicup_zh/runs/main",
},
"mlpromise_en": {
    "splits_dir": "inputs/mlpromise_en/splits",
    "decisions_root": "inputs/mlpromise_en/runs",
},
"mlpromise_fr": {
    "splits_dir": "inputs/mlpromise_fr/splits",
    "decisions_root": "inputs/mlpromise_fr/runs",
},
"mlpromise_ja": {
    "splits_dir": "inputs/mlpromise_ja/splits",
    "decisions_root": "inputs/mlpromise_ja/runs",
},
"mlpromise_ko": {
    "splits_dir": "inputs/mlpromise_ko/splits",
    "decisions_root": "inputs/mlpromise_ko/runs",
},
```

Retain each existing `load`, `probs_dir`, and `description` entry.

- [ ] **Step 3: Make the primary analysis root explicit**

In `analysis/load.py`, import `AICUP_PRIMARY_RUN_ROOT` and set:

```python
REAL_ROOT = AICUP_PRIMARY_RUN_ROOT
```

In `analysis/audit.py`, import `AICUP_SPLITS_DIR` and set:

```python
SPLITS_DIR = AICUP_SPLITS_DIR
```

Update `analysis/__main__.py` help text to say the default is the curated AI
CUP primary run and that a custom root must hold `predictions/`.

- [ ] **Step 4: Update Chinese and multilingual arm roots**

In `analysis/legality_cost.py`, replace the seven `Arm.root` values with:

```python
"inputs/aicup_zh/runs/main"
"inputs/aicup_zh/runs/structural"
"inputs/aicup_zh/runs/deberta_v2_320m/lambda_0.0"
"inputs/aicup_zh/runs/deberta_v2_320m/lambda_0.3"
"inputs/aicup_zh/runs/electra_180g_large/lambda_0.0"
"inputs/aicup_zh/runs/electra_180g_large/lambda_0.3"
"inputs/aicup_zh/runs/rbt_base"
```

Remove the empty-root special case from `arm_root`:

```python
def arm_root(arm, root=REPO_ROOT) -> Path:
    return Path(root) / arm.root
```

In `analysis/multilingual_mechanism.py`, use:

```python
ARMS = {
    "aicup_zh": "inputs/aicup_zh/runs/main",
    "mlpromise_en": "inputs/mlpromise_en/runs/roberta_large/lambda_0.0",
    "mlpromise_fr": "inputs/mlpromise_fr/runs/xlm_roberta_large/lambda_0.0",
    "mlpromise_ja": "inputs/mlpromise_ja/runs/xlm_roberta_large/lambda_0.0",
    "mlpromise_ko": "inputs/mlpromise_ko/runs/xlm_roberta_large/lambda_0.0",
}
```

- [ ] **Step 5: Refactor external arms to use corpus identifiers**

Import `decisions_root` beside `splits_dir`, define:

```python
CORPORA = (
    ("mlpromise_en", "English"),
    ("mlpromise_fr", "French"),
    ("mlpromise_ja", "Japanese"),
    ("mlpromise_ko", "Korean"),
)

ORDER = {
    "mlpromise_en": (
        "roberta_large",
        "deberta_v3_large",
        "electra_large_discriminator",
        "roberta_base",
    ),
}
```

In `build_report`, iterate `for corpus, name in CORPORA`, compute
`directory = decisions_root(corpus)`, use `ORDER.get(corpus, DEFAULT_ORDER)`,
and delete the now-redundant `CORPUS_IDS` mapping.

- [ ] **Step 6: Make `reproduce.py` inventory the complete input boundary**

Replace the corpus-index constant and inventory helper with:

```python
INPUTS_ROOT = ROOT / "inputs"
CORPUS_INDEX_PATH = INPUTS_ROOT / "aicup_zh" / "corpus_index.json.gz"


def _actual_artifact_paths(root: Path) -> set[str]:
    inputs_root = root / "inputs"
    return {
        path.relative_to(root).as_posix()
        for path in inputs_root.rglob("*")
        if path.is_file()
    }
```

In `verify_inputs`, call `_corpus_index_has_raw_text(root /
"inputs/aicup_zh/corpus_index.json.gz")`. Keep missing/extra, checksum,
reference, raw-text, and prediction-count checks unchanged.

- [ ] **Step 7: Update stale internal path examples**

Change path examples and comments in `paper/artifacts.py`,
`paper/provenance.py`, `paper/corpus.py`, and analysis module docstrings to the
new `inputs/<corpus>/...` names. Do not alter function signatures or generated
table text.

- [ ] **Step 8: Run the fast contract suite**

Run:

```bash
rtk pytest -q \
  tests/test_reproduction_release.py::test_release_groups_all_curated_inputs_by_corpus \
  tests/test_reproduction_release.py::test_release_verifies_the_exact_curated_input_capsule \
  tests/test_reproduction_release.py::test_release_rejects_an_artifact_whose_bytes_do_not_match_the_manifest
```

Expected: 3 passed. A manifest inventory failure means a path was missed; a
checksum failure means an artifact's bytes changed and the migration must stop
for investigation.

---

### Task 4: Document and fully verify the standalone capsule

**Files:**
- Modify: `ntcir19-esg-validity-layer/README.md`
- Test: `tests/test_reproduction_release.py`

**Interfaces:**
- Consumes: the working corpus-first runtime and manifest from Tasks 2–3
- Produces: user-facing layout documentation and evidence that all eight manuscript tables remain byte-identical

- [ ] **Step 1: Add the curated-input layout to the README**

After the Quick Start section, add:

```markdown
## Repository layout

All immutable reproduction inputs live under `inputs/`, grouped by corpus.
Each corpus owns its split manifests and run predictions; the AI CUP corpus
also carries the text-free corpus index used for the published audit.

```text
inputs/<corpus>/splits/
inputs/<corpus>/runs/<arm>/predictions/
inputs/aicup_zh/corpus_index.json.gz
```

`analysis/` regenerates the results, `paper/` contains the shared schema and
artifact readers, and `reference/` contains the manuscript-facing files used
for byte comparison.
```

- [ ] **Step 2: Scan for obsolete first-level input paths**

Run:

```bash
rtk rg -n '\b(runs_en|runs_fr|runs_ja|runs_ko|splits_en|splits_fr|splits_ja|splits_ko)\b|ROOT / "artifacts"|REPO_ROOT / "splits"|REAL_ROOT = REPO_ROOT$' ntcir19-esg-validity-layer --glob '*.py' --glob '*.md'
```

Expected: no matches. Any match must be updated or deliberately explained if
it describes historical source-workspace provenance rather than a capsule path.

- [ ] **Step 3: Verify the final first-level directories**

Run:

```bash
rtk proxy find ntcir19-esg-validity-layer -mindepth 1 -maxdepth 1 -type d -printf '%f\n' | sort
```

Expected:

```text
analysis
figures
inputs
paper
reference
```

- [ ] **Step 4: Run fast input verification directly**

Run:

```bash
rtk python ntcir19-esg-validity-layer/reproduce.py --verify-inputs
```

Expected final JSON:

```json
{"artifact_files": 311, "prediction_files": 300, "raw_text_files": 0, "source_commit": "babd85e61368037d235607ac5ebc798378a7ef75", "status": "ok"}
```

- [ ] **Step 5: Run full reproduction outside the capsule**

Create a temporary directory with `mktemp -d`, pass its explicit path to the
command, and remove only that validated temporary directory after the test:

```bash
rtk proxy bash -lc 'release_check_dir=$(mktemp -d); python ntcir19-esg-validity-layer/reproduce.py --output-dir "$release_check_dir/outputs"; release_status=$?; test -n "$release_check_dir" && test "$release_check_dir" != / && rm -rf -- "$release_check_dir"; exit "$release_status"'
```

Expected final JSON:

```json
{"figure_source_verified": true, "status": "ok", "tables_verified": 8}
```

- [ ] **Step 6: Run the complete public contract test file**

Run:

```bash
rtk pytest -q tests/test_reproduction_release.py
```

Expected: 4 passed.

- [ ] **Step 7: Inspect the final working-tree delta without changing unrelated state**

Run:

```bash
rtk git status --short
rtk git diff -- tests/test_reproduction_release.py
rtk git diff --cached -- tests/test_reproduction_release.py
```

Expected: existing manuscript and `.gitignore` changes remain present and
untouched; the contract test contains only the planned capsule-path changes.
Because the capsule is intentionally ignored, report its changed layout and
verification results explicitly in the handoff rather than claiming those
files are tracked by the parent repository.
