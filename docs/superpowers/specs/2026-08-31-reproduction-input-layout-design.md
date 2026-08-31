# Reproduction Capsule Input Layout Design

## Goal

Reduce the number of first-level directories in `ntcir19-esg-validity-layer`
without deleting or changing any released prediction, split, corpus-index, or
reference bytes. The capsule must remain CPU-only, self-contained, and runnable
with the existing `python reproduce.py` interface.

The current layout exposes sixteen first-level directories. Twelve of them are
curated reproduction inputs separated by artifact type or language. The new
layout groups all curated inputs first by corpus and then by purpose, leaving
five first-level directories: `analysis`, `paper`, `inputs`, `figures`, and
`reference`.

## Target layout

```text
ntcir19-esg-validity-layer/
├── analysis/
├── paper/
├── inputs/
│   ├── aicup_zh/
│   │   ├── corpus_index.json.gz
│   │   ├── splits/
│   │   └── runs/
│   │       ├── main/predictions/
│   │       ├── structural/predictions/
│   │       ├── deberta_v2_320m/
│   │       ├── electra_180g_large/
│   │       └── rbt_base/
│   ├── mlpromise_en/
│   │   ├── splits/
│   │   └── runs/<backbone>/<lambda>/predictions/
│   ├── mlpromise_fr/
│   ├── mlpromise_ja/
│   └── mlpromise_ko/
├── figures/
└── reference/
```

The four ML-Promise corpus directories use the same `splits/` and `runs/`
shape as each other. `aicup_zh/runs/main` is the frozen primary study. Its
additional screening and structural arms retain their existing names below the
same `runs/` directory.

## Path migration

| Current path | New path |
|---|---|
| `artifacts/aicup_corpus_index.json.gz` | `inputs/aicup_zh/corpus_index.json.gz` |
| `splits/` | `inputs/aicup_zh/splits/` |
| `predictions/` | `inputs/aicup_zh/runs/main/predictions/` |
| `runs/` | `inputs/aicup_zh/runs/` |
| `splits_en/` | `inputs/mlpromise_en/splits/` |
| `runs_en/` | `inputs/mlpromise_en/runs/` |
| `splits_fr/` | `inputs/mlpromise_fr/splits/` |
| `runs_fr/` | `inputs/mlpromise_fr/runs/` |
| `splits_ja/` | `inputs/mlpromise_ja/splits/` |
| `runs_ja/` | `inputs/mlpromise_ja/runs/` |
| `splits_ko/` | `inputs/mlpromise_ko/splits/` |
| `runs_ko/` | `inputs/mlpromise_ko/runs/` |

No compatibility symlinks or duplicate copies will be retained because they
would preserve the crowded first level and could let the two layouts drift.

## Code and manifest changes

`paper.data` will define the AI CUP corpus index, split directory, and primary
run root from the new layout. `paper.corpus` will be the single mapping from a
corpus identifier to its split and run roots. Analysis modules will consume
those mappings or new paths instead of the old `runs_*`, `splits_*`, and root
`predictions/` names.

`reproduce.py` will inventory every file below `inputs/` and compare that set
to the artifact keys in `release_manifest.json`. This preserves rejection of
missing, extra, and byte-modified inputs while removing the hard-coded list of
twelve old roots. The 311 manifest artifact keys will be rewritten to their
new paths; their SHA-256 values and the pinned source commit will remain
unchanged. Reference paths and hashes remain unchanged.

The README will explain the corpus-first layout. The parent repository's
`tests/test_reproduction_release.py` will point to the current
`ntcir19-esg-validity-layer` directory and its checksum-mismatch fixture will
use the new input layout.

## Error handling and compatibility

The public commands remain unchanged:

```bash
python reproduce.py --verify-inputs
python reproduce.py
python reproduce.py --output-dir PATH
```

Old internal artifact paths are intentionally not supported after migration.
Manifest validation remains the compatibility boundary: an incomplete move,
an obsolete duplicate input, or an unlisted file below `inputs/` must fail with
an inventory mismatch. Modified bytes must fail with a checksum mismatch.

## Verification

Before moving files, run the current fast verification and record that it
reports 311 artifacts and 300 prediction files. After migration:

1. Assert that the only first-level directories are `analysis`, `paper`,
   `inputs`, `figures`, and `reference` (excluding generated `outputs`).
2. Run `python reproduce.py --verify-inputs` and require the same artifact
   counts, source commit, and `status: ok`.
3. Run the full reproduction into a temporary output directory and require all
   eight tables plus the Figure 1 source verification to match `reference/`.
4. Run `pytest -q tests/test_reproduction_release.py` from the parent
   repository.
5. Confirm that no old first-level input directory remains.

These checks prove that the change only reorganizes paths and does not alter
the released evidence or regenerated manuscript results.
