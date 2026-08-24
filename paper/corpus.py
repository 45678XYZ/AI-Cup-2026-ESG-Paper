"""Which set of labelled rows a run is about, and where its artifacts live.

Two corpora reach the pipeline. They share the label vocabulary -- the English
one is translated into it at load time, see ``paper/labels_en.py`` -- so
everything downstream of ``load_rows`` is language-blind, and this module is
the only place that has to name them.

Every path here exists to keep the two apart on disk. The frozen study's
directories are named by contract section 4 and cannot move, and the English
artifacts are named identically within them: a decision run writes
``predictions/pdf_group_seed42_M0.csv.gz`` for either corpus. Defaulting the
English output to the repository root would therefore not merely mix the two,
it would overwrite the frozen predictions with the same filenames.

The split manifest carries ``data_checksum``, and both ``run_training`` and
``run_decisions`` refuse a manifest whose checksum does not match the rows they
loaded. That guard is what makes ``--corpus`` safe: naming the wrong one fails
before any GPU time is spent rather than training or scoring one language
against another's folds.
"""

from paper.data import load_dev


def _load_english():
    # Imported lazily: every Chinese training run would otherwise read and
    # translate the English release for nothing.
    from paper.data_en import load_english

    return load_english()


CORPORA = {
    "aicup_zh": {
        "load": load_dev,
        "splits_dir": "splits",
        "probs_dir": "probs",
        # The repository root: contract section 4 puts the frozen study's
        # predictions/ and results/ at the top level.
        "decisions_root": ".",
        "description": "AI CUP VeriPromiseESG, 2,000 rows, 49 reports (frozen study)",
    },
    "mlpromise_en": {
        "load": _load_english,
        "splits_dir": "splits_en",
        "probs_dir": "probs_en",
        "decisions_root": "runs_en",
        "description": "ML-Promise English, 400 rows, 9 reports (external replication)",
    },
}

DEFAULT = "aicup_zh"


def _entry(name: str) -> dict:
    if name not in CORPORA:
        raise ValueError(f"unknown corpus {name!r}; expected one of {sorted(CORPORA)}")
    return CORPORA[name]


def load_rows(name: str = DEFAULT) -> list[dict]:
    """The labelled rows of one corpus, in the frozen label vocabulary."""
    return _entry(name)["load"]()


def splits_dir(name: str = DEFAULT) -> str:
    return _entry(name)["splits_dir"]


def probs_dir(name: str = DEFAULT) -> str:
    return _entry(name)["probs_dir"]


def decisions_root(name: str = DEFAULT) -> str:
    """Directory holding ``predictions/`` and ``results/`` for this corpus."""
    return _entry(name)["decisions_root"]
