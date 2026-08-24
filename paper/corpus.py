"""Which set of labelled rows a run is about.

Two corpora reach the training path. They share the label vocabulary -- the
English one is translated into it at load time, see ``paper/labels_en.py`` --
so everything downstream of ``load_rows`` is language-blind, and this module is
the only place that has to name them.

The split manifest already carries ``data_checksum``, and ``run_training``
refuses a manifest whose checksum does not match the rows it loaded. That guard
is what makes ``--corpus`` safe: pointing it at the wrong corpus fails before
any GPU time is spent rather than training on one language against another's
folds.
"""

from paper.data import load_dev

# Directory suffixes keep the two corpora's artifacts apart on disk. The
# Chinese entries are empty strings because contract section 4 names those
# paths and they cannot move.
CORPORA = {
    "aicup_zh": {
        "load": load_dev,
        "splits_dir": "splits",
        "probs_dir": "probs",
        "description": "AI CUP VeriPromiseESG, 2,000 rows, 49 reports (frozen study)",
    },
    "mlpromise_en": {
        "load": None,          # bound below; importing it eagerly would make
                               # every training run read the English file
        "splits_dir": "splits_en",
        "probs_dir": "probs_en",
        "description": "ML-Promise English, 400 rows, 9 reports (external replication)",
    },
}

DEFAULT = "aicup_zh"


def _load_english():
    from paper.data_en import load_english

    return load_english()


CORPORA["mlpromise_en"]["load"] = _load_english


def load_rows(name: str = DEFAULT) -> list[dict]:
    """The labelled rows of one corpus, in the frozen label vocabulary."""
    if name not in CORPORA:
        raise ValueError(f"unknown corpus {name!r}; expected one of {sorted(CORPORA)}")
    return CORPORA[name]["load"]()


def splits_dir(name: str = DEFAULT) -> str:
    return CORPORA[name]["splits_dir"]


def probs_dir(name: str = DEFAULT) -> str:
    return CORPORA[name]["probs_dir"]
