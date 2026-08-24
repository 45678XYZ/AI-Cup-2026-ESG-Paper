"""Selecting a corpus, and the guard that makes selecting the wrong one safe.

``--corpus`` is the only switch that decides which four hundred or two thousand
rows a 30-fit campaign trains on. Getting it wrong has no visible symptom at
launch: the model loads, the folds parse, and the fits run to completion
against the wrong data. The checksum comparison in ``run_training`` is what
turns that into an immediate failure, so it is tested here rather than trusted.
"""

import json

import pytest

from paper import corpus
from paper.data import data_checksum as zh_checksum
from paper.data import load_dev
from paper.data_en import load_english


def test_the_two_corpora_are_the_ones_the_study_has():
    assert set(corpus.CORPORA) == {"aicup_zh", "mlpromise_en"}
    assert corpus.DEFAULT == "aicup_zh"


def test_the_default_is_the_frozen_study():
    """Every existing invocation keeps its meaning. A default that had to be
    passed would make the frozen path the special case."""
    assert corpus.load_rows() == load_dev()
    assert corpus.splits_dir() == "splits"
    assert corpus.probs_dir() == "probs"


def test_english_artifacts_land_somewhere_else():
    """Contract section 4 names ``splits/`` and ``probs/``. Writing English
    folds or bundles into them would be silent and unrecoverable from the tree."""
    assert corpus.splits_dir("mlpromise_en") == "splits_en"
    assert corpus.probs_dir("mlpromise_en") == "probs_en"
    assert corpus.splits_dir("mlpromise_en") != corpus.splits_dir("aicup_zh")
    assert corpus.probs_dir("mlpromise_en") != corpus.probs_dir("aicup_zh")


def test_an_unknown_corpus_is_refused():
    with pytest.raises(ValueError, match="unknown corpus"):
        corpus.load_rows("mlpromise_fr")


def test_both_corpora_speak_the_frozen_vocabulary():
    """The point of translating English at load time: nothing downstream of
    ``load_rows`` can tell which corpus it was handed."""
    from paper.labels import EVAL_FIELDS, FIELDS, is_valid_tuple

    for name in corpus.CORPORA:
        rows = corpus.load_rows(name)
        for row in rows:
            for field in FIELDS:
                assert row[field] in EVAL_FIELDS[field], (name, field, row["id"])
            assert is_valid_tuple(*(row[field] for field in FIELDS)), (name, row["id"])


def test_the_two_corpora_share_no_row_id():
    """A wrong-corpus join must return nothing rather than something."""
    assert not {r["id"] for r in load_dev()} & {r["id"] for r in load_english()}


def test_a_split_manifest_identifies_the_corpus_it_was_built_from():
    """The guard ``run_training`` relies on.

    It does not read a corpus name -- it compares checksums -- so this asserts
    the property that makes the comparison decisive: the two corpora cannot
    produce the same ``data_checksum``.
    """
    from paper.data_en import data_checksum as en_checksum

    assert zh_checksum() != en_checksum()


@pytest.mark.parametrize("name", sorted(corpus.CORPORA))
def test_the_shipped_manifests_match_their_corpus(name):
    """Each committed split manifest states the checksum of the rows it folds.

    A manifest that matched neither corpus would be unusable; one that matched
    the other would be worse.
    """
    from pathlib import Path

    from paper.data import REPO_ROOT

    directory = Path(REPO_ROOT) / corpus.CORPORA[name]["splits_dir"]
    manifests = sorted(directory.glob("*.json"))
    assert manifests, directory

    expected = data_checksum_of(name)
    for path in manifests:
        with open(path, encoding="utf-8") as f:
            split = json.load(f)
        assert split["data_checksum"] == expected, path.name


def data_checksum_of(name: str) -> str:
    if name == "aicup_zh":
        return zh_checksum()
    from paper.data_en import data_checksum as en_checksum

    return en_checksum()


@pytest.mark.parametrize("name", sorted(corpus.CORPORA))
def test_every_row_is_tested_exactly_once_per_seed(name):
    """The rotating design's defining property, asserted for both corpora.

    Nine reports fold less evenly than forty-nine, so this is worth checking on
    the English manifests rather than assuming the generator carried over.
    """
    from pathlib import Path

    from paper.data import REPO_ROOT

    rows = corpus.load_rows(name)
    directory = Path(REPO_ROOT) / corpus.CORPORA[name]["splits_dir"]
    for path in sorted(directory.glob("*.json")):
        with open(path, encoding="utf-8") as f:
            split = json.load(f)
        tested = [i for rot in split["rotations"] for i in rot["test_ids"]]
        assert len(tested) == len(rows), path.name
        assert set(tested) == {r["id"] for r in rows}, path.name


def test_the_english_document_disjoint_folds_are_uneven_and_that_is_the_data():
    """Nine reports into five folds cannot be balanced, and the paper has to
    say so rather than present the fold sizes as a design choice."""
    from pathlib import Path

    from paper.data import REPO_ROOT

    path = Path(REPO_ROOT) / "splits_en" / "pdf_group_seed42.json"
    with open(path, encoding="utf-8") as f:
        split = json.load(f)
    sizes = sorted(len(rot["test_ids"]) for rot in split["rotations"])
    assert sum(sizes) == 400
    assert max(sizes) - min(sizes) > 10, sizes
