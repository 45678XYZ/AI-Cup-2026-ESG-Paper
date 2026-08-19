"""Audit numbers the paper states about the data.

Asserted against the real dataset rather than a fixture: these are the values
Table 1 prints, and a silent change in dataset/ must fail here rather than
surface as a wrong number in the paper.
"""

from analysis.audit import dataset_audit, full_audit, load_test, split_audit
from paper.data import load_dev

DEV = load_dev()
TEST = load_test()
DATA = dataset_audit(DEV, TEST)
SPLITS = split_audit(dev=DEV)


def test_development_counts():
    d = DATA["development"]
    assert d["paragraphs"] == 2000
    assert d["pdfs"] == 49
    assert d["companies"] == 50
    assert d["labelled"] is True


def test_test_set_counts_and_absent_labels():
    t = DATA["test"]
    assert t["paragraphs"] == 2000
    assert t["pdfs"] == 49
    assert t["companies"] == 50
    # The competition test split ships no labels, so every label-derived cell
    # of Table 1 must stay empty for it rather than be silently filled with a
    # development number.
    assert t["labelled"] is False


def test_pdf_overlap_is_total():
    assert DATA["pdf_overlap"] == {"n_shared": 49, "dev_only": 0, "test_only": 0}


def test_paragraphs_per_pdf_spread():
    assert DATA["development"]["paragraphs_per_pdf"] == {
        "min": 4, "median": 39, "max": 91,
    }


def test_only_fifteen_of_seventeen_states_occur():
    d = DATA["development"]
    assert d["legal_states_observed"] == 15
    assert d["invalid_rows"] == 0


def test_misleading_has_two_rows_in_two_reports():
    rows = SPLITS["misleading_rows"]
    assert [r["id"] for r in rows] == ["10017", "11836"]
    assert len({r["pdf_url"] for r in rows}) == 2


def test_the_single_cross_split_duplicate_is_reported():
    assert DATA["duplicates"]["within_dev"] == []
    assert DATA["duplicates"]["dev_test"] == [["10404", "12550"]]


def test_most_calibration_partitions_cannot_see_misleading():
    cov = SPLITS["calibration_without_misleading"]
    assert cov["n_rotations"] == 30
    assert cov["n_without"] == 18
    assert cov["per_split"]["pdf_group_seed42"] == [1, 0, 0, 1, 0]


def test_full_audit_carries_the_data_checksum():
    audit = full_audit()
    assert audit["data_checksum"].startswith("sha256:")
    assert set(audit) >= {"data_checksum", "development", "test", "splits"}
