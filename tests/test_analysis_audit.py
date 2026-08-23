"""Audit numbers the paper states about the data.

Asserted against the real dataset rather than a fixture: these are the values
Table 1 prints, and a silent change in dataset/ must fail here rather than
surface as a wrong number in the paper.
"""

from analysis.audit import (
    company_key, dataset_audit, full_audit, load_test, split_audit,
)
from paper.data import load_dev

DEV = load_dev()
TEST = load_test()
DATA = dataset_audit(DEV, TEST)
SPLITS = split_audit(dev=DEV)


def test_development_counts():
    d = DATA["development"]
    assert d["paragraphs"] == 2000
    assert d["pdfs"] == 49
    assert d["companies"] == 49
    assert d["labelled"] is True


def test_test_set_counts_and_absent_labels():
    t = DATA["test"]
    assert t["paragraphs"] == 2000
    assert t["pdfs"] == 49
    assert t["companies"] == 49
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


# --- what a document-disjoint split does and does not buy -------------------


def test_audit_records_whether_a_company_spans_more_than_one_report():
    """Reviewers ask whether the document-disjoint result would survive a
    company-disjoint split. In this corpus that is a property of the data, not
    an experiment: if no company contributes two reports, splitting by document
    already splits by company. Auditing it beats asserting it."""
    audit = dataset_audit()
    company = audit["company_structure"]
    assert company["companies"] >= 1
    assert company["tickers"] >= 1
    assert company["companies_in_multiple_reports"] == 0
    # the converse is not guaranteed and must be reported as it is
    assert company["reports_with_multiple_companies"] == 0


def test_audit_lists_the_dataset_fields_so_absent_metadata_is_checkable():
    """A temporal split needs a date. Recording the columns that exist makes
    "the release carries no temporal field" a verifiable statement rather than
    an excuse."""
    audit = dataset_audit()
    fields = audit["dataset_fields"]
    assert "pdf_url" in fields and "promise_status" in fields
    assert not [f for f in fields if f in ("year", "date", "published_at")]


def test_one_company_is_spelled_two_ways_and_is_counted_once():
    """The release spells one company ``Wistron`` and ``wistron``, for
    paragraphs of the same report. Counting raw strings gives 50 companies
    against 49 reports, which reads as though some report were shared -- the
    opposite of what the corpus supports, and the exact claim Table 1 is used
    to make. Both spellings appear in all three partitions, so this is a
    property of the release rather than of one file."""
    dev, test = load_dev(), load_test()
    for rows in (dev, test):
        raw = {r["company"] for r in rows}
        assert len(raw) == 50
        assert len({company_key(r) for r in rows}) == 49
        assert {"Wistron", "wistron"} <= raw


def test_normalising_the_company_does_not_create_a_shared_report():
    """The fix must not buy a correct count at the price of the claim it
    supports: if the two spellings pointed at different reports, merging them
    would make one company span two, and document-disjoint would stop implying
    company-disjoint."""
    company = dataset_audit()["company_structure"]
    assert company["companies"] == 49
    assert company["companies_in_multiple_reports"] == 0
    # The same bug had a second symptom, and a worse one: keying by the raw
    # string made the Wistron report look like a report covering two companies.
    # _company_structure reports that "as found rather than assumed", so the
    # audit was stating a data quirk that does not exist.
    assert company["reports_with_multiple_companies"] == 0
