"""The English release, as loaded, and the properties the pre-registration rests on.

The go/no-go for this whole arm was whether the English annotation carries the
hierarchy. The published description of ML-Promise lists no ``N/A`` value and
states no dependency between the fields, so the answer had to come from the
data. It is asserted here rather than written down, because a fact that decides
whether an arm is worth running should fail loudly if a re-release changes it.
"""

import json

import pytest

from paper.data_en import (
    ENGLISH_PATH,
    audit,
    data_checksum,
    index_by_id,
    load_english,
    load_english_native,
    normalise,
    provenance,
)
from paper.labels import FIELDS, is_valid_tuple

ROWS = load_english()          # frozen vocabulary; what the pipeline consumes
NATIVE = load_english_native()  # the release's own spelling
AUDIT = audit()


def test_the_release_is_four_hundred_labelled_rows():
    assert AUDIT["n_rows"] == 400
    assert len(ROWS) == 400


def test_every_gold_row_obeys_the_hierarchy():
    """The go/no-go. 400 of 400, the same zero-violation property the AI CUP
    annotation has -- and the reason the 17-state machinery transfers at all."""
    assert AUDIT["hierarchy_violations"] == []


def test_the_hierarchy_holds_branch_by_branch():
    """Not just "no illegal tuple" but the three implications separately, so a
    release that broke only one of them says which."""
    ps_no = [r for r in ROWS if r["promise_status"] == "No"]
    assert ps_no
    for row in ps_no:
        assert row["verification_timeline"] == "N/A", row["id"]
        assert row["evidence_status"] == "N/A", row["id"]
        assert row["evidence_quality"] == "N/A", row["id"]

    no_evidence = [r for r in ROWS
                   if r["promise_status"] == "Yes" and r["evidence_status"] == "No"]
    assert no_evidence
    for row in no_evidence:
        assert row["evidence_quality"] == "N/A", row["id"]

    with_evidence = [r for r in ROWS
                     if r["promise_status"] == "Yes" and r["evidence_status"] == "Yes"]
    assert with_evidence
    for row in with_evidence:
        assert row["evidence_quality"] in ("Clear", "Not Clear", "Misleading"), row["id"]


def test_a_not_applicable_timeline_means_no_promise():
    """The property that keeps the collapsed ``evidence_status`` unambiguous."""
    for row in ROWS:
        if row["verification_timeline"] == "N/A":
            assert row["promise_status"] == "No", row["id"]


def test_the_stray_trailing_space_is_stripped():
    """One row spells the timeline ``"2 to 5 years "``.

    Left alone it is a sixth class of a five-class field: present in gold,
    predicted by nothing, and scored -- macro-F1 averages over classes, so it
    would take a fixed slice off the field under every method alike and read as
    a property of the task. Asserted from the raw file so this test starts
    failing if the release is fixed upstream, which is the moment to drop the
    normalisation rather than keep working around nothing.
    """
    with open(ENGLISH_PATH, encoding="utf-8") as f:
        raw = json.load(f)
    untrimmed = [r for r in raw
                 if any(str(r[f]) != str(r[f]).strip() for f in FIELDS)]
    assert untrimmed, (
        "the release no longer carries an untrimmed label; the strip in "
        "paper.data_en.normalise is now guarding nothing and can go")

    timelines = set(AUDIT["labels_native"]["verification_timeline"])
    assert len(timelines) == 5, sorted(timelines)
    assert all(t == t.strip() for t in timelines)


def test_row_ids_come_from_content_not_position():
    """A re-release that reorders the file must not renumber every row.

    Position-derived ids would make every artifact keyed to an ordering the
    authors never promised to keep.
    """
    first, last = NATIVE[0], NATIVE[-1]
    with open(ENGLISH_PATH, encoding="utf-8") as f:
        raw = json.load(f)
    reversed_ids = {normalise(r)["id"] for r in reversed(raw)}
    assert first["id"] in reversed_ids
    assert last["id"] in reversed_ids


def test_row_ids_are_unique_and_cannot_collide_with_ai_cup_ids():
    """AI CUP ids are decimal strings like ``"10001"``. An English id that
    looked like one would let a wrong-language join return rows instead of
    failing."""
    ids = [r["id"] for r in ROWS]
    assert len(set(ids)) == len(ids)
    assert all(i.startswith("en") for i in ids)
    assert not any(i.isdigit() for i in ids)
    assert len(index_by_id(ROWS)) == 400


def test_editing_a_paragraph_moves_only_that_row_id():
    edited = [dict(r) for r in json.load(open(ENGLISH_PATH, encoding="utf-8"))]
    edited[7]["data"] = edited[7]["data"] + " (edited)"
    moved = [i for i, (a, b) in enumerate(zip(NATIVE, [normalise(r) for r in edited]))
             if a["id"] != b["id"]]
    assert moved == [7]


def test_there_are_nine_source_reports():
    """The number behind every interval this arm could compute.

    Forty-nine in the Chinese study, nine here, and the resampling unit is the
    report. This is why the pre-registration declares the English official-style
    contrasts descriptive in advance rather than reporting wide intervals
    afterwards and calling them inconclusive.
    """
    assert AUDIT["n_source_reports"] == 9
    assert AUDIT["rows_per_report"] == {"min": 30, "max": 58}


def test_fifteen_of_the_seventeen_legal_states_occur_in_gold():
    """Two legal states are never annotated, both of them ``Misleading`` ones.

    The Chinese study reports the same shape of fact and treats it as a
    statement about class rarity rather than about the decoder, and so must this.
    """
    assert AUDIT["distinct_gold_tuples"] == 15
    seen = {tuple(r[f] for f in FIELDS) for r in ROWS}
    assert all(is_valid_tuple(*t) for t in seen)


def test_misleading_is_too_rare_to_support_any_claim():
    """Four rows against the Chinese study's two. Still four."""
    assert AUDIT["labels_canonical"]["evidence_quality"]["Misleading"] == 4


def test_the_checksum_is_stable_and_distinct_from_the_chinese_one():
    from paper.data import data_checksum as zh_checksum

    assert data_checksum(ROWS) == data_checksum(load_english())
    assert data_checksum(ROWS) != zh_checksum()
    assert AUDIT["data_checksum"] == data_checksum(ROWS)


def test_the_vendored_file_matches_its_recorded_hash():
    """The file is redistributed under CC BY-NC-SA 4.0 and is never rewritten;
    all normalisation happens at load. If the bytes move, the provenance record
    stops describing what is in the tree."""
    from paper.data import file_sha256

    prov = provenance()
    assert file_sha256(ENGLISH_PATH) == prov["sha256"]
    assert prov["n_rows"] == 400
    assert prov["licence"] == "CC BY-NC-SA 4.0"


@pytest.mark.parametrize("field", FIELDS)
def test_no_label_is_empty_or_padded(field):
    values = set(AUDIT["labels_canonical"][field])
    assert "" not in values
    assert all(v == v.strip() for v in values)


def test_the_redistributed_data_carries_its_attribution_where_it_can_be_found():
    """CC BY-NC-SA 4.0 asks a redistributor to state creator, source, licence
    and modifications. This repository is public, so cloning it delivers the
    dataset from us rather than from its authors, and the obligation applies.

    Asserted rather than trusted because the notice is the one artifact here
    that no other test would miss: nothing imports it, nothing computes from
    it, and a rename or a stale claim would be invisible until someone asked.
    """
    from paper.data_en import REPO_ROOT

    notice = REPO_ROOT / "dataset" / "mlpromise_english.NOTICE"
    assert notice.exists(), "the licence requires an attribution notice"
    text = notice.read_text(encoding="utf-8")

    prov = provenance()
    for required in (prov["licence"], prov["source_drive_folder"],
                     "Seki", "EMNLP 2025", "mlpromise_english.json"):
        assert required in text, required

    # The notice claims the file is unmodified. That claim is the reason the
    # recorded sha256 means anything, so it has to match what the loader does.
    assert "None." in text
    from paper.data import file_sha256

    assert file_sha256(ENGLISH_PATH) == prov["sha256"]

    readme = (REPO_ROOT / "README.md").read_text(encoding="utf-8")
    assert "mlpromise_english.NOTICE" in readme, (
        "the notice exists but nothing at the top level points at it, so a "
        "reader of the repository would not find it")
