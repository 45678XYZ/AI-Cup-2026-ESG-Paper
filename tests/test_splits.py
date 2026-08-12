from collections import Counter

from paper.data import load_dev
from paper.splits import build_split
from paper.train_config import N_FOLDS

ROWS = load_dev()
BY_ID = {r["id"]: r for r in ROWS}


def _split(protocol, seed=42):
    return build_split(protocol, seed, ROWS)


PDF = _split("pdf_group")
ROW = _split("row_strat")


def test_partitions_are_three_way_disjoint_and_exhaustive():
    for split in (PDF, ROW):
        for r in split["rotations"]:
            tr, ca, te = (set(r[f"{n}_ids"]) for n in ("train", "calibration", "test"))
            assert not (tr & ca) and not (tr & te) and not (ca & te)
            assert tr | ca | te == set(split["canonical_row_order"])


def test_every_row_is_tested_exactly_once():
    for split in (PDF, ROW):
        seen = [i for r in split["rotations"] for i in r["test_ids"]]
        assert Counter(seen) == Counter(split["canonical_row_order"])


def test_rotation_geometry():
    """Test = fold k, Calibration = fold k+1, Train = the other three."""
    for split in (PDF, ROW):
        assert [r["k"] for r in split["rotations"]] == list(range(N_FOLDS))
        for r in split["rotations"]:
            assert len(r["train_ids"]) > len(r["calibration_ids"])
            assert len(r["train_ids"]) > len(r["test_ids"])


def test_pdf_group_is_document_disjoint():
    for r in PDF["rotations"]:
        trp, cap, tep = (set(r[f"{n}_pdfs"]) for n in ("train", "calibration", "test"))
        assert not (trp & cap) and not (trp & tep) and not (cap & tep)
        assert r["checks"]["pdfs_disjoint"] is True


def test_row_strat_guarantees_same_document_coverage():
    """Contract §2 invariant 4: this is what makes 'seen report, unseen
    paragraph' a property of the protocol rather than of the dataset."""
    for r in ROW["rotations"]:
        cov = r["checks"]["same_document_coverage"]
        assert cov["satisfied"] and cov["violating_pdfs"] == []
        train_pdfs = {BY_ID[i]["pdf_url"] for i in r["train_ids"]}
        held_pdfs = {BY_ID[i]["pdf_url"] for i in r["calibration_ids"] + r["test_ids"]}
        assert held_pdfs <= train_pdfs


def test_row_strat_is_not_document_disjoint():
    """The two protocols must actually differ, or RQ3 has nothing to compare."""
    assert any(not r["checks"]["pdfs_disjoint"] for r in ROW["rotations"])


def test_declared_support_matches_the_data():
    for split in (PDF, ROW):
        for r in split["rotations"]:
            for name in ("train", "calibration", "test"):
                ids = r[f"{name}_ids"]
                for field, declared in r["support"][name].items():
                    actual = Counter(BY_ID[i][field] for i in ids)
                    assert all(declared[c] == actual.get(c, 0) for c in declared)


def test_ids_stay_strings():
    for split in (PDF, ROW):
        assert all(isinstance(i, str) for i in split["canonical_row_order"])
        assert all(isinstance(i, str) for r in split["rotations"] for i in r["test_ids"])


def test_generation_is_deterministic():
    assert build_split("pdf_group", 42, ROWS)["rotations"][0]["test_ids"] == \
        PDF["rotations"][0]["test_ids"]


def test_seeds_produce_different_partitions():
    """Seed-to-seed std is the paper's uncertainty estimate; if the seed did
    not move the split, that estimate would silently narrow."""
    for protocol in ("pdf_group", "row_strat"):
        a = set(_split(protocol, 42)["rotations"][0]["test_ids"])
        b = set(_split(protocol, 123)["rotations"][0]["test_ids"])
        assert a != b
        assert len(a & b) / len(a) < 0.6


def test_absent_calibration_classes_are_reported():
    """Misleading (n=2, in two different PDFs) is absent from most calibration
    partitions by construction; the fallback must be announced, not discovered."""
    triggered = [r for r in PDF["rotations"] if r["calibration_absent_classes"]]
    assert triggered, "expected at least one rotation to need the bias fallback"
    for r in triggered:
        for field, classes in r["calibration_absent_classes"].items():
            calib_present = {BY_ID[i][field] for i in r["calibration_ids"]}
            train_present = {BY_ID[i][field] for i in r["train_ids"]}
            for c in classes:
                assert c not in calib_present and c in train_present
