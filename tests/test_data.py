from paper.data import canonical_row_order, data_checksum, load_dev
from paper.labels import row_to_state_id

ROWS = load_dev()


def test_development_size():
    assert len(ROWS) == 2000


def test_ids_are_strings_and_unique():
    """Contract §1.1: positional alignment is banned, so ids must survive intact."""
    assert all(isinstance(r["id"], str) for r in ROWS)
    assert len(set(r["id"] for r in ROWS)) == 2000


def test_canonical_order_is_train_then_val():
    order = canonical_row_order(ROWS)
    assert order[0] == "10001"
    assert order[1000] == "11001"
    assert order[-1] == "12000"


def test_checksum_is_stable():
    assert data_checksum(ROWS) == data_checksum(load_dev())
    assert data_checksum(ROWS).startswith("sha256:")


def test_every_gold_row_is_a_legal_state():
    """Gold labels must never be hierarchy-invalid; if this fails the label
    space in paper/labels.py is wrong, not the data."""
    assert all(row_to_state_id(r) >= 0 for r in ROWS)


def test_observed_state_coverage():
    """Only 15 of the 17 legal states occur, and Misleading has n=2."""
    observed = {row_to_state_id(r) for r in ROWS}
    assert len(observed) == 15
    assert sum(r["evidence_quality"] == "Misleading" for r in ROWS) == 2
