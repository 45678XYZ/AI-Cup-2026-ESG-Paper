import json
from dataclasses import asdict, astuple

from contracts.export_states import build as build_states_export
from paper.data import REPO_ROOT
from paper.labels import (
    EVAL_FIELDS,
    INVALID_STATE_ID,
    STATE_TO_ID,
    STATES,
    is_valid_tuple,
    tuple_to_state_id,
)


def test_exactly_seventeen_states():
    assert len(STATES) == 17
    assert len(STATE_TO_ID) == 17


def test_canonical_order_is_frozen():
    """state_id 0 and 16 are cited in contract artifacts; pin both ends."""
    assert astuple(STATES[0]) == ("No", "N/A", "N/A", "N/A", "ps_no")
    assert astuple(STATES[16]) == ("Yes", "more_than_5_years", "Yes", "Misleading", "es_yes")
    assert tuple_to_state_id("No", "N/A", "N/A", "N/A") == 0


def test_hierarchy_rules_hold_for_every_state():
    for s in STATES:
        if s.ps == "No":
            assert (s.vt, s.es, s.eq) == ("N/A", "N/A", "N/A")
        else:
            assert s.vt != "N/A"
            assert s.es in ("Yes", "No")
            assert (s.eq == "N/A") == (s.es == "No")


def test_illegal_tuples_are_rejected():
    # PS=No with a substantive timeline
    assert tuple_to_state_id("No", "already", "N/A", "N/A") == INVALID_STATE_ID
    # PS=Yes with an N/A timeline
    assert tuple_to_state_id("Yes", "N/A", "Yes", "Clear") == INVALID_STATE_ID
    # ES=No with a substantive quality
    assert not is_valid_tuple("Yes", "already", "No", "Clear")


def test_label_order_matches_probability_columns():
    """Column j of every .npy is EVAL_FIELDS[field][j] (contract §1.3)."""
    assert EVAL_FIELDS["promise_status"] == ["Yes", "No"]
    assert EVAL_FIELDS["evidence_quality"] == ["Clear", "Not Clear", "Misleading", "N/A"]
    assert EVAL_FIELDS["verification_timeline"][-1] == "N/A"


def test_states_json_matches_its_source():
    """contracts/states.json is a published copy of this module; this assertion
    is what stops the copy from drifting."""
    with open(REPO_ROOT / "contracts" / "states.json", encoding="utf-8") as f:
        published = json.load(f)
    assert published == build_states_export()
    assert published["n_states"] == 17
    assert [s["state_id"] for s in published["states"]] == list(range(17))
    assert published["states"][0] == {"state_id": 0, **asdict(STATES[0])}
