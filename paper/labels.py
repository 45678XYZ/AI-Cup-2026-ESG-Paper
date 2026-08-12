"""Frozen label enumerations and the canonical 17-state space.

Interface contract §1.3 freezes everything in this module. The column order of
every probability array and every ``state_id`` in every contract artifact is
defined here. Changing any ordering is a major contract version bump.
"""

from dataclasses import dataclass

# Column order of the .npy probability arrays: index j <-> EVAL_FIELDS[field][j]
EVAL_FIELDS = {
    "promise_status":        ["Yes", "No"],
    "verification_timeline": ["already", "within_2_years", "between_2_and_5_years", "more_than_5_years", "N/A"],
    "evidence_status":       ["Yes", "No", "N/A"],
    "evidence_quality":      ["Clear", "Not Clear", "Misleading", "N/A"],
}

# Official AI CUP weighting of the four field-level macro-F1 scores.
FIELD_WEIGHTS = {
    "promise_status":        0.20,
    "verification_timeline": 0.15,
    "evidence_status":       0.30,
    "evidence_quality":      0.35,
}

FIELDS = list(EVAL_FIELDS)

LABEL2ID = {f: {lab: i for i, lab in enumerate(labs)} for f, labs in EVAL_FIELDS.items()}
ID2LABEL = {f: {i: lab for i, lab in enumerate(labs)} for f, labs in EVAL_FIELDS.items()}
NUM_LABELS = {f: len(labs) for f, labs in EVAL_FIELDS.items()}

# Short field aliases used as column prefixes in predictions CSVs (contract §4.1).
FIELD_ALIAS = {
    "promise_status": "ps",
    "verification_timeline": "vt",
    "evidence_status": "es",
    "evidence_quality": "eq",
}


@dataclass(frozen=True)
class State:
    ps: str
    vt: str
    es: str
    eq: str
    kind: str  # ps_no | es_no | es_yes


def build_states() -> list[State]:
    """The 17 label tuples permitted by the task hierarchy.

    1 (PS=No) + 4 timelines x (1 ES=No + 3 evidence qualities) = 17.
    Emission order is the canonical ``state_id`` order and must not change.
    """
    states = [State("No", "N/A", "N/A", "N/A", "ps_no")]
    vt_labels = [x for x in EVAL_FIELDS["verification_timeline"] if x != "N/A"]
    eq_labels = [x for x in EVAL_FIELDS["evidence_quality"] if x != "N/A"]
    for vt in vt_labels:
        states.append(State("Yes", vt, "No", "N/A", "es_no"))
        for eq in eq_labels:
            states.append(State("Yes", vt, "Yes", eq, "es_yes"))
    return states


STATES = build_states()

# (ps, vt, es, eq) -> state_id; an absent tuple means the row is hierarchy-invalid.
STATE_TO_ID = {(s.ps, s.vt, s.es, s.eq): i for i, s in enumerate(STATES)}

INVALID_STATE_ID = -1


def tuple_to_state_id(ps: str, vt: str, es: str, eq: str) -> int:
    """Return the canonical state_id, or INVALID_STATE_ID for illegal tuples."""
    return STATE_TO_ID.get((ps, vt, es, eq), INVALID_STATE_ID)


def row_to_state_id(row: dict) -> int:
    return tuple_to_state_id(
        row["promise_status"],
        row["verification_timeline"],
        row["evidence_status"],
        row["evidence_quality"],
    )


def is_valid_tuple(ps: str, vt: str, es: str, eq: str) -> bool:
    return (ps, vt, es, eq) in STATE_TO_ID
