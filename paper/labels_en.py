"""The ML-Promise English vocabulary, and its mapping onto the frozen space.

``paper/labels.py`` is frozen by interface contract 1.3, and twenty-two modules
import from it. Giving English a second label space would mean threading a
label module through every one of them -- the decision rules, the calibrators,
the validator, the analysis -- to serve four hundred rows. This module takes
the other route: the English release is translated into the frozen vocabulary
at load time, and the rest of the study does not learn that a second language
exists.

**This is a change of convention, not of content.** The two releases annotate
the same four questions and differ in how they spell the answers:

    verification_timeline   Already -> already, Less than 2 years ->
                            within_2_years, and so on. Pure renaming.

    evidence_status         English has no ``N/A``. Where the Chinese
                            annotation writes ``N/A`` -- "there is no promise,
                            so the evidence question does not arise" -- the
                            English writes ``No``, the same token it uses for
                            "a promise with no evidence".

The second one is the only mapping that is not field-local, and the reason it
is safe is a property of the *Chinese* data: there, ``evidence_status = N/A``
holds on exactly the 373 rows where ``promise_status = No``, and on no others.
The Chinese ``N/A`` is therefore already a restatement of ``promise_status``,
so recovering it from ``promise_status`` here invents nothing that the frozen
annotation does not already assert. The map is bijective given
``promise_status`` and ``to_native`` inverts it exactly.

What it costs, measured rather than asserted: scoring the Chinese M0
predictions under the English convention instead would move the invalid-tuple
rate from 12.55% to 12.35%. Two violation modes shrink -- ``promise_status =
Yes`` with ``evidence_status = N/A`` cannot be expressed, and ``(No, N/A, No,
N/A)`` is legal in English and illegal in Chinese. 0.2 points, well under every
effect the study reports, and it goes the other way from the direction that
would flatter the method. Translating *into* the frozen space rather than out
of it means the English arm carries the larger illegal region, not the smaller.
"""

from paper.labels import EVAL_FIELDS as CANONICAL_FIELDS
from paper.labels import FIELDS

# The release's own vocabulary, recorded so the audit can report the data as
# published rather than as translated. Nothing computes from this except the
# mapping's own tests.
NATIVE_FIELDS = {
    "promise_status":        ["Yes", "No"],
    "verification_timeline": ["Already", "Less than 2 years", "2 to 5 years",
                              "More than 5 years", "N/A"],
    "evidence_status":       ["Yes", "No"],
    "evidence_quality":      ["Clear", "Not Clear", "Misleading", "N/A"],
}

# Field-local renamings. ``evidence_status`` is deliberately absent: its
# English ``No`` maps to two different canonical values depending on
# ``promise_status``, so there is no field-local entry that could be written.
RENAMES = {
    "promise_status": {"Yes": "Yes", "No": "No"},
    "verification_timeline": {
        "Already": "already",
        "Less than 2 years": "within_2_years",
        "2 to 5 years": "between_2_and_5_years",
        "More than 5 years": "more_than_5_years",
        "N/A": "N/A",
    },
    "evidence_quality": {
        "Clear": "Clear", "Not Clear": "Not Clear",
        "Misleading": "Misleading", "N/A": "N/A",
    },
}

# The one field whose value depends on a second field.
CONTEXTUAL_FIELD = "evidence_status"


def to_canonical(row: dict) -> dict:
    """One release row's four labels, in the frozen vocabulary.

    Raises on an unknown value rather than passing it through: a label this
    module has not seen is a changed release, and letting it reach the frozen
    space would produce an illegal tuple attributed to the model.
    """
    out = dict(row)
    for field, mapping in RENAMES.items():
        value = row[field]
        if value not in mapping:
            raise ValueError(
                f"{field}={value!r} is not a value of the English release this "
                f"mapping was written against ({sorted(mapping)}). The release "
                "has changed; update paper/labels_en.py before loading it."
            )
        out[field] = mapping[value]

    evidence_status = row[CONTEXTUAL_FIELD]
    if evidence_status not in NATIVE_FIELDS[CONTEXTUAL_FIELD]:
        raise ValueError(
            f"{CONTEXTUAL_FIELD}={evidence_status!r} is not one of "
            f"{NATIVE_FIELDS[CONTEXTUAL_FIELD]}"
        )
    out[CONTEXTUAL_FIELD] = "N/A" if row["promise_status"] == "No" else evidence_status
    return out


def to_native(row: dict) -> dict:
    """The inverse of :func:`to_canonical`, for round-trip checking."""
    inverse = {f: {v: k for k, v in m.items()} for f, m in RENAMES.items()}
    out = dict(row)
    for field, mapping in inverse.items():
        out[field] = mapping[row[field]]
    out[CONTEXTUAL_FIELD] = (
        "No" if row[CONTEXTUAL_FIELD] == "N/A" else row[CONTEXTUAL_FIELD]
    )
    return out


def unmapped_canonical_values() -> dict:
    """Canonical labels no English row can produce.

    Empty for three fields. ``verification_timeline`` and ``evidence_quality``
    are fully covered, which is what lets the two studies' per-class tables be
    read side by side. Reported by the audit rather than assumed, because a
    class the English arm can never predict is a class its macro-F1 scores at
    zero for reasons that have nothing to do with the model.
    """
    reachable = {f: set(m.values()) for f, m in RENAMES.items()}
    reachable[CONTEXTUAL_FIELD] = set(NATIVE_FIELDS[CONTEXTUAL_FIELD]) | {"N/A"}
    return {
        field: sorted(set(CANONICAL_FIELDS[field]) - reachable[field])
        for field in FIELDS
    }
