"""The English-to-frozen-vocabulary mapping.

The mapping is the whole reason the rest of the study does not have to learn
that a second language exists, so it is worth more assertions than its size
suggests. Two properties carry it: it is reversible, and the one non-field-local
step restates something the *Chinese* annotation already asserts.
"""

import pytest

from paper import labels as zh
from paper import labels_en as en
from paper.data import load_dev


def test_the_chinese_not_applicable_is_already_a_restatement_of_promise_status():
    """Why recovering ``evidence_status`` from ``promise_status`` invents nothing.

    In the frozen annotation ``evidence_status = N/A`` holds on exactly the
    rows where ``promise_status = No`` -- 373 of them, the same 373, no
    exceptions. The mapping therefore reproduces a rule the frozen data already
    obeys rather than imposing one on the English data.

    If this ever stops holding, the mapping's justification is gone and the
    English arm needs its own label space after all.
    """
    rows = load_dev()
    na = {r["id"] for r in rows if r["evidence_status"] == "N/A"}
    no_promise = {r["id"] for r in rows if r["promise_status"] == "No"}
    assert na == no_promise
    assert len(na) == 373


def test_the_mapping_is_reversible_on_every_tuple_the_release_can_hold():
    """Nothing is lost in translation, so the audit can report the data as
    published -- but the domain of that guarantee needs naming.

    ``evidence_status`` is recovered from ``promise_status``, so the round trip
    is exact only where the two agree, which is to say on tuples that obey the
    hierarchy. On ``(No, *, Yes, *)`` -- no promise, yet evidence -- it is not:
    the forward map writes ``N/A`` and the inverse reads back ``No``. That
    tuple is not reversible and must not be, because it is not a thing the
    annotation says; asserting over the full Cartesian product would be
    demanding an inverse for inputs the release cannot contain.
    """
    checked = 0
    for ps in en.NATIVE_FIELDS["promise_status"]:
        for vt in en.NATIVE_FIELDS["verification_timeline"]:
            for es in en.NATIVE_FIELDS["evidence_status"]:
                for eq in en.NATIVE_FIELDS["evidence_quality"]:
                    if ps == "No" and es != "No":
                        continue
                    row = {"promise_status": ps, "verification_timeline": vt,
                           "evidence_status": es, "evidence_quality": eq}
                    assert en.to_native(en.to_canonical(row)) == row
                    checked += 1
    assert checked == 60


def test_the_release_only_contains_tuples_the_round_trip_covers():
    """The exclusion above is not a gap: no row in the release falls in it."""
    from paper.data_en import load_english_native

    native = load_english_native()
    outside = [r["id"] for r in native
               if r["promise_status"] == "No" and r["evidence_status"] != "No"]
    assert outside == []
    for row in native:
        assert en.to_native(en.to_canonical(row))["evidence_status"] == \
            row["evidence_status"], row["id"]


def test_evidence_status_is_the_only_field_whose_map_needs_a_second_field():
    """It is absent from RENAMES on purpose: English ``No`` becomes ``No`` or
    ``N/A`` depending on ``promise_status``, so no field-local entry exists."""
    assert en.CONTEXTUAL_FIELD == "evidence_status"
    assert "evidence_status" not in en.RENAMES
    assert set(en.RENAMES) == set(zh.FIELDS) - {"evidence_status"}

    promise = {"promise_status": "Yes", "verification_timeline": "Already",
               "evidence_status": "No", "evidence_quality": "N/A"}
    assert en.to_canonical(promise)["evidence_status"] == "No"

    no_promise = {**promise, "promise_status": "No",
                  "verification_timeline": "N/A"}
    assert en.to_canonical(no_promise)["evidence_status"] == "N/A"


def test_every_rename_lands_inside_the_frozen_vocabulary():
    for field, mapping in en.RENAMES.items():
        assert set(mapping) == set(en.NATIVE_FIELDS[field]), field
        assert set(mapping.values()) <= set(zh.EVAL_FIELDS[field]), field


def test_an_unrecognised_label_is_refused_rather_than_passed_through():
    """A changed release must stop the run, not reach the frozen space and be
    reported as an illegal tuple the model produced."""
    row = {"promise_status": "Yes", "verification_timeline": "Within 2 years",
           "evidence_status": "Yes", "evidence_quality": "Clear"}
    with pytest.raises(ValueError, match="verification_timeline"):
        en.to_canonical(row)

    row = {"promise_status": "Yes", "verification_timeline": "Already",
           "evidence_status": "Unknown", "evidence_quality": "Clear"}
    with pytest.raises(ValueError, match="evidence_status"):
        en.to_canonical(row)


def test_the_english_arm_can_reach_every_frozen_label():
    """A canonical class no English row could produce would score zero for
    reasons unrelated to the model, and its macro-F1 column would be
    uninterpretable. There are none."""
    assert en.unmapped_canonical_values() == {f: [] for f in zh.FIELDS}


def test_translating_into_the_frozen_space_keeps_the_larger_illegal_region():
    """The direction of the translation is the conservative one.

    English natively admits 2 x 5 x 2 x 4 = 80 combinations against the frozen
    space's 120, so scoring in English would give the method a smaller set of
    ways to be wrong. Mapping into the frozen space rather than out of it means
    the English arm is judged against 103 illegal combinations, not 63.
    """
    frozen = 1
    for labels in zh.EVAL_FIELDS.values():
        frozen *= len(labels)
    native = 1
    for labels in en.NATIVE_FIELDS.values():
        native *= len(labels)
    assert frozen == 120 and native == 80
    assert frozen - len(zh.STATES) == 103
