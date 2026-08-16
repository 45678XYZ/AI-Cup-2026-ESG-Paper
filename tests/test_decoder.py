import numpy as np

from paper.decoder import decode
from paper.labels import EVAL_FIELDS, FIELDS, LABEL2ID, STATES, tuple_to_state_id


def _scores(n=1, fill=-10.0):
    return {field: np.full((n, len(labels)), fill, dtype=float)
            for field, labels in EVAL_FIELDS.items()}


def test_every_canonical_state_decodes_to_itself():
    for state_id, state in enumerate(STATES):
        scores = _scores()
        values = (state.ps, state.vt, state.es, state.eq)
        for field, value in zip(FIELDS, values):
            scores[field][0, LABEL2ID[field][value]] = 10.0
        pred = decode(scores)[0]
        assert tuple_to_state_id(*(pred[f] for f in FIELDS)) == state_id


def test_joint_evidence_can_overturn_a_marginal_parent_argmax():
    scores = _scores(fill=0.0)
    scores["promise_status"][0] = [0.0, 0.1]  # independent choice: No
    scores["verification_timeline"][0, LABEL2ID["verification_timeline"]["already"]] = 3.0
    scores["evidence_status"][0, LABEL2ID["evidence_status"]["Yes"]] = 3.0
    scores["evidence_quality"][0, LABEL2ID["evidence_quality"]["Clear"]] = 3.0
    pred = decode(scores)[0]
    assert pred == {
        "promise_status": "Yes", "verification_timeline": "already",
        "evidence_status": "Yes", "evidence_quality": "Clear",
    }


def test_decoder_outputs_only_legal_states_for_arbitrary_scores():
    rng = np.random.default_rng(7)
    scores = {field: rng.normal(size=(200, len(labels)))
              for field, labels in EVAL_FIELDS.items()}
    assert all(tuple_to_state_id(*(pred[f] for f in FIELDS)) >= 0 for pred in decode(scores))
