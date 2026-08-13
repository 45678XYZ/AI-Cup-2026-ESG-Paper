"""The seven decision rules of the controlled comparison.

M0-M6 differ in exactly two respects — how the class biases are estimated, and
how the four fields are turned into one label tuple — and in nothing else. They
consume identical probabilities on identical rows, so a difference between two
of them is attributable to the decision stage rather than to the model.

Everything here operates on **scores**, not probabilities:

    s_t,c(x) = log p_t,c(x) + b_t,c

which is the form the plan defines the bias in, and the form the 17-state
decoder needs (a state's score is the sum of its four field scores; that sum is
only meaningful in log space). Because ``log`` is monotone and the bias is zero
for M0 and M1, feeding scores rather than probabilities to the argmax rules
cannot change their output — asserted in ``tests/test_methods.py`` rather than
assumed, since it is what makes M0 comparable to everything else.
"""

from dataclasses import dataclass

import numpy as np

from paper.labels import FIELDS
from paper.projection import independent_argmax, project

# A probability that underflowed to exactly zero in float32 still has to yield a
# finite score, or one impossible class would drag a whole state's sum to -inf
# and the decoder could never choose it. Flooring at the smallest value the
# input format can represent keeps that from happening without introducing a
# tunable constant: it is a property of float32, not a choice.
#
# Real softmax output from a 24-layer encoder underflows routinely. The
# synthetic fixtures never do -- their noise floor is bounded away from zero --
# so this path is exercised by tests rather than discovered on B's data.
PROB_FLOOR = float(np.finfo(np.float32).tiny)


@dataclass(frozen=True)
class Method:
    """One cell of the factorial comparison (plan §3.4)."""

    method_id: str
    calibration: str | None   # None | "global" | "conditional"
    output_rule: str          # "independent" | "projection" | "decoder"

    @property
    def guarantees_valid_state(self) -> bool:
        return self.output_rule != "independent"


METHODS = {
    m.method_id: m
    for m in (
        Method("M0", None, "independent"),
        Method("M1", None, "projection"),
        Method("M2", "global", "projection"),
        Method("M3", "conditional", "projection"),
        Method("M4", None, "decoder"),
        Method("M5", "global", "decoder"),
        Method("M6", "conditional", "decoder"),
    )
}

METHOD_IDS = list(METHODS)


def log_scores(probs, biases=None) -> dict:
    """``log p + b`` for every field, in float64.

    ``probs`` maps each field to an ``(N, C)`` array; ``biases`` maps each field
    to a length-C vector, or is None for the uncalibrated methods. The result is
    float64 because the decoder sums four of these per state and the inputs are
    float32 -- accumulating in the input precision would let rounding decide
    between two close states.
    """
    out = {}
    for field in FIELDS:
        p = np.asarray(probs[field], dtype=np.float64)
        s = np.log(np.maximum(p, PROB_FLOOR))
        if biases is not None:
            b = np.asarray(biases[field], dtype=np.float64)
            if b.shape != (p.shape[1],):
                raise ValueError(
                    f"{field}: bias has shape {b.shape}, expected {(p.shape[1],)}"
                )
            s = s + b
        out[field] = s
    return out


def decide(method, probs, biases=None) -> list[dict]:
    """Apply one method's output rule, returning field->label per row.

    ``biases`` must be None exactly when the method is uncalibrated; passing
    biases to M0/M1/M4 would quietly turn them into a different method, and the
    results table would not show it.
    """
    if isinstance(method, str):
        method = METHODS[method]
    if (biases is None) != (method.calibration is None):
        raise ValueError(
            f"{method.method_id} has calibration={method.calibration!r} but "
            f"biases were {'not ' if biases is None else ''}supplied"
        )

    scores = log_scores(probs, biases)
    if method.output_rule == "independent":
        return independent_argmax(scores)
    if method.output_rule == "projection":
        return project(scores)
    if method.output_rule == "decoder":
        from paper.decoder import decode_valid_states

        return decode_valid_states(scores)
    raise ValueError(f"unknown output rule: {method.output_rule!r}")
