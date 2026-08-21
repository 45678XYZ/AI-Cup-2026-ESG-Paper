"""Paired PDF-cluster bootstrap for method contrasts (plan section 4.5).

Paragraphs from one report are not independent -- they share an author, a
template and a topic -- so the resampling unit is the PDF, not the row. Two
choices below are load-bearing:

  * **One resample per iteration, applied to all three seeds.** The seeds share
    the same 2,000 rows and the same 49 reports and differ only in how folds
    were drawn, so a shared resample keeps the pairing as tight as possible and
    matches the plan's "compute the difference on the same resample, then
    average the differences across seeds".
  * **Both methods scored on the same resample.** The present-labels-only
    convention makes the scored class set depend on which rows were drawn;
    because gold is identical for both arms of a pair, both are scored over the
    same classes and the difference stays a like-for-like comparison.
"""

import numpy as np

from analysis.metrics import weighted_macro_f1

N_BOOT = 10_000

# Fixed so a rerun reproduces the published intervals exactly. Changing it
# after the 8/23 results freeze changes every CI in the paper.
BOOTSTRAP_SEED = 20260814

CI_PERCENTILES = (2.5, 97.5)


def resample_indices(clusters, rng) -> np.ndarray:
    """Row positions from a with-replacement draw of whole PDFs."""
    picked = rng.integers(0, len(clusters), size=len(clusters))
    return np.concatenate([clusters[i] for i in picked])


def _mean_difference(sets_a, sets_b, idx=None, score=weighted_macro_f1) -> float:
    return float(np.mean([
        score(gold_a, pred_a, idx) - score(gold_b, pred_b, idx)
        for (gold_a, pred_a), (gold_b, pred_b) in zip(sets_a, sets_b)
    ]))


def paired_delta(sets_a, sets_b, clusters, n_boot=N_BOOT, seed=BOOTSTRAP_SEED,
                 score=weighted_macro_f1) -> dict:
    """Bootstrap the seed-averaged difference ``A - B`` in weighted macro-F1.

    ``sets_a`` / ``sets_b`` are lists of ``(gold, pred)`` arrays, one per seed,
    all aligned to the same row order by ``analysis.load.load_aligned``.
    """
    if len(sets_a) != len(sets_b):
        raise ValueError("the two method sets cover different numbers of seeds")
    for (gold_a, _), (gold_b, _) in zip(sets_a, sets_b):
        if not np.array_equal(gold_a, gold_b):
            raise ValueError("paired arms disagree on gold; the rows are not aligned")

    rng = np.random.default_rng(seed)
    draws = np.empty(n_boot, dtype=float)
    for b in range(n_boot):
        draws[b] = _mean_difference(sets_a, sets_b,
                                    resample_indices(clusters, rng), score)

    low, high = np.percentile(draws, CI_PERCENTILES)
    return {
        "delta": _mean_difference(sets_a, sets_b, score=score),
        "ci_low": float(low),
        "ci_high": float(high),
        "p_value": _two_sided_p(draws),
        "n_boot": n_boot,
        "per_seed_delta": [
            weighted_macro_f1(gold_a, pred_a) - weighted_macro_f1(gold_b, pred_b)
            for (gold_a, pred_a), (gold_b, pred_b) in zip(sets_a, sets_b)
        ],
    }


def _two_sided_p(draws) -> float:
    """Two-sided bootstrap p, smoothed so it is never reported as exactly 0.

    ``(1 + count) / (1 + B)`` keeps the value at or above ``1/(B+1)``: 10,000
    resamples cannot support a claim finer than that, and a printed ``p = 0``
    would imply one.
    """
    n = len(draws)
    at_or_below = (1 + int((draws <= 0).sum())) / (n + 1)
    at_or_above = (1 + int((draws >= 0).sum())) / (n + 1)
    return float(min(1.0, 2 * min(at_or_below, at_or_above)))


def holm(pvalues: dict) -> dict:
    """Holm-Bonferroni adjusted p-values for the pre-specified contrasts.

    Applied only to the contrasts plan section 4.4 names in advance. Running it
    over a wider set of post-hoc comparisons and reporting whatever survives is
    the practice the plan explicitly forbids.
    """
    ordered = sorted(pvalues.items(), key=lambda kv: kv[1])
    n, running, adjusted = len(ordered), 0.0, {}
    for i, (key, p) in enumerate(ordered):
        running = max(running, (n - i) * p)
        adjusted[key] = min(1.0, running)
    return adjusted
