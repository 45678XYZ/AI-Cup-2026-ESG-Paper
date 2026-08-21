"""Cross-seed aggregation and the paper's pre-specified contrasts.

Two rules from the plan are enforced structurally rather than by discipline:

  * per-fold F1 is never averaged -- each seed is one score over all 2,000
    concatenated test rows, and only those three numbers are averaged
    (section 4.2, rule 7);
  * only the five contrasts named in advance by section 4.4 are tested, and
    Holm is applied across exactly those five. Widening the set and reporting
    what survives is what the plan forbids.
"""

import numpy as np

from analysis.bootstrap import BOOTSTRAP_SEED, N_BOOT, holm, paired_delta
from analysis.load import METHODS, load_all
from analysis.metrics import (
    consistent_weighted_macro_f1,
    field_macro_f1,
    weighted_macro_f1,
)
from paper.data import load_dev
from paper.labels import EVAL_FIELDS, FIELDS, INVALID_STATE_ID, tuple_to_state_id
from paper.train_config import SEEDS

# Plan section 4.4. Adding a row here means re-running Holm over a larger
# family and saying so in the paper.
CONTRASTS = (
    ("M1", "M0", "hierarchy legalisation"),
    ("M3", "M2", "conditional vs global calibration, under projection"),
    ("M4", "M1", "17-state decoding vs projection"),
    ("M6", "M3", "adding the decoder under conditional calibration"),
    ("M6", "M5", "conditional vs global calibration, under decoding"),
)


def mean_std(values):
    """Mean and *sample* std (ddof=1) across seeds.

    ddof=1 because the three seeds are a sample of the pipeline's variability,
    not the population. The caption must say the std is seed-to-seed variation
    of the whole pipeline -- fold assignment and training together -- and not
    model stability (plan section 4.2).
    """
    arr = np.asarray(values, dtype=float)
    return float(arr.mean()), (float(arr.std(ddof=1)) if arr.size > 1 else 0.0)


def misleading_free_index(order, dev):
    """Row positions excluding the two gold ``Misleading`` paragraphs.

    The only statistical treatment the plan permits for a class with n=2: a
    sensitivity metric computed without it. No significance claim about
    ``Misleading`` itself is allowed.
    """
    by_id = {r["id"]: r for r in dev}
    return np.array(
        [i for i, row_id in enumerate(order)
         if by_id[row_id]["evidence_quality"] != "Misleading"],
        dtype=np.int64,
    )


def _state_ids(codes):
    """Per-row state_id for an ``(N, 4)`` class-code array; -1 when illegal."""
    names = [[EVAL_FIELDS[f][c] for f, c in zip(FIELDS, row)] for row in codes]
    return np.array([tuple_to_state_id(*n) for n in names], dtype=np.int64)


def method_summary(sets_by_seed, idx=None, no_misleading_idx=None) -> dict:
    """Aggregate one method across seeds.

    ``sets_by_seed`` is a list of ``(gold, pred)`` arrays, one per seed. Each
    entry is already a single score over all 2,000 concatenated test rows, so
    averaging them is a seed average and never a fold average.
    """
    scores = [weighted_macro_f1(g, p, idx) for g, p in sets_by_seed]
    mean, std = mean_std(scores)

    per_field = {}
    for field in FIELDS:
        values = [field_macro_f1(g, p, idx)[field] for g, p in sets_by_seed]
        per_field[field] = mean_std(values)[0]

    tuple_acc, invalid = [], []
    for gold, pred in sets_by_seed:
        gold_ids, pred_ids = _state_ids(gold), _state_ids(pred)
        sel = slice(None) if idx is None else idx
        tuple_acc.append(float((gold_ids[sel] == pred_ids[sel]).mean()))
        invalid.append(float((pred_ids[sel] == INVALID_STATE_ID).mean()))

    # The secondary metric, per method. It has no column of its own in Table 2
    # -- the column count is D's page budget -- so this is where the caption
    # gets the one number that locates it: how far the unconstrained arm falls
    # when its ancestor-unsupported fields stop earning partial credit.
    constrained = [consistent_weighted_macro_f1(g, p, idx) for g, p in sets_by_seed]

    out = {
        "weighted_macro_f1_per_seed": scores,
        "weighted_macro_f1_mean": mean,
        "weighted_macro_f1_std": std,
        "consistent_weighted_macro_f1_mean": float(np.mean(constrained)),
        "per_field_mean": per_field,
        "tuple_exact_match_mean": float(np.mean(tuple_acc)),
        "invalid_tuple_rate_mean": float(np.mean(invalid)),
    }
    if no_misleading_idx is not None:
        sensitivity = [weighted_macro_f1(g, p, no_misleading_idx)
                       for g, p in sets_by_seed]
        out["weighted_macro_f1_mean_no_misleading"] = float(np.mean(sensitivity))
    return out


def protocol_summary(protocol, order, root, clusters, n_boot=N_BOOT,
                     seeds=SEEDS, dev=None, bootstrap_seed=BOOTSTRAP_SEED) -> dict:
    """Everything Table 2 needs for one protocol."""
    dev = dev if dev is not None else load_dev()
    by_seed = {seed: load_all(protocol, seed, order, root) for seed in seeds}
    no_mis = misleading_free_index(order, dev)

    methods = {
        method: method_summary(
            [by_seed[seed][method] for seed in seeds], no_misleading_idx=no_mis,
        )
        for method in METHODS
    }

    def family(score):
        """One Holm family: the same pre-specified contrasts, one metric.

        The two metrics answer different questions -- weighted macro-F1 is the
        competition's ranking rule, its path-constrained variant asks the same
        question of predictions the hierarchy actually supports -- so they are
        corrected separately. Pooling them into a family of ten would treat them
        as ten attempts at one question and inflate the correction against
        contrasts that were specified once, not twice.

        The secondary metric was chosen to differ from the primary one in
        exactly one respect, so a contrast the two families resolve differently
        is evidence about consistency and not about the shape of the metric.
        """
        rows = {}
        for a, b, description in CONTRASTS:
            rows[f"{a}-{b}"] = {
                **paired_delta(
                    [by_seed[s][a] for s in seeds], [by_seed[s][b] for s in seeds],
                    clusters, n_boot=n_boot, seed=bootstrap_seed, score=score,
                ),
                "description": description,
            }
        adjusted = holm({key: row["p_value"] for key, row in rows.items()})
        for key, row in rows.items():
            row["p_holm"] = adjusted[key]
        return rows

    return {"protocol": protocol, "seeds": list(seeds), "methods": methods,
            "contrasts": family(weighted_macro_f1),
            "consistent_contrasts": family(consistent_weighted_macro_f1)}


def regime_comparison(summaries, order, root, clusters, n_boot=N_BOOT,
                      seeds=SEEDS, bootstrap_seed=BOOTSTRAP_SEED) -> dict:
    """Table 3: same-document (row_strat) against document-disjoint (pdf_group).

    The three reported rows are M0 plus the best-scoring calibrated projection
    (M2/M3) and the best-scoring valid-state decoder (M5/M6), selected on the
    document-disjoint protocol so the choice is made once rather than per
    column. Delta is the gap between two estimation targets, **not** a bias
    estimate -- the caption must say so.
    """
    disjoint = summaries["pdf_group"]["methods"]
    best_projection = max(("M2", "M3"),
                          key=lambda m: disjoint[m]["weighted_macro_f1_mean"])
    best_decoder = max(("M5", "M6"),
                       key=lambda m: disjoint[m]["weighted_macro_f1_mean"])

    rows = {}
    for label, method in (("M0", "M0"),
                          ("Best calibrated projection", best_projection),
                          ("Best valid-state decoder", best_decoder)):
        same = [load_all("row_strat", s, order, root, methods=(method,))[method]
                for s in seeds]
        disj = [load_all("pdf_group", s, order, root, methods=(method,))[method]
                for s in seeds]
        delta = paired_delta(same, disj, clusters, n_boot=n_boot,
                             seed=bootstrap_seed)
        rows[label] = {
            "method": method,
            "same_document": float(np.mean([weighted_macro_f1(g, p) for g, p in same])),
            "document_disjoint": float(np.mean([weighted_macro_f1(g, p) for g, p in disj])),
            **delta,
        }
    return rows
