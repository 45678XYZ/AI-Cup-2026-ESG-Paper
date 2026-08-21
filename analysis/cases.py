"""Why removing every invalid tuple barely moves weighted macro-F1.

Table 2 reports two things that look contradictory: structured decoding takes
the invalid rate from 12.6% to zero, and leaves the official metric alone. The
counts here are what reconcile them, and they are counts rather than prose
because plan section 7 covers this file too -- no number is transcribed.

Two questions, both answered per (protocol, seed):

* **Which rule does the model break?** The four fields are predicted by four
  independent heads, so nothing stops a head from filling in a branch its
  parent already closed. Naming the rule turns "12.6% invalid" into a failure
  mode a reader can picture.
* **What does the repair cost?** Projection overwrites a child field whenever
  the parent forbids it. When the parent is right that repairs a field; when
  the parent is wrong it destroys one. Reporting only the net would hide that
  both happen, and it is the mix -- not the net -- that explains why a metric
  computed field by field stays flat.
"""

import json
from pathlib import Path

import numpy as np

from analysis.load import load_aligned
from paper.data import load_dev
from paper.labels import FIELDS, ID2LABEL, is_valid_tuple
from paper.provenance import git_sha, now_iso
from paper.train_config import PROTOCOLS, SEEDS

BASELINE = "M0"          # the only arm that can emit an illegal tuple
PROJECTED = "M1"         # same probabilities, projection applied, no calibration

# Ordered: each row is attributed to the first rule it breaks, so the counts
# partition the invalid rows instead of double-counting a row that breaks two.
RULES = (
    ("promise_no_children_set",
     lambda ps, vt, es, eq: ps == "No" and (vt, es, eq) != ("N/A", "N/A", "N/A")),
    ("promise_yes_children_absent",
     lambda ps, vt, es, eq: ps == "Yes" and (vt == "N/A" or es == "N/A")),
    ("evidence_no_quality_set",
     lambda ps, vt, es, eq: es == "No" and eq != "N/A"),
    ("evidence_yes_quality_missing",
     lambda ps, vt, es, eq: es == "Yes" and eq == "N/A"),
)


def _labels(row):
    return tuple(ID2LABEL[field][int(row[j])] for j, field in enumerate(FIELDS))


def violation_breakdown(pred) -> dict:
    """How many rows are illegal, and which rule each one breaks."""
    pred = np.asarray(pred)
    counts = {name: 0 for name, _ in RULES}
    n_invalid = 0
    for i in range(len(pred)):
        ps, vt, es, eq = _labels(pred[i])
        if is_valid_tuple(ps, vt, es, eq):
            continue
        n_invalid += 1
        for name, breaks in RULES:
            if breaks(ps, vt, es, eq):
                counts[name] += 1
                break
    return {
        "n_rows": int(len(pred)),
        "n_invalid": n_invalid,
        "invalid_rate": n_invalid / len(pred) if len(pred) else 0.0,
        "by_rule": counts,
    }


def projection_ledger(gold, before, after) -> dict:
    """What the repair costs, field by field, on the rows it touches.

    ``wrong_either_way`` matters: a field can be overwritten from one wrong
    label to another, which changes the prediction without changing the score.
    Folding it into either column would overstate the rule's effect.
    """
    gold, before, after = np.asarray(gold), np.asarray(before), np.asarray(after)
    repaired = destroyed = either_way = rows_touched = 0
    for i in range(len(gold)):
        touched = False
        for j in range(len(FIELDS)):
            was, now, truth = int(before[i, j]), int(after[i, j]), int(gold[i, j])
            if was == now:
                continue
            touched = True
            if now == truth:
                repaired += 1
            elif was == truth:
                destroyed += 1
            else:
                either_way += 1
        rows_touched += touched
    return {
        "rows_touched": rows_touched,
        "fields_repaired": repaired,
        "fields_destroyed": destroyed,
        "fields_wrong_either_way": either_way,
        "net_fields": repaired - destroyed,
    }


def case_analysis(protocol, seed, order, root, dev=None,
                  baseline=BASELINE, projected=PROJECTED) -> dict:
    """The qualitative record for one (protocol, seed)."""
    if baseline != BASELINE:
        raise ValueError(
            f"only {BASELINE} can emit an invalid tuple; {baseline} is legal by "
            "construction, so a violation breakdown of it is meaningless"
        )
    dev = dev if dev is not None else load_dev()
    gold, raw = load_aligned(protocol, seed, baseline, order, root)
    _, projected_pred = load_aligned(protocol, seed, projected, order, root)
    return {
        "protocol": protocol,
        "seed": seed,
        "baseline": baseline,
        "projected": projected,
        "independent": violation_breakdown(raw),
        "after_projection": violation_breakdown(projected_pred),
        "projection": projection_ledger(gold, raw, projected_pred),
    }


def write_case_analysis(out_dir, order, root, dev=None,
                        protocols=PROTOCOLS, seeds=SEEDS) -> Path:
    """Emit ``case_analysis.json`` for every (protocol, seed).

    Supplementary rather than contract-4: the frozen deliverable list in §5 is
    the three tabulars, their captions and the manifest. This sits beside them
    because D writes the Discussion from it, and because a rule breakdown that
    lives only in a console is not evidence anyone can re-check.
    """
    dev = dev if dev is not None else load_dev()
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    runs = [case_analysis(protocol, seed, order, root, dev=dev)
            for protocol in protocols for seed in seeds]

    totals = {
        "n_rows": sum(r["independent"]["n_rows"] for r in runs),
        "n_invalid": sum(r["independent"]["n_invalid"] for r in runs),
        "by_rule": {
            name: sum(r["independent"]["by_rule"][name] for r in runs)
            for name, _ in RULES
        },
        "fields_repaired": sum(r["projection"]["fields_repaired"] for r in runs),
        "fields_destroyed": sum(r["projection"]["fields_destroyed"] for r in runs),
        "fields_wrong_either_way": sum(
            r["projection"]["fields_wrong_either_way"] for r in runs),
    }
    totals["net_fields"] = totals["fields_repaired"] - totals["fields_destroyed"]
    totals["invalid_rate"] = (totals["n_invalid"] / totals["n_rows"]
                              if totals["n_rows"] else 0.0)

    path = out_dir / "case_analysis.json"
    with open(path, "w", encoding="utf-8") as f:
        json.dump({"generated_at": now_iso(), "git_sha": git_sha(),
                   "totals": totals, "runs": runs}, f,
                  ensure_ascii=False, indent=1)
    return path
